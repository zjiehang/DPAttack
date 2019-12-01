import math
from collections import defaultdict
from contextlib import contextmanager
from typing import Optional, Union

import torch
from tabulate import tabulate
from torch.utils.data import DataLoader

from dpattack.libs.luna import (Aggregator, CherryPicker, TrainingStopObserver,
                                as_table, cast_list, create_folder_for_file,
                                fetch_best_ckpt_name, idx_to_msk,
                                log_config, ram_pop, ram_write, show_mean_std,
                                time, time_stamp)
from dpattack.libs.luna.public import auto_create
from dpattack.models import PosTagger, WordParser, WordTagParser
from dpattack.models.tagger import PosTagger
from dpattack.task import ParserTask
from dpattack.utils.corpus import Corpus
from dpattack.utils.data import TextDataset, collate_fn
from dpattack.utils.embedding_searcher import (EmbeddingSearcher, cos_dist,
                                               euc_dist)
from dpattack.utils.metric import Metric, ParserMetric
from dpattack.utils.parser_helper import load_parser
from dpattack.utils.tag_tool import gen_tag_dict, train_gram_tagger
from dpattack.utils.vocab import Vocab
import random
from config import Config
from functools import lru_cache
from nltk import TrigramTagger, CRFTagger
from .hack_util import HACK_TAGS, v
from dpattack.libs.nlpaug.augmenter.word import ContextualWordEmbsAug


class IHack:
    def __init__(self):
        self.config: Config

        self.train_corpus: Corpus
        self.corpus: Corpus

        self.task: ParserTask

        self.__nn_tagger: PosTagger = None
        self.__trigram_tagger: TrigramTagger = None
        self.__crf_tagger: CRFTagger = None
        self.__tag_dict: dict = None
        self.__bert_aug: ContextualWordEmbsAug = None

        self.embed_searcher: EmbeddingSearcher

        self.loader: DataLoader

    @property
    def nn_tagger(self) -> PosTagger:
        if self.__nn_tagger is None:
            self.__nn_tagger = PosTagger.load(
                fetch_best_ckpt_name(self.config.tagger_model))
        return self.__nn_tagger

    @property
    def trigram_tagger(self) -> TrigramTagger:
        if self.__trigram_tagger is None:
            self.__trigram_tagger = auto_create("trigram_tagger",
                                                lambda: train_gram_tagger(
                                                    self.train_corpus, ngram=3),
                                                cache=True, path=self.config.workspace + '/saved_vars')
        return self.__trigram_tagger

    @property
    def crf_tagger(self) -> CRFTagger:
        if self.__crf_tagger is None:
            self.__crf_tagger = CRFTagger()
            self.__crf_tagger.set_model_file(self.config.crf_tagger_path)
        return self.__crf_tagger

    @property
    def tag_dict(self) -> dict:
        if self.__tag_dict is None:
            self.__tag_dict = auto_create("tagdict3",
                                          lambda: gen_tag_dict(
                                              self.train_corpus, self.vocab, 3, False),
                                          cache=True, path=self.config.workspace + '/saved_vars')
            self.__tag_dict = {k: torch.tensor(v)
                               for k, v in self.tag_dict.items()}
        return self.__tag_dict

    @property
    def bert_aug(self) -> ContextualWordEmbsAug:
        if self.__bert_aug is None:
            self.__bert_aug = ContextualWordEmbsAug(model_path=self.config.path,
                                                    top_k=512)
        return self.__bert_aug

    @property
    def vocab(self) -> Vocab:
        return self.task.vocab

    @property
    def parser(self) -> Union[WordTagParser, WordParser]:
        return self.task.model

    def init_logger(self, config):
        if config.logf == 'on':
            if config.hk_use_worker == 'on':
                worker_info = "-{}@{}".format(config.hk_num_worker,
                                              config.hk_worker_id)
            else:
                worker_info = ""
            log_config('{}-{}-{}{}'.format(config.mode,
                                           config.input, config.hk_tag_type, worker_info),
                       log_path=config.workspace,
                       default_target='cf')
            from dpattack.libs.luna import log
        else:
            log = print

        log('[General Settings]')
        log(config)
        log('[Hack Settings]')
        for arg in config.kwargs:
            if arg.startswith('hk'):
                log(arg, '\t', config.kwargs[arg])
        log('------------------')

    def setup(self, config):
        self.config = config

        print("Load the models")
        vocab = torch.load(config.vocab)  # type: Vocab
        parser = load_parser(fetch_best_ckpt_name(config.parser_model))

        self.task = ParserTask(vocab, parser)

        print("Load the dataset")

        self.train_corpus = Corpus.load(config.ftrain)

        if config.hk_training_set == 'on':
            self.corpus = self.train_corpus
        else:
            self.corpus = Corpus.load(config.fdata)
        dataset = TextDataset(vocab.numericalize(self.corpus, True))
        # set the data loader
        self.loader = DataLoader(dataset=dataset,
                                 collate_fn=collate_fn)

        def embed_backward_hook(module, grad_in, grad_out):
            ram_write('embed_grad', grad_out[0])

        self.parser.embed.register_backward_hook(embed_backward_hook)
        self.parser.eval()

        self.embed_searcher = EmbeddingSearcher(
            embed=self.parser.embed.weight,
            idx2word=lambda x: self.vocab.words[x],
            word2idx=lambda x: self.vocab.word_dict[x]
        )

        random.seed(1)
        torch.manual_seed(1)

    def hack(self, instance, **kwargs):
        raise NotImplementedError

    @lru_cache(maxsize=None)
    def __gen_tag_mask(self, tags: tuple, tsr_device, tsr_size):
        word_idxs = []
        for tag in tags:
            if tag in self.tag_dict:
                word_idxs.extend(self.tag_dict[tag])
        legal_tag_index = torch.tensor(word_idxs, device=tsr_device)
        legal_tag_mask = torch.zeros(tsr_size, device=tsr_device)\
            .index_fill_(0, legal_tag_index, 1.).byte()
        return legal_tag_mask

    @torch.no_grad()
    def find_replacement(self,
                         changed, must_tags, dist_measure,
                         forbidden_idxs__,
                         repl_method='tagdict',
                         words=None, word_sid=None,  # Only need when using a tagger
                         ) -> (Optional[torch.Tensor], dict):
        if must_tags is None:
            must_tags = tuple(self.vocab.tags)
        if isinstance(must_tags, str):
            must_tags = (must_tags,)

        if repl_method == 'lstm':
            # Pipeline:
            #    256 minimum dists
            # -> Filtered by a NN tagger
            # -> Smallest one
            words = words.repeat(64, words.size(1))
            dists, idxs = self.embed_searcher.find_neighbours(
                changed, 64, dist_measure, False)
            for i, ele in enumerate(idxs):
                words[i][word_sid] = ele
            self.nn_tagger.eval()
            s_tags = self.nn_tagger(words)
            pred_tags = s_tags.argmax(-1)[:, word_sid]
            pred_tags = pred_tags.cpu().numpy().tolist()
            new_word_vid = None
            for i, ele in enumerate(pred_tags):
                if self.vocab.tags[ele] in must_tags:
                    if idxs[i] not in forbidden_idxs__:
                        new_word_vid = idxs[i]
                        break
            return new_word_vid, {"avgd": dists.mean().item(),
                                  "mind": dists.min().item()}
        elif repl_method in ['3gram', 'crf']:
            # Pipeline:
            #    256 minimum dists
            # -> Filtered by a Statistical tagger
            # -> Smallest one
            tagger = self.trigram_tagger if repl_method == '3gram' else self.crf_tagger
            word_texts = self.vocab.id2word(words)
            word_sid = word_sid.item()

            dists, idxs = self.embed_searcher.find_neighbours(
                changed, 64, dist_measure, False)

            cands = []
            for ele in cast_list(idxs):
                cand = word_texts.copy()
                cand[word_sid] = self.vocab.words[ele]
                cands.append(cand)

            pred_tags = tagger.tag_sents(cands)
            s_tags = [ele[word_sid][1] for ele in pred_tags]

            new_word_vid = None
            for i, ele in enumerate(s_tags):
                if ele in must_tags:
                    if idxs[i] not in forbidden_idxs__:
                        new_word_vid = idxs[i]
                        break
            return new_word_vid, {"avgd": dists.mean().item(),
                                  "mind": dists.min().item()}
        elif repl_method == 'tagdict':
            # Pipeline:
            #    All dists
            # -> Filtered by a tag dict
            # -> Smallest one
            dist = {'euc': euc_dist, 'cos': cos_dist}[
                dist_measure](changed, self.parser.embed.weight)
            # Mask illegal words by its POS
            tag_mask = self.__gen_tag_mask(must_tags, dist.device, dist.size())

            dist.masked_fill_(1 - tag_mask, 1000.)
            for ele in forbidden_idxs__:
                dist[ele] = 1000.
            mindist = dist.min()
            if abs(mindist - 1000.) < 0.001:
                new_word_vid = None
            else:
                new_word_vid = dist.argmin()
            return new_word_vid, {}
        elif repl_method == 'bert':
            # Pipeline:
            #    Bert select 256 words
            # -> Filtered by a tagger
            # -> Smallest one
            rpl, _ = self.bert_aug.substitute('it is black friday', [0], n=256)
            raise NotImplementedError
