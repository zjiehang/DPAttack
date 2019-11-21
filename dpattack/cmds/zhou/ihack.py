import math
from collections import defaultdict
from contextlib import contextmanager
from typing import Optional, Union

import torch
from tabulate import tabulate
from torch.utils.data import DataLoader

from dpattack.libs.luna import (Aggregator, CherryPicker, TrainingStopObserver,
                                as_table, cast_list, create_folder_for_file,
                                fetch_best_ckpt_name, idx_to_msk, log,
                                log_config, ram_read, ram_write, show_mean_std,
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
from nltk import TrigramTagger


# Code for fucking VSCode debug console


class V:
    def __sub__(self, tsr):
        for ele in tsr.__repr__().split('\n'):
            print(ele)


v = V()


class _Tags:

    def __getitem__(self, k):
        assert k
        ret = []
        if 'n' in k:
            ret += ['NN', 'NNS', 'NNP', 'NNPS']
        if 'j' in k:
            ret += ['JJ', 'JJR', 'JJS']
        if 'v' in k:
            ret += ['VB', 'VBD', 'VBG', 'VBN', 'VBZ', 'VBP']
        if 'i' in k:
            ret += ['i']
        if 'r' in k:
            ret += ['RB', 'RBR', 'RBS']
        return tuple(ret)


HACK_TAGS = _Tags()


class IHack:
    def __init__(self):
        self.config: Config

        self.corpus: Corpus

        self.task: ParserTask

        self.nn_tagger: PosTagger
        self.gram_tagger: TrigramTagger
        self.tag_dict: dict

        self.embed_searcher: EmbeddingSearcher

        self.loader: DataLoader

    @property
    def vocab(self) -> Vocab:
        return self.task.vocab

    @property
    def parser(self) -> Union[WordTagParser, WordParser]:
        return self.task.model

    def setup(self, config):
        self.config = config

        print("Load the models")
        vocab = torch.load(config.vocab)  # type: Vocab
        parser = load_parser(fetch_best_ckpt_name(config.parser_model))
        self.nn_tagger = PosTagger.load(
            fetch_best_ckpt_name(config.tagger_model))

        self.task = ParserTask(vocab, parser)

        print("Load the dataset")

        train_corpus = Corpus.load(config.ftrain)

        self.tag_dict = auto_create("tagdict3",
                                    lambda: gen_tag_dict(
                                        train_corpus, vocab, 3, False),
                                    cache=True, path=config.workspace + '/saved_vars')
        self.tag_dict = {k: torch.tensor(v) for k, v in self.tag_dict.items()}

        self.gram_tagger = auto_create("trigram_tagger",
                                       lambda: train_gram_tagger(
                                           train_corpus, ngram=3),
                                       cache=True, path=config.workspace + '/saved_vars')

        if config.hk_training_set == 'on':
            self.corpus = train_corpus
        else:
            self.corpus = Corpus.load(config.fdata)
        dataset = TextDataset(vocab.numericalize(self.corpus, True))
        # set the data loader
        self.loader = DataLoader(dataset=dataset,
                                 collate_fn=collate_fn)

        def embed_backward_hook(module, grad_in, grad_out):
            ram_write('embed_grad', grad_out[0])

        self.parser.embed.register_backward_hook(embed_backward_hook)

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
            new_word_vid = None
            for i, ele in enumerate(pred_tags):
                if self.vocab.tags[ele.item()] in must_tags:
                    if idxs[i] not in forbidden_idxs__:
                        new_word_vid = idxs[i]
                        break
            return new_word_vid, {"avgd": dists.mean().item(),
                                  "mind": dists.min().item()}
        elif repl_method == '3gram':
            # Pipeline:
            #    256 minimum dists
            # -> Filtered by a 3-GRAM tagger
            # -> Smallest one
            prefix = (self.vocab.words[words[0][word_sid - 2].item()],
                      self.vocab.words[words[0][word_sid - 1].item()])
            dists, idxs = self.embed_searcher.find_neighbours(
                changed, 64, dist_measure, False)
            sents = [(*prefix, self.vocab.words[ele]) for ele in cast_list(idxs)]

            pred_tags = self.gram_tagger.tag_sents(sents)
            s_tags = [ele[2][1] for ele in pred_tags]

            new_word_vid = None
            for i, ele in enumerate(s_tags):
                if ele in must_tags:
                    if idxs[i] not in forbidden_idxs__:
                        new_word_vid = idxs[i]
                        break
            return new_word_vid, {}
        elif repl_method == 'tagdict':
            # Pipeline:
            #    All dists
            # -> Filtered by a tag dict
            # -> Smallest one
            # vals, idxs = self.embed_searcher.find_neighbours(changed, 10, 'euc', False)
            # print("⊥ {:.2f}, - {:.2f}, {} ~ {}, {}, {}".format(
            #     vals.min().item(), vals.mean().item(),
            #     *[self.vocab.words[idxs[i].item()] for i in range(4)]))
            # show_mean_std(embed[word_vid])
            # show_mean_std(max_grad)
            dist = {'euc': euc_dist, 'cos': cos_dist}[
                dist_measure](changed, self.parser.embed.weight)
            # print('>>> before moving')
            # self.embed_searcher.find_neighbours(embed[word_vid],10, 'euc', True)
            # print('>>> after moving')
            # self.embed_searcher.find_neighbours(changed, 10, 'euc', True)

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
            raise NotImplementedError
