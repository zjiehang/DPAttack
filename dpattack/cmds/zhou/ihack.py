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
from dpattack.utils.tag_tool import gen_tag_dict
from dpattack.utils.vocab import Vocab
import random
from config import Config


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

        self.tagger: PosTagger
        self.tag_filter: dict

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
        self.tagger = PosTagger.load(fetch_best_ckpt_name(config.tagger_model))

        self.task = ParserTask(vocab, parser)

        print("Load the dataset")

        train_corpus = Corpus.load(config.ftrain)

        self.tag_dict = auto_create("tagdict3",
                                    lambda: gen_tag_dict(
                                        train_corpus, vocab, 3, False),
                                    cache=True, path=config.workspace + '/saved_vars')
        self.tag_dict = {k: torch.tensor(v) for k, v in self.tag_dict.items()}

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

    def hack(self, instance, **kwargs):
        raise NotImplementedError

    @torch.no_grad()
    def find_replacement(self,
                         changed, must_tag, dist_measure,
                         forbidden_idxs__,
                         repl_method='tagdict', 
                         words=None, word_sid=None, # Only need when using a tagger
                         ) -> (Optional[torch.Tensor], dict):
        if repl_method == 'tagger':
            # Pipeline:
            #    256 minimum dists
            # -> Filtered by a tagger
            # -> Smallest one
            words = words.repeat(128, words.size(1))
            dists, idxs = self.embed_searcher.find_neighbours(changed, 128, dist_measure, False)
            for i, ele in enumerate(idxs):
                words[i][word_sid] = ele
            self.tagger.eval()
            s_tags = self.tagger(words)
            pred_tags = s_tags.argmax(-1)[:, word_sid]
            new_word_vid = None
            for i, ele in enumerate(pred_tags):
                if self.vocab.tags[ele.item()] == must_tag:
                    new_word_vid = idxs[i]
                    if new_word_vid.item() not in forbidden_idxs__:
                        break
            return new_word_vid, {"avgd": dists.mean().item(),
                                  "mind": dists.min().item()}
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
            dist = {'euc': euc_dist, 'cos': cos_dist}[dist_measure](changed, self.parser.embed.weight)
            # print('>>> before moving')
            # self.embed_searcher.find_neighbours(embed[word_vid],10, 'euc', True)
            # print('>>> after moving')
            # self.embed_searcher.find_neighbours(changed, 10, 'euc', True)

            # Mask illegal words by its POS
            legal_tag_index = self.tag_dict[must_tag].to(dist.device)
            legal_tag_mask = dist.new_zeros(dist.size()) \
                .index_fill_(0, legal_tag_index, 1.).byte()
            dist.masked_fill_(1 - legal_tag_mask, 1000.)
            for ele in forbidden_idxs__:
                dist[ele] = 1000.
            new_word_vid = dist.argmin()
            return new_word_vid, {"avgd": 0, "mind": 0}
        elif repl_method == 'bert':
            # Pipeline:
            #    Bert select 256 words
            # -> Filtered by a tagger
            # -> Smallest one
            raise NotImplementedError
