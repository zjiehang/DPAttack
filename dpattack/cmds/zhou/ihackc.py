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
from dpattack.models import CharParser
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
            ret += ['IN']
        if 'r' in k:
            ret += ['RB', 'RBR', 'RBS']
        return tuple(ret)


HACK_TAGS = _Tags()


class IHackC:
    def __init__(self):
        self.config: Config

        self.corpus: Corpus

        self.task: ParserTask

        self.embed_searcher: EmbeddingSearcher

        self.loader: DataLoader

    @property
    def vocab(self) -> Vocab:
        return self.task.vocab

    @property
    def parser(self) -> CharParser:
        return self.task.model

    def init_logger(self, config):
        if config.logf == 'on':
            if config.hk_use_worker == 'on':
                worker_info = "-{}@{}".format(config.hk_num_worker,
                                              config.hk_worker_id)
            else:
                worker_info = ""
            log_config('{}'.format(config.mode),
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

        train_corpus = Corpus.load(config.ftrain)

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

        self.parser.char_lstm.embed.register_backward_hook(embed_backward_hook)
        # self.parser.embed.register_backward_hook(embed_backward_hook)
        self.parser.eval()

        self.embed_searcher = EmbeddingSearcher(
            embed=self.parser.char_lstm.embed.weight,
            idx2word=lambda x: self.vocab.chars[x],
            word2idx=lambda x: self.vocab.char_dict[x]
        )

        random.seed(1)
        torch.manual_seed(1)

    def hack(self, instance, **kwargs):
        raise NotImplementedError
