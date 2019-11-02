# -*- coding: utf-8 -*-

from datetime import datetime, timedelta

from dpattack.libs.luna import fetch_best_ckpt_name
from dpattack.utils.parser_helper import init_parser, load_parser
from dpattack.utils.metric import ParserMetric
from dpattack.utils.corpus import Corpus
from dpattack.utils.pretrained import Pretrained
from dpattack.utils.vocab import Vocab
from dpattack.models import WordParser, WordTagParser, WordCharParser, CharParser, PosTagger
from dpattack.utils.data import TextDataset, batchify
from dpattack.task import ParserTask, TaggerTask

from dpattack.utils.data import TextDataset, batchify

import torch


class Predict(object):

    def __call__(self, config):
        print("Load the models")
        vocab = torch.load(config.vocab)
        parser = load_parser(fetch_best_ckpt_name(config.parser_model))
        task = ParserTask(vocab, parser)
        if config.pred_tag:
            tagger = PosTagger.load(fetch_best_ckpt_name(config.tagger_model))
        else:
            tagger = None

        print("Load the dataset")
        corpus = Corpus.load(config.fdata)
        dataset = TextDataset(vocab.numericalize(corpus, False))
        # set the data loader
        loader = batchify(dataset, config.batch_size)

        print("Make predictions on the dataset")
        corpus.heads, corpus.rels = task.predict(loader, tagger)

        print(f"Save the predicted result to {config.fpred}")
        corpus.save(config.fpred)
