# -*- coding: utf-8 -*-
from dpattack.libs.luna import fetch_best_ckpt_name
from dpattack.utils.parser_helper import load_parser
from dpattack.utils.corpus import Corpus
from dpattack.models import PosTagger
from dpattack.task import ParserTask

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
        dataset = TextDataset(vocab.numericalize(corpus, training=False))
        # set the data loader
        loader = batchify(dataset, config.batch_size)

        print("Make predictions on the dataset")
        corpus.tags, corpus.heads, corpus.rels = task.predict(loader, tagger)

        saved_path = '{}/raw_result.conllx'.format(config.result_path)
        print(f"Save the predicted result to {saved_path}")
        corpus.save(saved_path)
