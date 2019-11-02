import torch
import torch.nn as nn
from datetime import datetime, timedelta
from dpattack.utils.parser_helper import init_parser, load_parser
from dpattack.utils.metric import ParserMetric
from dpattack.utils.corpus import Corpus
from dpattack.utils.pretrained import Pretrained
from dpattack.utils.vocab import Vocab
from dpattack.models import WordParser, WordTagParser, WordCharParser, CharParser, PosTagger
from dpattack.utils.data import TextDataset, batchify
from dpattack.task import ParserTask, TaggerTask

from torch.utils.data import DataLoader

# base class for Attack
# two subclass: BlackBoxAttack,WhiteBoxAttack
class Attack(object):
    def __init__(self):
        self.criterion = nn.CrossEntropyLoss()

    def __call__(self, config):
        pass

    def attack(self, *args, **kwargs):
        pass

    def pre_attack(self, config):
        print("Load the models")
        self.vocab = torch.load(config.vocab)
        self.parser = load_parser(config.parser_model)
        self.tagger = PosTagger.load(config.tagger_model)
        self.model = ParserTask(self.vocab, self.parser)

        print("Load the dataset")
        corpus = Corpus.load(config.fdata)
        dataset = TextDataset(self.vocab.numericalize(corpus, True))
        loader = DataLoader(dataset=dataset, collate_fn=collate_fn)
        return corpus, loader

    def get_seqs_name(self, seqs):
        # assert seqs.shape
        if len(seqs.shape) == 2:
            return self.vocab.word2id(seqs.squeeze(0))
        else:
            return self.vocab.word2id(seqs)

    def get_tags_name(self, tags):
        if len(tags.shape) == 2:
            return self.vocab.tag2id(tags.squeeze(0))
        else:
            return self.vocab.tag2id(tags)

    def decode(self, s_arc, s_rel):
        pred_arcs = s_arc.argmax(dim=-1)
        pred_rels = s_rel[torch.arange(len(s_rel)), pred_arcs].argmax(dim=-1)
        return pred_arcs, pred_rels

    def get_loss(self, s_arc, s_rel, gold_arcs, gold_rels):
        s_rel = s_rel[torch.arange(len(s_rel)), gold_arcs]

        arc_loss = self.criterion(s_arc, gold_arcs)
        rel_loss = self.criterion(s_rel, gold_rels)
        loss = arc_loss + rel_loss
        return loss

    def get_mask(self, words, pad_index, punct = False, punct_list = None):
        '''
        get the mask of a sentence, mask all <pad>
        the starting of sentence is <ROOT>, mask[0] is always False
        :param words: sentence
        :param pad_index: pad index
        :param punct: whether to ignore the punctuation, when punct is False, take all the punctuation index to False(for evaluation)
        punct is True for getting loss
        :param punct_list: only used when punct is False
        :return:
        For example, for a sentence:  <ROOT>     no      ,       it      was     n't     Black   Monday  .
        when punct is True,
        The returning value is       [False    True     True    True    True    True    True    True    True]
        when punct is False,
        The returning value is      [False    True     False    True    True    True    True    True    False]
        '''
        mask = words.ne(pad_index)
        mask[0] = False
        if not punct:
            puncts = words.new_tensor(punct_list)
            mask &= words.unsqueeze(-1).ne(puncts).all(-1)
        return mask


