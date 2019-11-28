# -*- coding: utf-8 -*-

from collections import namedtuple
from tabulate import tabulate

Sentence = namedtuple(typename='Sentence',
                      field_names=['ID', 'FORM', 'LEMMA', 'CPOS',
                                   'POS', 'FEATS', 'HEAD', 'DEPREL',
                                   'PHEAD', 'PDEPREL'],
                      defaults=[None]*10)


def sent_print(sent: Sentence, format='table'):
    """
        conll:
            The raw conll format (10 columns)
        tablev:
            A minimal version of conll format (5 columns, the transpose of tableh ↓)
        tableh:
            --  -  ---  ---  ---  -----  ------  -
            7   7  7    7    7    7      0       7
            RB  ,  PRP  VBD  RB   NNP    NNP     .
            No  ,  it   was  n't  Black  Monday  .
            1   2  3    4    5    6      7       8
            --  -  ---  ---  ---  -----  ------  -
    """
    if format == 'conll':
        table = []
        for i in range(len(sent.ID)):
            table.append([sent.ID[i], sent.FORM[i], sent.LEMMA[i], sent.CPOS[i],
                          sent.POS[i], sent.FEATS[i], sent.HEAD[i], sent.DEPREL[i],
                          sent.PHEAD[i], sent.PDEPREL[i]])
        print(tabulate(table))
    elif format == 'tablev':
        table = []
        for i in range(len(sent.ID)):
            table.append([sent.ID[i], sent.FORM[i], sent.POS[i],
                          sent.HEAD[i], sent.DEPREL[i]])
        print(tabulate(table))
    elif format == 'tableh':
        table = [sent.HEAD, sent.POS, sent.FORM, sent.ID]
        print(tabulate(table))
    else:
        raise NotImplementedError


def init_sentence(seqs, tags, arcs, rels, pred_arcs=None):
    length = len(seqs)
    ID = tuple(i for i in range(1, length+1))
    FORM = tuple(seqs)
    CPOS = tuple(tags)
    POS = tuple(tags)
    HEAD = tuple(arcs)
    DEPREL = tuple(rels)
    if pred_arcs is None:
        LEMMA, FEATS, PHEAD, PDEPREL = map(lambda x: tuple('_' for _ in range(x)), [length]*4)
    else:
        LEMMA, FEATS, PDEPREL = map(lambda x: tuple('_' for _ in range(x)), [length] * 3)
        PHEAD = pred_arcs
    return Sentence(ID, FORM, LEMMA, CPOS, POS, FEATS, HEAD, DEPREL, PHEAD, PDEPREL)


class Corpus(object):
    ROOT = '<ROOT>'

    def __init__(self, sentences):
        super(Corpus, self).__init__()

        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __repr__(self):
        return '\n'.join(
            '\n'.join('\t'.join(map(str, i))
                      for i in zip(*(f for f in sentence if f))) + '\n'
            for sentence in self
        )

    def __getitem__(self, index):
        return self.sentences[index]

    @property
    def words(self):
        return [[self.ROOT] + list(sentence.FORM) for sentence in self]

    @property
    def tags(self):
        return [[self.ROOT] + list(sentence.CPOS) for sentence in self]

    @property
    def heads(self):
        return [[0] + list(map(int, sentence.HEAD)) for sentence in self]

    @property
    def rels(self):
        return [[self.ROOT] + list(sentence.DEPREL) for sentence in self]

    @words.setter
    def words(self, sequences):
        self.sentences = [sentence._replace(FORM=sequence)
                          for sentence, sequence in zip(self, sequences)]

    @tags.setter
    def tags(self, sequences):
        self.sentences = [sentence._replace(POS=sequence)
                          for sentence, sequence in zip(self, sequences)]

    @heads.setter
    def heads(self, sequences):
        self.sentences = [sentence._replace(HEAD=sequence)
                          for sentence, sequence in zip(self, sequences)]

    @rels.setter
    def rels(self, sequences):
        self.sentences = [sentence._replace(DEPREL=sequence)
                          for sentence, sequence in zip(self, sequences)]

    @classmethod
    def load(cls, fname):
        start, sentences = 0, []
        with open(fname, 'r') as f:
            lines = [line for line in f if not line.startswith('#')]
        for i, line in enumerate(lines):
            if len(line) <= 1:
                sentence = Sentence(*zip(*[l.split() for l in lines[start:i]]))
                sentences.append(sentence)
                start = i + 1
        corpus = cls(sentences)

        return corpus

    def save(self, fname):
        with open(fname, 'w') as f:
            f.write(f"{self}\n")

    def append(self, sentence):
        self.sentences.append(sentence)
