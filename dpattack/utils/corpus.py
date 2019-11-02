# -*- coding: utf-8 -*-

from collections import namedtuple


Sentence = namedtuple(typename='Sentence',
                      field_names=['ID', 'FORM', 'LEMMA', 'CPOS',
                                   'POS', 'FEATS', 'HEAD', 'DEPREL',
                                   'PHEAD', 'PDEPREL'],
                      defaults=[None]*10)


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

    def update_with_dict(self, index, tag, arc, rel, seq = None):
        update_dict = {}
        update_dict['POS'] = tag
        update_dict['HEAD'] = arc
        update_dict['DEPREL'] = rel
        if seq is not None:
            update_dict['FORM'] = seq
        self.sentences[index] = self.sentences[index]._replace(**update_dict)

    def update_corpus(self, index, tag, arc, rel, seq = None):
        # check length between current sequence and origin sequence
        # if current sequence is longer than origin sequence, update the new class
        current_length = len(tag)
        origin_length = len(self.sentences[index].FORM)
        diff = current_length - origin_length

        if diff == 0:
            self.update_with_dict(index, tag, arc, rel, seq)
        else:
            self.sentences[index] = self.create_sentence(diff, self.sentences[index], tag, arc, rel, seq)

    def create_sentence(self, length, sentence, tag, arc, rel, seq):
        ID = sentence.ID[:length] if length < 0 else sentence.ID + tuple(str(index) for index in range(len(sentence.ID)+1,len(sentence.ID)+1+length))
        FORM = seq
        LEMMA = sentence.LEMMA[:length] if length < 0 else sentence.LEMMA + tuple('_' for _ in range(length))
        CPOS = tag
        POS = tag
        FEATS = sentence.FEATS[:length] if length < 0 else sentence.FEATS + tuple('_' for _ in range(length))
        HEAD = arc
        DEPREL = rel
        PHEAD = sentence.PHEAD[:length] if length < 0 else sentence.PHEAD + tuple('_' for _ in range(length))
        PDEPREL = sentence.PDEPREL[:length] if length < 0 else sentence.PDEPREL + tuple('_' for _ in range(length))
        return Sentence(ID, FORM, LEMMA, CPOS, POS, FEATS, HEAD, DEPREL, PHEAD, PDEPREL)

    @classmethod
    def load(cls, fname):
        start, sentences = 0, []
        with open(fname, 'r') as f:
            lines = [line for line in f]
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
