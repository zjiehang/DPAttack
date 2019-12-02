import torch
import numpy as np
from dpattack.utils.corpus import Corpus
from dpattack.utils.tag_tool import gen_tag_dict

class RandomWordAug(object):
    def __init__(self, vocab):
        vocab = torch.load(vocab)
        self.word_to_choice = vocab.words[2:]

    def substitute(self, seqs, tags, idxes):
        attack_seqs = seqs.copy()
        for idx in idxes:
            attack_seqs[idx] = np.random.choice(self.word_to_choice)
        return attack_seqs

