import torch
import numpy as np
from dpattack.utils.corpus import Corpus
from dpattack.utils.tag_tool import gen_tag_dict

class RandomTagAug(object):
    def __init__(self,vocab, ftrain):
        np.random.seed(123)
        vocab = torch.load(vocab)
        train_corpus = Corpus.load(ftrain)
        self.tag_dict = gen_tag_dict(train_corpus,vocab,threshold=2,verbose=False)

    def get_tag_dict(self,corpus,vocab):
        tag_dict = dict()
        train_words = vocab.n_train_words

        for word_seq,tag_seq in zip(corpus.words,corpus.tags):
            for word,tag in zip(word_seq[1:], tag_seq[1:]):
                word_id = vocab.word_dict.get(word.lower(), vocab.unk_index)
                if word_id != vocab.unk_index and word_id < train_words:
                    if tag in tag_dict:
                        tag_dict[tag].append(word.lower())
                    else:
                        tag_dict[tag] = []
                        tag_dict[tag].append(word.lower())
        for key,value in tag_dict.items():
            tag_dict[key] = list(set(value))
        return tag_dict


    def substitute(self, seqs, tags, idxes):
        attack_seqs = seqs.copy()
        for idx in idxes:
            tag = tags[idx]
            attack_seqs[idx] = np.random.choice(self.tag_dict[tag])
        return attack_seqs


    # def substitute(self, origin_seq, aug_idxes = None):
    #     origin_seq = origin_seq.split()
    #     seq_with_tag = self.stanford_pos_tagger.tag(origin_seq)
    #     if aug_idxes is None:
    #         aug_idxes = self.get_aug_idxes(origin_seq)
    #     aug_seq = origin_seq.copy()
    #     for idxes in aug_idxes:
    #         tag = seq_with_tag[idxes][1]
    #         aug_seq[idxes] = random.choice(self.tag_dict.get(tag))
    #     return ' '.join(aug_seq)
