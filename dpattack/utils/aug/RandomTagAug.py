import random
import torch
from dpattack.utils.corpus import Corpus
from dpattack.utils.vocab import Vocab
from nltk.tag import StanfordPOSTagger

STANFORD_POS_TAGGGER_PATH = '/disks/sdb/zjiehang/DependencyParsing/pretrained_model/stanford-postagger-full-2017-06-09/'
STANFORD_POS_TAGGGER_MODEL_PATH = STANFORD_POS_TAGGGER_PATH + 'models/english-bidirectional-distsim.tagger'
STANFORD_POS_TAGGGER_JAR_PATH = STANFORD_POS_TAGGGER_PATH + 'stanford-postagger-3.8.0.jar'

class RandomTagAug(object):
    def __init__(self,vocab,ftrain,revised_rate,aug_min=1):
        self.revised_rate = revised_rate
        self.aug_min = aug_min
        vocab = torch.load(vocab)
        train_corpus = Corpus.load(ftrain)
        self.tag_dict = self.get_tag_dict(train_corpus,vocab)

        self.stanford_pos_tagger = StanfordPOSTagger(STANFORD_POS_TAGGGER_MODEL_PATH, path_to_jar=STANFORD_POS_TAGGGER_JAR_PATH)

    def get_tag_dict(self,corpus,vocab):
        tag_dict = {}
        train_words = vocab.n_train_words

        for word_seq,tag_seq in zip(corpus.words,corpus.tags):
            for word,tag in zip(word_seq[1:], tag_seq[1:]):
                word_id = vocab.word_dict.get(word.lower(), vocab.unk_index)
                if word_id != vocab.unk_index and word_id < train_words:
                    if tag in tag_dict:
                        tag_dict[tag].append(word)
                    else:
                        tag_dict[tag] = []
                        tag_dict[tag].append(word)
        for key,value in tag_dict.items():
            tag_dict[key] = list(set(value))
        return tag_dict


    def substitute(self, origin_seq, aug_idxes = None):
        origin_seq = origin_seq.split()
        seq_with_tag = self.stanford_pos_tagger.tag(origin_seq)
        if aug_idxes is None:
            aug_idxes = self.get_aug_idxes(origin_seq)
        aug_seq = origin_seq.copy()
        for idxes in aug_idxes:
            tag = seq_with_tag[idxes][1]
            aug_seq[idxes] = random.choice(self.tag_dict.get(tag))
        return ' '.join(aug_seq)

    def get_aug_idxes(self,seq):
        length = len(seq)
        random_prob = [random.random() for i in range(length)]
        aug_idxes = [index for index, prob in enumerate(random_prob) if prob <= self.revised_rate]
        if len(aug_idxes) < self.aug_min:
            for i in range(self.aug_min):
                random_index = random.randint(0,length - 1)
                while random_index in aug_idxes:
                    random_index = random.randint(0, length - 1)
                aug_idxes.append(random_index)
        return aug_idxes