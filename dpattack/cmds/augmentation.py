import logging
# logging.basicConfig(level=logging.INFO,filename='log/augmentation.log')
from dpattack.utils.corpus import Corpus
import numpy as np
import random
import torch
import math
import copy
from dpattack.utils.utils import get_blackbox_augmentor

# for training data Augmentation
class Augmentation(object):
    def __call__(self, config):
        self.vocab = torch.load(config.vocab)
        # get augmentor
        self.aug = get_blackbox_augmentor(config)

        # load training data
        corpus = Corpus.load(config.ftrain)
        training_data_number = len(corpus.sentences)
        # random prob to decide whether to change a specific training data
        # if prob[index] < augmentation_rate, augmented.
        prob = np.random.uniform(0.0,1.0,size=(training_data_number,))
        for index in range(training_data_number):
            if index % 1000 == 0:
                print("{} sentences have processed! ".format(index))

            if prob[index] < config.augmentation_rate:
                sentence = corpus.sentences[index]
                sentence_length = len(sentence.FORM)
                # get revised number to be revised
                # revised_number = revised_rate * len(sentence), make sure that revised_number is at least 1.
                revised_number = math.floor(config.revised_rate * sentence_length)
                if revised_number == 0:
                    revised_number = 1
                # generate the word index to be augmented
                augmentation_index = random.sample(list(range(sentence_length)),revised_number)

                # get augmentation sequence
                origin_seq = ' '.join(sentence.FORM)
                aug_seq = tuple(self.generate_attack_seq(origin_seq, augmentation_index))

                # add the new sentence to training set
                new_sentence = copy.deepcopy(sentence)
                new_sentence = new_sentence._replace(FORM = aug_seq)
                corpus.sentences.append(new_sentence)

        saved_file = '{}/ptb_train_{}_{}.sd'.format(config.augmentation_dir,config.method,config.augmentation_rate)

        print("Complete! {} sentences have processed!".format(training_data_number))
        print("Current training data number is {}.".format(len(corpus.sentences)))
        print("The augmentation data are saved to file {}".format(saved_file))
        corpus.save(saved_file)

    def generate_attack_seq(self, origin_seq, index):
        '''
        :param origin_seq:  origin sequence
        :param index: the word index to be augmented
        :return:
        '''
        try:
            # specify the attack index
            attack_seq = self.aug.substitute(origin_seq, aug_idxes = index)
        except Exception:
            # if exception throws, choose the random index to be attack
            try:
                attack_seq = self.aug.substitute(origin_seq)
            except Exception:
                attack_seq = origin_seq

        return attack_seq.split()