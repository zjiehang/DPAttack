import numpy as np
import random

class CharTypoAug(object):
    def __init__(self, char_dict):
        self.char_list = [key for key in char_dict.keys() if key.isdigit() or key.isalpha()]
        self.trisection = [1.0 / 3.0, 2.0 / 3.0, 1.0]
        self.function = [self.substitute, self.insert, self.delete]

    def get_aug_idxes(self, seq, indexes):
        origin_seq = seq.split()
        if len(indexes) == 0:
            return origin_seq

        attack_seq = origin_seq.copy()
        random_value = np.random.rand(len(indexes))
        for count, index in enumerate(indexes):
            if random_value[count] < self.trisection[0]:
                attack_seq[index] = self.function[0](origin_seq[index])
            elif random_value[count] < self.trisection[1]:
                attack_seq[index] = self.function[1](origin_seq[index])
            else:
                attack_seq[index] = self.function[2](origin_seq[index])
        return attack_seq

    def substitute(self, words):
        word_index = random.randint(0,len(words)-1)
        random_char = random.choice(self.char_list)
        while random_char == words[word_index]:
            random_char = random.choice(self.char_list)
        return words[:word_index] + random_char + words[word_index+1:] if word_index < len(words) - 1 else words[:word_index] + random_char

    def insert(self, words):
        word_index = random.randint(0,len(words)-1)
        random_char = random.choice(self.char_list)
        return words[:word_index] + random_char + words[word_index:]

    def delete(self, words):
        if len(words) == 1:
            return self.substitute(words)
        word_index = random.randint(0,len(words)-1)
        return words[:word_index] + words[word_index+1:] if word_index < len(words) - 1 else words[:word_index]
