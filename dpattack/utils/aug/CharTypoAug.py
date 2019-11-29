import numpy as np
import random

class CharTypoAug(object):
    def __init__(self, char_dict):
        self.char_list = [key for key in char_dict.keys() if key.isdigit() or key.isalpha()]
        self.function = [self.substitute]

    def get_typos(self, seq, indexes):
        origin_seq = seq.split()
        if len(indexes) == 0:
            return origin_seq

        attack_seq = origin_seq.copy()
        for index in indexes:
            attack_seq[index] = random.choice(self.function)(origin_seq[index])
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
