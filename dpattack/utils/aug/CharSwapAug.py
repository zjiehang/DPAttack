import random

class CharSwapAug(object):
    def __init__(self,revised_rate,aug_min=1):
        self.revised_rate = revised_rate
        self.aug_min = aug_min

    def substitute(self, origin_seq, aug_idxes = None):
        origin_seq = origin_seq.split()
        if aug_idxes is None:
            aug_idxes = self.get_aug_idxes(origin_seq)

        aug_seq = origin_seq.copy()
        for idxes in aug_idxes:
            aug_seq[idxes] = self.get_swap_result(origin_seq[idxes])
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

    def get_swap_result(self, word):
        word_length = len(word)
        if word_length == 1:
            return word
        elif word_length == 2:
            return word[1] + word[0]
        else:
            first_index = random.randint(0,word_length-1)
            second_index = random.randint(0,word_length-1)

            if first_index>second_index:
                first_index,second_index=second_index,first_index

            if word[first_index].isupper():
                second_index_char = word[second_index].upper()
            elif word[first_index].islower():
                second_index_char = word[second_index].lower()
            else:
                second_index_char = word[second_index]

            if word[second_index].isupper():
                first_index_char = word[first_index].upper()
            elif word[second_index].islower():
                first_index_char = word[first_index].lower()
            else:
                first_index_char = word[first_index]
            swap_word = word[0:first_index]+second_index_char+word[first_index+1:second_index]+first_index_char+word[second_index+1:]
            return swap_word
