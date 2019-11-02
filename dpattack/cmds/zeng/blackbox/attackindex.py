'''
package for generate word indexes to be attacked in a sentence
For insert, check
For delete,
For substitute, two method: unk(replace each word to <unk>) and pos_tag
'''
import math
import torch
import numpy as np

class AttackIndex(object):
    def __init__(self, config):
        self.revised_rate = config.revised_rate

    def get_attack_index(self, *args, **kwargs):
        pass

    def get_number(self, revised_rate, length):
        number = math.floor(revised_rate * length)
        if number == 0:
            number = 1
        return number

    def get_random_index_by_length_rate(self, index, revised_rate, length):
        number = self.get_number(revised_rate, length)
        if len(index) <= number:
            return index
        else:
            return np.random.choice(index, number)


class AttackIndexInserting(AttackIndex):
    def __init__(self, config):
        super(AttackIndexInserting, self).__init__(config)
        self.AuxiliaryVerb = ["do", "did", "does", "have", "has", "had"]

    def get_attack_index(self, seqs, tags, arcs):
        index = []
        length = len(tags)
        for i in range(length):
            if tags[i] == 'NN' or tags[i] == 'NNS':
                # current index is a NN, check the word before it
                if self.check_noun(tags, i):
                    index.append(i)
            elif tags[i].startswith('VB'):
                # current index is a VB, check whether this VB is modified by a RB
                if self.check_verb(seqs, tags, arcs, i):
                    index.append(i + 1)
        index = list(set(index))
        return index
        #return self.get_random_index_by_length_rate(index, self.revised_rate, length)

    def check_noun(self, tags, i):
        if i == 0:
            return True
        else:
            tag_before_word_i = tags[i-1]
            if not tag_before_word_i.startswith('NN') and not tag_before_word_i.startswith('JJ'):
                return True
            return False

    def check_verb(self, seqs, tags, arcs, i):
        if seqs[i] in self.AuxiliaryVerb:
            return False
        for tag, arc in zip(tags, arcs):
            if tag.startswith('RB') and (arc - 1) == i:
                return False
        return True


class AttackIndexDeleting(AttackIndex):
    def __init__(self, config):
        super(AttackIndexDeleting, self).__init__(config)

    def get_attack_index(self):
        pass


class AttackIndexUnkReplacement(AttackIndex):
    def __init__(self, config, parser = None):
        super(AttackIndexUnkReplacement, self).__init__(config)

        self.parser = parser

    def get_attack_index(self, seqs, seq_idx, tags, tag_idx, arcs, mask):
        length = seq_idx.shape[0] - 1
        # for metric when change a word to <unk>
        # change each word to <unk> in turn, taking the worst case.
        # For a seq_index [<ROOT>   1   2   3   4   5]
        # seq_idx_unk is
        #  [[<ROOT>    <unk>    2   3   4   5]
        #   [<ROOT>    1    <unk>   3   4   5]
        #   [<ROOT>    1    2   <unk>   4   5]
        #   [<ROOT>    1    2   3   <unk>   5]
        #   [<ROOT>    1    2   3   4   <unk>]]
        seq_idx_unk = self.generate_unk_seqs(seq_idx, length)
        tag_idx_unk = self.generate_unk_tags(tag_idx, length)
        score_arc, score_rel = self.parser.decorator_forward(seq_idx_unk, tag_idx_unk)
        pred_arc = score_arc.argmax(dim=-1)
        non_equal_numbers = self.calculate_non_equal_numbers(pred_arc[:,mask], arcs[mask])
        sorted_index = sorted(range(length), key=lambda k: non_equal_numbers[k], reverse=True)
        number = self.get_number(self.revised_rate, length)
        return sorted_index[:number]

    def generate_unk_seqs(self, seq, length):
        '''
        :param seq: seq_idx [<ROOT>   1   2   3   4   5], shape: [length + 1]
        :param length: sentence length
        :return:
            [[<ROOT>    <unk>    2   3   4   5]
            [<ROOT>    1    <unk>   3   4   5]
            [<ROOT>    1    2   <unk>   4   5]
            [<ROOT>    1    2   3   <unk>   5]
            [<ROOT>    1    2   3   4   <unk>]]
            shape: [length, length + 1]
        '''
        seqs_repeat = seq.repeat(length + 1, 1)
        diag_element = torch.diag(seqs_repeat, 1) - 1
        diag_matrix = torch.diag(diag_element, 1)
        return (seqs_repeat - diag_matrix)[:-1]

    def generate_unk_tags(self, tag, length):
        return tag.repeat(length, 1)

    def calculate_non_equal_numbers(self, pred_arc, gold_arc):
        '''
        :param pred_arc: pred arc 
        :param gold_arc: gold arc
        :return: the error numbers list
        '''
        non_equal_numbers = [torch.sum(torch.ne(arc, gold_arc)).item() for arc in pred_arc]
        return non_equal_numbers


class AttackIndexPosTag(AttackIndex):
    def __init__(self, config):
        super(AttackIndexPosTag, self).__init__(config)
        self.pos_tag = config.blackbox_pos_tag

    def get_attack_index(self, seqs, seq_idx, tags, tag_idx, arcs, mask):
        index = [index for index, tag in enumerate(tags) if tag.startswith(self.pos_tag)]
        return self.get_random_index_by_length_rate(index, self.revised_rate, len(tags))
