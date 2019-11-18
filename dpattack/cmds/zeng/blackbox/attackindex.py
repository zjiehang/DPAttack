'''
package for generate word indexes to be attacked in a sentence
For insert, check
For delete,
For substitute, two method: unk(replace each word to <unk>) and pos_tag
'''
import math
import torch
import numpy as np
from dpattack.utils.constant import CONSTANT
from dpattack.utils.parser_helper import is_chars_judger
from dpattack.libs.luna.pytorch import cast_list
from dpattack.models.char import CharParser
from transformers import GPT2Tokenizer,GPT2LMHeadModel


class AttackIndex(object):
    def __init__(self, config):
        self.revised_rate = config.revised_rate
        self.config = config

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

class AttackIndexRandomGenerator(AttackIndex):
    def __init__(self, config):
        super(AttackIndexRandomGenerator, self).__init__(config)

    def get_attack_index(self, seqs, seq_idx, tags, tag_idx, chars, arcs, mask):
        sentence_length = len(seqs)
        number = self.get_number(self.revised_rate, sentence_length)
        return self.get_random_index_to_be_attacked(tags, sentence_length, number)

    def get_random_index_to_be_attacked(self, tags, length, number):
        if self.config.input == 'char':
            word_index = list(range(length))
        else:
            word_index = [index for index, tag in enumerate(tags) if tag in CONSTANT.REAL_WORD_TAGS]
        if len(word_index) <= number:
            return word_index
        else:
            return np.random.choice(word_index, number, replace=False)


class AttackIndexInserting(AttackIndex):
    def __init__(self, config):
        super(AttackIndexInserting, self).__init__(config)

    def get_attack_index(self, seqs, seq_idx, tags, tag_idx, chars, arcs, mask):
        index = []
        length = len(tags)
        for i in range(length):
            if tags[i] in CONSTANT.NOUN_TAG:
                # current index is a NN, check the word before it
                if self.check_noun(tags, i):
                    index.append(i - 1)
            elif tags[i].startswith(CONSTANT.VERB_TAG):
                # current index is a VB, check whether this VB is modified by a RB
                if self.check_verb(seqs[i-1], tags, arcs, i):
                    index.append(i)
        index = list(set(index))
        return index
        #return self.get_random_index_by_length_rate(index, self.revised_rate, length)

    def check_noun(self, tags, i):
        if i == 0:
            return True
        else:
            tag_before_word_i = tags[i-1]
            if not tag_before_word_i.startswith(CONSTANT.NOUN_TAG[0]) and not tag_before_word_i.startswith(CONSTANT.ADJ_TAG):
                return True
            return False

    def check_verb(self, verb, tags, arcs,i):
        if verb in CONSTANT.AUXILIARY_VERB:
            return False
        for tag, arc in zip(tags, arcs):
            if tag.startswith(CONSTANT.ADV_TAG) and arc == i:
                return False
        return True


class AttackIndexDeleting(AttackIndex):
    def __init__(self, config):
        super(AttackIndexDeleting, self).__init__(config)

    def get_attack_index(self, seqs, seq_idx, tags, tag_idx, chars, arcs, mask):
        index = []
        length = len(tags)
        for i in range(length):
            if tags[i].startswith(CONSTANT.ADJ_TAG) or tags[i].startswith(CONSTANT.ADV_TAG):
                if self.check_modifier(arcs,i):
                    index.append(i)
        return index

    def check_modifier(self, arcs, index):
        for arc in arcs:
            if arc == index:
                return False
        return True

class AttackIndexUnkReplacement(AttackIndex):
    def __init__(self, config, vocab = None, parser = None):
        super(AttackIndexUnkReplacement, self).__init__(config)

        self.parser = parser
        self.vocab = vocab
        self.unk_chars = self.get_unk_chars_idx(self.vocab.UNK)

    def get_attack_index(self, seqs, seq_idx, tags, tag_idx, chars, arcs, mask):
        length = torch.sum(mask).item()
        index_to_be_replace = cast_list(mask.squeeze(0).nonzero())
        # for metric when change a word to <unk>
        # change each word to <unk> in turn, taking the worst case.
        # For a seq_index [<ROOT>   1   2   3   ,   5]
        # seq_idx_unk is
        #  [[<ROOT>    <unk>    2   3   ,   5]
        #   [<ROOT>    1    <unk>   3   ,   5]
        #   [<ROOT>    1    2   <unk>   ,   5]
        #   [<ROOT>    1    2   3   ,   <unk>]]
        seq_idx_unk = self.generate_unk_seqs(seq_idx, length, index_to_be_replace)
        if is_chars_judger(self.parser):
            char_idx_unk = self.generate_unk_chars(chars, length, index_to_be_replace)
            score_arc, score_rel = self.parser.forward(seq_idx_unk, char_idx_unk)
        else:
            tag_idx_unk = self.generate_unk_tags(tag_idx, length)
            score_arc, score_rel = self.parser.forward(seq_idx_unk, tag_idx_unk)
        pred_arc = score_arc.argmax(dim=-1)
        non_equal_numbers = self.calculate_non_equal_numbers(pred_arc[:,mask.squeeze(0)], arcs[mask])
        sorted_index = sorted(range(length), key=lambda k: non_equal_numbers[k], reverse=True)
        number = self.get_number(self.revised_rate, length)
        if isinstance(self.parser, CharParser):
            return [index_to_be_replace[index] - 1 for index in sorted_index[:number]]
        else:
            return self.get_index_to_be_attacked(sorted_index,tags,index_to_be_replace,number)

    def generate_unk_seqs(self, seq, length, index_to_be_replace):
        '''
        :param seq: seq_idx [<ROOT>   1   2   3   4   5], shape: [length + 1]
        :param length: sentence length
        :return:
        # for metric when change a word to <unk>
        # change each word to <unk> in turn, taking the worst case.
        # For a seq_index [<ROOT>   1   2   3   ,   5]
        # seq_idx_unk is
        #  [[<ROOT>    <unk>    2   3   ,   5]
        #   [<ROOT>    1    <unk>   3   ,   5]
        #   [<ROOT>    1    2   <unk>   ,   5]
        #   [<ROOT>    1    2   3   ,   <unk>]]
            shape: [length, length + 1]
        '''
        unk_seqs = seq.repeat(length, 1)
        for count, index in enumerate(index_to_be_replace):
            unk_seqs[count, index] = self.vocab.unk_index
        return unk_seqs

    def generate_unk_tags(self, tag, length):
        return tag.repeat(length, 1)

    def generate_unk_chars(self, char, length, index_to_be_replace):
        unk_chars = char.repeat(length, 1, 1)
        for count, index in enumerate(index_to_be_replace):
            unk_chars[count, index] = self.unk_chars
        return unk_chars

    def calculate_non_equal_numbers(self, pred_arc, gold_arc):
        '''
        :param pred_arc: pred arc 
        :param gold_arc: gold arc
        :return: the error numbers list
        '''
        non_equal_numbers = [torch.sum(torch.ne(arc, gold_arc)).item() for arc in pred_arc]
        return non_equal_numbers

    def get_unk_chars_idx(self, UNK_TOKEN):
        unk_chars = self.vocab.char2id([UNK_TOKEN]).squeeze(0)
        if torch.cuda.is_available():
            unk_chars = unk_chars.cuda()
        return unk_chars
    
    def get_index_to_be_attacked(self, sorted_index,tags,index_to_be_replace,number):
        attacked_list = []
        for index in sorted_index:
            if tags[index_to_be_replace[index]] in CONSTANT.REAL_WORD_TAGS:
                attacked_list.append(index_to_be_replace[index] - 1)
                if len(attacked_list) >= number:
                    break
        return attacked_list

class AttackIndexPosTag(AttackIndex):
    def __init__(self, config):
        super(AttackIndexPosTag, self).__init__(config)
        self.pos_tag = config.blackbox_pos_tag

    def get_attack_index(self, seqs, seq_idx, tags, tag_idx, chars, arcs, mask):
        index = [index - 1 for index, tag in enumerate(tags) if tag.startswith(self.pos_tag)]
        return self.get_random_index_by_length_rate(index, self.revised_rate, len(tags))


class AttackIndexInsertingPunct(AttackIndex):
    def __init__(self, config, vocab):
        super(AttackIndexInsertingPunct, self).__init__(config)

        self.tokenizer = GPT2Tokenizer.from_pretrained(config.language_model_path)
        self.model = GPT2LMHeadModel.from_pretrained(config.language_model_path)
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.vocab = vocab
        self.puncts = self.vocab.id2word(self.vocab.puncts)

    def get_sentence_score(self, sentence):
        tokenize_input = self.tokenizer.tokenize(sentence)
        tensor_input = torch.tensor([[self.tokenizer.eos_token_id] + self.tokenizer.convert_tokens_to_ids(tokenize_input)])
        if torch.cuda.is_available():
            tensor_input = tensor_input.cuda()
        output = self.model(tensor_input, labels=tensor_input)
        loss, logits = output[:2]
        return -loss.item() * len(tokenize_input)

    def get_attack_index(self, seqs, seq_idx, tags, tag_idx, chars, arcs, mask):
        comma_insert_index, comma_insert_seqs = self.duplicate_sentence_with_comma_insertion(seqs, len(seqs))
        if len(comma_insert_index) == 0:
            return []
        with torch.no_grad():
            seq_scores = [self.get_sentence_score(seq) for seq in comma_insert_seqs]
        sorted_index = sorted(range(len(seq_scores)), key=lambda k: seq_scores[k], reverse=True)
        number = self.get_number(self.revised_rate, len(seqs))
        return [comma_insert_index[index] for index in sorted_index[:number]]

    def duplicate_sentence_with_comma_insertion(self, seqs, length):
        duplicate_seqs_list = []
        comma_insert_index = []
        for index in range(1, length):
            if seqs[index] not in self.puncts and seqs[index - 1] not in self.puncts:
                duplicate_seqs = seqs.copy()
                duplicate_seqs.insert(index, CONSTANT.COMMA)
                duplicate_seqs_list.append(duplicate_seqs)
                comma_insert_index.append(index)
        return comma_insert_index, [' '.join(seq) for seq in duplicate_seqs_list]


class AttackIndexDeletingPunct(AttackIndex):
    def __init__(self, config, vocab):
        super(AttackIndexDeletingPunct, self).__init__(config)

        self.tokenizer = GPT2Tokenizer.from_pretrained(config.language_model_path)
        self.model = GPT2LMHeadModel.from_pretrained(config.language_model_path)
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.vocab = vocab
        self.puncts = self.vocab.id2word(self.vocab.puncts)

    def get_sentence_score(self, sentence):
        tokenize_input = self.tokenizer.tokenize(sentence)
        tensor_input = torch.tensor([[self.tokenizer.eos_token_id] + self.tokenizer.convert_tokens_to_ids(tokenize_input)])
        if torch.cuda.is_available():
            tensor_input = tensor_input.cuda()
        output = self.model(tensor_input, labels=tensor_input)
        loss, logits = output[:2]
        return -loss.item() * len(tokenize_input)

    def get_attack_index(self, seqs, seq_idx, tags, tag_idx, chars, arcs, mask):
        comma_insert_index, comma_insert_seqs = self.duplicate_sentence_with_comma_insertion(seqs, len(seqs), arcs)
        if len(comma_insert_index) == 0:
            return []
        with torch.no_grad():
            seq_scores = [self.get_sentence_score(seq) for seq in comma_insert_seqs]
        sorted_index = sorted(range(len(seq_scores)), key=lambda k: seq_scores[k], reverse=True)
        number = self.get_number(self.revised_rate, len(seqs))
        return [comma_insert_index[index] for index in sorted_index[:number]]

    def duplicate_sentence_with_comma_insertion(self, seqs, length, arcs):
        duplicate_seqs_list = []
        punct_delete_index = []
        for index in range(length - 1):
            if seqs[index] in self.puncts:
                if self.check_arcs(index, arcs):
                    duplicate_seqs = seqs.copy()
                    del duplicate_seqs[index]
                    duplicate_seqs_list.append(duplicate_seqs)
                    punct_delete_index.append(index)
        return punct_delete_index, [' '.join(seq) for seq in duplicate_seqs_list]

    def check_arcs(self, index, arcs):
        target_index = index + 1
        for arc in arcs:
            if target_index == arc:
                return False
        return True