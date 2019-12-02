import math
import torch
import random
import numpy as np
import time
from collections import defaultdict
from dpattack.cmds.zeng.attack import Attack
from dpattack.utils.corpus import Corpus, init_sentence, Sentence
from dpattack.utils.aug import RandomWordAug, RandomTagAug
from dpattack.libs.luna.pytorch import cast_list
from dpattack.utils.parser_helper import is_chars_judger
from dpattack.utils.constant import CONSTANT
from dpattack.utils.metric import ParserMetric as Metric

class BlackBoxSubTree(Attack):
    def __init__(self):
        super(BlackBoxSubTree, self).__init__()
        self.min_span_length = 4
        self.max_span_length = 8

        self.reivsed = 3
        self.candidates = 256
        self.isPermutation = False

    def get_span_length(self, span, length):
        if isinstance(span, int):
            return span
        elif isinstance(span, float):
            span_length = math.floor(span * length)
            if span_length == 0:
                span_length += 1
            return span_length
        else:
            raise TypeError('Error in type of span length! ')

    def get_attack_seq_generator(self, config):
        return RandomWordAug(config.vocab)

    def __call__(self, config):
        random.seed(1)
        np.random.seed(1)
        torch.manual_seed(1)

        corpus, loader = self.pre_attack(config)
        self.parser.eval()

        attack_corpus = Corpus([])

        # attack seq generator
        self.attack_seq_generator = self.get_attack_seq_generator(config)
        #self.attack_index = AttackIndexUnkReplacement(config, self.vocab, self.parser)
        with torch.no_grad():
            self.attack(config, loader, corpus, attack_corpus)

        if config.save_result_to_file:
            # corpus_save_path = '{}/{}'.format(config.result_path,'origin.conllx')
            # corpus.save(corpus_save_path)
            # print('Result before attacking has saved in {}'.format(corpus_save_path))
            attack_corpus_save_path = self.get_attack_corpus_saving_path(config)
            attack_corpus.save(attack_corpus_save_path)
            print('Result after attacking has saved in {}'.format(attack_corpus_save_path))

    def get_attack_corpus_saving_path(self, config):
        path = "{}/subtree/{}_{}_{}.conllx".format(config.result_path, 'permutation' if self.isPermutation else 'random',self.reivsed, self.candidates)
        return path

    def attack(self, config, loader, corpus, attack_corpus):
        success = 0
        all_number = 0
        raw_metric_all = Metric()
        attack_metric_all = Metric()
        span_number = 0

        for index, (seq_idx, tag_idx, chars, arcs, rels) in enumerate(loader):
            start_time = time.time()
            mask = self.get_mask(seq_idx, self.vocab.pad_index, punct_list=self.vocab.puncts)
            seqs = self.get_seqs_name(seq_idx)
            tags = self.get_tags_name(tag_idx)

            sent = corpus[index]
            sentence_length = len(sent.FORM)
            spans = self.gen_spans(sent)
            roots = (arcs == 0).squeeze().nonzero().squeeze().tolist()

            valid_spans = self.get_valid_spans(spans, roots, sentence_length)
            if len(valid_spans) >= 2 and valid_spans[-1][0] > valid_spans[0][1]:
                filter_valid_spans = self.filter_spans(valid_spans)

                spans_list_to_attack = self.get_span_to_attack(filter_valid_spans)
                all_number += 1
                succeed_flag = False

                raw_s_rac, raw_s_rel = self.parser.forward(seq_idx, is_chars_judger(self.parser, tag_idx, chars))
                raw_mask = torch.ones_like(mask, dtype=mask.dtype)
                raw_mask[0, 0] = False
                raw_pred_arc, raw_pred_rel = self.task.decode(raw_s_rac, raw_s_rel, raw_mask, mst=config.mst)

                for spans_to_attack in spans_list_to_attack:
                    src_span = spans_to_attack[0]
                    tgt_span = spans_to_attack[1]

                    mask_idxes = list(range(0, tgt_span[0])) + list(range(tgt_span[1]+1,sentence_length+1))
                    new_mask = self.update_mask(mask, mask_idxes)

                    # for batch compare , no used task.evaluate
                    raw_non_equal_number = torch.sum(torch.ne(raw_pred_arc[new_mask], arcs[new_mask])).item()

                    indexes = self.get_attack_index(self.vocab.id2rel(rels), list(range(src_span[0],src_span[1]+1)), self.reivsed, self.candidates)
                    attack_seqs = [self.attack_seq_generator.substitute(seqs, tags, index) for index in indexes]
                    attack_seq_idx = torch.cat([self.vocab.word2id(attack_seq).unsqueeze(0) for attack_seq in attack_seqs],dim=0)
                    if torch.cuda.is_available():
                        attack_seq_idx = attack_seq_idx.cuda()
                    attack_mask = torch.ones_like(attack_seq_idx, dtype=mask.dtype)
                    attack_mask[:, 0] = 0
                    if is_chars_judger(self.parser):
                        attack_chars_idx = torch.cat([self.get_chars_idx_by_seq(attack_seq) for attack_seq in attack_seqs],dim=0)
                        attack_s_arc, attack_s_rel = self.parser.forward(attack_seq_idx, attack_chars_idx)
                    else:
                        attack_tags_idx = tag_idx.repeat(self.candidates, 1)
                        attack_s_arc, attack_s_rel = self.parser.forward(attack_seq_idx, attack_tags_idx)

                    attack_pred_arc, attack_pred_rel = self.task.decode(attack_s_arc, attack_s_rel, attack_mask, mst=config.mst)
                    attack_pred_arc_tgt = torch.split(attack_pred_arc[new_mask.repeat(self.candidates, 1)], [torch.sum(new_mask)] * self.candidates)
                    attack_non_equal_number_index = [count for count, pred in enumerate(attack_pred_arc_tgt) if torch.sum(torch.ne(pred, arcs[new_mask])).item() > raw_non_equal_number]

                    if len(attack_non_equal_number_index) != 0:
                        print("Sentence {} attacked succeeded! Time: {:.2f}".format(index + 1, time.time()-start_time))
                        success += 1
                        non_equal_numbers = [torch.sum(torch.ne(attack_pred_arc_tgt[non_equal_number_index], arcs[new_mask])).item() for non_equal_number_index in attack_non_equal_number_index]
                        attack_succeed_index = sorted(range(len(non_equal_numbers)), key=lambda k: non_equal_numbers[k], reverse=True)[0]
                        attack_succeed_index = attack_non_equal_number_index[attack_succeed_index]
                        attack_metric = Metric()
                        attack_metric(attack_pred_arc[attack_succeed_index].unsqueeze(0)[mask], attack_pred_rel[attack_succeed_index].unsqueeze(0)[mask],arcs[mask], rels[mask])
                        if config.save_result_to_file:
                            attack_seq = attack_seqs[attack_succeed_index]
                            for span in range(src_span[0],src_span[1]+1):
                                attack_seq[span] = "@" + attack_seq[span]
                            for span in range(tgt_span[0],tgt_span[1]+1):
                                attack_seq[span] = "#" + attack_seq[span]
                            attack_corpus.append(init_sentence(seqs[1:],
                                                               attack_seq[1:],
                                                               tags[1:],
                                                               cast_list(arcs)[1:],
                                                               self.vocab.id2rel(rels)[1:],
                                                               cast_list(attack_pred_arc[attack_succeed_index])[1:],
                                                               self.vocab.id2rel(attack_pred_rel[attack_succeed_index])[1:]))
                        succeed_flag = True
                        break

                raw_metric = Metric()
                raw_metric(raw_pred_arc[mask], raw_pred_rel[mask], arcs[mask], rels[mask])
                if not succeed_flag:
                    print("Sentence {} attacked failed! Time: {:.2f}".format(index + 1, time.time() - start_time))
                    attack_metric = raw_metric
                raw_metric_all += raw_metric
                attack_metric_all += attack_metric
            else:
                print("Sentence {} doesn't has enough valid spans. Time: {:.2f}".format(index + 1, time.time()-start_time))
        print("Before: {} After:{}".format(raw_metric_all, attack_metric_all))
        print("All: {}, Success: {}, Success Rate:{:.2f}%".format(all_number, success, success / all_number * 100))
        print("Average: {}".format(span_number / all_number))

    def get_valid_spans(self, spans, roots, length):
        return [span for index, span in enumerate(spans) if
                #index not in roots
                #and
                self.get_span_length(self.min_span_length, length) <= span[1] + 1 - span[0] <= self.get_span_length(self.max_span_length, length)]

    def filter_spans(self, valid_spans):
        filter_valid_spans = []
        for span_to_check in valid_spans:
            span_to_remain = True
            for span in valid_spans:
                if span != span_to_check:
                    if span[0] <= span_to_check[0] <= span[1] and span[0] <= span_to_check[1] <= span[1]:
                        span_to_remain = False
                        break
            if span_to_remain:
                filter_valid_spans.append(span_to_check)
        return filter_valid_spans

    def get_span_to_attack(self, valid_spans):
        return self.get_permutation_spans_to_attack(valid_spans) if self.isPermutation else self.get_random_spans_to_attack(valid_spans)
        # span_end_dict = defaultdict(lambda: list())
        # for span in valid_spans:
        #     span_end_dict[span[1]].append(span)
        # if len(span_end_dict) < 2:
        #     return []
        # attack_span_list = self.get_permutation_spans_to_attack(span_end_dict) if self.isPermutation else self.get_random_spans_to_attack(span_end_dict)
        # return attack_span_list

    def get_permutation_spans_to_attack(self, valid_spans):
        attack_span_list = []
        for span_start in valid_spans:
            for span_end in valid_spans:
                if span_start!=span_end:
                    attack_span_list.append((span_start, span_end))
        return attack_span_list
        # attack_span_list = []
        # span_end_key = list(span_end_dict.keys())
        # for key1 in span_end_key:
        #     for key2 in span_end_key:
        #         if key1!=key2:
        #             for value1 in span_end_dict[key1]:
        #                 for value2 in span_end_dict[key2]:
        #                     attack_span_list.append((value1, value2))
        # return attack_span_list

    def get_random_spans_to_attack(self, valid_spans):
        span_index = np.random.choice(list(range(len(valid_spans))),2,replace=False)
        return [(valid_spans[span_index[0]], valid_spans[span_index[1]])]
        # src_span_end, tgt_span_end = np.random.choice(list(span_end_dict.keys()),2,replace=False)
        # src_span = random.choice(span_end_dict[src_span_end])
        # tgt_span = random.choice(span_end_dict[tgt_span_end])
        # return [(src_span, tgt_span)]

    def update_mask(self, mask, mask_idxes):
        new_mask = mask.clone()
        for idx in mask_idxes:
            new_mask[0, idx] = 0
        return new_mask

    def decode_by_pred_score(self, s_arc, s_rel, mask):
        if s_arc.shape[0] != mask.shape[0]:
            s_arc, s_rel = s_arc[:,mask.squeeze(0)], s_rel[:,mask.squeeze(0)]
        else:
            s_arc, s_rel = s_arc[mask], s_rel[mask]
        pred_arc = s_arc.argmax(dim=-1)
        return pred_arc

    def get_attack_index(self, rels, change_indexes, revised, candidates):
        idxes = [idx for idx in change_indexes if rels[idx] != CONSTANT.PUNCT]
        # idxes = [idx for idx in mask_idxes if tags[idx] in CONSTANT.REAL_WORD_TAGS]
        return [idxes if len(idxes)<=revised else np.random.choice(idxes,revised,replace=False).tolist() for _ in range(candidates)]

    def get_chars_idx_by_seq(self, sentence):
        chars = self.vocab.char2id(sentence).unsqueeze(0)
        if torch.cuda.is_available():
            chars = chars.cuda()
        return chars

    def turn_tensor_to_list(self, tensor_or_list):
        if isinstance(tensor_or_list, torch.Tensor):
            return cast_list(tensor_or_list)
        elif isinstance(tensor_or_list, list):
            return tensor_or_list
        else:
            return tensor_or_list

    def compare_result(self, attacks, raw, gold_arc, mask_indexes):
        # uas between gold and raw pred
        raw_non_equal = torch.sum(torch.ne(raw, gold_arc)).item()

        for index, attack in enumerate(attacks):
            # usa betweeen attack and gold
            ne_flag = torch.ne(attack, gold_arc)
            attack_non_equal = torch.sum(ne_flag).item()
            # judge whether the prediction of attack refer to the mask indexes
            ne_index = ne_flag.nonzero().squeeze().tolist()
            if isinstance(ne_index,int):
                ne_index=[ne_index]
            # if the  prediction of attack result refer to the mask indexes, decrease one
            for index in ne_index:
                if attack[index] in mask_indexes:
                    attack_non_equal -= 1
            if attack_non_equal > raw_non_equal:
                return index
        return CONSTANT.FALSE_TOKEN

    # HIGHLIGHT: SPAN OPERATION
    def gen_spans(self, sent: Sentence):
        """
        Sample of a sentence (starting at 0):
              ID = ('1', '2', '3', '4', '5', '6', '7', '8')
            HEAD = ('7', '7', '7', '7', '7', '7', '0', '7')

        Return(ROOT included):
            [(0, 8), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (1, 8), (8, 8)]
        """
        ids = [0] + list(map(int, sent.ID))
        heads = [-1] + list(map(int, sent.HEAD))
        sent_len = len(ids)
        # print(ids, heads)
        l_children = [[] for _ in range(sent_len)]
        r_children = [[] for _ in range(sent_len)]

        for tid, hid in enumerate(heads):
            if hid != -1:
                if hid > tid:
                    l_children[hid].append(tid)
                else:
                    r_children[hid].append(tid)

        # for i in range(sent_len):
        #     print(ids[i], heads[i], l_children[ids[i]], r_children[ids[i]])

        # Find left/right-most span index
        def _find_span_id(idx, dir='l'):
            if dir == 'l':
                if len(l_children[idx]) == 0:
                    return idx
                else:
                    return _find_span_id(l_children[idx][0], 'l')
            else:
                if len(r_children[idx]) == 0:
                    return idx
                else:
                    return _find_span_id(r_children[idx][-1], 'r')

        spans = [(_find_span_id(idx, 'l'), _find_span_id(idx, 'r'))
                 for idx in range(sent_len)]
        # print(headed_span)

        # headed_span_length = [right - left + 1 for left, right in headed_span]
        # print(headed_span_length)
        return spans
