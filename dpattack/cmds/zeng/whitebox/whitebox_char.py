from dpattack.cmds.zeng.attack import Attack
from dpattack.utils.corpus import Corpus,init_sentence
from dpattack.utils.metric import ParserMetric as Metric
from dpattack.libs.luna.pytorch import cast_list,idx_to_msk
from dpattack.utils.parser_helper import is_chars_judger
from dpattack.utils.constant import CONSTANT
import torch.nn.functional as F
import torch
import random
import math
import numpy as np
from collections import defaultdict

torch.backends.cudnn.enabled = False

# BlackBoxAttack class
class WhiteBoxAttack_Char(Attack):
    def __init__(self):
        super(WhiteBoxAttack_Char, self).__init__()

        self.embed_grad = dict()
        self.attack_epochs = 50

    def pre_attack(self, config):
        corpus,loader = super().pre_attack(config)
        self.parser.eval()
        return corpus, loader

    def char_embed_backward_hook(self, module, grad_in, grad_out):
        self.embed_grad['char_embed_grad'] = grad_out[0]

    def __call__(self, config):
        random.seed(1)
        np.random.seed(1)
        torch.manual_seed(1)

        #self.distance = F.cosine_similarity
        self.succeed_number = 0

        corpus, loader = self.pre_attack(config)
        print(self.vocab.char_dict)
        self.punct_rel_idx = self.vocab.rel_dict[CONSTANT.PUNCT]
        self.punct_idx = [value for key, value in self.vocab.char_dict.items() if not (key.isdigit() or key.isalpha())]
        # for saving
        attack_corpus = Corpus([])

        self.embedding_weight = self.parser.char_lstm.embed.weight
        self.parser.char_lstm.embed.register_backward_hook(self.char_embed_backward_hook)
        # attack seq generator
        # self.attack_seq_generator = self.get_attack_seq_generator(config)
        self.attack(loader, config, attack_corpus)

        # save to file
        if config.save_result_to_file:
            attack_corpus_save_path = self.get_attack_corpus_saving_path(config)
            attack_corpus.save(attack_corpus_save_path)
            print('Result after attacking has saved in {}'.format(attack_corpus_save_path))

    def get_attack_corpus_saving_path(self, config):
        if config.input == 'char':
            attack_corpus_save_path = '{}/white_typo_{}_{}.conllx'.format(config.result_path,
                                                                          config.blackbox_index if config.blackbox_index == 'unk' else config.blackbox_pos_tag,
                                                                          config.revised_rate)
        else:
            if config.blackbox_method == 'substitute':
                attack_corpus_save_path = '{}/black_{}_{}_{}.conllx'.format(config.result_path,
                                                                            config.blackbox_method,
                                                                            config.blackbox_index if config.blackbox_index == 'unk' else config.blackbox_pos_tag,
                                                                            config.revised_rate)
            else:
                attack_corpus_save_path = '{}/black_{}_{}.conllx'.format(config.c,
                                                                        config.blackbox_method,
                                                                         config.revised_rate)
        return attack_corpus_save_path

    def get_number(self, revised_rate, length):
        number = math.floor(revised_rate * length)
        if number == 0:
            number = 1
        return number

    def attack_for_each_process(self, config, loader, attack_corpus):
        revised_numbers = 0

        # three metric:
        # metric_before_attack: the metric before attacking(origin)
        # metric_after_attack: the metric after black box attacking
        metric_before_attack = Metric()
        metric_after_attack = Metric()

        for index, (seq_idx, tag_idx, chars, arcs, rels) in enumerate(loader):
            mask = self.get_mask(seq_idx, self.vocab.pad_index, punct_list=self.vocab.puncts)
            seqs = self.get_seqs_name(seq_idx)
            tags = self.get_tags_name(tag_idx)

            number = self.get_number(config.revised_rate, len(seqs) - 1)
            # attack for one sentence
            raw_metric, attack_metric,\
            attack_seq, revised_number = self.attack_for_each_sentence(config, seqs, seq_idx, tags, tag_idx, chars, arcs, rels, mask, number)

            metric_before_attack += raw_metric
            metric_after_attack += attack_metric

            if config.save_result_to_file:
                # all result ignores the first token <ROOT>
                attack_seq_idx = self.vocab.word2id(attack_seq).unsqueeze(0)
                if torch.cuda.is_available():
                    attack_seq_idx = attack_seq_idx.cuda()
                attack_chars = self.get_chars_idx_by_seq(attack_seq)
                _, attack_arc, attack_rel = self.task.predict([(attack_seq_idx, tag_idx, attack_chars)],mst=config.mst)
                attack_corpus.append(init_sentence(seqs[1:],
                                                   attack_seq[1:],
                                                   tags[1:],
                                                   cast_list(arcs)[1:],
                                                   self.vocab.id2rel(rels)[1:],
                                                   attack_arc,
                                                   attack_rel))

            revised_numbers += revised_number
            print("Sentence: {}, Revised: {} Before: {} After: {} ".format(index + 1, revised_number, metric_before_attack, metric_after_attack))

        return metric_before_attack, metric_after_attack, revised_numbers

    def get_whitebox_loss(self, arsc, s_arc):
        msk_gold_arc = idx_to_msk(arsc, num_classes=s_arc.size(1))
        s_gold = s_arc[msk_gold_arc]
        s_other = s_arc.masked_fill(msk_gold_arc, -1000.)
        max_s_other, _ = s_other.max(1)
        margin = s_gold - max_s_other
        margin[margin < -1] = -1
        loss = margin.sum()
        return loss

    def attack_for_each_sentence(self, config, seq, seq_idx, tag, tag_idx, chars, arcs, rels, mask, number):
        '''
        :param seqs:
        :param seq_idx:
        :param tags:
        :param tag_idx:
        :param arcs:
        :param rels:
        :return:
        '''
        # seq length: ignore the first token (ROOT) of each sentence
        # for metric before attacking
        # loss, raw_metric = self.task.evaluate([seq_idx,tag_idx, chars,arcs, rels])
        self.parser.eval()
        _, raw_metric = self.task.evaluate([(seq_idx, tag_idx, chars, arcs, rels)],mst=config.mst)
        # score_arc_before_attack, score_rel_before_attack = self.parser.forward(seq_idx, is_chars_judger(self.parser, tag_idx, chars))
        # raw_metric = self.get_metric(score_arc_before_attack[mask], score_rel_before_attack[mask], arcs[mask], rels[mask])

        # pre-process word mask
        word_index_grad_neednot_consider = cast_list(rels.eq(self.punct_rel_idx).squeeze().nonzero())
        word_index_grad_neednot_consider.append(0)

        # pre-process char mask
        char_mask = chars.gt(0)[0]
        sorted_lens, indices = torch.sort(char_mask.sum(dim=1), descending=True)
        inverse_indices = indices.argsort()

        char_mask_max = torch.max(char_mask.sum(dim=1))
        char_mask = char_mask[:,:char_mask_max]
        # delete the root token
        char_mask[0, :] = False
        # delete punct
        punct_idx_list = cast_list(rels.eq(self.punct_rel_idx).nonzero())
        char_mask[punct_idx_list, :] = False
        # the index in origin char
        # char_indexes = cast_list(char_mask.nonzero())

        attack_chars = chars.clone()

        forbidden_idx = defaultdict(lambda: set())
        char_idx = dict()
        revised_number = 0
        for i in range(self.attack_epochs):
            self.parser.zero_grad()
            s_arc, s_rel = self.parser.forward(seq_idx, is_chars_judger(self.parser, tag_idx, attack_chars))
            loss = self.get_whitebox_loss(arcs[mask], s_arc[mask])
            loss.backward()

            charembed_grad = self.embed_grad['char_embed_grad'][inverse_indices]
            wordembed_grad = self.parser.word_embed_grad[0]
            word_grad_norm = wordembed_grad.norm(dim=1)
            word_grad_norm[word_index_grad_neednot_consider] = -10000.0

            if i == 0:
                current_norm_indexes = cast_list(word_grad_norm.topk(number)[1])
                for index in current_norm_indexes:
                    forbidden_idx[index].update(self.punct_idx.copy())
                    revised_number += 1
                    char_grad = charembed_grad[index][char_mask[index]]
                    char_grad_norm = char_grad.norm(dim=1)
                    char_index = char_grad_norm.topk(1)[1].item()
                    char_idx[index] = char_index
            # if number == 1:
            #     if len(forbidden_idx.keys()) == 1:
            #         current_norm_indexes = list(forbidden_idx.keys())
            #     else:
            #         current_norm_indexes = cast_list(word_grad_norm.topk(1)[1])
            #         for index in current_norm_indexes:
            #             revised_number += 1
            #             forbidden_idx[index].update(self.punct_idx.copy())
            #             char_grad = charembed_grad[index][char_mask[index]]
            #             char_grad_norm = char_grad.norm(dim=1)
            #             char_index = char_grad_norm.topk(1)[1].item()
            #             char_idx[index] = char_index
            # elif number > 1:
            #     current_norm_indexes = cast_list(word_grad_norm.topk(2)[1])
            #     for count, index in enumerate(current_norm_indexes):
            #         if index in forbidden_idx:
            #             continue
            #         else:
            #             if len(forbidden_idx) < number:
            #                 revised_number += 1
            #                 forbidden_idx[index].update(self.punct_idx.copy())
            #                 char_grad = charembed_grad[index][char_mask[index]]
            #                 char_grad_norm = char_grad.norm(dim=1)
            #                 char_index = char_grad_norm.topk(1)[1].item()
            #                 char_idx[index] = char_index
            #             else:
            #                 current_norm_indexes[count] = np.random.choice(list(forbidden_idx.keys()))
            #     while current_norm_indexes[0] == current_norm_indexes[1]:
            #         current_norm_indexes[1] = np.random.choice(list(forbidden_idx.keys()))
            for index in forbidden_idx.keys():
                raw_index = attack_chars[0, index, char_idx[index]].item()
                # add raw index to be forbidden
                # if raw_index_char is a alpha, including its lower and upper letter
                self.add_raw_index_to_be_forbidden(forbidden_idx, index, raw_index)
                replace_index = self.find_neighbors(raw_index, charembed_grad[index, char_idx[index]],list(forbidden_idx[index]))
                attack_chars[0, index, char_idx[index]] = replace_index

            self.parser.eval()

            _, attack_metric = self.task.evaluate([(seq_idx, tag_idx, attack_chars, arcs, rels)], mst=config.mst)

            if attack_metric.uas < raw_metric.uas:
                self.succeed_number += 1
                print("Succeed", end=" ")
                break
        attack_seq = [Corpus.ROOT] + [self.vocab.id2char(chars) for chars in attack_chars[0,1:]]

        return raw_metric, attack_metric, attack_seq, revised_number

    def add_raw_index_to_be_forbidden(self, forbidden_idx, index, raw_index):
        forbidden_idx[index].add(raw_index)
        # raw_char = self.vocab.chars[raw_index]
        # if raw_char.islower():
        #     forbidden_idx[index].add(self.vocab.char_dict[raw_char.upper()])
        # elif raw_char.isupper():
        #     forbidden_idx[index].add(self.vocab.char_dict[raw_char.lower()])


    @torch.no_grad()
    def find_neighbors(self, raw_index, grads, forbidden):
        raw_embedding = self.embedding_weight[raw_index]
        delta = raw_embedding -  F.normalize(grads,dim=0)
        diffs = -torch.sqrt(torch.sum(torch.pow(delta - self.embedding_weight, 2),dim=1))
        diffs[forbidden] = -10000.0
        return diffs.topk(1)[1].item()


    def attack(self, loader, config, attack_corpus):
        metric_before_attack, metric_after_attack, revised_numbers = self.attack_for_each_process(config, loader, attack_corpus)

        print("Before attacking: {}".format(metric_before_attack))
        print("Black box attack. Method: {}, Rate: {}, Modified:{:.2f}".format(config.blackbox_method,config.revised_rate,revised_numbers/len(loader.dataset.lengths)))
        print("After attacking: {}".format(metric_after_attack))
        print("UAS Drop Rate: {:.2f}%".format((metric_before_attack.uas - metric_after_attack.uas)*100))
        print("Success Rate: {:.2f}%".format(self.succeed_number / len(loader.dataset.lengths)*100))


    def get_metric(self, s_arc,s_rel, gold_arc, gold_rel):
        pred_arc, pred_rel = self.decode(s_arc, s_rel)
        metric = Metric()
        metric(pred_arc, pred_rel, gold_arc, gold_rel)
        return metric

    def get_chars_idx_by_seq(self, sentence):
        chars = self.vocab.char2id(sentence).unsqueeze(0)
        if torch.cuda.is_available():
            chars = chars.cuda()
        return chars
