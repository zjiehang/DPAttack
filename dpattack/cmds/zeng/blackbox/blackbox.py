from dpattack.cmds.zeng.attack import Attack
from dpattack.cmds.zeng.blackbox.blackboxmethod import Substituting, CharTypo, InsertingPunct, DeletingPunct
from dpattack.utils.corpus import Corpus,init_sentence
from dpattack.utils.metric import ParserMetric as Metric
from dpattack.libs.luna.pytorch import cast_list
from dpattack.utils.parser_helper import is_chars_judger
import torch
import random
import numpy as np


# BlackBoxAttack class
class BlackBoxAttack(Attack):
    def __init__(self):
        super(BlackBoxAttack, self).__init__()

    def pre_attack(self, config):
        corpus,loader = super().pre_attack(config)
        self.parser.eval()
        return corpus, loader

    def __call__(self, config):
        random.seed(1)
        np.random.seed(1)
        torch.manual_seed(1)

        corpus, loader = self.pre_attack(config)
        # for saving
        attack_corpus = Corpus([])

        # attack seq generator
        self.attack_seq_generator = self.get_attack_seq_generator(config)
        self.attack(loader, config, attack_corpus)

        # save to file
        if config.save_result_to_file:
            attack_corpus_save_path = self.get_attack_corpus_saving_path(config)
            attack_corpus.save(attack_corpus_save_path)
            print('Result after attacking has saved in {}'.format(attack_corpus_save_path))

    def get_attack_corpus_saving_path(self, config):
        if config.input == 'char':
            attack_corpus_save_path = '{}/black_typo_{}_{}.conllx'.format(config.result_path,
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

    def get_attack_seq_generator(self, config):
        method = config.blackbox_method
        input_type = config.input
        if input_type == 'char':
            return CharTypo(config, self.vocab, parser=self.parser)
        else:
            if method == 'insert':
                return InsertingPunct(config, self.vocab)
            elif method == 'substitute':
                return Substituting(config, self.task, self.vocab, parser=self.parser)
            elif method == 'delete':
                return DeletingPunct(config, self.vocab)

    def attack_for_each_process(self, config, loader, attack_corpus):
        revised_numbers = 0

        # three metric:
        # metric_before_attack: the metric before attacking(origin)
        # metric_after_attack: the metric after black box attacking
        raw_metric_all = Metric()
        attack_metric_all = Metric()
        success_numbers = 0

        for index, (seq_idx, tag_idx, chars, arcs, rels) in enumerate(loader):
            if index == 0:
                continue
            mask = self.get_mask(seq_idx, self.vocab.pad_index, punct_list=self.vocab.puncts)
            seqs = self.get_seqs_name(seq_idx)
            tags = self.get_tags_name(tag_idx)

            # attack for one sentence
            raw_metric, attack_metric, \
            attack_seq, attack_arc, attack_rel, \
            revised_number = self.attack_for_each_sentence(config, seqs, seq_idx, tags, tag_idx, chars, arcs, rels, mask)

            raw_metric_all += raw_metric
            attack_metric_all += attack_metric

            if attack_metric.uas < raw_metric.uas:
                success_numbers += 1

            if config.save_result_to_file:
                # all result ignores the first token <ROOT>
                attack_corpus.append(init_sentence(seqs[1:],
                                                   attack_seq[1:],
                                                   tags[1:],
                                                   cast_list(arcs)[1:],
                                                   self.vocab.id2rel(rels)[1:],
                                                   attack_arc,
                                                   attack_rel))

            revised_numbers += revised_number
            print("Sentence: {}, Revised: {} Before: {} After: {} ".format(index + 1, revised_number, raw_metric_all, attack_metric_all))

        return raw_metric_all, attack_metric_all, revised_numbers, success_numbers

    def attack_for_each_sentence(self, config, seq, seq_idx, tag, tag_idx, chars, arcs, rels, mask):
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
        with torch.no_grad():
            # for metric before attacking
            raw_loss, raw_metric = self.task.evaluate([(seq_idx, tag_idx, chars, arcs, rels)],mst=config.mst)
            # score_arc_before_attack, score_rel_before_attack = self.parser.forward(seq_idx, is_chars_judger(self.parser, tag_idx, chars))

            # for metric after attacking
            # generate the attack sentence under attack_index
            attack_seq, attack_mask, attack_gold_arc, attack_gold_rel, revised_number = self.attack_seq_generator.generate_attack_seq(' '.join(seq[1:]), seq_idx, tag, tag_idx, chars, arcs, rels, mask, raw_metric)
            # get the attack seq idx and tag idx
            attack_seq_idx = self.vocab.word2id(attack_seq).unsqueeze(0)
            if torch.cuda.is_available():
                attack_seq_idx = attack_seq_idx.cuda()

            if is_chars_judger(self.parser):
                attack_chars = self.get_chars_idx_by_seq(attack_seq)
                attack_loss, attack_metric = self.task.evaluate([(attack_seq_idx, None, attack_chars, arcs, rels)], mst=config.mst)
                _, attack_arc, attack_rel = self.task.predict([(attack_seq_idx, tag_idx, attack_chars)], mst=config.mst)
            else:
                attack_tag_idx = tag_idx.clone()
                attack_loss, attack_metric = self.task.evaluate([(attack_seq_idx, attack_tag_idx, None, arcs, rels)], mst=config.mst)
                _, attack_arc, attack_rel = self.task.predict([(attack_seq_idx, attack_tag_idx, None)], mst=config.mst)
            return raw_metric, attack_metric, attack_seq, attack_arc[0], attack_rel[0], revised_number

    def update_metric(self, metric, s_arc,s_rel, gold_arc, gold_rel):
        pred_arc, pred_rel = self.decode(s_arc, s_rel)
        metric(pred_arc, pred_rel, gold_arc, gold_rel)

    def attack(self, loader, config, attack_corpus):
        metric_before_attack, metric_after_attack, revised_numbers, success_numbers = self.attack_for_each_process(config, loader, attack_corpus)

        print("Before attacking: {}".format(metric_before_attack))
        print("Black box attack. Method: {}, Rate: {}, Modified:{:.2f}".format(config.blackbox_method,config.revised_rate,revised_numbers/len(loader.dataset.lengths)))
        print("After attacking: {}".format(metric_after_attack))
        print("UAS Drop Rate: {:.2f}%, success rate: {:.2f}%".format((metric_before_attack.uas - metric_after_attack.uas) * 100, success_numbers/len(loader.dataset.lengths)*100))

    def get_chars_idx_by_seq(self, sentence):
        chars = self.vocab.char2id(sentence).unsqueeze(0)
        if torch.cuda.is_available():
            chars = chars.cuda()
        return chars