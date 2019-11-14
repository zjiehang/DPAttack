import logging
logging.basicConfig(level=logging.ERROR)
from dpattack.cmds.zeng.attack import Attack
from dpattack.cmds.zeng.blackbox.blackboxmethod import Substituting, Inserting, Deleting,CharTypo
from dpattack.utils.corpus import Corpus,init_sentence
from dpattack.utils.metric import ParserMetric as Metric
from dpattack.libs.luna.pytorch import cast_list
from dpattack.utils.parser_helper import is_chars_judger
import torch


# BlackBoxAttack class
class BlackBoxAttack(Attack):
    def __init__(self):
        super(BlackBoxAttack, self).__init__()

    def pre_attack(self, config):
        loader = super().pre_attack(config)
        self.parser.eval()
        self.tagger.eval()
        return loader

    def __call__(self, config):
        loader = self.pre_attack(config)
        # for saving
        attack_corpus = Corpus([])

        # ROOT tag id
        self.ROOT_TAG = self.vocab.tag_dict[Corpus.ROOT]
        # attack seq generator
        self.attack_seq_generator = self.get_attack_seq_generator(config)
        self.attack(loader, config, attack_corpus)

        # save to file
        if config.save_result_to_file:
            # corpus_save_path = '{}/{}'.format(config.result_path,'origin.conllx')
            # corpus.save(corpus_save_path)
            # print('Result before attacking has saved in {}'.format(corpus_save_path))

            attack_corpus_save_path = '{}/black_{}_{}_{}.conllx'.format(config.result_path,
                                                                        config.blackbox_method,
                                                                        config.blackbox_index if config.blackbox_index == 'unk' else config.blackbox_pos_tag,
                                                                        config.revised_rate)
            attack_corpus.save(attack_corpus_save_path)
            print('Result after attacking has saved in {}'.format(attack_corpus_save_path))

    def get_attack_seq_generator(self, config):
        method = config.blackbox_method
        input_type = config.input
        if input_type == 'char':
            return CharTypo(config, self.vocab, self.tagger, self.ROOT_TAG, parser = self.parser)
        else:
            if method == 'insert':
                return Inserting(config, self.vocab, self.tagger, self.ROOT_TAG)
            elif method == 'substitute':
                return Substituting(config, self.vocab, self.tagger, self.ROOT_TAG, parser=self.parser)
            elif method == 'delete':
                return Deleting(config, self.vocab, self.tagger, self.ROOT_TAG)

    def attack_for_each_process(self, config, loader, attack_corpus):
        #recode revised_number for all sentences
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
            if config.pred_tag:
                tag_idx = self.tagger.decorator_forward(seq_idx, self.ROOT_TAG)

            # attack for one sentence
            score_arc_before_attack, score_rel_before_attack,\
            score_arc_after_attack, score_rel_after_attack, \
            attack_seq, attack_pred_tag, attack_mask, attack_gold_arc, attack_gold_rel, revised_number = self.attack_for_each_sentence(seqs, seq_idx, tags, tag_idx, chars, arcs, rels, mask)

            self.update_metric(metric_before_attack, score_arc_before_attack[mask], score_rel_before_attack[mask], arcs[mask], rels[mask])
            self.update_metric(metric_after_attack, score_arc_after_attack[attack_mask],score_rel_after_attack[attack_mask],attack_gold_arc[attack_mask],attack_gold_rel[attack_mask])

            if config.save_result_to_file:
                # all result ignores the first token <ROOT>
                pred_arc_after_attack, pred_rel_after_attack = self.decode(score_arc_after_attack.squeeze(0)[1:],score_rel_after_attack.squeeze(0)[1:])
                attack_corpus.append(init_sentence(attack_seq[1:],
                                                   self.vocab.id2tag(attack_pred_tag.squeeze(0)[1:]),
                                                   cast_list(pred_arc_after_attack),
                                                   self.vocab.id2rel(pred_rel_after_attack)))

            revised_numbers += revised_number
            print("Sentence: {}, Revised: {} Before: {} After: {} ".format(index + 1, revised_number, metric_before_attack, metric_after_attack))

        return metric_before_attack, metric_after_attack, revised_numbers

    def attack_for_each_sentence(self, seq, seq_idx, tag, tag_idx, chars, arcs, rels, mask):
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
            score_arc_before_attack, score_rel_before_attack = self.parser.forward(seq_idx, is_chars_judger(self.parser, tag_idx, chars))

            # for metric after attacking
            # generate the attack sentence under attack_index
            attack_seq, attack_mask, attack_gold_arc, attack_gold_rel, revised_number = self.attack_seq_generator.generate_attack_seq(' '.join(seq[1:]), seq_idx, tag, tag_idx, chars, arcs, rels, mask)
            # get the attack seq idx and tag idx
            attack_seq_idx = self.vocab.word2id(attack_seq).unsqueeze(0)
            if torch.cuda.is_available():
                attack_seq_idx = attack_seq_idx.cuda()
            attack_tag_idx = self.tagger.decorator_forward(attack_seq_idx, self.ROOT_TAG)
            if is_chars_judger(self.parser):
                attack_chars = self.get_chars_idx_by_seq(attack_seq)
                score_arc_after_attack, score_rel_after_attack = self.parser.forward(attack_seq_idx, attack_chars)
            else:
                score_arc_after_attack, score_rel_after_attack = self.parser.forward(attack_seq_idx, attack_tag_idx)

            return score_arc_before_attack, score_rel_before_attack, \
                   score_arc_after_attack, score_rel_after_attack, \
                   attack_seq, attack_tag_idx, attack_mask, attack_gold_arc, attack_gold_rel, revised_number

    def update_metric(self, metric, s_arc,s_rel, gold_arc, gold_rel):
        pred_arc, pred_rel = self.decode(s_arc, s_rel)
        metric(pred_arc, pred_rel, gold_arc, gold_rel)

    def attack(self, loader, config, attack_corpus):
        metric_before_attack, metric_after_attack, revised_numbers = self.attack_for_each_process(config, loader, attack_corpus)

        print("Before attacking: {}".format(metric_before_attack))
        print("Black box attack. Method: {}, Rate: {}, Modified:{:.2f}".format(config.blackbox_method,config.revised_rate,revised_numbers/len(loader.dataset.lengths)))
        print("After attacking: {}".format(metric_after_attack))

    def get_chars_idx_by_seq(self, sentence):
        chars = self.vocab.char2id(sentence).unsqueeze(0)
        if torch.cuda.is_available():
            chars = chars.cuda()
        return chars
