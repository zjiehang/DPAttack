# logging.basicConfig(level=logging.INFO,filename='log/attack.log')
from dpattack.cmds.zeng.attack import Attack
from dpattack.cmds.zeng.blackbox import Substituting, Inserting, Deleting
from dpattack.utils.parser_helper import load_parser
from dpattack.models.tagger import PosTagger
from dpattack.utils.corpus import Corpus
from dpattack.utils.metric import ParserMetric as Metric
import torch
import copy


# BlackBoxAttack class
class BlackBoxAttack(Attack):
    def __init__(self):
        super(BlackBoxAttack, self).__init__()

    def __call__(self, config):
        print("Load the models")
        self.vocab = torch.load(config.vocab)

        # load pretrained dpattack
        self.parser = load_parser(config.parser_model)
        self.tagger = PosTagger.load(config.tagger_model)
        self.parser.eval()
        self.tagger.eval()

        print("Load the dataset")
        corpus = Corpus.load(config.fdata)
        # for saving the attacking result(including prediction result)
        attack_corpus = copy.deepcopy(corpus)

        self.ROOT_TAG = self.vocab.tag_dict[Corpus.ROOT]

        # dataset = DataSet(corpus, self.vocab)

        self.attack_seq_generator = self.get_attack_seq_generator(config)

        self.attack(dataset, config, corpus, attack_corpus)

        # save to file
        if config.save_result_to_file:
            corpus_save_path = '{}/{}'.format(config.result_path,'origin.conllx')
            corpus.save(corpus_save_path)
            print('Result before attacking has saved in {}'.format(corpus_save_path))

            attack_corpus_save_path = '{}/black_{}_{}_{}.conllx'.format(config.result_path,
                                                                        config.blackbox_method,
                                                                        config.blackbox_index if config.blackbox_index == 'unk' else config.blackbox_pos_tag,
                                                                        config.revised_rate)
            attack_corpus.save(attack_corpus_save_path)
            print('Result after attacking has saved in {}'.format(attack_corpus_save_path))

    def get_attack_seq_generator(self, config):
        method = config.blackbox_method
        if method == 'insert':
            return Inserting(config, self.vocab, self.tagger, self.ROOT_TAG)
        elif method == 'substitute':
            return Substituting(config, self.vocab, self.tagger, self.ROOT_TAG, parser=self.parser)
        elif method == 'delete':
            return Deleting(config, self.vocab, self.tagger, self.ROOT_TAG)

    def attack_for_each_process(self, config, dataset, corpus, attack_corpus):
        #recode revised_number for all sentences
        revised_numbers = 0

        # three metric:
        # metric_before_attack: the metric before attacking(origin)
        # metric_after_attack: the metric after black box attacking
        metric_before_attack = Metric()
        metric_after_attack = Metric()

        for index, (seqs, seq_idx, tags, tag_idx, arcs, rels) in dataset[1:20]:
            mask = self.get_mask(seq_idx, self.vocab.pad_index, punct_list=self.vocab.puncts)

            if config.pred_tag:
                tag_idx = self.tagger.decorator_forward(seq_idx, self.ROOT_TAG)
            # attack for one sentence
            score_arc_before_attack, score_rel_before_attack,\
            score_arc_after_attack, score_rel_after_attack, \
            attack_seq, attack_pred_tag, attack_mask, gold_arc, gold_rel, revised_number = self.attack_for_each_sentence(seqs, seq_idx, tags, tag_idx, arcs, rels, mask)

            self.update_metric(metric_before_attack, score_arc_before_attack[mask], score_rel_before_attack[mask], arcs[mask], rels[mask])
            self.update_metric(metric_after_attack, score_arc_after_attack[attack_mask],score_rel_after_attack[attack_mask],gold_arc[attack_mask],gold_rel[attack_mask])

            if config.save_result_to_file:
                # all result ignores the <ROOT>
                pred_arc_before_attack, pred_rel_before_attack = self.decode(score_arc_before_attack[1:], score_rel_before_attack[1:])
                corpus.update_corpus(index,
                                     self.vocab.id2tag(tag_idx[1:]),
                                     pred_arc_before_attack.tolist(),
                                     self.vocab.id2rel(pred_rel_before_attack))

                pred_arc_after_attack, pred_rel_after_attack = self.decode(score_arc_after_attack[1:],score_rel_after_attack[1:])
                attack_corpus.update_corpus(index,
                                            self.vocab.id2tag(attack_pred_tag[1:]),
                                            pred_arc_after_attack.tolist(),
                                            self.vocab.id2rel(pred_rel_after_attack),
                                            seq=attack_seq[1:])

            revised_numbers += revised_number
            print("Sentence: {}, Revised: {} Before: {} After: {} ".format(index + 1, revised_number, metric_before_attack, metric_after_attack))

        return metric_before_attack, metric_after_attack, revised_numbers

    def attack_for_each_sentence(self, seq, seq_idx, tag, tag_idx, arcs, rels, mask):
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
            score_arc_before_attack, score_rel_before_attack = self.parser.decorator_forward(seq_idx, tag_idx)

            # for metric after attacking
            # generate the attack sentence under attack_index
            attack_seq, attack_mask, gold_arc, gold_rel, revised_number = self.attack_seq_generator.generate_attack_seq(seq, seq_idx, tag, tag_idx, arcs, rels, mask)
            # get the attack seq idx and tag idx
            attack_seq_idx = self.vocab.word2id(attack_seq)
            # tag is equal to the origin tag if checking(If not checking, tag should be get by STANFORD POSTAGGER)
            attack_tag_idx = self.tagger.decorator_forward(attack_seq_idx, self.ROOT_TAG)

            score_arc_after_attack, score_rel_after_attack = self.parser.decorator_forward(attack_seq_idx, attack_tag_idx)

            return score_arc_before_attack, score_rel_before_attack, \
                   score_arc_after_attack, score_rel_after_attack, \
                   attack_seq, attack_tag_idx, attack_mask, gold_arc, gold_rel, revised_number

    def update_metric(self, metric, s_arc,s_rel, gold_arc, gold_rel):
        pred_arc, pred_rel = self.decode(s_arc, s_rel)
        metric(pred_arc, pred_rel, gold_arc, gold_rel)

    def attack(self, dataset, config, corpus, attack_corpus):
        metric_before_attack, metric_after_attack, revised_numbers = self.attack_for_each_process(config, dataset, corpus, attack_corpus)

        print("Before attacking: {}".format(metric_before_attack))
        print("Black box attack. Method: {}, Rate: {}, Modified:{:.2f}".format(config.blackbox_method,config.revised_rate,revised_numbers/dataset.data_numbers))
        print("After attacking: {}".format(metric_after_attack))


