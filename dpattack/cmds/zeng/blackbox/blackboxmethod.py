from dpattack.utils.utils import get_blackbox_augmentor
from dpattack.utils.corpus import Corpus

class BlackBoxMethod(object):
    def __init__(self, tagger, vocab, ROOT_TAG):
        self.tagger = tagger
        self.ROOT_TAG = ROOT_TAG
        self.vocab = vocab

        self.FALSE_TOKEN = -1

    def generate_attack_seq(self, *args, **kwargs):
        pass

    def check_pos_tag(self, *args, **kwargs):
        pass

    def update_mask_arc_rel(self, mask, arc, rel, revised_list):
        pass

    def copy_str_to_list(self, seq):
        if isinstance(seq, str):
            return seq.split()
        elif isinstance(seq, list):
            return seq

    def insert_token_to_str_list(self, seq, index, token):
        seq_list = seq
        if isinstance(seq, str):
            seq_list = seq.split()
        seq_list.insert(index, token)
        return seq_list

    def duplicate_sentence_with_candidate_replacement(self, seq, candidate, index):
        candidates_number = len(candidate)
        duplicate_list = self.duplicate_sentence(self.copy_str_to_list(seq), candidates_number)
        for i in range(candidates_number):
            duplicate_list[i][index] = candidate[i]
        return duplicate_list

    def duplicate_sentence(self, seq, times):
        return [seq.copy() for _ in range(times)]


class Substituting(BlackBoxMethod):
    def __init__(self, config, vocab, tagger, ROOT_TAG, parser=None):
        super(Substituting, self).__init__(tagger, vocab, ROOT_TAG)
        self.index = self.get_index(config, parser)
        self.aug = get_blackbox_augmentor(config.blackbox_model, config.path, config.revised_rate, vocab=vocab, ftrain=config.ftrain)

    def get_index(self, config, parser=None):
        if config.blackbox_index == 'pos':
            return AttackIndexPosTag(config)
        else:
            if parser is None:
                print('unk replacement can not missing dpattack')
                exit()
            return AttackIndexUnkReplacement(config, parser=parser)

    def generate_attack_seq(self, seqs, seq_idx, tags, tag_idx, arcs, rels, mask):
        # generate word index to be attacked
        attack_index = self.index.get_attack_index(seqs, seq_idx, tags, tag_idx, arcs, mask)
        # generate word candidates to be attacked
        candidates, indexes = self.substituting(seqs, attack_index)
        # check candidates by pos_tagger
        attack_seq, revised_list = self.check_pos_tag(seqs, tag_idx, candidates, indexes)
        #mask = self.update_mask(mask, revised_list)
        return [Corpus.ROOT] + attack_seq, mask, arcs, rels, len(revised_list)

    def substituting(self, seq, index):
        try:
            # generate the attack sentence by index
            candidates, revised_indexes = self.aug.substitute(seq, aug_idxes=index)
        except Exception:
            try:
                # if error happens, generate the attack sentence by random
                candidates, revised_indexes = self.aug.substitute(seq)
            except Exception:
                candidates = None
                revised_indexes = []
        return candidates, revised_indexes

    def update_mask_arc_rel(self, mask, arc, rel, revised_list):
        return mask, arc, rel

    def check_pos_tag(self, seqs, tag_idx, candidates, indexes):
        final_attack_seq = self.copy_str_to_list(seqs)
        revised_index_list = []
        for count, index in enumerate(indexes):
            succeed = self.check_pos_tag_under_each_index(seqs, tag_idx, candidates[count], index)
            if succeed!=self.FALSE_TOKEN:
                final_attack_seq[index] = candidates[count][succeed]
                revised_index_list.append(index)
        return final_attack_seq, revised_index_list

    def check_pos_tag_under_each_index(self, seqs, tag_idx, candidate, index):
        attack_seq_list = self.duplicate_sentence_with_candidate_replacement(seqs, candidate, index)
        # change seq list to idx
        # add <ROOT> token to the first token of each setence
        attack_seq_idx = torch.cat([self.vocab.word2id([Corpus.ROOT] + s).unsqueeze(0) for s in attack_seq_list], dim=0)
        attack_tag_idx = self.tagger.decorator_forward(attack_seq_idx, self.ROOT_TAG)
        tag_equal_flag = torch.eq(attack_tag_idx[:, index + 1], tag_idx[index + 1])
        if torch.sum(tag_equal_flag) != 0:
            attack_succeed_index = tag_equal_flag.nonzero().squeeze(0)
            return attack_succeed_index[0]
        return self.FALSE_TOKEN


class Inserting(BlackBoxMethod):
    def __init__(self, config, vocab, tagger, ROOT_TAG):
        super(Inserting, self).__init__(tagger, vocab, ROOT_TAG)
        self.index = AttackIndexInserting(config)
        self.aug = get_blackbox_augmentor(config.blackbox_model, config.path, config.revised_rate, vocab=vocab,ftrain=config.ftrain)

    def generate_attack_seq(self, seqs, seq_idx, tags, tag_idx, arcs, rels, mask):
        # generate word index to be attacked
        attack_index = self.index.get_attack_index(self.copy_str_to_list(seqs), tags, arcs)
        # generate word candidates to be attacked
        candidates, indexes = self.inserting(seqs, attack_index)
        # check candidates by pos_tagger
        attack_seq, revised_list = self.check_pos_tag(seqs, tags, candidates, indexes)
        mask, arc, rel = self.update_mask_arc_rel(mask, arcs, rels, revised_list)
        return [Corpus.ROOT] + attack_seq, mask, arc, rel, len(revised_list)

    def inserting(self, seq, index):
        try:
            # generate the attack sentence by index
            candidates, revised_indexes = self.aug.insert(seq, aug_idxes = index)
        except Exception:
            candidates = None
            revised_indexes = []
        return candidates, revised_indexes

    def check_pos_tag(self, seqs, tags, candidates, indexes):
        final_attack_seq = self.copy_str_to_list(seqs)
        revised_index_list = []
        for count, index in enumerate(indexes):
            succeed = self.check_pos_tag_under_each_index(seqs, tags, candidates[count], index)
            if succeed!=self.FALSE_TOKEN:
                final_attack_seq.insert(index, candidates[count][succeed])
                for i in range(len(revised_index_list)):
                    revised_index_list[i] += 1
                revised_index_list.append(index)
        return final_attack_seq, revised_index_list

    def check_pos_tag_under_each_index(self, seqs, tags, candidate, index):
        seqs_insert_token = self.insert_token_to_str_list(seqs, index, self.vocab.UNK)
        attack_seq_list = self.duplicate_sentence_with_candidate_replacement(seqs_insert_token, candidate, index)
        # change seq list to idx
        # add <ROOT> token to the first token of each setence
        attack_seq_idx = torch.cat([self.vocab.word2id([Corpus.ROOT] + s).unsqueeze(0) for s in attack_seq_list], dim=0)
        attack_tag_idx = self.tagger.decorator_forward(attack_seq_idx, self.ROOT_TAG)
        gold_tag = self.get_gold_tag_under_inserting(tags, index)
        gold_tag_idx = self.vocab.tag2id(gold_tag)
        tag_equal_flag = torch.eq(attack_tag_idx[:, index + 1], gold_tag_idx[index + 1])
        if torch.sum(tag_equal_flag) != 0:
            attack_succeed_index = tag_equal_flag.nonzero().squeeze(0)
            return attack_succeed_index[0]
        return self.FALSE_TOKEN

    def get_gold_tag_under_inserting(self, tag, index):
        gold_tag = tag.copy()
        if tag[index] == 'NN' or tag[index] == 'NNS':
            gold_tag.insert(index, 'JJ')
        elif tag[index - 1].startswith('VB'):
            gold_tag.insert(index, 'RB')
        return [Corpus.ROOT] + gold_tag

    def insert_list_to_tensor(self, tensor, revised_list):
        tensor_in_list = tensor.tolist()
        for revised in revised_list:
            tensor_in_list.insert(revised, 0)
        return torch.tensor(tensor_in_list,dtype=tensor.dtype)

    def update_mask_arc_rel(self, mask, arc, rel, revised_list):
        return self.insert_list_to_tensor(mask, revised_list), \
               self.insert_list_to_tensor(arc, revised_list), \
               self.insert_list_to_tensor(rel, revised_list)

class Deleting(BlackBoxMethod):
    def __init__(self, config, vocab, tagger, ROOT_TAG):
        super(Deleting, self).__init__(tagger, vocab, ROOT_TAG)
        self.index = AttackIndexDeleting(config)



