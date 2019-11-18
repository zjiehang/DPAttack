from dpattack.utils.utils import get_blackbox_augmentor
from dpattack.utils.corpus import Corpus
from dpattack.cmds.zeng.blackbox.attackindex import *
from dpattack.libs.luna.pytorch import cast_list
from dpattack.utils.aug import CharTypoAug


class BlackBoxMethod(object):
    def __init__(self, vocab):
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
        super(Substituting, self).__init__(vocab)
        self.tagger = tagger
        self.ROOT_TAG = ROOT_TAG
        self.index = self.get_index(config, vocab, parser)
        self.aug = get_blackbox_augmentor(config.blackbox_model, config.path, config.revised_rate, vocab=vocab, ftrain=config.ftrain)

    def get_index(self, config, vocab=None, parser=None):
        if config.mode == 'augmentation':
            return AttackIndexRandomGenerator(config)
        if config.blackbox_index == 'pos':
            return AttackIndexPosTag(config)
        else:
            if parser is None and vocab is None:
                print('unk replacement can not missing dpattack and vocab')
                exit()
            return AttackIndexUnkReplacement(config, vocab=vocab, parser=parser)

    def generate_attack_seq(self, seqs, seq_idx, tags, tag_idx, chars, arcs, rels, mask):
        # generate word index to be attacked
        attack_index = self.index.get_attack_index(self.copy_str_to_list(seqs), seq_idx, tags, tag_idx, chars, arcs, mask)
        # generate word candidates to be attacked
        candidates, indexes = self.substituting(seqs, attack_index)
        # check candidates by pos_tagger
        attack_seq, revised_list = self.check_pos_tag(seqs, tag_idx, candidates, indexes)
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
            if succeed != self.FALSE_TOKEN:
                final_attack_seq[index] = candidates[count][succeed]
                revised_index_list.append(index)
        return final_attack_seq, revised_index_list

    def check_pos_tag_under_each_index(self, seqs, tag_idx, candidate, index):
        if len(candidate) == 0:
            return self.FALSE_TOKEN
        attack_seq_list = self.duplicate_sentence_with_candidate_replacement(seqs, candidate, index)
        # change seq list to idx
        # add <ROOT> token to the first token of each setence
        attack_seq_idx = torch.cat([self.vocab.word2id([Corpus.ROOT] + s).unsqueeze(0) for s in attack_seq_list], dim=0)
        attack_tag_idx = self.tagger.decorator_forward(attack_seq_idx, self.ROOT_TAG)
        tag_equal_flag = torch.eq(attack_tag_idx[:, index + 1], tag_idx[0, index + 1])
        if torch.sum(tag_equal_flag) != 0:
            attack_succeed_index = tag_equal_flag.nonzero().squeeze(0)
            return attack_succeed_index[0].item()
        return self.FALSE_TOKEN


class Inserting(BlackBoxMethod):
    def __init__(self, config, vocab, tagger, ROOT_TAG):
        super(Inserting, self).__init__(vocab)
        self.tagger = tagger
        self.ROOT_TAG = ROOT_TAG
        self.index = AttackIndexInserting(config)
        self.aug = get_blackbox_augmentor(config.blackbox_model, config.path, config.revised_rate, vocab=vocab,ftrain=config.ftrain)

    def generate_attack_seq(self, seqs, seq_idx, tags, tag_idx, chars, arcs, rels, mask):
        # seq_idx, tag_idx, arcs, rels, mask = map(lambda x:x.squeeze(0) if len(x.shape)==2 else x,[seq_idx, tag_idx, arcs, rels, mask])
        # generate word index to be attacked
        gold_arcs = cast_list(arcs)
        attack_index = self.index.get_attack_index(self.copy_str_to_list(seqs), seq_idx, tags, tag_idx, chars, arcs, mask)
        # generate word candidates to be attacked
        candidates, indexes = self.inserting(seqs, attack_index)
        # check candidates by pos_tagger
        attack_seq, attack_mask, attack_gold_arc, attack_gold_rel, revised_number = self.check_pos_tag(seqs,
                                                                                                       tags,
                                                                                                       cast_list(mask),
                                                                                                       gold_arcs,
                                                                                                       self.vocab.id2rel(rels),
                                                                                                       candidates,
                                                                                                       indexes)


        attack_mask = torch.tensor(attack_mask,dtype=mask.dtype)
        attack_gold_arc = torch.tensor(attack_gold_arc, dtype=arcs.dtype)
        attack_gold_rel = self.vocab.rel2id(attack_gold_rel)
        attack_mask, attack_gold_arc, attack_gold_rel = map(lambda x:x.unsqueeze(0) if len(x.shape)==1 else x, [attack_mask, attack_gold_arc, attack_gold_rel])
        attack_mask, attack_gold_arc, attack_gold_rel = map(lambda x:x.cuda() if torch.cuda.is_available() else x, [attack_mask, attack_gold_arc, attack_gold_rel])

        return [Corpus.ROOT] + attack_seq, attack_mask, attack_gold_arc, attack_gold_rel, revised_number

    def inserting(self, seq, index):
        try:
            # generate the attack sentence by index
            candidates, revised_indexes = self.aug.insert(seq, aug_idxes = index)
        except Exception:
            candidates = None
            revised_indexes = []
        return candidates, revised_indexes

    def check_pos_tag(self, seqs, tags, mask, arcs, rels, candidates, indexes):
        final_attack_seq = self.copy_str_to_list(seqs)
        #final_attack_tag = tags.copy()
        #revised_index_list = []
        revised_number = 0
        for count, index in enumerate(indexes):
            succeed = self.check_pos_tag_under_each_index(seqs, tags, candidates[count], index)
            if succeed != self.FALSE_TOKEN:
                final_attack_seq.insert(index, candidates[count][succeed])
                mask.insert(index + 1, 0)
                arcs = self.get_gold_arc_under_inserting(tags, arcs, index)
                rels = self.get_gold_rel_under_inserting(tags, rels, index)
                revised_number += 1

        return final_attack_seq, mask, arcs, rels, revised_number

    def check_pos_tag_under_each_index(self, seqs, tags, candidate, index):
        seqs_insert_token = self.insert_token_to_str_list(seqs, index, self.vocab.UNK)
        attack_seq_list = self.duplicate_sentence_with_candidate_replacement(seqs_insert_token, candidate, index)
        # change seq list to idx
        # add <ROOT> token to the first token of each setence
        attack_seq_idx = torch.cat([self.vocab.word2id([Corpus.ROOT] + s).unsqueeze(0) for s in attack_seq_list], dim=0)
        attack_tag_idx = self.tagger.decorator_forward(attack_seq_idx, self.ROOT_TAG, return_device='cpu')
        gold_tag_idx = self.get_gold_tag_idx_under_inserting(tags, index)
        tag_equal_flag = torch.eq(attack_tag_idx[:, index + 1], gold_tag_idx[index + 1])
        if torch.sum(tag_equal_flag) != 0:
            attack_succeed_index = tag_equal_flag.nonzero().squeeze(0)
            return attack_succeed_index[0].item()
        return self.FALSE_TOKEN

    def get_gold_tag_under_inserting(self, tag, index):
        gold_tag = tag.copy()
        if tag[index + 1] in CONSTANT.NOUN_TAG:
            gold_tag.insert(index + 1, CONSTANT.ADJ_TAG)
        elif tag[index].startswith(CONSTANT.VERB_TAG):
            gold_tag.insert(index + 1, CONSTANT.ADV_TAG)
        return gold_tag

    def get_gold_arc_under_inserting(self, tag, arcs, index):
        if tag[index + 1] in CONSTANT.NOUN_TAG:
            arcs.insert(index + 1, index + 1)
        elif tag[index].startswith(CONSTANT.VERB_TAG):
            arcs.insert(index + 1, index)
        return [arc + 1 if arc > index else arc for arc in arcs]


    def get_gold_rel_under_inserting(self, tag, rel, index):
        if tag[index + 1] in CONSTANT.NOUN_TAG:
            rel.insert(index + 1, CONSTANT.JJ_REL_MODIFIER)
        elif tag[index].startswith(CONSTANT.VERB_TAG):
            rel.insert(index + 1, CONSTANT.RB_REL_MODIFIER)
        return rel

    def get_gold_tag_idx_under_inserting(self, tag, index):
        gold_tag = self.get_gold_tag_under_inserting(tag, index)
        return self.vocab.tag2id(gold_tag)


    def update_mask(self, mask, revised_list):
        if len(revised_list)==0:
            return mask
        mask_in_list = cast_list(mask)
        for revised in revised_list:
            mask_in_list.insert(revised, 0)
        return torch.tensor(mask_in_list,dtype=mask.dtype)

    def update_arcs(self,arcs,tags,revised_list):
        if len(revised_list)==0:
            return arcs
        arcs_in_list = cast_list(arcs)
        for revised in revised_list:
            arcs_in_list = [arcs + 1 if arcs > revised else arcs for arcs in arcs_in_list]
            arcs_in_list.insert(revised+1 if tags[revised].startswith(CONSTANT.ADJ_TAG) else revised-1,revised)
        return torch.tensor(arcs_in_list,dtype=arcs.dtype)

    def update_rels(self,rels,tags,revised_list):
        if len(revised_list)==0:
            return rels
        rels_in_list = self.vocab.id2rel(rels)
        for revised in revised_list:
            rels_in_list.insert(CONSTANT.JJ_REL_MODIFIER if tags[revised].startswith(CONSTANT.ADJ_TAG) else CONSTANT.RB_REL_MODIFIER,revised)
        return self.vocab.rel2id(rels_in_list)


class InsertingPunct(BlackBoxMethod):
    def __init__(self, config, vocab):
        super(InsertingPunct, self).__init__(vocab)

        self.index = AttackIndexInsertingPunct(config, vocab)

    def generate_attack_seq(self, seqs, seq_idx, tags, tag_idx, chars, arcs, rels, mask):
        attack_index = self.index.get_attack_index(self.copy_str_to_list(seqs), seq_idx, tags, tag_idx, chars, arcs, mask)

        attack_index.sort(reverse=True)
        attack_mask = cast_list(mask)
        attack_arcs = cast_list(arcs)
        attack_rels = cast_list(rels)
        attack_seqs = self.copy_str_to_list(seqs)

        for index in attack_index:
            attack_seqs.insert(index, CONSTANT.COMMA)
            attack_mask.insert(index + 1, 0)
            attack_rels.insert(index + 1, 0)
            attack_arcs.insert(index + 1, 0)
            attack_arcs = [arc + 1 if arc > index else arc for arc in attack_arcs]

        attack_mask = torch.tensor(attack_mask, dtype=mask.dtype)
        attack_arcs = torch.tensor(attack_arcs, dtype=arcs.dtype)
        attack_rels = torch.tensor(attack_rels, dtype=rels.dtype)
        attack_mask, attack_arcs, attack_rels = map(lambda x: x.unsqueeze(0) if len(x.shape) == 1 else x,
                                                            [attack_mask, attack_arcs, attack_rels])
        attack_mask, attack_arcs, attack_rels = map(lambda x: x.cuda() if torch.cuda.is_available() else x,
                                                            [attack_mask, attack_arcs, attack_rels])
        return [Corpus.ROOT] + attack_seqs, attack_mask, attack_arcs, attack_rels, len(attack_index)


class Deleting(BlackBoxMethod):
    def __init__(self, config, vocab):
        super(Deleting, self).__init__(vocab)
        self.index = AttackIndexDeleting(config)

    def generate_attack_seq(self, seqs, seq_idx, tags, tag_idx, chars, arcs, rels, mask):
        seq_idx, tag_idx, arcs, rels, mask = map(lambda x:x.squeeze(0) if len(x.shape)==2 else x,[seq_idx, tag_idx, arcs, rels, mask])

        gold_arcs = cast_list(arcs)
        attack_index = self.index.get_attack_index(self.copy_str_to_list(seqs), seq_idx, tags, tag_idx, chars, arcs, mask)
        attack_index.sort(reverse=True)

        attack_seq = [Corpus.ROOT] + seqs.split()
        attack_mask = cast_list(mask)
        attack_gold_arc = gold_arcs.copy()
        attack_gold_rel = cast_list(rels)

        for index in attack_index:
            del attack_seq[index]
            del attack_mask[index]
            del attack_gold_arc[index]
            del attack_gold_rel[index]
            attack_gold_arc = [arc - 1 if arc > index else arc for arc in attack_gold_arc]

        attack_mask = torch.tensor(attack_mask,dtype=mask.dtype)
        attack_gold_arc = torch.tensor(attack_gold_arc, dtype=arcs.dtype)
        attack_gold_rel = torch.tensor(attack_gold_rel, dtype=rels.dtype)
        attack_mask, attack_gold_arc, attack_gold_rel = map(lambda x:x.unsqueeze(0) if len(x.shape)==1 else x, [attack_mask, attack_gold_arc, attack_gold_rel])
        attack_mask, attack_gold_arc, attack_gold_rel = map(lambda x:x.cuda() if torch.cuda.is_available() else x, [attack_mask, attack_gold_arc, attack_gold_rel])

        return attack_seq, attack_mask, attack_gold_arc, attack_gold_rel, len(attack_index)


class DeletingPunct(BlackBoxMethod):
    def __init__(self, config, vocab):
        super(DeletingPunct, self).__init__(vocab)

        self.index = AttackIndexDeletingPunct(config, vocab)

    def generate_attack_seq(self, seqs, seq_idx, tags, tag_idx, chars, arcs, rels, mask):
        gold_arcs = cast_list(arcs)
        attack_index = self.index.get_attack_index(self.copy_str_to_list(seqs), seq_idx, tags, tag_idx, chars, arcs, mask)

        attack_index.sort(reverse=True)
        attack_mask = cast_list(mask)
        attack_arcs = gold_arcs.copy()
        attack_rels = cast_list(rels)
        attack_seqs = self.copy_str_to_list(seqs)

        for index in attack_index:
            del attack_seqs[index]
            del attack_mask[index + 1]
            del attack_arcs[index + 1]
            del attack_rels[index + 1]
            attack_arcs = [arc - 1 if arc > index else arc for arc in attack_arcs]

        attack_mask = torch.tensor(attack_mask, dtype=mask.dtype)
        attack_arcs = torch.tensor(attack_arcs, dtype=arcs.dtype)
        attack_rels = torch.tensor(attack_rels, dtype=rels.dtype)
        attack_mask, attack_arcs, attack_rels = map(lambda x: x.unsqueeze(0) if len(x.shape) == 1 else x,
                                                            [attack_mask, attack_arcs, attack_rels])
        attack_mask, attack_arcs, attack_rels = map(lambda x: x.cuda() if torch.cuda.is_available() else x,
                                                            [attack_mask, attack_arcs, attack_rels])
        return [Corpus.ROOT] + attack_seqs, attack_mask, attack_arcs, attack_rels, len(attack_index)


class CharTypo(BlackBoxMethod):
    def __init__(self, config, vocab, parser=None):
        super(CharTypo,self).__init__(vocab)
        self.index = self.get_index(config, vocab, parser)
        self.aug = CharTypoAug(vocab.char_dict)

    def get_index(self, config, vocab=None, parser=None):
        if config.mode == 'augmentation':
            return AttackIndexRandomGenerator(config)
        if config.blackbox_index == 'pos':
            return AttackIndexPosTag(config)
        else:
            if parser is None and vocab is None:
                print('unk replacement can not missing dpattack and vocab')
                exit()
            return AttackIndexUnkReplacement(config, vocab=vocab, parser=parser)

    def generate_attack_seq(self, seqs, seq_idx, tags, tag_idx, chars, arcs, rels, mask):
        # generate word index to be attacked
        attack_index = self.index.get_attack_index(self.copy_str_to_list(seqs), seq_idx, tags, tag_idx, chars, arcs, mask)

        attack_seq = self.aug.get_typos(seqs, attack_index)

        return [Corpus.ROOT] + attack_seq, mask, arcs, rels, len(attack_index)