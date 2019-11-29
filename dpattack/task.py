# -*- coding: utf-8 -*-

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from dpattack.utils.metric import ParserMetric, TaggerMetric
from dpattack.utils.parser_helper import is_chars_judger
from dpattack.libs.luna import cast_list


class Task(object):
    def __init__(self, vocab, model):
        self.vocab = vocab
        self.model = model

    def train(self, loader, **kwargs):
        pass

    @torch.no_grad()
    def evaluate(self, loader, **kwargs):
        pass

    @torch.no_grad()
    def predict(self, loader, **kwargs):
        pass


class ParserTask(Task):
    def __init__(self, vocab, model):
        super(ParserTask, self).__init__(vocab, model)

        self.vocab = vocab
        self.model = model
        self.criterion = nn.CrossEntropyLoss()

    def train(self, loader):
        self.model.train()

        for words, tags, chars, arcs, rels in loader:
            self.optimizer.zero_grad()

            mask = words.ne(self.vocab.pad_index)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            # tags = self.get_tag(words, tags, mask)
            s_arc, s_rel = self.model(
                words, is_chars_judger(self.model, tags, chars))
            s_arc, s_rel = s_arc[mask], s_rel[mask]
            gold_arcs, gold_rels = arcs[mask], rels[mask]

            loss = self.get_loss(s_arc, s_rel, gold_arcs, gold_rels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()
            self.scheduler.step()

    @torch.no_grad()
    def evaluate(self, loader, punct=False, tagger=None, mst=False):
        self.model.eval()

        loss, metric = 0, ParserMetric()

        for words, tags, chars, arcs, rels in loader:
            mask = words.ne(self.vocab.pad_index)
            # ignore the first token of each sentence
            mask[:, 0] = 0

            tags = self.get_tags(words, tags, mask, tagger)
            
            s_arc, s_rel = self.model(
                words, is_chars_judger(self.model, tags, chars))

            loss += self.get_loss(s_arc[mask], s_rel[mask], arcs[mask], rels[mask])
            pred_arcs, pred_rels = self.decode(s_arc, s_rel, mask, mst)

            # ignore all punctuation if not specified
            if not punct:
                puncts = words.new_tensor(self.vocab.puncts)
                mask &= words.unsqueeze(-1).ne(puncts).all(-1)
            pred_arcs, pred_rels = pred_arcs[mask], pred_rels[mask]
            gold_arcs, gold_rels = arcs[mask], rels[mask]

            metric(pred_arcs, pred_rels, gold_arcs, gold_rels)
        loss /= len(loader)

        return loss, metric

    # WARNING: DIRTY CODE >>>>>>>>>>>>>>>>>>>>>>>>>>>
    @torch.no_grad()
    def partial_evaluate(self, instance: tuple,
                         mask_idxs: List[int],
                         punct=False, tagger=None, mst=False):
        self.model.eval()

        loss, metric = 0, ParserMetric()

        words, tags, chars, arcs, rels = instance

        mask = words.ne(self.vocab.pad_index)
        # ignore the first token of each sentence
        mask[:, 0] = 0
        decode_mask = mask.clone()

        tags = self.get_tags(words, tags, mask, tagger)
        # ignore all punctuation if not specified
        if not punct:
            puncts = words.new_tensor(self.vocab.puncts)
            mask &= words.unsqueeze(-1).ne(puncts).all(-1)
        s_arc, s_rel = self.model(
            words, is_chars_judger(self.model, tags, chars))

        # mask given indices
        for idx in mask_idxs:
            mask[0][idx] = 0

        pred_arcs, pred_rels = self.decode(s_arc, s_rel, decode_mask, mst)

        pred_arcs, pred_rels = pred_arcs[mask], pred_rels[mask]
        gold_arcs, gold_rels = arcs[mask], rels[mask]

        # exmask = torch.ones_like(gold_arcs, dtype=torch.uint8)

        # for i, ele in enumerate(cast_list(gold_arcs)):
        #     if ele in mask_idxs:
        #         exmask[i] = 0
        # for i, ele in enumerate(cast_list(pred_arcs)):
        #     if ele in mask_idxs:
        #         exmask[i] = 0
        # gold_arcs = gold_arcs[exmask]
        # pred_arcs = pred_arcs[exmask]
        # gold_rels = gold_rels[exmask]
        # pred_rels = pred_rels[exmask]

        # loss += self.get_loss(s_arc, s_rel, gold_arcs, gold_rels)
        metric(pred_arcs, pred_rels, gold_arcs, gold_rels)

        return metric
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    @torch.no_grad()
    def predict(self, loader, tagger=None, mst=False):
        self.model.eval()

        all_tags, all_arcs, all_rels = [], [], []
        for words, tags, chars in loader:
            mask = words.ne(self.vocab.pad_index)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            lens = mask.sum(dim=1).tolist()

            tags = self.get_tags(words, tags, mask, tagger)
            s_arc, s_rel = self.model(
                words, is_chars_judger(self.model, tags, chars))

            pred_arcs, pred_rels = self.decode(s_arc, s_rel, mask, mst)
            tags, pred_arcs, pred_rels = tags[mask], pred_arcs[mask], pred_rels[mask]
            

            all_tags.extend(torch.split(tags, lens))
            all_arcs.extend(torch.split(pred_arcs, lens))
            all_rels.extend(torch.split(pred_rels, lens))
        all_tags = [self.vocab.id2tag(seq) for seq in all_tags]
        all_arcs = [seq.tolist() for seq in all_arcs]
        all_rels = [self.vocab.id2rel(seq) for seq in all_rels]

        return all_tags, all_arcs, all_rels

    def get_loss(self, s_arc, s_rel, gold_arcs, gold_rels):
        s_rel = s_rel[torch.arange(len(s_rel)), gold_arcs]
        # s_rel = s_rel[torch.arange(len(gold_arcs)), gold_arcs]

        arc_loss = self.criterion(s_arc, gold_arcs)
        rel_loss = self.criterion(s_rel, gold_rels)
        loss = arc_loss + rel_loss

        return loss

    def decode(self, s_arc, s_rel, mask, mst):
        from dpattack.utils.alg import eisner
        if mst:
            pred_arcs = eisner(s_arc, mask)
        else:
            pred_arcs = s_arc.argmax(dim = -1)
        # pred_arcs = s_arc.argmax(dim=-1)
        # pred_rels = s_rel[torch.arange(len(s_rel)), pred_arcs].argmax(dim=-1)
        pred_rels = s_rel.argmax(-1)
        pred_rels = pred_rels.gather(-1, pred_arcs.unsqueeze(-1)).squeeze(-1)

        return pred_arcs, pred_rels

    def get_tags(self, words, tags, mask, tagger):
        if tagger is None:
            return tags
        else:
            tagger = tagger.eval()
            lens = mask.sum(dim=1).tolist()
            s_tags = tagger(words)
            pred_tags = s_tags[mask].argmax(-1)
            pred_tags = torch.split(pred_tags, lens)
            pred_tags = pad_sequence(pred_tags, True)
            pred_tags = torch.cat(
                [torch.zeros_like(pred_tags[:, :1]), pred_tags], dim=1)
            return pred_tags


class TaggerTask(Task):
    def __init__(self, vocab, model):
        super(TaggerTask, self).__init__(vocab, model)

        self.vocab = vocab
        self.model = model
        self.criterion = nn.CrossEntropyLoss()

    def train(self, loader, **kwargs):
        self.model.train()

        for words, tags, chars, arcs, rels in loader:
            self.optimizer.zero_grad()

            mask = words.ne(self.vocab.pad_index)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            s_tags = self.model(words)
            s_tags = s_tags[mask]
            gold_tags = tags[mask]

            loss = self.get_loss(s_tags, gold_tags)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()
            self.scheduler.step()

    @torch.no_grad()
    def evaluate(self, loader, punct=False, **kwargs):
        self.model.eval()

        loss, metric = 0, TaggerMetric()

        for words, tags, chars, arcs, rels in loader:
            mask = words.ne(self.vocab.pad_index)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            # ignore all punctuation if not specified
            s_tags = self.model(words)
            s_tags = s_tags[mask]
            gold_tags = tags[mask]
            pred_tags = self.decode(s_tags)

            loss += self.get_loss(s_tags, gold_tags)
            metric(pred_tags, gold_tags)
        loss /= len(loader)

        return loss, metric

    @torch.no_grad()
    def predict(self, loader, **kwargs):
        self.model.eval()

        all_tags = []
        for words, tags, chars in loader:
            mask = words.ne(self.vocab.pad_index)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            lens = mask.sum(dim=1).tolist()
            s_tags = self.model(words)
            s_tags = s_tags[mask]
            pred_tags = self.decode(s_tags)

            all_tags.extend(torch.split(pred_tags, lens))
        all_tags = [self.vocab.id2tag(seq) for seq in all_tags]
        return all_tags

    def get_loss(self, s_tags, gold_tags):
        loss = self.criterion(s_tags, gold_tags)
        return loss

    def decode(self, s_tags):
        pred_tags = s_tags.argmax(dim=-1)
        return pred_tags
