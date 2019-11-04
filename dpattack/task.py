# -*- coding: utf-8 -*-

from dpattack.utils.metric import ParserMetric,TaggerMetric

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence


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

        for words, tags, arcs, rels in loader:
            self.optimizer.zero_grad()

            mask = words.ne(self.vocab.pad_index)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            tags = self.get_tag(words, tags, mask)
            s_arc, s_rel = self.model(words, tags)
            s_arc, s_rel = s_arc[mask], s_rel[mask]
            gold_arcs, gold_rels = arcs[mask], rels[mask]

            loss = self.get_loss(s_arc, s_rel, gold_arcs, gold_rels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()
            self.scheduler.step()

    @torch.no_grad()
    def evaluate(self, loader, punct=False, tagger=None):
        self.model.eval()

        loss, metric = 0, ParserMetric()

        for words, tags, arcs, rels in loader:
            mask = words.ne(self.vocab.pad_index)
            # ignore the first token of each sentence
            mask[:, 0] = 0

            tags = self.get_tag(words, tags, mask, tagger)
            # ignore all punctuation if not specified
            if not punct:
                puncts = words.new_tensor(self.vocab.puncts)
                mask &= words.unsqueeze(-1).ne(puncts).all(-1)
            s_arc, s_rel = self.model(words, tags)
            s_arc, s_rel = s_arc[mask], s_rel[mask]
            gold_arcs, gold_rels = arcs[mask], rels[mask]
            pred_arcs, pred_rels = self.decode(s_arc, s_rel)

            loss += self.get_loss(s_arc, s_rel, gold_arcs, gold_rels)
            metric(pred_arcs, pred_rels, gold_arcs, gold_rels)
        loss /= len(loader)

        return loss, metric

    @torch.no_grad()
    def predict(self, loader, tagger=None):
        self.model.eval()

        all_tags, all_arcs, all_rels = [], [], []
        for words, tags in loader:
            mask = words.ne(self.vocab.pad_index)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            lens = mask.sum(dim=1).tolist()
            tags = self.get_tag(words, tags, mask, tagger)

            s_arc, s_rel = self.model(words, tags)
            tags, s_arc, s_rel = tags[mask], s_arc[mask], s_rel[mask]
            pred_arcs, pred_rels = self.decode(s_arc, s_rel)

            all_tags.extend(torch.split(tags, lens))
            all_arcs.extend(torch.split(pred_arcs, lens))
            all_rels.extend(torch.split(pred_rels, lens))
        all_tags = [self.vocab.id2tag(seq) for seq in all_tags]
        all_arcs = [seq.tolist() for seq in all_arcs]
        all_rels = [self.vocab.id2rel(seq) for seq in all_rels]

        return all_tags, all_arcs, all_rels

    def get_loss(self, s_arc, s_rel, gold_arcs, gold_rels):
        s_rel = s_rel[torch.arange(len(s_rel)), gold_arcs]

        arc_loss = self.criterion(s_arc, gold_arcs)
        rel_loss = self.criterion(s_rel, gold_rels)
        loss = arc_loss + rel_loss

        return loss

    def decode(self, s_arc, s_rel):
        pred_arcs = s_arc.argmax(dim=-1)
        pred_rels = s_rel[torch.arange(len(s_rel)), pred_arcs].argmax(dim=-1)

        return pred_arcs, pred_rels

    def get_tag(self, words, tags, mask, tagger = None):
        if tagger is None:
            return tags
        else:
            tagger = tagger.eval()
            lens = mask.sum(dim=1).tolist()
            s_tags = tagger(words)
            pred_tags = s_tags[mask].argmax(-1)
            pred_tags = torch.split(pred_tags, lens)
            pred_tags = pad_sequence(pred_tags, True)
            pred_tags = torch.cat([tags[:,0].unsqueeze(1),pred_tags],dim=1)
            return pred_tags


class TaggerTask(Task):
    def __init__(self, vocab, model):
        super(TaggerTask, self).__init__(vocab, model)

        self.vocab = vocab
        self.model = model
        self.criterion = nn.CrossEntropyLoss()

    def train(self, loader, **kwargs):
        self.model.train()

        for words, tags, arcs, rels in loader:
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

        for words, tags, arcs, rels in loader:
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
        for words, tags in loader:
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
