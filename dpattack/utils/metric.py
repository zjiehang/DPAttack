# -*- coding: utf-8 -*-


class Metric(object):
    def __init__(self, eps=1e-5):
        pass

    def __repr__(self):
        pass

    def __lt__(self, other):
        pass

    def __le__(self, other):
        pass

    def __ge__(self, other):
        pass

    def __gt__(self, other):
        pass

    @property
    def score(self):
        pass


class ParserMetric(Metric):

    def __init__(self, eps=1e-5):
        super(ParserMetric, self).__init__()

        self.eps = eps
        self.total = 0.0
        self.correct_arcs = 0.0
        self.correct_rels = 0.0

    def __repr__(self):
        return f"UAS: {self.uas:.2%} LAS: {self.las:.2%}"

    def __add__(self, other: "ParserMetric"):
        metric = ParserMetric()
        metric.eps = self.eps
        metric.total = self.total + other.total
        metric.correct_arcs = self.correct_arcs + other.correct_arcs
        metric.correct_rels = self.correct_rels + other.correct_rels
        return metric

    def __call__(self, pred_arcs, pred_rels, gold_arcs, gold_rels):
        arc_mask = pred_arcs.eq(gold_arcs)
        rel_mask = pred_rels.eq(gold_rels) & arc_mask

        self.total += len(arc_mask)
        self.correct_arcs += arc_mask.sum().item()
        self.correct_rels += rel_mask.sum().item()

    def __lt__(self, other):
        return self.score < other

    def __le__(self, other):
        return self.score <= other

    def __ge__(self, other):
        return self.score >= other

    def __gt__(self, other):
        return self.score > other

    @property
    def score(self):
        return self.las

    @property
    def uas(self):
        return self.correct_arcs / (self.total + self.eps)

    @property
    def las(self):
        return self.correct_rels / (self.total + self.eps)

    def update(self, metric):
        self.correct_arcs += metric.correct_arcs
        self.correct_rels += metric.correct_rels

        self.total += metric.total


class TaggerMetric(Metric):

    def __init__(self, eps=1e-5):
        super(TaggerMetric, self).__init__()

        self.eps = eps
        self.total = 0.0
        self.correct_tags = 0.0
        

    def __repr__(self):
        return f"TAGS: {self.tags:.2%}"

    def __call__(self, pred_tags, gold_tags):
        tag_mask = pred_tags.eq(gold_tags)

        self.total += len(tag_mask)
        self.correct_tags += tag_mask.sum().item()

    def __lt__(self, other):
        return self.score < other

    def __le__(self, other):
        return self.score <= other

    def __ge__(self, other):
        return self.score >= other

    def __gt__(self, other):
        return self.score > other

    @property
    def score(self):
        return self.tags

    @property
    def tags(self):
        return self.correct_tags / (self.total + self.eps)

    def update(self, metric):
        self.correct_tags += metric.correct_tags
        self.total += metric.total