# -*- coding: utf-8 -*-
import math
import random
import re
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, List, Optional, Union
import numpy as np

import torch
from tabulate import tabulate
from torch.utils.data import DataLoader

from dpattack.libs.luna import (Aggregator, CherryPicker, Color, TrainingStopObserver, as_table,
                                auto_create, cast_list, fetch_best_ckpt_name, flt2str, idx_to_msk,
                                log, log_config, ram_append, ram_pop, ram_read, ram_reset,
                                ram_write, show_mean_std, show_num_list, time, ram_has)
from dpattack.models import PosTagger, WordParser, WordTagParser
from dpattack.task import ParserTask
from dpattack.utils.corpus import Corpus, Sentence, sent_print
from dpattack.utils.data import TextDataset, collate_fn
from dpattack.utils.embedding_searcher import (EmbeddingSearcher, cos_dist, euc_dist)
from dpattack.utils.metric import Metric, ParserMetric
from dpattack.utils.parser_helper import load_parser
from dpattack.utils.tag_tool import gen_tag_dict
from dpattack.utils.vocab import Vocab

from .ihack import HACK_TAGS, IHack, v
from .hack_util import elder_select
from .treeops import gen_spans, filter_spans
from dpattack.cmds.zhou.treeops import check_gap, ex_span_idx, get_gap
from dpattack.cmds.zhou.hack_util import gen_idxs_to_substitute, subsitute_by_idxs, subsitute_by_idxs_2
from functools import lru_cache


class HackSubtree(IHack):

    # tag_type: njvri
    @lru_cache(maxsize=None)
    def njvri_subidxs(self, njvri_tags):
        ret = []
        for t in njvri_tags:
            for ptb_tag in HACK_TAGS[t]:
                if ptb_tag in self.tag_dict:
                    ret.extend(cast_list(self.tag_dict[ptb_tag]))
        return ret

    # tag_type: univ
    @lru_cache(maxsize=None)
    def univ_subidxs(self, univ_tag):
        ret = []
        for ptbtag in HACK_TAGS.uni2ptb(univ_tag):
            if ptbtag in self.tag_dict:
                ret.extend(cast_list(self.tag_dict[ptbtag]))
        return ret

    def __call__(self, config):
        self.init_logger(config)
        self.setup(config)
        self.blackbox_sub_idxs = [
            ele for ele in range(self.vocab.n_words) if ele not in
            [self.vocab.pad_index, self.vocab.unk_index, self.vocab.word_dict['<root>']]
        ]

        agg = Aggregator()
        for sid, (words, tags, chars, arcs, rels) in enumerate(self.loader):
            # if sid > 100:
            #     continue
            # zjh = [
            #     20, 41, 46, 117, 137, 143, 183, 198, 258, 295, 310, 350, 410, 421, 464, 485, 512,
            #     528, 544, 600, 601, 681, 702, 728, 735, 738, 762, 783, 794, 803, 805, 821, 844, 845,
            #     921, 931, 937, 939, 948, 962, 968, 975, 1019, 1044, 1068, 1069, 1096, 1104, 1121,
            #     1122, 1138, 1142, 1155, 1163, 1180, 1197, 1224, 1228, 1270, 1272, 1292, 1306, 1315,
            #     1317, 1342, 1345, 1393, 1400, 1431, 1478, 1503, 1522, 1524, 1526, 1608, 1677, 1729,
            #     1759, 1775, 1795, 1811, 1831, 1925, 1929, 1983, 1984, 2026, 2031, 2176, 2234, 2291,
            #     2296, 2318, 2327, 2330, 2342, 2343, 2355, 2360
            # ]
            # if sid + 1 not in zjh:
            #     continue
            # if sid > 600 - 1:
            #     continue

            words_text = self.vocab.id2word(words[0])
            tags_text = self.vocab.id2tag(tags[0])
            log('****** {}: \n{}\n{}'.format(sid, " ".join(words_text), " ".join(tags_text)))

            result = self.meta_hack(instance=(words, tags, chars, arcs, rels),
                                    sentence=self.corpus[sid])

            if result is None:
                continue

            # yapf: disable
            agg.aggregate(
                ("iters", result['iters']),
                ("time", result['time']),
                ("succ", result['succ']),
                ('best_iter', result['best_iter']),
                ("changed", result['num_changed']),
                ("att_id", result['att_id']),
                ("meta_time", result['meta_time']),
                ("meta_trial_pair", result['meta_trial_pair']),
                ("meta_total_pair", result['meta_total_pair']),
                ("meta_succ_trial_pair", result['meta_succ_trial_pair']),
                ("meta_succ_total_pair", result['meta_succ_total_pair']),
            )

            # # WARNING: SOME SENTENCE NOT SHOWN!
            if result:
                log('Show result from iter {}:'.format(result['best_iter']))
                log(result['logtable'])

            log('Aggregated result: '
                'iters(avg) {:.1f}, time(avg) {:.1f}s, meta_time(avg) {:.1f}s, '
                'succ rate {:.2f}% ({}/{}), best_iter(avg) {:.1f}, best_iter(std) {:.1f}, '
                'changed(avg) {:.1f}, '
                'total pair {}, trial pair {}, '
                'succ total pair {}, succ trial pair {}, '
                'succ att id {}'.format(
                    agg.mean('iters'), agg.mean('time'), agg.mean('meta_time'),
                    agg.mean('succ') * 100, agg.sum('succ'), agg.size,
                    agg.mean('best_iter'), agg.std('best_iter'),
                    agg.mean('changed'),
                    agg.sum('meta_total_pair'), agg.sum('meta_trial_pair'),
                    agg.sum('meta_succ_total_pair'), agg.sum('meta_succ_trial_pair'),
                    agg.aggregated(key='att_id', reduce=np.nanmean)
                ))
            log()
            # yapf: enable

            # exit()  # HIGHLIGHT:

    # WARNING:
    # By default, the returned value does not contain the <ROOT> node!!!
    def compute_margin(self, instance, mask_idxs=[]):
        self.parser.zero_grad()
        words, tags, chars, arcs, rels = instance
        s_arc, s_rel = self.parser(words, tags)

        mask = words.ne(self.vocab.pad_index)
        mask[0, 0] = 0
        for mask_idx in mask_idxs:
            mask[0, mask_idx] = 0
        s_arc, s_rel = s_arc[mask], s_rel[mask]
        gold_arcs, gold_rels = arcs[mask], rels[mask]  # shape like [7,7,7,0,3]

        # max margin loss
        msk_gold_arc = idx_to_msk(gold_arcs, num_classes=s_arc.size(1))
        s_gold = s_arc[msk_gold_arc]
        s_other = s_arc.masked_fill(msk_gold_arc, -1000.)
        max_s_other, _ = s_other.max(1)
        margin = s_gold - max_s_other
        return margin

    # WARNING: The returned grads contain the <root>
    def backward_loss(self, instance, mask_idxs, verbose=False) -> torch.Tensor:
        # margin = gold - max_non_gold, bigger is better.
        # when attacking, we attempt to decrease it.
        margin = self.compute_margin(instance, mask_idxs)
        margin[margin < -1] = -1
        # if verbose:
        #     log("\t> ", flt2str(margin, cat=" ", fmt=":4.1f"), color=Color.red)

        # loss = margin[]
        if self.config.hks_loss == 'sum':
            loss = margin.sum()
        elif self.config.hks_loss == 'min':
            loss = margin[margin > 0].min()
        else:
            raise Exception()
        loss.backward()

        return ram_pop('embed_grad')

    def select_src_span(self, instance, sentence, tgt_span, mode):
        minl = self.config.hks_min_span_len
        maxl = self.config.hks_max_span_len
        gap = self.config.hks_span_gap
        if mode == "cls":
            spans = filter_spans(gen_spans(sentence), minl, maxl, True)
            src_span = None
            src_picker = CherryPicker(lower_is_better=True)
            for span in spans:
                stgap = get_gap(span, tgt_span)
                if stgap >= gap:
                    src_picker.add(stgap, span)
            if src_picker.size == 0:
                log('Source span not found')
                return None
            _, _, src_span = src_picker.select_best_point()
            return src_span
        elif mode == 'rdm':
            spans = filter_spans(gen_spans(sentence), minl, maxl, True)
            src_spans = []
            for span in spans:
                if check_gap(span, tgt_span, self.config.hks_span_gap):
                    src_spans.append(span)
            if len(src_spans) == 0:
                return None
            return random.choice(src_spans)
        elif mode == 'nom':
            sent_len = instance[0].size(1)
            spans = filter_spans(gen_spans(sentence), minl, maxl, True)
            embed_grad = self.backward_loss(instance, ex_span_idx(tgt_span, sent_len))
            grad_norm = embed_grad.norm(dim=2)  # 1 x sent_len, <root> included

            src_picker = CherryPicker(lower_is_better=False)
            for span in spans:
                if check_gap(span, tgt_span, self.config.hks_span_gap):
                    src_picker.add(grad_norm[0][span[0]:span[1] + 1].sum(), span)
            if src_picker.size == 0:
                return None
            _, _, src_span = src_picker.select_best_point()
            return src_span
        else:
            raise Exception

    def select_tgt_span(self, instance, sentence, mode):
        minl = self.config.hks_min_span_len
        maxl = self.config.hks_max_span_len
        if mode == 'vul':
            # Compute the ``vulnerable'' values of each words
            words, tags, chars, arcs, rels = instance
            # sent_len = words.size(1)

            # The returned margins does not contain <ROOT>
            margins = self.compute_margin(instance)
            log(margins)

            # Count the vulnerable words in each span,
            # Select the most vulerable span as target.
            vul_margins = [0]
            for ele in margins:
                if 0 < ele < 1:
                    vul_margins.append(100)
                elif 1 <= ele < 2:
                    vul_margins.append(1)
                else:
                    vul_margins.append(-1)
            spans = filter_spans(gen_spans(sentence), minl, maxl, True)
            span_vuls = list()
            # span_ratios = list()
            for span in spans:
                span_vuls.append(sum(vul_margins[span[0]:span[1] + 1]))
                # span_ratios.append(span_vuls[-1] / (span[1] + 1 - span[0]))

            pairs = []
            for tid, t in enumerate(spans):
                for s in spans:
                    if check_gap(s, t, self.config.hks_span_gap):
                        pairs.append((span_vuls[tid], (t, s)))
            if len(pairs) == 0:
                return None
            spairs = sorted(pairs, key=lambda x: x[0], reverse=True)
            return list(zip(*spairs))[1]

            tgt_picker = CherryPicker(lower_is_better=False)
            for span_i, span in enumerate(spans):
                tgt_picker.add(span_vuls[span_i], span)
            if tgt_picker.size == 0:
                log('Target span not found')
                return None
            _, _, tgt_span = tgt_picker.select_best_point()
            return tgt_span

        elif mode == "rdm":
            spans = filter_spans(gen_spans(sentence), minl, maxl, True)
            if len(spans) == 0:
                return None
            return random.choice(spans)
        else:
            raise Exception

    # There may be two versions of a ``phrase jacobian norm''
    #   1st: - compute partial(y_[s0, s1], x[t0, t1]) to get the gradient
    #        - compute each norm and sum
    #        -> the norm = ||\sigma G_ts||
    #   2nd: - compute partial(y_i, x_i) to get a gradient matrix NxNxD
    #        - compute the norm of each element -> NxN
    #        - compute the influence of a span on another by summing the rectange area
    #        -> the norm = \sigma ||G_ts||
    def select_by_jacobian(self, instance, sentence, mode):
        sent_len = instance[0].size(1)
        spans = filter_spans(gen_spans(sentence), self.config.hks_min_span_len,
                             self.config.hks_max_span_len, True)
        if mode == 'jacobian1':
            table = []
            pairs = []
            for tgt_span in spans:
                embed_grad = self.backward_loss(instance, ex_span_idx(tgt_span, sent_len))
                grad_norm = embed_grad.norm(dim=-1)  # 1 x (sen_len - 1)
                row = [tgt_span]
                for src_span in spans:
                    if check_gap(src_span, tgt_span, self.config.hks_span_gap):
                        # <root> not included
                        norm = grad_norm[0][src_span[0] - 1:src_span[1]].sum().item()
                        pairs.append((norm, (tgt_span, src_span)))
                        row.append("{:4.2f}".format(norm))
                    else:
                        row.append("-")
                table.append(row)
            log(tabulate(table, headers=["t↓"] + spans))
            if len(pairs) == 0:
                return None
            spairs = sorted(pairs, key=lambda x: x[0], reverse=True)

            return list(zip(*spairs))[1]
        elif mode == 'jacobian2':
            # log('valid', spans)
            # margins = self.compute_margin(instance)
            # log(margins)
            norms = []
            # WARNING: This is rather slow!
            idxs_to_backward = []
            for span in spans:
                idxs_to_backward.extend(list(range(span[0], span[1] + 1)))
            for i in range(1, sent_len):
                if i in idxs_to_backward:
                    embed_grad = self.backward_loss(instance,
                                                    [_ for _ in range(1, sent_len) if _ != i])
                    grad_norm = embed_grad.norm(dim=-1)[:, 1:]
                    norms.append(grad_norm)
                else:
                    norms.append(torch.zeros_like(instance[0]).float()[:, 1:])
            norms = torch.cat(norms)  # size: sent_len x sent_len
            # log("t↓", flt2str(list(range(sent_len)), cat=" ", fmt=":3"))
            # for i, norm in enumerate(norms):
            #     log("{:2}".format(i + 1),
            #         flt2str(cast_list(norm), cat=" ", fmt=":3.1f"),
            #         flt2str(sum(norm), fmt=":3.1f"))

            table = []
            for t in spans:
                row = [t]
                for s in spans:
                    if check_gap(s, t, self.config.hks_span_gap):
                        row.append("{:4.2f}".format(norms[t[0] - 1:t[1],
                                                          s[0] - 1:s[1]].sum().item()))
                    else:
                        row.append('-')
                table.append(row)
            log(tabulate(table, headers=["t↓"] + spans))

            pairs = []
            for t in spans:
                for s in spans:
                    if check_gap(s, t, self.config.hks_span_gap):
                        pairs.append((norms[t[0] - 1:t[1], s[0] - 1:s[1]].sum().item(), (t, s)))
            if len(pairs) == 0:
                return None
            spairs = sorted(pairs, key=lambda x: x[0], reverse=True)

            return list(zip(*spairs))[1]

    def select_by_deltalogit(self, instance, sentence):
        minl = self.config.hks_min_span_len
        maxl = self.config.hks_max_span_len
        # Compute the ``vulnerable'' values of each words
        words, tags, chars, arcs, rels = instance
        # sent_len = words.size(1)

        # The returned margins does not contain <ROOT>
        margins = self.compute_margin(instance)
        log(margins)

        # Count the vulnerable words in each span,
        # Select the most vulerable span as target.
        vul_scores = [0]
        for ele in margins:
            if ele > 0:
                vul_scores.append(10 - ele)
            else:
                vul_scores.append(0)
        spans = filter_spans(gen_spans(sentence), minl, maxl, True)
        span_vuls = list()
        # span_ratios = list()
        for span in spans:
            span_vuls.append(sum(vul_scores[span[0]:span[1] + 1]))
            # span_ratios.append(span_vuls[-1] / (span[1] + 1 - span[0]))

        pairs = []
        for tid, t in enumerate(spans):
            for s in spans:
                if check_gap(s, t, self.config.hks_span_gap):
                    pairs.append((span_vuls[tid], (t, s)))
        if len(pairs) == 0:
            return None
        spairs = sorted(pairs, key=lambda x: x[0], reverse=True)
        return list(zip(*spairs))[1]

    def select_by_random(self, instance, sentence):
        spans = filter_spans(gen_spans(sentence), self.config.hks_min_span_len,
                             self.config.hks_max_span_len, True)
        paired = []
        for tgt_span in spans:
            for src_span in spans:
                if check_gap(tgt_span, src_span, self.config.hks_span_gap):
                    paired.append((tgt_span, src_span))
        if len(paired) == 0:
            return None
        return paired

    def white_hack(self, instance, sentence, tgt_span, src_span):
        words, tags, chars, arcs, rels = instance
        sent_len = words.size(1)

        raw_words = words.clone()
        var_words = words.clone()

        raw_metric = self.task.partial_evaluate(instance=(raw_words, tags, None, arcs, rels),
                                                mask_idxs=ex_span_idx(tgt_span, sent_len),
                                                mst=self.config.hks_mst == 'on')
        _, raw_arcs, _ = self.task.predict([(raw_words, tags, None)])

        forbidden_idxs__ = [self.vocab.unk_index, self.vocab.pad_index]
        change_positions__ = set()
        if self.config.hks_max_change > 0.9999:
            max_change_num = int(self.config.hks_max_change)
        else:
            max_change_num = max(1, int(self.config.hks_max_change * words.size(1)))
        iter_change_num = min(max_change_num, self.config.hks_iter_change)

        picker = CherryPicker(lower_is_better=True)
        t0 = time.time()
        picker.add(raw_metric, {"num_changed": 0, "logtable": 'No modification'})
        log('iter -1, uas {:.4f}'.format(raw_metric.uas))
        succ = False
        for iter_id in range(self.config.hks_steps):
            result = self.single_hack(instance=(var_words, tags, None, arcs, rels),
                                      raw_words=raw_words,
                                      raw_metric=raw_metric,
                                      raw_arcs=raw_arcs,
                                      src_span=src_span,
                                      tgt_span=tgt_span,
                                      iter_id=iter_id,
                                      forbidden_idxs__=forbidden_idxs__,
                                      change_positions__=change_positions__,
                                      max_change_num=max_change_num,
                                      iter_change_num=iter_change_num)
            if result['code'] == 200:
                var_words = result['words']
                picker.add(result['attack_metric'], {
                    'logtable': result['logtable'],
                    "num_changed": len(change_positions__)
                })
                if result['attack_metric'].uas < raw_metric.uas - 0.00001:
                    succ = True
                    log('Succeed in step {}'.format(iter_id))
                    break
            elif result['code'] == 404:
                log('FAILED')
                break
        t1 = time.time()
        best_iter, best_attack_metric, best_info = picker.select_best_point()

        return defaultdict(
            lambda: -1, {
                "succ": 1 if succ else 0,
                "raw_metric": raw_metric,
                "attack_metric": best_attack_metric,
                "iters": iter_id,
                "best_iter": best_iter,
                "num_changed": best_info['num_changed'],
                "time": t1 - t0,
                "logtable": best_info['logtable']
            })

    def select_span_pair(self, instance, sentence):
        if self.config.hks_span_selection in \
                ["vulnom", "rdmnom", "vulcls", "rdmcls", "rdmrdm", "rdmnom"]:
            tgt_mode = self.config.hks_span_selection[:3]
            src_mode = self.config.hks_span_selection[3:]
            tgt_span = self.select_tgt_span(instance, sentence, tgt_mode)
            if tgt_span is None:
                log('Target span not found')
                return
            src_span = self.select_src_span(instance, sentence, tgt_span, src_mode)
            if src_span is None:
                log('Source span not found')
                return
            ret = [(tgt_span, src_span)]
        elif self.config.hks_span_selection in ['jacobian1', 'jacobian2']:
            ret = self.select_by_jacobian(instance, sentence, self.config.hks_span_selection)
        elif self.config.hks_span_selection in ['random']:
            ret = self.select_by_random(instance, sentence)
        elif self.config.hks_span_selection in ['deltalogit']:
            ret = self.select_by_deltalogit(instance, sentence)
        else:
            raise Exception

        if ret is None:
            log('Span pair not found')
            return
        return ret

    def random_hack(self, instance, sentence, tgt_span_lst, src_span):
        t0 = time.time()
        words, tags, chars, arcs, rels = instance
        sent_len = words.size(1)

        raw_words_lst = cast_list(words)
        idxs = gen_idxs_to_substitute(list(range(src_span[0], src_span[1] + 1)),
                                      int(self.config.hks_max_change), self.config.hks_cand_num)
        if self.config.hks_blk_repl_tag == 'any':
            cand_words_lst = subsitute_by_idxs(raw_words_lst, idxs, self.blackbox_sub_idxs)
        elif re.match("[njvri]+", self.config.hks_blk_repl_tag):
            cand_words_lst = subsitute_by_idxs(raw_words_lst, idxs,
                                               self.njvri_subidxs(self.config.hks_blk_repl_tag))
        elif re.match("keep", self.config.hks_blk_repl_tag):
            tag_texts = cast_list(tags[0])
            vocab_idxs_lst = []
            for i in range(src_span[0], src_span[1] + 1):
                vocab_idxs_lst.append(self.univ_subidxs(HACK_TAGS.ptb2uni(self.vocab.tags[tag_texts[i]])))
            cand_words_lst = subsitute_by_idxs_2(raw_words_lst, idxs, src_span[0], vocab_idxs_lst)
        else:
            raise Exception

        # First index is raw words
        all_words = torch.tensor([raw_words_lst] + cand_words_lst, device=words.device)
        raw_words = all_words[0:1]
        cand_words = all_words[1:]

        tgt_idxs = []
        for tgt_span in tgt_span_lst:
            tgt_idxs.extend([i for i in range(tgt_span[0], tgt_span[1] + 1) if i != tgt_span[2]])
        ex_tgt_idxs = [i for i in range(sent_len) if i not in tgt_idxs]

        pred_arcs, pred_rels, gold_arcs, gold_rels = self.task.partial_evaluate(
            (all_words, tags.expand_as(all_words), None, arcs.expand_as(all_words),
             rels.expand_as(all_words)),
            mask_idxs=ex_tgt_idxs,
            mst=False,
            return_metric=False)

        # raw_metric = ParserMetric()
        # raw_metric(pred_arcs[0], pred_rels[0], gold_arcs[0], gold_rels[0])
        raw_metric = self.task.partial_evaluate((raw_words, tags, None, arcs, rels),
                                                ex_tgt_idxs,
                                                mst=True)

        succ = False
        arc_delta = (pred_arcs[1:] - pred_arcs[0]).abs().sum(1)
        if arc_delta.sum() == 0:
            # direct forward to the last attacked sentence
            att_id = self.config.hks_cand_num - 1
        else:
            for att_id in range(0, self.config.hks_cand_num):
                if arc_delta[att_id] != 0:
                    att_metric = self.task.partial_evaluate(
                        (cand_words[att_id:att_id + 1], tags, None, arcs, rels),
                        ex_tgt_idxs,
                        mst=True)

                    if att_metric.uas < raw_metric.uas - 0.0001:
                        succ = True
                        break

        # ATT_ID will be equal to att_id when failing
        t1 = time.time()

        _, raw_arcs, raw_rels = self.task.predict([(raw_words, tags, None)], mst=True)
        _, att_arcs, att_rels = self.task.predict([(cand_words[att_id:att_id + 1], tags, None)],
                                                  mst=True)

        if not succ:
            info = 'Nothing'
        else:
            for tgt_span in tgt_span_lst:
                if tgt_span == src_span:
                    continue
                raw_span_corr = 0
                att_span_corr = 0
                for i in range(tgt_span[0], tgt_span[1] + 1):
                    if i == tgt_span[2]:
                        continue
                    if raw_arcs[0][i - 1] == arcs[0][i - 1]:
                        raw_span_corr += 1
                    if att_arcs[0][i - 1] == arcs[0][i - 1]:
                        att_span_corr += 1
                if att_span_corr < raw_span_corr:
                    break

            info = tabulate(self._gen_log_table(words, cand_words[att_id:att_id + 1], tags, arcs,
                                                rels, raw_arcs, att_arcs, src_span, tgt_span),
                            floatfmt='.6f')

        return defaultdict(
            lambda: -1, {
                "succ": 1 if succ else 0,
                "att_id": att_id if att_id < self.config.hks_cand_num - 1 else np.nan,
                "num_changed": self.config.hks_max_change,
                "time": t1 - t0,
                "logtable": info
            })

    def meta_hack(self, instance, sentence):
        t0 = time.time()
        ram_reset("hk")

        words, tags, chars, arcs, rels = instance
        sent_len = words.size(1)

        words_text = self.vocab.id2word(words[0])

        succ = False

        if self.config.hks_color == "black":
            spans = filter_spans(gen_spans(sentence), self.config.hks_min_span_len,
                                 self.config.hks_max_span_len, True)
            hack_result = None
            for pair_id, src_span in enumerate(spans):
                log('[Chosen source] ', src_span, ' '.join(words_text[src_span[0]:src_span[1] + 1]))
                tgt_span_lst = [
                    ele for ele in spans if check_gap(ele, src_span, self.config.hks_span_gap)
                ]
                if len(tgt_span_lst) == 0:
                    continue
                hack_result = self.random_hack(instance, sentence, tgt_span_lst, src_span)
                if hack_result['succ'] == 1:
                    succ = True
                    log("Succ on source span {}.\n".format(src_span))
                    break
                log("Fail on source span {}.\n".format(src_span))
            if hack_result is None:
                log("Not enough span pairs")
                return None
        elif self.config.hks_color in ['white', 'grey']:
            pairs = self.select_span_pair(instance, sentence)
            if pairs is None:
                return None
            for pair_id, (tgt_span, src_span) in enumerate(pairs[:self.config.hks_topk_pair]):
                log('[Chosen span] ', 'tgt-', tgt_span,
                    ' '.join(words_text[tgt_span[0]:tgt_span[1] + 1]), 'src-', src_span,
                    ' '.join(words_text[src_span[0]:src_span[1] + 1]))

                if self.config.hks_color == "white":
                    hack_result = self.white_hack(instance, sentence, tgt_span, src_span)
                elif self.config.hks_color == "grey":
                    hack_result = self.random_hack(instance, sentence, [tgt_span], src_span)
                else:
                    raise Exception
                if hack_result['succ'] == 1:
                    succ = True
                    log("Succ on the span pair {}/{}.\n".format(pair_id, len(pairs)))
                    break
                log("Fail on the span pair {}/{}.\n".format(pair_id, len(pairs)))

            hack_result['meta_trial_pair'] = pair_id + 1
            hack_result['meta_total_pair'] = len(pairs)
            hack_result['meta_succ_trial_pair'] = pair_id + 1 if succ else 0
            hack_result['meta_succ_total_pair'] = len(pairs) if succ else 0

        t1 = time.time()
        log('Sentence cost {:.1f}s'.format(t1 - t0))
        hack_result['meta_time'] = t1 - t0

        return hack_result

    # yapf: disable
    def single_hack(self, instance,
                    src_span, tgt_span,
                    iter_id,
                    raw_words, raw_metric, raw_arcs,
                    forbidden_idxs__: list,
                    change_positions__: set,
                    max_change_num,
                    iter_change_num,
                    verbose=False):
        # yapf: enable
        words, tags, chars, arcs, rels = instance
        sent_len = words.size(1)

        # Backward loss
        embed_grad = self.backward_loss(instance=instance,
                                        mask_idxs=ex_span_idx(tgt_span, sent_len),
                                        verbose=True)
        grad_norm = embed_grad.norm(dim=2)
        if self.config.hks_word_random == "on":
            grad_norm = torch.rand(grad_norm.size(), device=grad_norm.device)

        position_mask = [False for _ in range(words.size(1))]
        # Mask some positions
        for i in range(sent_len):
            if rels[0][i].item(
            ) == self.vocab.rel_dict['punct'] or not src_span[0] <= i <= src_span[1]:
                position_mask[i] = True
        # Check if the number of changed words exceeds the max value
        if len(change_positions__) >= max_change_num:
            for i in range(sent_len):
                if i not in change_positions__:
                    position_mask[i] = True

        for i in range(sent_len):
            if position_mask[i]:
                grad_norm[0][i] = -(grad_norm[0][i] + 1000)

        # print(grad_norm)

        # Select a word and forbid itself
        word_sids = []  # type: list[torch.Tensor]
        word_vids = []  # type: list[torch.Tensor]
        new_word_vids = []  # type: list[torch.Tensor]

        # _, topk_idxs = grad_norm[0].topk(min(max_change_num, len(src_idxs)))
        # for ele in topk_idxs:
        #     word_sids.append(ele)
        _, topk_idxs = grad_norm[0].sort(descending=True)
        selected_words = elder_select(ordered_idxs=cast_list(topk_idxs),
                                      num_to_select=iter_change_num,
                                      selected=change_positions__,
                                      max_num=max_change_num)
        # The position mask will ensure that at least one word is legal,
        # but the second one may not be allowed
        selected_words = [ele for ele in selected_words if position_mask[ele] is False]
        word_sids = torch.tensor(selected_words)

        for word_sid in word_sids:
            word_vid = words[0][word_sid]
            emb_to_rpl = self.parser.embed.weight[word_vid]

            if self.config.hks_step_size > 0:
                word_grad = embed_grad[0][word_sid]
                delta = word_grad / \
                    torch.norm(word_grad) * self.config.hks_step_size
                changed = emb_to_rpl - delta

                tag_type = self.config.hks_constraint
                if tag_type == 'any':
                    must_tag = None
                elif tag_type == 'same':
                    must_tag = self.vocab.tags[tags[0][word_sid].item()]
                elif re.match("[njvri]+", tag_type):
                    must_tag = HACK_TAGS[tag_type]
                else:
                    raise Exception
                forbidden_idxs__.append(word_vid)
                change_positions__.add(word_sid.item())
                new_word_vid, repl_info = self.find_replacement(
                    changed,
                    must_tag,
                    dist_measure=self.config.hks_dist_measure,
                    forbidden_idxs__=forbidden_idxs__,
                    repl_method='tagdict',
                    words=words,
                    word_sid=word_sid)
            else:
                new_word_vid = random.randint(0, self.vocab.n_words)
                while new_word_vid in [self.vocab.pad_index, self.vocab.word_dict["<root>"]]:
                    new_word_vid = random.randint(0, self.vocab.n_words - 1)
                new_word_vid = torch.tensor(new_word_vid, device=words.device)
                repl_info = {}
                change_positions__.add(word_sid.item())

            word_vids.append(word_vid)
            new_word_vids.append(new_word_vid)

            # log(delta @ (self.parser.embed.weight[new_word_vid] - self.parser.embed.weight[word_vid]))
            # log()

        new_words = words.clone()
        for i in range(len(word_vids)):
            if new_word_vids[i] is not None:
                new_words[0][word_sids[i]] = new_word_vids[i]
        """
            Evaluating the result
        """
        # print('START EVALUATING')
        # print([self.vocab.words[ele] for ele in self.forbidden_idxs__])
        metric = self.task.partial_evaluate(instance=(new_words, tags, None, arcs, rels),
                                            mask_idxs=ex_span_idx(tgt_span, sent_len),
                                            mst=self.config.hks_mst == 'on')
        att_tags, att_arcs, att_rels = self.task.predict([(new_words, tags, None)],
                                                         mst=self.config.hks_mst == 'on')

        # if verbose:
        #     print('$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        #     print('Iter {}'.format(iter_id))
        #     print(tabulate(_gen_log_table(), floatfmt=('.6f')))
        #     print('^^^^^^^^^^^^^^^^^^^^^^^^^^^')

        rpl_detail = ""
        for i in range(len(word_sids)):
            if new_word_vids[i] is not None:
                rpl_detail += "{}:{}->{}  ".format(
                    self.vocab.words[raw_words[0][word_sids[i]].item()],
                    self.vocab.words[word_vids[i].item()],
                    self.vocab.words[new_word_vids[i].item()])

        log(
            "iter {}, uas {:.4f}, ".format(iter_id, metric.uas),
            "mind {:6.3f}, avgd {:6.3f}, ".format(repl_info['mind'], repl_info['avgd'])
            if 'mind' in repl_info else '', rpl_detail)
        if metric.uas >= raw_metric.uas - .00001:
            info = 'Nothing'
        else:
            info = tabulate(self._gen_log_table(raw_words, new_words, tags, arcs, rels, raw_arcs,
                                                att_arcs, src_span, tgt_span),
                            floatfmt='.6f')
        return {
            'code': 200,
            'words': new_words,
            'attack_metric': metric,
            'logtable': info,
        }

    def _gen_log_table(self, raw_words, new_words, tags, arcs, rels, raw_arcs, att_arcs, src_span,
                       tgt_span):
        sent_len = raw_words.size(1)
        new_words_text = [self.vocab.words[i.item()] for i in new_words[0]]
        raw_words_text = [self.vocab.words[i.item()] for i in raw_words[0]]
        tags_text = [self.vocab.tags[i.item()] for i in tags[0]]

        table = []
        for i in range(sent_len):
            gold_arc = int(arcs[0][i])
            raw_arc = 0 if i == 0 else raw_arcs[0][i - 1]
            att_arc = 0 if i == 0 else att_arcs[0][i - 1]
            if src_span[0] <= i <= src_span[1]:
                span_symbol = "Ⓢ"
            elif tgt_span[0] <= i <= tgt_span[1]:
                span_symbol = "Ⓣ"
            else:
                span_symbol = ""
            mask_symbol = '&' if tgt_span[0] <= att_arc <= tgt_span[1] or tgt_span[
                0] <= i <= tgt_span[1] else ""

            table.append([
                "{}{}".format(i, span_symbol),
                raw_words_text[i],
                '>{}'.format(new_words_text[i]) if raw_words_text[i] != new_words_text[i] else "*",
                tags_text[i],
                gold_arc,
                raw_arc,
                '>{}{}'.format(att_arc, mask_symbol) if att_arc != raw_arc else '*',
                # grad_norm[0][i].item()
            ])
        return table
