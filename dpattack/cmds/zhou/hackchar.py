# -*- coding: utf-8 -*-
import math
from collections import defaultdict, deque
from contextlib import contextmanager
from typing import Optional, Union

import torch
from tabulate import tabulate
from torch.utils.data import DataLoader
from torch.distributions import Categorical

from dpattack.libs.luna import (
    Aggregator, CherryPicker, TrainingStopObserver, as_table, cast_list,
    create_folder_for_file, fetch_best_ckpt_name, idx_to_msk, log, log_config,
    ram_pop, ram_write, show_mean_std, time, time_stamp, locate_chunk)
from dpattack.libs.luna.public import auto_create
from dpattack.models import PosTagger, WordParser, WordTagParser
from dpattack.task import ParserTask
from dpattack.utils.corpus import Corpus
from dpattack.utils.data import TextDataset, collate_fn
from dpattack.utils.embedding_searcher import (EmbeddingSearcher, cos_dist,
                                               euc_dist)
from dpattack.utils.metric import Metric, ParserMetric
from dpattack.utils.parser_helper import load_parser
from dpattack.utils.tag_tool import gen_tag_dict
from dpattack.utils.vocab import Vocab

from .ihackc import IHackC, v, HACK_TAGS

torch.backends.cudnn.enabled = False


def young_select(ordered_idxs=[5, 2, 1, 3, 0, 4],
                 num_to_select=3,
                 selected={2, 3, 4},
                 max_num=4):
    """
    selected = set()
    new_select, exc = young_select([5, 2, 1, 3, 0, 4], selected=selected)
    assert new_select == [5, 2, 1] and exc == []
    selected.update(set(new_select))
    for ele in exc:
        selected.remove(ele)
    new_select, exc = young_select([5, 4, 0, 3, 1, 2], selected=selected)
    assert new_select == [5, 4, 0] and exc == [2]
    selected.update(set(new_select))
    for ele in exc:
        selected.remove(ele)
    new_select, exc = young_select([3, 4, 0, 1, 2, 5], selected=selected)
    assert(new_select == [3, 4, 0] and exc == [5])
    """
    new_select = ordered_idxs[:num_to_select]
    # Remove words selected in this iteration
    ex_cands = [ele for ele in selected if ele not in new_select]
    # Sort other words
    sorted_ex_cands = [ele for ele in ordered_idxs if ele in ex_cands]
    return new_select, sorted_ex_cands[max_num - num_to_select:]


def elder_select(ordered_idxs=[5, 2, 1, 3, 0, 4],
                 num_to_select=3,
                 selected={2, 3, 4},
                 max_num=5):
    """
    selected = set()
    new_select = elder_select([5, 2, 1, 3, 0, 4], selected=selected)
    assert(new_select == [5, 2, 1])
    selected.update(set(new_select))
    new_select = elder_select([5, 2, 0, 3, 1, 4], selected=selected)
    assert(new_select == [5, 2, 0])
    selected.update(set(new_select))
    new_select = elder_select([3, 4, 0, 1, 2, 5], selected=selected)
    assert(new_select == [3, 0, 1])
    """
    ret = []
    total_num = len(selected)
    for ele in ordered_idxs:
        if len(ret) == num_to_select:
            break
        if ele in selected:
            ret.append(ele)
        else:
            if total_num == max_num:
                continue
            else:
                ret.append(ele)
                total_num += 1
    return ret


class HackChar(IHackC):

    def __call__(self, config):
        assert config.input == 'char'
        self.init_logger(config)
        self.setup(config)

        if self.config.hk_use_worker == 'on':
            start_sid, end_sid = locate_chunk(
                len(self.loader), self.config.hk_num_worker, self.config.hk_worker_id)
            log('Run code on a chunk [{}, {})'.format(start_sid, end_sid))

        raw_metrics = ParserMetric()
        attack_metrics = ParserMetric()

        agg = Aggregator()
        for sid, (words, tags, chars, arcs, rels) in enumerate(self.loader):
            # if sid in [0, 1, 2, 3, 4]:
            #     continue
            # if sid > config.hk_sent_num:
            #     continue
            if self.config.hk_use_worker == 'on':
                if sid < start_sid or sid >= end_sid:
                    continue
            if self.config.hk_training_set == 'on' and words.size(1) > 50:
                log('Skip sentence {} whose length is {}(>50).'.format(
                    sid, words.size(1)))
                continue

            if words.size(1) < 5:
                log('Skip sentence {} whose length is {}(<5).'.format(
                    sid, words.size(1)))
                continue

            words_text = self.vocab.id2word(words[0])
            tags_text = self.vocab.id2tag(tags[0])
            log('****** {}: \n{}\n{}'.format(
                sid, " ".join(words_text), " ".join(tags_text)))

            # hack it!
            result = self.hack(instance=(words, tags, chars, arcs, rels))

            # aggregate information
            raw_metrics += result['raw_metric']
            attack_metrics += result['attack_metric']
            agg.aggregate(
                ("iters", result['iters']),
                ("time", result['time']),
                ("fail", abs(result['attack_metric'].uas -
                             result['raw_metric'].uas) < 1e-4),
                ('best_iter', result['best_iter']),
                ("changed", result['num_changed'])
            )

            # log some information
            log('Show result from iter {}, changed num {}:'.format(
                result['best_iter'], result['num_changed']))
            log(result['logtable'])

            log('Aggregated result: {} --> {}, '
                'iters(avg) {:.1f}, time(avg) {:.1f}s, '
                'fail rate {:.2f}, best_iter(avg) {:.1f}, best_iter(std) {:.1f}, '
                'changed(avg) {:.1f}'.format(
                    raw_metrics, attack_metrics,
                    agg.mean('iters'), agg.mean('time'),
                    agg.mean('fail'), agg.mean(
                        'best_iter'), agg.std('best_iter'),
                    agg.mean('changed')
                ))
            log()

    def hack(self, instance):
        words, tags, chars, arcs, rels = instance
        _, raw_metric = self.task.evaluate([(words, tags, chars, arcs, rels)],
                                           mst=self.config.hkc_mst == 'on')
        _, raw_arcs, _ = self.task.predict([(words, tags, chars)],
                                           mst=self.config.hkc_mst == 'on')

        # char_grads, word_grad = self.backward_loss(words, chars, arcs, rels)

        forbidden_idxs__ = defaultdict(lambda: deque(maxlen=5))    # word_sid -> deque()
        change_positions__ = dict()  # word_sid -> char_wid
        if self.config.hkc_max_change > 0.9999:
            max_change_num = int(self.config.hkc_max_change)
        else:
            max_change_num = max(1, int(self.config.hkc_max_change * words.size(1)))
        iter_change_num = min(max_change_num, self.config.hkc_iter_change)

        raw_chars = chars.clone()
        var_chars = chars.clone()

        # HIGHLIGHT: ITERATION
        t0 = time.time()
        picker = CherryPicker(lower_is_better=True,
                              compare_fn=lambda m1, m2: m1.uas - m2.uas)
        # iter 0 -> raw
        picker.add(raw_metric, {
            "num_changed": 0,
            "logtable": 'No modification'
        })
        for iter_id in range(1, self.config.hkc_steps):
            result = self.single_hack(
                words, tags, var_chars, arcs, rels,
                raw_chars=raw_chars, raw_metric=raw_metric, raw_arcs=raw_arcs,
                verbose=False,
                max_change_num=max_change_num,
                iter_change_num=iter_change_num,
                iter_id=iter_id,
                forbidden_idxs__=forbidden_idxs__,
                change_positions__=change_positions__,
            )
            # Fail
            if result['code'] == 404:
                log('Stop in step {}, info: {}'.format(
                    iter_id, result['info']))
                break
            # Success
            if result['code'] == 200:
                picker.add(result['attack_metric'], {
                    # "words": result['new_words'],
                    "logtable": result['logtable'],
                    "num_changed": len(change_positions__)
                })
                if result['attack_metric'].uas < raw_metric.uas - self.config.hkw_eps:
                    log('Succeed in step {}'.format(iter_id))
                    break
            var_chars = result['chars']
        t1 = time.time()

        best_iter, best_attack_metric, best_info = picker.select_best_point()

        return {
            "raw_metric": raw_metric,
            "attack_metric": best_attack_metric,
            "iters": iter_id,
            "best_iter": best_iter,
            "num_changed": best_info['num_changed'],
            "time": t1 - t0,
            "logtable": best_info['logtable']
        }

    def backward_loss(self, words, chars, arcs, rels) -> torch.Tensor:
        self.parser.zero_grad()
        mask = words.ne(self.vocab.pad_index)
        mask[:, 0] = 0
        s_arc, s_rel = self.parser(words, chars)
        s_arc, s_rel = s_arc[mask], s_rel[mask]
        gold_arcs, gold_rels = arcs[mask], rels[mask]  # shape like [7,7,7,0,3]

        # max margin loss
        msk_gold_arc = idx_to_msk(gold_arcs, num_classes=s_arc.size(1))
        s_gold = s_arc[msk_gold_arc]
        s_other = s_arc.masked_fill(msk_gold_arc, -1000.)
        max_s_other, _ = s_other.max(1)
        margin = s_gold - max_s_other
        margin[margin < -1] = -1
        loss = margin.sum()
        loss.backward()

        # pre-process char mask
        char_mask = chars.gt(0)[0]
        sorted_lens, indices = torch.sort(
            char_mask.sum(dim=1), descending=True)
        inverse_indices = indices.argsort()
        char_embed_grad = ram_pop('embed_grad')[inverse_indices].unsqueeze(0)
        word_embed_grad = self.parser.word_embed_grad

        return char_embed_grad, word_embed_grad

    def single_hack(self,
                    words, tags, chars, arcs, rels,
                    raw_chars, raw_metric, raw_arcs,
                    forbidden_idxs__,
                    change_positions__,
                    verbose=False,
                    max_change_num=1,
                    iter_change_num=1,
                    iter_id=-1):
        sent_len = words.size(1)

        """
            Loss back-propagation
        """
        char_grads, word_grads = self.backward_loss(words, chars, arcs, rels)

        """
            Select and change a word
        """
        word_grad_norm = word_grads.norm(dim=-1)   # 1 x length
        # 1 x length x max_char_length
        char_grad_norm = char_grads.norm(dim=-1)
        woca_grad_norm = char_grad_norm.max(dim=-1)    # 1 x length

        position_mask = [False for _ in range(words.size(1))]
        # Mask some positions by POS & <UNK>
        for i in range(sent_len):
            if rels[0][i].item() == self.vocab.rel_dict['punct']:
                position_mask[i] = True
        # Check if the number of changed words exceeds the max value
        if len(change_positions__) >= max_change_num:
            for i in range(sent_len):
                if i not in change_positions__:
                    position_mask[i] = True
        if all(position_mask):
            return {"code": 404, "info": "Constrained by tags, no valid word to replace."}

        for i in range(sent_len):
            if position_mask[i]:
                word_grad_norm[0][i] = -(word_grad_norm[0][i] + 1000)
                char_grad_norm[0][i] = -(char_grad_norm[0][i] + 1000)

        # Select a word and forbid itself
        word_sids = []
        char_wids = []
        char_vids = []
        new_char_vids = []

        if self.config.hkc_selection == 'elder':
            _, topk_idxs = word_grad_norm[0].sort(descending=True)
            selected_words = elder_select(ordered_idxs=cast_list(topk_idxs),
                                          num_to_select=iter_change_num,
                                          selected=change_positions__,
                                          max_num=max_change_num)
            selected_chars = []
            for ele in selected_words:
                if ele in change_positions__:
                    selected_chars.append(change_positions__[ele])
                else:
                    wcid = char_grad_norm[0][ele].argmax()
                    selected_chars.append(wcid)
            word_sids = torch.tensor(selected_words)
            char_wids = torch.tensor(selected_chars)
        elif self.config.hkc_selection == 'young':
            _, topk_idxs = word_grad_norm[0].sort(descending=True)
            selected_words, ex_words = young_select(ordered_idxs=cast_list(topk_idxs),
                                                    num_to_select=iter_change_num,
                                                    selected=change_positions__,
                                                    max_num=max_change_num)
            for ele in ex_words:
                change_positions__.pop(ele)
                forbidden_idxs__.pop(ele)
                chars[0][ele] = raw_chars[0][ele]
                log('Drop elder replacement', self.vocab.words[words[i].item()])
            selected_chars = []
            for ele in selected_words:
                if ele in change_positions__:
                    selected_chars.append(change_positions__[ele])
                else:
                    wcid = char_grad_norm[0][ele].argmax()
                    selected_chars.append(wcid)
            word_sids = torch.tensor(selected_words)
            char_wids = torch.tensor(selected_chars)
        else:
            raise Exception

        for word_sid, char_wid in zip(word_sids, char_wids):

            char_grad = char_grads[0][word_sid][char_wid]
            char_vid = chars[0][word_sid][char_wid]

            emb_to_rpl = self.parser.char_lstm.embed.weight[char_vid]
            forbidden_idxs__[word_sid.item()].append(char_vid.item())
            change_positions__[word_sid.item()] = char_wid.item()

            # Find a word to change with dynamically step
            # Note that it is possible that all words found are not as required, e.g.
            #   all neighbours have different tags.
            delta = char_grad / torch.norm(char_grad) * self.config.hkc_step_size
            changed = emb_to_rpl - delta

            dist = {'euc': euc_dist, 'cos': cos_dist}[
                self.config.hkc_dist_measure](changed, self.parser.char_lstm.embed.weight)
            vals, idxs = dist.sort()
            for ele in idxs:
                if ele.item() not in forbidden_idxs__[word_sid.item()]:
                    new_char_vid = ele
                    break

            char_vids.append(char_vid)
            new_char_vids.append(new_char_vid)

        new_chars = chars.clone()
        for i in range(len(word_sids)):
            new_chars[0][word_sids[i]][char_wids[i]] = new_char_vids[i]

        # log(dict(forbidden_idxs__))
        # log(change_positions__)

        """
            Evaluating the result
        """
        # print('START EVALUATING')
        # print([self.vocab.words[ele] for ele in self.forbidden_idxs__])
        new_chars_text = [self.vocab.id2char(ele) for ele in new_chars[0]]
        # print(new_words_txt)
        loss, metric = self.task.evaluate(
            [(words, tags, new_chars, arcs, rels)],
            mst=self.config.hkc_mst == 'on')

        def _gen_log_table():
            new_words_text = [self.vocab.id2char(ele) for ele in new_chars[0]]
            raw_words_text = [self.vocab.id2char(ele) for ele in raw_chars[0]]
            tags_text = [self.vocab.tags[ele.item()] for ele in tags[0]]
            _, att_arcs, _ = self.task.predict([(words, tags, new_chars)],
                                               mst=self.config.hkc_mst == 'on')

            table = []
            for i in range(sent_len):
                gold_arc = int(arcs[0][i])
                raw_arc = 0 if i == 0 else raw_arcs[0][i - 1]
                att_arc = 0 if i == 0 else att_arcs[0][i - 1]

                relevant_mask = '&' if \
                    raw_words_text[att_arc] != new_words_text[att_arc] \
                    or raw_words_text[i] != new_words_text[i] \
                    else ""
                table.append([
                    i,
                    raw_words_text[i],
                    '>{}'.format(
                        new_words_text[i]) if raw_words_text[i] != new_words_text[i] else "*",
                    tags_text[i],
                    gold_arc,
                    raw_arc,
                    '>{}{}'.format(
                        att_arc, relevant_mask) if att_arc != raw_arc else '*',
                    word_grad_norm[0][i].item()
                ])
            return table

        if verbose:
            print('$$$$$$$$$$$$$$$$$$$$$$$$$$$')
            print('Iter {}'.format(iter_id))
            print(tabulate(_gen_log_table(), floatfmt=('.6f')))
            print('^^^^^^^^^^^^^^^^^^^^^^^^^^^')

        rpl_detail = ""
        for i in range(len(word_sids)):
            rpl_detail += "{}:{}->{}  ".format(
                self.vocab.id2char(raw_chars[0, word_sids[i]]),
                self.vocab.id2char(chars[0, word_sids[i]]),
                self.vocab.id2char(new_chars[0, word_sids[i]]))
        log("iter {}, uas {:.4f}, ".format(iter_id, metric.uas),
            # "mind {:6.3f}, avgd {:6.3f}, ".format(
            #     repl_info['mind'], repl_info['avgd']) if 'mind' in repl_info else '',
            rpl_detail
            )
        if metric.uas >= raw_metric.uas - .00001:
            logtable = 'Nothing'
        else:
            logtable = tabulate(_gen_log_table(), floatfmt='.6f')
        return {
            'code': 200,
            'chars': new_chars,
            'attack_metric': metric,
            'logtable': logtable,
            # "forbidden_idxs__": forbidden_idxs__,
            # "change_positions__": change_positions__,
        }
