# -*- coding: utf-8 -*-
import math
import random
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, List, Optional, Union

import torch
from tabulate import tabulate
from torch.utils.data import DataLoader

from dpattack.libs.luna import (Aggregator, CherryPicker, TrainingStopObserver,
                                as_table, auto_create, cast_list,
                                fetch_best_ckpt_name, idx_to_msk, log,
                                log_config, ram_pop, ram_write, show_mean_std,
                                time)
from dpattack.models import PosTagger, WordParser, WordTagParser
from dpattack.task import ParserTask
from dpattack.utils.corpus import Corpus, Sentence, sent_print
from dpattack.utils.data import TextDataset, collate_fn
from dpattack.utils.embedding_searcher import (EmbeddingSearcher, cos_dist,
                                               euc_dist)
from dpattack.utils.metric import Metric, ParserMetric
from dpattack.utils.parser_helper import load_parser
from dpattack.utils.tag_tool import gen_tag_dict
from dpattack.utils.vocab import Vocab

from .ihack import HACK_TAGS, IHack, v
from .treeops import gen_spans


class HackOutside(IHack):

    def __call__(self, config):
        if config.logf == 'on':
            log_config('hackoutside',
                       log_path=config.workspace,
                       default_target='cf')
            from dpattack.libs.luna import log
        else:
            log = print

        log('[General Settings]')
        log(config)
        log('[Hack Settings]')
        for arg in config.kwargs:
            if arg.startswith('hks'):
                log(arg, '\t', config.kwargs[arg])
        log('------------------')

        self.setup(config)

        raw_metrics = ParserMetric()
        attack_metrics = ParserMetric()

        agg = Aggregator()
        for sid, (words, tags, chars, arcs, rels) in enumerate(self.loader):
            # if sid > 100:
            #     continue

            words_text = self.vocab.id2word(words[0])
            tags_text = self.vocab.id2tag(tags[0])
            log('****** {}: \n{}\n{}'.format(
                sid, " ".join(words_text), " ".join(tags_text)))

            result = self.hack(instance=(words, tags, chars, arcs, rels),
                               sentence=self.corpus[sid])

            if result is None:
                continue
            else:
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

            # WARNING: SOME SENTENCE NOT SHOWN!
            if result:
                log('Show result from iter {}:'.format(result['best_iter']))
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

            # exit()

    def backward_loss(self, instance, mask_idxs) -> torch.Tensor:
        self.parser.zero_grad()
        words, tags, chars, arcs, rels = instance
        mask = words.ne(self.vocab.pad_index)
        mask[0, 0] = 0
        for mask_idx in mask_idxs:
            mask[0, mask_idx] = 0
        s_arc, s_rel = self.parser(words, tags)
        s_arc, s_rel = s_arc[mask], s_rel[mask]
        gold_arcs, gold_rels = arcs[mask], rels[mask]  # shape like [7,7,7,0,3]

        # max margin loss
        msk_gold_arc = idx_to_msk(gold_arcs, num_classes=s_arc.size(1))
        s_gold = s_arc[msk_gold_arc]
        s_other = s_arc.masked_fill(msk_gold_arc, -1000.)
        max_s_othe__global_ramamr.max(1)
        margin = s_gold - max_s_other
        margin[margin < -1] = -1
        loss = margin.sum()
        # raw_embed = parser.embed(raw_words)
        # change_penalty = torch.norm(current_embed - raw_embed,
        #                             p=2, dim=1).sum()
        # print(loss.item(), change_penalty.item())
        # loss += change_penalty
        loss.backward()

        return ram_pop('embed_grad')

    def hack(self, instance, sentence):
        words, tags, chars, arcs, rels = instance
        sent_len = words.size(1)
        spans = gen_spans(sentence)
        valid_spans = list(
            filter(lambda ele: 5 <= ele[1] - ele[0] <= 8, spans))
        if len(valid_spans) == 0:
            log("Attack error, no valid spans".format())
            return None
        chosen_span = random.choice(valid_spans)
        # chosen_span = (18, 24)
        if chosen_span[1] - chosen_span[0] + 1 > 0.5 * sent_len:
            return None
        # sent_print(sentence, 'tablev')
        # print('spans', spans)
        # print('valid', valid_spans)
        log('chosen span: ', chosen_span,
            ' '.join(self.vocab.id2word(words[0])[chosen_span[0]: chosen_span[1] + 1]))

        raw_words = words.clone()
        var_words = words.clone()
        words_text = self.vocab.id2word(words[0])
        tags_text = self.vocab.id2tag(tags[0])

        # eval_idxs = [eval_idx for eval_idx in range(sent_len)
        #               if not chosen_span[0] <= eval_idx <= chosen_span[1]]
        mask_idxs = list(range(chosen_span[0], chosen_span[1] + 1))
        raw_metric = self.task.partial_evaluate(
            instance=(raw_words, tags, None, arcs, rels),
            mask_idxs=mask_idxs)
        _, raw_arcs, _ = self.task.predict([(raw_words, tags, None)])

        forbidden_idxs__ = [self.vocab.unk_index, self.vocab.pad_index]
        change_positions__ = set()
        if isinstance(self.config.hko_max_change, int):
            max_change_num = self.config.hko_max_change
        elif isinstance(self.config.hk_max_change, float):
            max_change_num = int(self.config.hk_max_change * words.size(1))
        else:
            raise Exception("hk_max_change must be a float or an int")

        picker = CherryPicker(lower_is_better=True)
        t0 = time.time()
        for iter_id in range(self.config.hko_steps):
            result = self.single_hack(
                instance=(var_words, tags, None, arcs, rels),
                raw_words=raw_words, raw_metric=raw_metric, raw_arcs=raw_arcs,
                mask_idxs=mask_idxs,
                iter_id=iter_id,
                forbidden_idxs__=forbidden_idxs__,
                change_positions__=change_positions__,
                max_change_num=max_change_num
            )
            if result['code'] == 200:
                var_words = result['words']
                picker.add(result['attack_metric'],
                           {'logtable': result['logtable'],
                            "num_changed": len(change_positions__)})
            elif result['code'] == 404:
                print('FAILED')
                break
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

    def single_hack(self, instance,
                    mask_idxs,
                    iter_id,
                    raw_words, raw_metric, raw_arcs,
                    forbidden_idxs__: list,
                    change_positions__: set,
                    max_change_num,
                    verbose=False):
        words, tags, chars, arcs, rels = instance
        sent_len = words.size(1)

        # Backward loss
        embed_grad = self.backward_loss(instance=instance, mask_idxs=mask_idxs)

        grad_norm = embed_grad.norm(dim=2)

        position_mask = [False for _ in range(words.size(1))]
        # Mask some positions
        for i in range(sent_len):
            if i not in mask_idxs or rels[0][i].item() == self.vocab.rel_dict['punct']:
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

        _, topk_idxs = grad_norm[0].topk(max_change_num)
        for ele in topk_idxs:
            word_sids.append(ele)

        for word_sid in word_sids:
            word_grad = embed_grad[0][word_sid]
            word_vid = words[0][word_sid]
            emb_to_rpl = self.parser.embed.weight[word_vid]

            # Find a word to change
            delta = word_grad / \
                torch.norm(word_grad) * self.config.hko_step_size
            changed = emb_to_rpl - delta

            # must_tag = self.vocab.tags[tags[0][word_sid].item()]
            must_tag = None
            # must_tag = HACK_TAGS['njr']
            # must_tag = 'CD'
            forbidden_idxs__.append(word_vid)
            change_positions__.add(word_sid.item())
            new_word_vid, repl_info = self.find_replacement(
                changed, must_tag, dist_measure=self.config.hko_dist_measure,
                forbidden_idxs__=forbidden_idxs__,
                repl_method='tagdict',
                words=words, word_sid=word_sid
            )
            word_vids.append(word_vid)
            new_word_vids.append(new_word_vid)

        new_words = words.clone()
        for i in range(len(word_vids)):
            if new_word_vids[i] is not None:
                new_words[0][word_sids[i]] = new_word_vids[i]

        """
            Evaluating the result
        """
        # print('START EVALUATING')
        # print([self.vocab.words[ele] for ele in self.forbidden_idxs__])
        metric = self.task.partial_evaluate(
            instance=(new_words, tags, None, arcs, rels),
            mask_idxs=mask_idxs)

        def _gen_log_table():
            new_words_text = [self.vocab.words[i.item()] for i in new_words[0]]
            raw_words_text = [self.vocab.words[i.item()] for i in raw_words[0]]
            tags_text = [self.vocab.tags[i.item()] for i in tags[0]]
            att_tags, att_arcs, att_rels = self.task.predict(
                [(new_words, tags, None)])

            table = []
            for i in range(sent_len):
                gold_arc = int(arcs[0][i])
                raw_arc = 0 if i == 0 else raw_arcs[0][i - 1]
                att_arc = 0 if i == 0 else att_arcs[0][i - 1]
                mask_symbol = '&' if att_arc in mask_idxs or i in mask_idxs else ""

                table.append([
                    "{}{}".format(i, "@" if i in mask_idxs else ""),
                    raw_words_text[i],
                    '>{}'.format(
                        new_words_text[i]) if raw_words_text[i] != new_words_text[i] else "*",
                    tags_text[i],
                    gold_arc,
                    raw_arc,
                    '>{}{}'.format(
                        att_arc, mask_symbol) if att_arc != raw_arc else '*',
                    grad_norm[0][i].item()
                ])
            return table

        if verbose:
            print('$$$$$$$$$$$$$$$$$$$$$$$$$$$')
            print('Iter {}'.format(iter_id))
            print(tabulate(_gen_log_table(), floatfmt=('.6f')))
            print('^^^^^^^^^^^^^^^^^^^^^^^^^^^')

        rpl_detail = ""
        for i in range(len(word_sids)):
            if new_word_vids[i] is not None:
                rpl_detail += "{}:{}->{}  ".format(
                    self.vocab.words[raw_words[0][word_sids[i]].item()],
                    self.vocab.words[word_vids[i].item()],
                    self.vocab.words[new_word_vids[i].item()])

        log("iter {}, uas {:.4f}, ".format(iter_id, metric.uas),
            "mind {:6.3f}, avgd {:6.3f}, ".format(
                repl_info['mind'], repl_info['avgd']) if 'mind' in repl_info else '',
            rpl_detail
            )
        if metric.uas >= raw_metric.uas - .00001:
            info = 'Nothing'
        else:
            info = tabulate(_gen_log_table(), floatfmt='.6f')
        return {
            'code': 200,
            'words': new_words,
            'attack_metric': metric,
            'logtable': info,
        }
