# -*- coding: utf-8 -*-
import math
from collections import defaultdict
from contextlib import contextmanager
from typing import Optional, Union

import torch
from tabulate import tabulate
from torch.utils.data import DataLoader

from dpattack.libs.luna import (
    Aggregator, CherryPicker, TrainingStopObserver, as_table, cast_list,
    create_folder_for_file, fetch_best_ckpt_name, idx_to_msk, log, log_config,
    ram_read, ram_write, show_mean_std, time, time_stamp, locate_chunk)
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

from .ihack import IHack, v, HACK_TAGS


class HackWhole(IHack):

    def __call__(self, config):
        if config.logf == 'on':
            if config.hk_use_worker == 'on':
                worker_info = "-{}@{}".format(config.hk_num_worker,
                                              config.hk_worker_id)
            else:
                worker_info = ""
            log_config('hackwhole-{}-{}{}'.format(config.input, config.hk_tag_type, worker_info),
                       log_path=config.workspace,
                       default_target='cf')
            from dpattack.libs.luna import log
        else:
            log = print

        log('[General Settings]')
        log(config)
        log('[Hack Settings]')
        for arg in config.kwargs:
            if arg.startswith('hk'):
                log(arg, '\t', config.kwargs[arg])
        log('------------------')

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
                log('Skip sentence {} whose length is {}.'.format(sid, words.size(1)))
                continue

            words_text = self.vocab.id2word(words[0])
            tags_text = self.vocab.id2tag(tags[0])
            log('****** {}: \n{}\n{}'.format(
                sid, " ".join(words_text), " ".join(tags_text)))

            # hact it!
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
        _, raw_metric = self.task.evaluate([(words, tags, chars, arcs, rels)])
        _, raw_arcs, _ = self.task.predict([(words, tags, None)])

        # Setup some states before attacking a sentence
        # WARNING: Operations on variables naming like "abc__" passed to a function
        #  are in-placed! Never try to save an internal state of the variable.
        forbidden_idxs__ = [self.vocab.unk_index, self.vocab.pad_index]
        change_positions__ = set()
        if isinstance(self.config.hk_max_change, int):
            max_change_num = self.config.hk_max_change
        elif isinstance(self.config.hk_max_change, float):
            max_change_num = int(self.config.hk_max_change * words.size(1))
        else:
            raise Exception("hk_max_change must be a float or an int")
        var_words = words.clone()
        raw_words = words.clone()

        # HIGHLIGHT: ITERATION
        t0 = time.time()
        picker = CherryPicker(lower_is_better=True)
        # iter 0 -> raw
        picker.add(raw_metric, {
            "num_changed": 0,
            "logtable": 'No modification'
        })
        for iter_id in range(1, self.config.hk_steps):
            result = self.single_hack(
                var_words, tags, arcs, rels,
                raw_words=raw_words, raw_metric=raw_metric, raw_arcs=raw_arcs,
                verbose=False,
                max_change_num=max_change_num,
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
                if result['attack_metric'].uas < raw_metric.uas - self.config.hk_eps:
                    log('Succeed in step {}'.format(iter_id))
                    break
            var_words = result['words']
            # forbidden_idxs__ = result['forbidden_idxs__']
            # change_positions__ = result['change_positions__']
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

    def backward_loss(self, words, tags, arcs, rels,
                      loss_based_on) -> torch.Tensor:
        self.parser.zero_grad()
        mask = words.ne(self.vocab.pad_index)
        mask[:, 0] = 0
        s_arc, s_rel = self.parser(words, tags)
        s_arc, s_rel = s_arc[mask], s_rel[mask]
        gold_arcs, gold_rels = arcs[mask], rels[mask]  # shape like [7,7,7,0,3]

        if loss_based_on == 'logit':
            # max margin loss
            msk_gold_arc = idx_to_msk(gold_arcs, num_classes=s_arc.size(1))
            s_gold = s_arc[msk_gold_arc]
            s_other = s_arc.masked_fill(msk_gold_arc, -1000.)
            max_s_other, _ = s_other.max(1)
            margin = s_gold - max_s_other
            margin[margin < -1] = -1
            loss = margin.sum()
            # current_embed = parser.embed(words)[0]
            # raw_embed = parser.embed(raw_words)
            # change_penalty = torch.norm(current_embed - raw_embed,
            #                             p=2, dim=1).sum()
            # print(loss.item(), change_penalty.item())
            # loss += change_penalty
            loss.backward()
        elif loss_based_on == 'prob':
            # cross entropy loss
            loss = - self.task.criterion(s_arc, gold_arcs)
            loss.backward()
        else:
            raise Exception
        return ram_read('embed_grad')

    # TODO: 1. Dynamically increase the step size?
    # TODO: 2. Choose some words to change together?
    def single_hack(self,
                    words, tags, arcs, rels,
                    raw_words, raw_metric, raw_arcs,
                    forbidden_idxs__,
                    change_positions__,
                    verbose=False,
                    max_change_num=1,
                    iter_id=-1):
        sent_len = words.size(1)

        """
            Loss back-propagation
        """
        embed_grad = self.backward_loss(words, tags, arcs, rels,
                                        loss_based_on=self.config.hk_loss_based_on)
        # Sometimes the loss/grad will be zero.
        # Especially in the case of applying pgd_freq>1 to small sentences:
        # e.g., the uas of projected version may be 83.33%
        # while under the case of the unprojected version, the loss is 0.
        if torch.sum(embed_grad) == 0.0:
            return {"code": 404, "info": "Loss is zero."}

        """
            Select and change a word
        """
        grad_norm = embed_grad.norm(dim=2)

        position_mask = [False for _ in range(words.size(1))]
        # Mask some positions by POS & <UNK>
        for i in range(sent_len):
            if self.vocab.tags[tags[0][i]] not in HACK_TAGS[self.config.hk_tag_type]:
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
                grad_norm[0][i] = -(grad_norm[0][i] + 1000)

        # Select a word and forbid itself
        word_sids = []  # type: list[torch.Tensor]
        word_vids = []  # type: list[torch.Tensor]
        new_word_vids = []  # type: list[torch.Tensor]

        _, topk_idxs = grad_norm[0].topk(2)
        for ele in topk_idxs:
            word_sids.append(ele)

        for word_sid in word_sids:
            word_grad = embed_grad[0][word_sid]

            word_vid = words[0][word_sid]
            emb_to_rpl = self.parser.embed.weight[word_vid]
            forbidden_idxs__.append(word_vid.item())
            change_positions__.add(word_sid.item())
            # print(self.change_positions__)

            # Find a word to change with dynamically step
            # Note that it is possible that all words found are not as required, e.g.
            #   all neighbours have different tags.
            delta = word_grad / \
                torch.norm(word_grad) * self.config.hk_step_size
            changed = emb_to_rpl - delta

            must_tag = self.vocab.tags[tags[0][word_sid].item()]
            new_word_vid, repl_info = self.find_replacement(
                changed, must_tag, dist_measure=self.config.hk_dist_measure,
                forbidden_idxs__=forbidden_idxs__,
                repl_method=self.config.hk_repl_method,
                words=words, word_sid=word_sid
            )

            word_vids.append(word_vid)
            new_word_vids.append(new_word_vid)

        new_words = words.clone()
        exist_change = False
        for i in range(len(word_vids)):
            if new_word_vids[i] is not None:
                new_words[0][word_sids[i]] = new_word_vids[i]
                exist_change = True
        
        if not exist_change:
            log('Attack failed.')
            return {'code': 404,
                    'info': 'Neighbours of the selected words have different tags.'}            
        

        # if new_word_vid is None:
        #     log('Attack failed.')
        #     return {'code': 404,
        #             'info': 'Neighbours of the selected words have different tags.'}

        # new_words = words.clone()
        # new_words[0][word_sid] = new_word_vid

        """
            Evaluating the result
        """
        # print('START EVALUATING')
        # print([self.vocab.words[ele] for ele in self.forbidden_idxs__])
        new_words_text = [self.vocab.words[i.item()] for i in new_words[0]]
        # print(new_words_txt)
        loss, metric = self.task.evaluate(
            [(new_words, tags, None, arcs, rels)])

        def _gen_log_table():
            new_words_text = [self.vocab.words[i.item()] for i in new_words[0]]
            raw_words_text = [self.vocab.words[i.item()] for i in raw_words[0]]
            tags_text = [self.vocab.tags[i.item()] for i in tags[0]]
            _, att_arcs, _ = self.task.predict([(new_words, tags, None)])

            table = []
            for i in range(sent_len):
                gold_arc = int(arcs[0][i])
                raw_arc = 0 if i == 0 else raw_arcs[0][i - 1]
                att_arc = 0 if i == 0 else att_arcs[0][i - 1]

                relevant_mask = '&' if raw_words[0][att_arc] != new_words[0][att_arc] else ""
                table.append([
                    i,
                    raw_words_text[i],
                    '>{}'.format(
                        new_words_text[i]) if raw_words_text[i] != new_words_text[i] else "*",
                    tags_text[i],
                    gold_arc,
                    raw_arc,
                    '>{}{}'.format(att_arc, relevant_mask) if att_arc != raw_arc else '*',
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
            logtable = 'Nothing'
        else:
            logtable = tabulate(_gen_log_table(), floatfmt='.6f')
        return {
            'code': 200,
            'words': new_words,
            'attack_metric': metric,
            'logtable': logtable,
            # "forbidden_idxs__": forbidden_idxs__,
            # "change_positions__": change_positions__,
        }
