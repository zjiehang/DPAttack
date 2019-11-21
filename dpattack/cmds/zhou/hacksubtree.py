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
                                log_config, ram_read, ram_write, show_mean_std,
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


class HackSubtree(IHack):

    def __call__(self, config):
        if config.logf == 'on':
            log_config('hacksubtree',
                       log_path=config.workspace,
                       default_target='cf')
        else:
            log = print

        for arg in config.kwargs:
            if arg.startswith('hks'):
                log(arg, '\t', config.kwargs[arg])
        log('------------------')

        self.setup(config)

        for sid, (words, tags, chars, arcs, rels) in enumerate(self.loader):
            if sid < 2 or sid > 10:
                continue
            self.hack(instance=(words, tags, chars, arcs, rels),
                      sentence=self.corpus[sid])
            exit()

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
        max_s_other, _ = s_other.max(1)
        margin = s_gold - max_s_other
        margin[margin < -1] = -1
        loss = margin.sum()
        # raw_embed = parser.embed(raw_words)
        # change_penalty = torch.norm(current_embed - raw_embed,
        #                             p=2, dim=1).sum()
        # print(loss.item(), change_penalty.item())
        # loss += change_penalty
        loss.backward()

        return ram_read('embed_grad')

    def hack(self, instance, sentence):
        words, tags, chars, arcs, rels = instance
        sent_len = words.size(1)
        spans = gen_spans(sentence)
        valid_spans = list(
            filter(lambda ele: 5 <= ele[1] - ele[0] <= 8, spans))
        if len(valid_spans) == 0:
            log("Sentence {} donot have valid spans".format(sid))
            return
        chosen_span = random.choice(valid_spans)
        sent_print(sentence, 'tableh')
        print('spans', spans)
        print('valid', valid_spans)
        print('chosen', chosen_span)

        raw_words = words.clone()
        var_words = words.clone()
        words_text = self.vocab.id2word(words[0])
        tags_text = self.vocab.id2tag(tags[0])

        # eval_idxs = [eval_idx for eval_idx in range(sent_len)
        #               if not chosen_span[0] <= eval_idx <= chosen_span[1]]
        mask_idxs = list(range(chosen_span[0], chosen_span[1]))
        _, raw_metric = self.task.partial_evaluate(
            instance=(raw_words, tags, None, arcs, rels),
            mask_idxs=mask_idxs)

        forbidden_idxs__ = [self.vocab.unk_index, self.vocab.pad_index]
        for iter_id in range(50):
            result = self.single_hack(
                instance=(var_words, tags, None, arcs, rels),
                raw_words=raw_words, raw_metric=raw_metric,
                mask_idxs=mask_idxs,
                iter_id=iter_id,
                forbidden_idxs__=forbidden_idxs__
            )
            var_words = result['words']
            # Maybe it is a reference?
            forbidden_idxs__ = result['forbidden_idxs__']

    def single_hack(self, instance,
                    mask_idxs,
                    iter_id,
                    raw_words, raw_metric,
                    forbidden_idxs__: list,
                    verbose=True,
                    step_size=8,
                    dist_measure='euc'):
        words, tags, chars, arcs, rels = instance
        sent_len = words.size(1)

        # Backward loss
        embed_grad = self.backward_loss(
            instance=instance, mask_idxs=mask_idxs)

        grad_norm = embed_grad.norm(dim=2)

        for i in range(sent_len):
            if i not in mask_idxs:
                grad_norm[0][i] = -(grad_norm[0][i] + 1000)
        # print(grad_norm)

        # Select a word and forbid itself
        word_sid = grad_norm[0].argmax()
        max_grad = embed_grad[0][word_sid]
        word_vid = words[0][word_sid]
        emb_to_rpl = self.parser.embed.weight[word_vid]

        # Find a word to change
        delta = max_grad / torch.norm(max_grad) * step_size
        changed = emb_to_rpl - delta

        must_tag = self.vocab.tags[tags[0][word_sid].item()]
        # must_tag = None
        # must_tag = 'CD'
        forbidden_idxs__.append(word_vid)
        new_word_vid, repl_info = self.find_replacement(
            changed, must_tag, dist_measure='euc',
            forbidden_idxs__=forbidden_idxs__,
            repl_method='tagdict',
            words=words, word_sid=word_sid
        )

        new_words = words.clone()
        new_words[0][word_sid] = new_word_vid

        """
            Evaluating the result
        """
        # print('START EVALUATING')
        # print([self.vocab.words[ele] for ele in self.forbidden_idxs__])
        loss, metric = self.task.partial_evaluate(
            instance=(new_words, tags, None, arcs, rels),
            mask_idxs=mask_idxs)

        def _gen_log_table():
            new_words_text = [self.vocab.words[i.item()] for i in new_words[0]]
            raw_words_text = [self.vocab.words[i.item()] for i in raw_words[0]]
            tags_text = [self.vocab.tags[i.item()] for i in tags[0]]
            pred_tags, pred_arcs, pred_rels = self.task.predict(
                [(new_words, tags, None)])

            table = []
            for i in range(sent_len):
                gold_arc = int(arcs[0][i])
                pred_arc = 0 if i == 0 else pred_arcs[0][i - 1]
                table.append([
                    "{}{}".format(i, "@" if i in mask_idxs else ""),
                    raw_words_text[i],
                    '>{}'.format(
                        new_words_text[i]) if raw_words_text[i] != new_words_text[i] else "*",
                    tags_text[i],
                    gold_arc,
                    '>{}'.format(pred_arc) if pred_arc != gold_arc else '*',
                    grad_norm[0][i].item()
                ])
            return table

        if verbose:
            print('$$$$$$$$$$$$$$$$$$$$$$$$$$$')
            print('Iter {}'.format(iter_id))
            # print('After Attacking: \n\t{}\n\t{}'.format(
            #     " ".join(new_words_text), " ".join(tags_text)))
            # print('{} --> {}'.format(
            #     self.vocab.words[word_vid.item()],
            #     self.vocab.words[new_word_vid.item()]
            # ))
            print(tabulate(_gen_log_table(), floatfmt=('.6f')))
            print('^^^^^^^^^^^^^^^^^^^^^^^^^^^')

        log('iter {}, uas {:.4f}, mind {:6.3f}, avgd {:6.3f}'.format(
            iter_id, metric.uas, repl_info['mind'], repl_info['avgd']))
        if metric.uas >= raw_metric.uas - .00001:
            info = 'Nothing'
        else:
            info = tabulate(_gen_log_table(), floatfmt='.6f')
        return {'code': 200, 'words': new_words,
                "forbidden_idxs__": forbidden_idxs__,
                'attack_metric': metric, 'info': info}


# HIGHLIGHT: SPAN OPERATION
