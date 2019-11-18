# -*- coding: utf-8 -*-
import math
from contextlib import contextmanager
from typing import Union, Optional

from torch.utils.data import DataLoader

from dpattack.models import WordTagParser, WordParser, PosTagger
from dpattack.utils.data import TextDataset, collate_fn
from dpattack.utils.embedding_searcher import EmbeddingSearcher, cos_dist, euc_dist
from dpattack.utils.parser_helper import load_parser
from dpattack.task import ParserTask
from dpattack.utils.metric import Metric, ParserMetric
from dpattack.utils.corpus import Corpus, Sentence, sent_print
from tabulate import tabulate
import torch
from collections import defaultdict
from dpattack.libs.luna import log_config, log, fetch_best_ckpt_name, show_mean_std, idx_to_msk, cast_list, as_table, \
    ram_write, ram_read, TrainingStopObserver, CherryPicker, Aggregator, time
from dpattack.utils.vocab import Vocab
import random

random.seed(1)


# Code for fucking VSCode debug console
class V:
    def __sub__(self, tsr):
        for ele in tsr.__repr__().split('\n'):
            print(ele)


v = V()


class HackSubtree:
    def __init__(self):
        self.task: ParserTask
        self.embed_searcher: EmbeddingSearcher
        # self.tagger: PosTagger

    @property
    def vocab(self) -> Vocab:
        return self.task.vocab

    @property
    def parser(self) -> Union[WordTagParser, WordParser]:
        return self.task.model

    def __call__(self, config):
        log_config('hacksubtree',
                   log_path=config.workspace,
                   default_target='cf')
        for arg in config.kwargs:
            if arg.startswith('hks'):
                log(arg, '\t', config.kwargs[arg])
        log('------------------')

        print("Load the models")
        vocab = torch.load(config.vocab)  # type: Vocab
        parser = load_parser(fetch_best_ckpt_name(config.parser_model))
        self.task = ParserTask(vocab, parser)

        # self.tagger = PosTagger.load(fetch_best_ckpt_name(config.tagger_model))

        print("Load the dataset")
        train_corpus = Corpus.load(config.ftrain)  # type:Corpus
        corpus = Corpus.load(config.fdata)
        dataset = TextDataset(vocab.numericalize(corpus, True))
        # set the data loader
        loader = DataLoader(dataset=dataset,
                            collate_fn=collate_fn)

        def embed_backward_hook(module, grad_in, grad_out):
            ram_write('embed_grad', grad_out[0])

        parser.embed.register_backward_hook(embed_backward_hook)

        self.embed_searcher = EmbeddingSearcher(
            embed=parser.embed.weight,
            idx2word=lambda x: vocab.words[x],
            word2idx=lambda x: vocab.word_dict[x]
        )

        # HIGHLIGHT:

        for sid, (words, tags, chars, arcs, rels) in enumerate(loader):
            if sid < 2 or sid > 10:
                continue
            sent = corpus[sid]
            sent_len = words.size(1)
            spans = gen_spans(sent)
            valid_spans = list(
                filter(lambda ele: 5 <= ele[1] - ele[0] <= 8, spans))
            if len(valid_spans) == 0:
                log("Sentence {} donot have valid spans".format(i))
                break
            chosen_span = random.choice(valid_spans)
            sent_print(sent, 'tableh')
            print('spans', spans)
            print('valid', valid_spans)
            print('chosen', chosen_span)

            raw_words = words.clone()
            var_words = words.clone()
            words_text = vocab.id2word(words[0])
            tags_text = vocab.id2tag(tags[0])

            # eval_idxes = [eval_idx for eval_idx in range(sent_len)
            #               if not chosen_span[0] <= eval_idx <= chosen_span[1]]
            mask_idxes = list(range(chosen_span[0], chosen_span[1]))
            _, raw_metric = self.task.partial_evaluate(
                instance=(raw_words, tags, None, arcs, rels),
                mask_idxes=mask_idxes)

            for iter_id in range(50):
                result = self.single_hack(
                    instance=(var_words, tags, None, arcs, rels),
                    raw_words=raw_words, raw_metric=raw_metric,
                    mask_idxes=mask_idxes,
                    iter_id=iter_id,
                )
                var_words = result['words']

    def backward_loss(self, instance, mask_idxes) -> torch.Tensor:
        self.parser.zero_grad()
        words, tags, chars, arcs, rels = instance
        mask = words.ne(self.vocab.pad_index)
        mask[0, 0] = 0
        for mask_idx in mask_idxes:
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

    def single_hack(self, instance,
                    mask_idxes,
                    iter_id,
                    raw_words, raw_metric,
                    verbose=True,
                    step_size=8,
                    dist_measure='euc'):
        words, tags, chars, arcs, rels = instance
        sent_len = words.size(1)

        # Backward loss
        embed_grad = self.backward_loss(
            instance=instance, mask_idxes=mask_idxes)

        grad_norm = embed_grad.norm(dim=2)

        for i in mask_idxes:
            grad_norm[0][i] = -(grad_norm[0][i] + 1000)
        print(grad_norm)

        # Select a word and forbid itself
        word_sid = grad_norm[0].argmax()
        max_grad = embed_grad[0][word_sid]
        word_vid = words[0][word_sid]
        emb_to_rpl = self.parser.embed.weight[word_vid]

        # Find a word to change
        delta = max_grad / torch.norm(max_grad) * step_size
        changed = emb_to_rpl - delta

        must_tag = self.vocab.tags[tags[0][word_sid].item()]
        new_word_vid, repl_info = self.find_replacement(changed, dist_measure)

        new_words = words.clone()
        new_words[0][word_sid] = new_word_vid

        """
            Evaluating the result
        """
        # print('START EVALUATING')
        # print([self.vocab.words[ele] for ele in self.forbidden_idxs])
        loss, metric = self.task.partial_evaluate(instance=(new_words, tags, None, arcs, rels),
                                                  mask_idxes=mask_idxes)

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
                    "{}{}".format(i, "@" if i in mask_idxes else ""),
                    new_words_text[i],
                    raw_words_text[i] if raw_words_text[i] != new_words_text[i] else "*",
                    tags_text[i],
                    gold_arc,
                    pred_arc if pred_arc != gold_arc else '*',
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
                'attack_metric': metric, 'info': info}

    @torch.no_grad()
    def find_replacement(self, changed,
                         dist_measure) -> (Optional[torch.Tensor], dict):
        dists, idxs = self.embed_searcher.find_neighbours(changed, 64,
                                                          dist_measure, False)
        new_word_vid = idxs[0]
        return new_word_vid, {"avgd": dists.mean().item(),
                              "mind": dists.min().item()}


# HIGHLIGHT: SPAN OPERATION
def gen_spans(sent: Sentence):
    """
    Sample of a sentence (starting at 0):
          ID = ('1', '2', '3', '4', '5', '6', '7', '8')
        HEAD = ('7', '7', '7', '7', '7', '7', '0', '7')

    Return(ROOT included): 
        [(0, 8), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (1, 8), (8, 8)]
    """
    ids = [0] + list(map(int, sent.ID))
    heads = [-1] + list(map(int, sent.HEAD))
    sent_len = len(ids)
    # print(ids, heads)
    l_children = [[] for _ in range(sent_len)]
    r_children = [[] for _ in range(sent_len)]

    for tid, hid in enumerate(heads):
        if hid != -1:
            if hid > tid:
                l_children[hid].append(tid)
            else:
                r_children[hid].append(tid)

    # for i in range(sent_len):
    #     print(ids[i], heads[i], l_children[ids[i]], r_children[ids[i]])

    # Find left/right-most span index
    def _find_span_id(idx, dir='l'):
        if dir == 'l':
            if len(l_children[idx]) == 0:
                return idx
            else:
                return _find_span_id(l_children[idx][0], 'l')
        else:
            if len(r_children[idx]) == 0:
                return idx
            else:
                return _find_span_id(r_children[idx][-1], 'r')

    spans = [(_find_span_id(idx, 'l'), _find_span_id(idx, 'r'))
             for idx in range(sent_len)]
    # print(headed_span)

    # headed_span_length = [right - left + 1 for left, right in headed_span]
    # print(headed_span_length)
    return spans


def subtree_distribution(corpus: Corpus):
    print(corpus[4])
    print(gen_spans(corpus[4]))
    exit()
    min_span_len = 5
    max_span_len = 10
    # num = 0
    for sid, sent in enumerate(corpus):
        if len(sent.ID) < 15:
            continue
        min_span_len = len(sent.ID) * 0.2
        max_span_len = len(sent.ID) * 0.4
        span = gen_spans(sent)
        span = list(filter(lambda ele: min_span_len <=
                           ele[1] - ele[0] <= max_span_len, span))
        # num += len(span)

        print(len(sent.ID), len(span))
        if sid == 10:
            print()
            exit()
