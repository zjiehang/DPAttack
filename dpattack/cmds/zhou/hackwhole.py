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
from dpattack.utils.corpus import Corpus
from tabulate import tabulate
import torch
from collections import defaultdict
from dpattack.libs.luna import log_config, log, fetch_best_ckpt_name, show_mean_std, idx_to_msk, cast_list, as_table, \
    ram_write, ram_read, TrainingStopObserver, CherryPicker, Aggregator, time
from dpattack.utils.vocab import Vocab

hack_tags = {
    "nj": ("NN", "JJ"),
    "njr": ("NN", "JJ", "RB"),
    "njvr": ('NN', 'JJ', 'VB', 'RB'),
    "exnjvr": ('NN', 'NNS', 'NNP', 'NNPS',
               'JJ', 'JJR', 'JJS',
               'VB', 'VBD', 'VBG', 'VBN', 'VBZ',
               'RB', 'RBR', 'RBS')
}


def mask_to_small(val):
    if val > 0:
        val = -val
    val = val - 1000
    return val


def mask_to_large(val):
    if val < 0:
        val = - val
    val = val + 1000
    return val


def generate_tag_filter(corpus, vocab):
    tag_filter = defaultdict(lambda: set())
    train_words = vocab.n_train_words

    for word_seq, tag_seq in zip(corpus.words, corpus.tags):
        for word, tag in zip(word_seq[1:], tag_seq[1:]):
            word_id = vocab.word_dict.get(word.lower(), vocab.unk_index)
            if word_id != vocab.unk_index and word_id < train_words:
                tag_filter[tag].add(word)
    for key, value in tag_filter.items():
        tag_filter[key] = vocab.word2id(list(value))
    return tag_filter


class HackWhole:

    def __init__(self):
        self.task: ParserTask
        self.tag_filter: dict
        self.embed_searcher: EmbeddingSearcher
        self.tagger: PosTagger

        self.contiguous_embed = {}
        self.__contiguous_embed_flag = None

    @property
    def vocab(self) -> Vocab:
        return self.task.vocab

    @property
    def parser(self) -> Union[WordTagParser, WordParser]:
        return self.task.model

    @contextmanager
    def use_contiguous_embed(self, flag=True):
        # whatever the flag, after execution it will be set to False
        if self.__contiguous_embed_flag is not None:
            raise Exception('Do not use a use_contiguous_embed '
                            'within a use_contiguous_embed')
        self.__contiguous_embed_flag = flag
        yield
        self.__contiguous_embed_flag = None

    def __call__(self, config):
        log_config('hackwhole-{}-{}-{}'.format(config.hk_loss_based_on,
                                               config.hk_step_size,
                                               config.hk_pgd_freq),
                   log_path=config.workspace,
                   default_target='cf')
        for arg in config.kwargs:
            if arg.startswith('hk'):
                log(arg, '\t', config.kwargs[arg])
        log('------------------')

        print("Load the models")
        vocab = torch.load(config.vocab)  # type: Vocab
        parser = load_parser(fetch_best_ckpt_name(config.parser_model))
        self.tagger = PosTagger.load(fetch_best_ckpt_name(config.tagger_model))

        self.task = ParserTask(vocab, parser)

        print("Load the dataset")

        train_corpus = Corpus.load(config.ftrain)
        self.tag_filter = generate_tag_filter(train_corpus, vocab)
        corpus = Corpus.load(config.fdata)
        dataset = TextDataset(vocab.numericalize(corpus, True))
        # set the data loader
        loader = DataLoader(dataset=dataset,
                            collate_fn=collate_fn)

        def embed_forward_hook(module, input, output):
            # output_bak = output.clone()
            # print('before: ', output[0, :, :3])
            if self.__contiguous_embed_flag:
                # print('USE CONTIGUOUS')
                for idx, val in self.contiguous_embed.items():
                    output[0][idx] = val
            # print('after: ', output[0, :, :3])
            # print(output - output_bak)

        def embed_backward_hook(module, grad_in, grad_out):
            ram_write('embed_grad', grad_out[0])

        parser.embed.register_forward_hook(embed_forward_hook)
        parser.embed.register_backward_hook(embed_backward_hook)

        self.embed_searcher = EmbeddingSearcher(
            embed=parser.embed.weight,
            idx2word=lambda x: vocab.words[x],
            word2idx=lambda x: vocab.word_dict[x]
        )

        raw_metrics = ParserMetric()
        attack_metrics = ParserMetric()

        log('dist measure', config.hk_dist_measure)
        agg = Aggregator()
        # batch size == 1
        for sid, (var_words, tags, chars, arcs, rels) in enumerate(loader):
            # if sid in [1, 2]:
            #     continue
            if sid > config.hk_sent_num:
                continue

            raw_words = var_words.clone()
            words_text = vocab.id2word(var_words[0])
            tags_text = vocab.id2tag(tags[0])
            _, raw_metric = self.task.evaluate(
                [(raw_words, tags, None, arcs, rels)])

            self.forbidden_idxs = [vocab.unk_index, vocab.pad_index]
            self.contiguous_embed = {}
            self.change_positions = set()
            log('****** {}: \n{}\n{}'.format(sid,
                                             " ".join(words_text), " ".join(tags_text)))

            picker = CherryPicker(lower_is_better=True)
            # iter 0 -> raw
            picker.add(raw_metric, 'No modification to the raw sentence')

            t0 = time.time()
            for iter_id in range(1, config.hk_steps):

                cgs_flg = iter_id % config.hk_pgd_freq != 0
                if not cgs_flg:
                    self.contiguous_embed = {}

                result = self.single_hack(
                    var_words, tags, arcs, rels,
                    target_tags=hack_tags[config.hk_tag_type],
                    raw_words=raw_words, raw_metric=raw_metric,
                    dist_measure=config.hk_dist_measure,
                    loss_based_on=config.hk_loss_based_on,
                    cgs_flg=cgs_flg,
                    step_size=config.hk_step_size,
                    verbose=False,
                    max_change_num=config.hk_max_change,
                    iter_id=iter_id
                )
                # Fail
                if result['code'] == 404:
                    log('Stop in step {}, info: {}'.format(
                        iter_id, result['info']))
                    break
                # Success
                if result['code'] == 200:
                    picker.add(result['attack_metric'], result['info'])
                    if result['attack_metric'].uas < raw_metric.uas - config.hk_eps:
                        log('Succeed in step {}'.format(iter_id))
                        break
                var_words = result['words']
            t1 = time.time()
            raw_metrics += raw_metric
            best_iter, best_attack_metric, best_info = picker.select_best_point()
            attack_metrics += best_attack_metric
            agg.aggregate(("iters", iter_id), ("time", t1 - t0),
                          ("fail", abs(best_attack_metric.uas - raw_metric.uas) < 1e-4),
                          ('best_iter', best_iter), ("changed", len(self.change_positions)))
            log('Show result from iter {}:'.format(best_iter))
            log(best_info)
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

    def backward_loss(self, words, tags, arcs, rels,
                      cgs_flg,
                      loss_based_on) -> torch.Tensor:
        self.parser.zero_grad()
        mask = words.ne(self.vocab.pad_index)
        mask[:, 0] = 0
        with self.use_contiguous_embed(cgs_flg):
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
            # with self.use_contiguous_embed(cgs_flg):
            #     current_embed = parser.embed(words)[0]
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

    @torch.no_grad()
    def find_replacement(self, words, word_sid,
                         changed, must_tag,
                         dist_measure,
                         repl_method='tagger') -> (Optional[torch.Tensor], dict):
        if repl_method == 'tagger':
            # Pipeline:
            #    256 minimum dists
            # -> Filtered by a tagger
            # -> Smallest one
            words = words.repeat(128, words.size(1))
            dists, idxs = self.embed_searcher.find_neighbours(changed, 128,
                                                              dist_measure, False)
            for i, ele in enumerate(idxs):
                words[i][word_sid] = ele
            self.tagger.eval()
            s_tags = self.tagger(words)
            pred_tags = s_tags.argmax(-1)[:, word_sid]
            new_word_vid = None
            for i, ele in enumerate(pred_tags):
                if self.vocab.tags[ele.item()] == must_tag:
                    new_word_vid = idxs[i]
                    if new_word_vid.item() not in self.forbidden_idxs:
                        break
            return new_word_vid, {"avgd": dists.mean().item(),
                                  "mind": dists.min().item()}
        elif repl_method == 'tagdict':
            # Pipeline:
            #    All dists
            # -> Filtered by a tag dict
            # -> Smallest one
            # vals, idxs = self.embed_searcher.find_neighbours(changed, 10, 'euc', False)
            # print("⊥ {:.2f}, - {:.2f}, {} ~ {}, {}, {}".format(
            #     vals.min().item(), vals.mean().item(),
            #     *[self.vocab.words[idxs[i].item()] for i in range(4)]))
            # show_mean_std(embed[word_vid])
            # show_mean_std(max_grad)
            dist = {'euc': euc_dist, 'cos': cos_dist}[
                dist_measure](changed, self.parser.embed.weight)
            # print('>>> before moving')
            # self.embed_searcher.find_neighbours(embed[word_vid],10, 'euc', True)
            # print('>>> after moving')
            # self.embed_searcher.find_neighbours(changed, 10, 'euc', True)

            # Mask illegal words by its POS
            legal_tag_index = self.tag_filter[must_tag].to(dist.device)
            legal_tag_mask = dist.new_zeros(dist.size()) \
                .index_fill_(0, legal_tag_index, 1.).byte()
            dist.masked_fill_(1 - legal_tag_mask, 1000.)
            for ele in self.forbidden_idxs:
                dist[ele] = 1000.
            new_word_vid = dist.argmin()
            return new_word_vid, {}
        elif repl_method == 'bert':
            # Pipeline:
            #    Bert select 256 words
            # -> Filtered by a tagger
            # -> Smallest one
            raise NotImplemented

    def single_hack(self,
                    words, tags, arcs, rels,
                    raw_words, raw_metric,
                    target_tags=('NN', 'JJ'),
                    dist_measure='euc',
                    loss_based_on='logit',
                    step_size=5,
                    cgs_flg=False,
                    verbose=False,
                    max_change_num=1,
                    iter_id=-1):
        sent_len = words.size(1)

        """
            Loss back-propagation
        """
        embed_grad = self.backward_loss(words, tags, arcs, rels,
                                        cgs_flg=cgs_flg,
                                        loss_based_on=loss_based_on)
        # Sometimes the loss/grad will be zero.
        # Especially in the case of applying pgd_freq>1 to small sentences:
        # e.g., the uas of projected version may be 83.33%
        # while under the case of the unprojected version, the loss is 0.
        if torch.sum(embed_grad) == 0.0:
            return {"code": 404, "info": "Loss is already zero."}

        """
            Select and change a word
        """
        grad_norm = embed_grad.norm(dim=2)

        position_mask = [False for _ in range(words.size(1))]
        # Mask some positions by POS
        for i in range(sent_len):
            if self.vocab.tags[tags[0][i]] not in target_tags:
                position_mask[i] = True
        # Check if the number of changed words exceeds the max value
        if len(self.change_positions) >= max_change_num:
            for i in range(sent_len):
                if i not in self.change_positions:
                    position_mask[i] = True
        if all(position_mask):
            return {"code": 404, "info": "No valid word to replace"}

        for i in range(sent_len):
            if position_mask[i]:
                grad_norm[0][i] = -(grad_norm[0][i] + 1000)

        # Select a word and forbid itself
        word_sid = grad_norm[0].argmax()
        max_grad = embed_grad[0][word_sid]

        word_vid = words[0][word_sid]
        if cgs_flg and word_sid.item() in self.contiguous_embed:
            emb_to_rpl = self.contiguous_embed[word_sid.item()]
        else:
            emb_to_rpl = self.parser.embed.weight[word_vid]
        self.forbidden_idxs.append(word_vid.item())
        self.change_positions.add(word_sid.item())
        # print(self.change_positions)

        # Find a word to change
        delta = max_grad / torch.norm(max_grad) * step_size
        changed = emb_to_rpl - delta

        must_tag = self.vocab.tags[tags[0][word_sid].item()]
        new_word_vid, repl_info = self.find_replacement(words, word_sid,
                                                        changed, must_tag,
                                                        dist_measure, 'tagger')

        # A forbidden word maybe chosen if all words are forbidden(1000.)
        if new_word_vid is None or new_word_vid.item() in self.forbidden_idxs:
            log('Attack failed.')
            return {'code': 404,
                    'info': "Selected word are forbidden"}

        # For contiguous version, the new_word is the projected version.
        if cgs_flg:
            self.contiguous_embed[word_sid.item()] = changed
        new_words = words.clone()
        new_words[0][word_sid] = new_word_vid

        """
            Evaluating the result
        """
        # print('START EVALUATING')
        # print([self.vocab.words[ele] for ele in self.forbidden_idxs])
        loss, metric = self.task.evaluate(
            [(new_words, tags, None, arcs, rels)])

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
                    i,
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
            print('Iter {}, cgs_flg={}'.format(iter_id, cgs_flg))
            # print('After Attacking: \n\t{}\n\t{}'.format(
            #     " ".join(new_words_text), " ".join(tags_text)))
            # print('{} --> {}'.format(
            #     self.vocab.words[word_vid.item()],
            #     self.vocab.words[new_word_vid.item()]
            # ))
            print(tabulate(_gen_log_table(), floatfmt=('.6f')))
            print('^^^^^^^^^^^^^^^^^^^^^^^^^^^')

        log('iter {}, pgd {}, uas {:.4f}, mind {:6.3f}, avgd {:6.3f}'.format(
            iter_id, int(not cgs_flg), metric.uas, repl_info['mind'], repl_info['avgd']))
        if metric.uas >= raw_metric.uas - .00001:
            info = 'Nothing'
        else:
            info = tabulate(_gen_log_table(), floatfmt='.6f')
        return {'code': 200, 'words': new_words,
                'attack_metric': metric, 'info': info}
