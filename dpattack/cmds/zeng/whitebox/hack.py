# -*- coding: utf-8 -*-

from dpattack.task import ParserTask
from dpattack.utils.metric import ParserMetric as Metric
from dpattack.utils.corpus import Corpus
from tabulate import tabulate
import torch
from collections import defaultdict
from dpattack.libs.luna import log_config, log
from dpattack.cmds.zeng.whitebox.base import WhiteBoxAttackBase


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


class WhiteHack(WhiteBoxAttackBase):

    def __init__(self):
        super(WhiteHack,self).__init__()

        self.vals = {}

    def extract_embed_grad(self, module, grad_in, grad_out):
        self.vals['embed_grad'] = grad_out[0]

    def __call__(self, config):
        loader = self.pre_attack(config)
        self.parser.register_backward_hook(self.extract_embed_grad)
        # log_config('whitelog.txt',
        #            log_path=config.workspace,
        #            default_target='cf')

        train_corpus = Corpus.load(config.ftrain)
        self.tag_filter = generate_tag_filter(train_corpus, self.vocab)
        # corpus = Corpus.load(config.fdata)
        # dataset = TextDataset(self.vocab.numericalize(corpus, True))
        # # set the data loader
        # loader = DataLoader(dataset=dataset,
        #                     collate_fn=collate_fn)

        # def embed_hook(module, grad_in, grad_out):
        #     self.vals["embed_grad"] = grad_out[0]

        # dpattack.pretrained.register_backward_hook(embed_hook)

        raw_metrics = Metric()
        attack_metrics = Metric()

        log('dist measure', config.hk_dist_measure)

        # batch size == 1
        for sid, (words, tags, arcs, rels) in enumerate(loader):
            # if sid > 10:
            #     break

            raw_words = words.clone()
            words_text = self.get_seqs_name(words)
            tags_text = self.get_tags_name(tags)

            log('****** {}: \n\t{}\n\t{}'.format(
                sid,
                " ".join(words_text),
                " ".join(tags_text)
            ))

            self.vals['forbidden'] = [self.vocab.unk_index, self.vocab.pad_index]
            for pgdid in range(100):
                result = self.single_hack(words, tags, arcs, rels,
                                          dist_measure=config.hk_dist_measure,
                                          raw_words=raw_words)
                if result['code'] == 200:
                    raw_metrics += result['raw_metric']
                    attack_metrics += result['attack_metric']
                    log('attack successfully at step {}'.format(pgdid))
                    break
                elif result['code'] == 404:
                    raw_metrics += result['raw_metric']
                    attack_metrics += result['raw_metric']
                    log('attack failed at step {}'.format(pgdid))
                    break
                elif result['code'] == 300:
                    if pgdid == 99:
                        raw_metrics += result['raw_metric']
                        attack_metrics += result['raw_metric']
                        log('attack failed at step {}'.format(pgdid))
                    else:
                        words = result['words']
            log()

            log('Aggregated result: {} --> {}'.format(raw_metrics, attack_metrics),
                target='cf')

    def single_hack(self,
                    words, tags, arcs, rels,
                    raw_words,
                    target_tags=['NN', 'JJ'],
                    dist_measure='enc',
                    verbose=False):
        vocab = self.model.vocab
        parser = self.model.model
        loss_fn = self.model.criterion
        embed = parser.embed.weight

        raw_loss, raw_metric = self.model.evaluate([(raw_words, tags, arcs, rels)])

        # backward the loss
        parser.zero_grad()
        mask = words.ne(vocab.pad_index)
        mask[:, 0] = 0
        s_arc, s_rel = parser(words, tags)
        s_arc, s_rel = s_arc[mask], s_rel[mask]
        gold_arcs, gold_rels = arcs[mask], rels[mask]
        loss = loss_fn(s_arc, gold_arcs)
        loss.backward()

        grad_norm = self.vals['embed_grad'].norm(dim=2)

        # Select a word to attack by its POS and norm
        exist_target_tag = False
        for i in range(tags.size(1)):
            if vocab.tags[tags[0][i]] not in target_tags:
                grad_norm[0][i] -= 99999.
            else:
                exist_target_tag = True
        if not exist_target_tag:
            return {"code": 404,
                    'raw_metric': raw_metric}
        word_sid = grad_norm[0].argmax()
        word_vid = words[0][word_sid]
        max_grad = self.vals['embed_grad'][0][word_sid]

        # Forbid the word itself to be selected
        self.vals['forbidden'].append(word_vid.item())

        # Find a word to change
        changed = embed[word_vid] - max_grad * 1
        dist = {
            'euc': lambda: (changed - embed).pow(2).sum(dim=1),
            'cos': lambda: torch.nn.functional.cosine_similarity(
                embed, changed.repeat(embed.size(0), 1), dim=1)
        }[dist_measure]()

        must_tag = vocab.tags[tags[0][word_sid].item()]
        legal_tag_index = self.tag_filter[must_tag].to(dist.device)
        legal_tag_mask = dist.new_zeros(dist.size()) \
            .index_fill_(0, legal_tag_index, 1.).byte()

        dist.masked_fill_(1 - legal_tag_mask, 99999.)

        for ele in self.vals['forbidden']:
            dist[ele] = 99999.
        word_vid_to_rpl = dist.argmin()
        # A forbidden word maybe chosen if all words are forbidden(99999.)
        if word_vid_to_rpl.item() in self.vals['forbidden']:
            log('Attack failed.')
            return {'code': 404,
                    'raw_metric': raw_metric}

        # =====================
        # Evaluating the result
        # =====================
        repl_words = words.clone()
        repl_words[0][word_sid] = word_vid_to_rpl

        repl_words_text = [vocab.words[i.item()] for i in repl_words[0]]
        raw_words_text = [vocab.words[i.item()] for i in raw_words[0]]
        tags_text = [vocab.tags[i.item()] for i in tags[0]]
        if verbose:
            print('After Attacking: \n\t{}\n\t{}'.format(
                " ".join(repl_words_text), " ".join(tags_text)
            ))
        pred_tags, pred_arcs, pred_rels = self.model.predict([(repl_words, tags)])
        loss, metric = self.model.evaluate([(repl_words, tags, arcs, rels)])
        table = []
        for i in range(words.size(1)):
            gold_arc = int(arcs[0][i])
            pred_arc = 0 if i == 0 else pred_arcs[0][i - 1]
            table.append([
                i,
                repl_words_text[i],
                raw_words_text[i] if raw_words_text[i] != repl_words_text[i] else "*",
                tags_text[i],
                gold_arc,
                pred_arc if pred_arc != gold_arc else '*',
                grad_norm[0][i].item()
            ])

        if verbose:
            print('{} --> {}'.format(
                vocab.words[word_vid.item()],
                vocab.words[word_vid_to_rpl.item()]
            ))
            print(tabulate(table, floatfmt=('.6f')))
            print(metric)
            print('**************************')
        if metric.uas > raw_metric.uas - 0.1:
            return {'code': 300,
                    'words': repl_words,
                    'raw_metric': raw_metric,
                    'attack_metric': metric}
        else:
            log(tabulate(table, floatfmt=('.6f')))
            log('Result {} --> {}'.format(raw_metric.uas, metric.uas),
                target='cf')
            return {'code': 200,
                    'words': repl_words,
                    'raw_metric': raw_metric,
                    'attack_metric': metric}
