from dpattack.libs.luna import fetch_best_ckpt_name, log_config
from dpattack.task import ParserTask
from dpattack.utils.corpus import Corpus, Sentence
import torch

from dpattack.utils.parser_helper import load_parser
from dpattack.utils.vocab import Vocab


class HackSubtree:
    def __init__(self):
        self.vals = {}
        self.task: ParserTask
        # self.tag_filter: dict

    def __call__(self, config):
        corpus = Corpus.load(config.fdata)

        print("Load the models")
        vocab = torch.load(config.vocab)  # type: Vocab
        parser = load_parser(fetch_best_ckpt_name(config.parser_model))
        self.task = ParserTask(vocab, parser)
        log_config('subtreelog.txt',
                   log_path=config.workspace,
                   default_target='cf')

        # for i in range
        subtree_distribution(corpus)

    def select_subtree(self, sent):
        pass


def gen_headed_span(sent: Sentence):
    """
    Sample of a sentence (starting at 0):
          ID=('1', '2', '3', '4', '5', '6', '7', '8')
        HEAD=('7', '7', '7', '7', '7', '7', '0', '7')
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

    headed_span = [(_find_span_id(idx, 'l'), _find_span_id(idx, 'r'))
                   for idx in range(sent_len)]
    # print(headed_span)

    # headed_span_length = [right - left + 1 for left, right in headed_span]
    # print(headed_span_length)
    return headed_span


def subtree_distribution(corpus: Corpus):
    print(corpus[4])
    print(gen_headed_span(corpus[4]))
    exit()
    min_span_len = 5
    max_span_len = 10
    # num = 0
    for sid, sent in enumerate(corpus):
        if len(sent.ID) < 15:
            continue
        min_span_len = len(sent.ID) * 0.2
        max_span_len = len(sent.ID) * 0.4
        span = gen_headed_span(sent)
        span = list(filter(lambda ele: min_span_len <= ele[1] - ele[0] <= max_span_len, span))
        # num += len(span)

        print(len(sent.ID), len(span))
        if sid == 10:
            print()
            exit()
