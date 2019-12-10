from dpattack.utils.corpus import Corpus, Sentence, sent_print
from collections import defaultdict


def check_gap(span_a, span_b, gap):
    return span_a[0] - span_b[1] >= gap or span_b[0] - span_a[1] >= gap


def get_gap(span_a, span_b):
    return max(span_a[0] - span_b[1], span_b[0] - span_a[1])


def filter_spans(spans, minl, maxl, trim=True, add_head=True):
    if add_head:
        hspans = []
        for sid, span in enumerate(spans):
            hspans.append((span[0], span[1], sid))
        spans = hspans
    spans = list(filter(lambda ele: minl <= ele[1] + 1 - ele[0] <= maxl, spans))
    if trim:
        ret = []
        for span in spans:
            if len(ret) != 0 and ret[-1][0] <= span[0] and ret[-1][1] >= span[1]:
                continue
            else:
                ret.append(span)
    else:
        ret = spans
    return ret


def ex_span_idx(span, sent_len):
    return [i for i in range(sent_len) if (not span[0] <= i <= span[1]) or i == span[2]]


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

    spans = [(_find_span_id(idx, 'l'), _find_span_id(idx, 'r')) for idx in range(sent_len)]
    # print(headed_span)

    # headed_span_length = [right - left + 1 for left, right in headed_span]
    # print(headed_span_length)
    return spans


def subtree_distribution(corpus: Corpus):
    dist = defaultdict(lambda: 0)

    for sid, sent in enumerate(corpus):
        spans = gen_spans(sent)
        dist[len(filter_spans(spans, 4, 12))] += 1
    total = sum(dist.values())
    for k in range(10):
        print(k, '->', dist[k] / total * 100)
    # print(dist)


if __name__ == "__main__":
    # Corpus.load('/home/zhouyi/en_ewt-ud-test.txt')
    # exit()

    # spans = [(2, 5), (7, 10), (11, 14), (16, 23), (19, 23), (20, 23), (28, 33), (38, 41)]
    # print(filter_spans(spans))

    corpus = Corpus.load("/disks/sdb/zjiehang/zhou_data_new/ptb/ptb_test_3.3.0.sd")
    subtree_distribution(corpus)
    # min_span_len = 5
    # max_span_len = 10
    # tt = 0
    # for sid, sent in enumerate(corpus):
    #     if len(sent.ID) < 15:
    #         continue
    #     # min_span_len = len(sent.ID) * 0.2
    #     # max_span_len = len(sent.ID) * 0.3
    #     min_span_len = 5
    #     max_span_len = 8
    #     span = gen_spans(sent)
    #     span = list(filter(lambda ele: min_span_len <= ele[1] - ele[0] <= max_span_len, span))
    #     # num += len(span)

    #     if len(span) >= 2:
    #         tt += 1
    #     print(len(sent.ID), len(span), min_span_len)
    #     # if sid == 100:
    #     #     print()
    #     #     break
    # print(tt)
