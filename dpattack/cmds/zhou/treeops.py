from dpattack.utils.corpus import Corpus, Sentence, sent_print


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


if __name__ == "__main__":
    # Corpus.load('/home/zhouyi/en_ewt-ud-test.txt')
    # exit()

    corpus = Corpus.load("/disks/sdb/zjiehang/zhou_data/ptb/ptb_test_3.3.0.sd")
    min_span_len = 5
    max_span_len = 10
    tt = 0
    for sid, sent in enumerate(corpus):
        if len(sent.ID) < 15:
            continue
        # min_span_len = len(sent.ID) * 0.2
        # max_span_len = len(sent.ID) * 0.3
        min_span_len = 5
        max_span_len = 8
        span = gen_spans(sent)
        span = list(filter(lambda ele: min_span_len <= ele[1] - ele[0] <= max_span_len, span))
        # num += len(span)

        if len(span) >= 2:
            tt += 1
        print(len(sent.ID), len(span), min_span_len)
        # if sid == 100:
        #     print()
        #     break
    print(tt)
