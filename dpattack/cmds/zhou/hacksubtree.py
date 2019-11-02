from dpattack.utils.corpus import Corpus, Sentence


class HackSubtree:
    def __init__(self):
        pass

    def __call__(self, config):
        corpus = Corpus.load(config.fdata)

        sent = corpus[777]  # type: Sentence

        # for i in range
        subtree_distribution(corpus)


def gen_headed_span(sent: Sentence):
    """
            Sample of a sentence (starting at 0):
                  ID=('1', '2', '3', '4', '5', '6', '7', '8')
                HEAD=('7', '7', '7', '7', '7', '7', '0', '7')
            """
    headed_span = [[i + 1, i + 1] for i in range(len(sent.ID))]
    for tid, hid in enumerate(sent.HEAD):
        # tid is the 0-started index,
        # hid is the 1-started index. (0 denotes ROOT)
        if hid == 0:
            continue
        hid = int(hid) - 1
        tid += 1
        if tid < hid:
            headed_span[hid][0] = min(headed_span[hid][0], tid)
        else:
            headed_span[hid][1] = max(headed_span[hid][1], tid)
    print(list(zip(sent.ID, sent.HEAD)))
    # print(sent.HEAD)
    print(headed_span)
    headed_span_length = [right - left + 1 for left, right in headed_span]
    # print(headed_span_length)
    return headed_span


def subtree_distribution(corpus: Corpus):
    gen_headed_span(corpus[3])
    exit()
    min_span_len = 3
    max_span_len = 5
    for sid, sent in enumerate(corpus):
        span = gen_headed_span(sent)
        span_len = [right - left + 1 for left, right in span]
        span = list(filter(lambda ele: min_span_len <= ele[1] - ele[0] <= max_span_len, span))
        print(span)
        if sid == 10:
            exit()
