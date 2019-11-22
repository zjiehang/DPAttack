import nltk
from nltk.corpus import brown
from dpattack.utils.corpus import Corpus
from dpattack.utils.vocab import Vocab
from collections import defaultdict
import torch
from nltk.tag import CRFTagger


def train_gram_tagger(train_corpus: Corpus, ngram=1):
    train_sents = gen_tagged_sents(train_corpus)
    if ngram in [1, 2, 3]:
        tagger = nltk.UnigramTagger(
            train_sents, backoff=nltk.DefaultTagger('NN'))
    if ngram in [2, 3]:
        tagger = nltk.BigramTagger(train_sents, backoff=tagger)
    if ngram in [3]:
        tagger = nltk.TrigramTagger(train_sents, backoff=tagger)
    return tagger


def train_crf_tagger(train_corpus: Corpus, model_file='crfmodel'):
    train_sents = gen_tagged_sents(train_corpus)
    crf_tagger = CRFTagger()
    crf_tagger.train(train_sents, model_file)
    return crf_tagger


def gen_tagged_sents(corpus: Corpus):
    all_sents = []
    for sentence in corpus:
        sent = []
        for i in range(len(sentence.ID)):
            sent.append((sentence.FORM[i], sentence.POS[i]))
        all_sents.append(sent)
    return all_sents


if __name__ == "__main__":
    train_corpus = Corpus.load(
        "/disks/sdb/zjiehang/zhou_data/ptb/ptb_train_3.3.0.sd")
    test_corpus = Corpus.load(
        "/disks/sdb/zjiehang/zhou_data/ptb/ptb_test_3.3.0.sd")

    # vocab = torch.load("/disks/sdb/zjiehang/zhou_data/ptb/vocab")
    # gen_tag_dict(train_corpus, vocab)

    """
        Below is performance benchmark of different taggers.
    """
    # tagger = train_gram_tagger(train_corpus, 1)   # 0.9279514501446616
    # tagger = train_gram_tagger(train_corpus, 2)   # 0.947216145649566
    # tagger = train_gram_tagger(train_corpus, 3)   # 0.9476395455507727
    # tagger = train_crf_tagger(train_corpus)       # 0.9715263566438501  -> long training time
    ## tagger = CRFTagger()
    # tagger.set_model_file("/disks/sdb/zjiehang/zhou_data/saved_models/crftagger")
    # print(tagger.evaluate(gen_tagged_sents(test_corpus)))

    """
        Below is time benchmark of different taggers tested on 2048 sentences.
               for-loop   batch
        CRF    0.915      0.936
        TRI    0.723      0.661
        BI     0.526      0.495
        UNI    0.283      0.278
    """
    # tagger = train_gram_tagger(train_corpus, 1)
    # tagger = CRFTagger()
    # tagger.set_model_file("/disks/sdb/zjiehang/zhou_data/saved_models/crftagger")
    # test_sents = gen_tagged_sents(test_corpus)
    # test_sents = [[ele[0] for ele in sent] for sent in test_sents]
    # from dpattack.libs.luna import time_record
    # with time_record():
    #     for i in range(2048):
    #         tagger.tag_sents(test_sents[i:i+1])  # crf 0.915 seconds 0.723   0.526  0.283
    # with time_record():
    #     tagger.tag_sents(test_sents[:2048])      # crf 0.936 seconds 0.661   0.495  0.278

    # tagger = nltk.BigramTagger([[('the', 'dt'), ('work', 'nn'), ('of', 'in')]])
    # print(tagger.tag_sents([('the', 'end', 'of')]))

    sent = [
        "at <UNK> p.m. , at the throw of the `` cooling off '' period , the average was down <UNK> points .".split(" ")]

    tagger = CRFTagger()
    tagger.set_model_file(
        "/disks/sdb/zjiehang/zhou_data/saved_models/crftagger")
    print(tagger.tag_sents(sent))

    from dpattack.libs.luna import auto_create
    tagger = auto_create("trigram_tagger",
                         lambda: train_gram_tagger(
                             train_corpus, ngram=3),
                         cache=True, path='/disks/sdb/zjiehang/zhou_data/saved_vars')
    print(tagger.tag_sents(sent))


def gen_tag_dict(corpus: Corpus, vocab: Vocab,
                 threshold=3,
                 verbose=True):
    """
        Rule:
            If a word has more than one tags, select the tag with the maximum
            occurence so that a word with several tags will be classified into
            the most frequent one. 
        Return:
            Dict[str, List[int]] 
                e.g.  { "NN": [3, 4, 99, ...], ... }
        A verbose sample:
            Words occur more than 0 times: 29266
            Words occur more than 1 times: 25194
            Words occur more than 2 times: 18505
            Words occur more than 3 times: 14935
            Words occur more than 4 times: 12663
            Words occur more than 5 times: 11051
            Words occur more than 6 times: 9805
            Words occur more than 7 times: 8895
            Words occur more than 8 times: 8184
            Words occur more than 9 times: 7564
            <PAD> : 
            <UNK> : 
            # : #
            $ : $, a$, c$, hk$, us$
            '' : ''
            , : ,
            -LRB- : -lcb-, -lrb-
            -RRB- : -rcb-, -rrb-
            . : !, ., ?
            : : -, --, ..., :, ;
            <ROOT> : 
            CC : &, 'n', and, but, nor, or, plus
            CD : '86, 0.02, 0.03, 0.05, 0.1, 0.10, 0.13, 0.19, 0.2, 0.24
            DT : a, all, an, another, any, both, each, every, neither, no
            EX : there
            FW : bono, etc., facto, markka, vs.
            IN : aboard, about, above, across, after, against, albeit, along, alongside, although
            JJ : 1/2-year, 10-a-share, 10-month, 10-year, 100-share, 10th, 12-month, 12-year, 120-day, 13-week
            JJR : better, bigger, broader, cheaper, closer, deeper, easier, fewer, firmer, freer
            JJS : best, biggest, brightest, broadest, busiest, cheapest, closest, earliest, fastest, finest
            LS : 
            MD : 'd, 'll, ca, can, could, may, might, must, ought, shall
            NN : %, *, **, a.m., abatement, ability, abortion, about-face, absence, absenteeism
            NNP : <UNK>, 1989a, a&m, a., a.c., a.g., a.h., a.p., ab, abb
            NNPS : afrikaners, airlines, airways, americans, americas, angels, arabs, asians, associates, bahamas
            NNS : '60s, '80s, 1920s, 1930s, 1940s, 1950s, 1960s, 1970s, abortions, abrasives
            PDT : 
            POS : ', 's
            PRP : 'em, he, herself, him, himself, i, it, itself, me, myself
            PRP$ : her, his, its, my, our, their, your
            RB : a.m, abroad, abruptly, absolutely, accordingly, accurately, actively, actually, additionally, adequately
            RBR : earlier, faster
            RBS : 
            RP : off, out, up
            SYM : e, f, ffr, x
            TO : na, to
            UH : oh, yes
            VB : abandon, abolish, absorb, accelerate, accept, accommodate, accompany, accomplish, achieve, acquire
            VBD : accepted, accounted, acknowledged, acted, added, advanced, advised, affirmed, aged, agreed
            VBG : abandoning, accelerating, accepting, accompanying, according, accumulating, accusing, achieving, acknowledging, acquiring
            VBN : abandoned, abolished, absorbed, abused, accompanied, accomplished, accrued, accumulated, accused, accustomed
            VBP : 'm, 're, 've, abound, accuse, acknowledge, agree, am, are, argue
            VBZ : accepts, accompanies, accuses, acknowledges, acquires, adds, admits, advises, affects, agrees
            WDT : whatever, which, whichever
            WP : what, who, whoever, whom
            WP$ : whose
            WRB : how, when, whenever, where, whereby, wherever, why
            `` : `, ``
    """
    count = torch.zeros((vocab.n_words, vocab.n_tags))
    for word_seq, tag_seq in zip(corpus.words, corpus.tags):
        for word, tag in zip(word_seq[1:], tag_seq[1:]):
            word_id = vocab.word_dict.get(word.lower(), vocab.unk_index)
            tag_id = vocab.tag_dict.get(tag)
            count[word_id, tag_id] += 1
    if verbose:
        for i in range(10):
            print("Words occur more than {} times: {}".format(
                i, torch.sum(count > i).item()))
    vals, idxs = count.max(1)
    vals = vals.tolist()
    idxs = idxs.tolist()

    tag_dict = defaultdict(lambda: list())
    for i in range(vocab.n_words):
        if vals[i] < threshold:
            continue
        else:
            tag_dict[vocab.tags[idxs[i]]].append(i)
    if verbose:
        for i in range(vocab.n_tags):
            print(vocab.tags[i], ":", ", ".join([vocab.words[e]
                                                 for e in tag_dict[i][:10]]))
    return dict(tag_dict)
