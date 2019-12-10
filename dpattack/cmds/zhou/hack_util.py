import numpy as np
from collections import defaultdict
from functools import lru_cache


def young_select(ordered_idxs=[5, 2, 1, 3, 0, 4], num_to_select=3, selected={2, 3, 4}, max_num=4):
    """
    selected = set()
    new_select, exc = young_select([5, 2, 1, 3, 0, 4], selected=selected)
    assert new_select == [5, 2, 1] and exc == []
    selected.update(set(new_select))
    for ele in exc:
        selected.remove(ele)
    new_select, exc = young_select([5, 4, 0, 3, 1, 2], selected=selected)
    assert new_select == [5, 4, 0] and exc == [2]
    selected.update(set(new_select))
    for ele in exc:
        selected.remove(ele)
    new_select, exc = young_select([3, 4, 0, 1, 2, 5], selected=selected)
    assert(new_select == [3, 4, 0] and exc == [5])
    """
    new_select = ordered_idxs[:num_to_select]
    # Remove words selected in this iteration
    ex_cands = [ele for ele in selected if ele not in new_select]
    # Sort other words
    sorted_ex_cands = [ele for ele in ordered_idxs if ele in ex_cands]
    return new_select, sorted_ex_cands[max_num - num_to_select:]


def elder_select(ordered_idxs=[5, 2, 1, 3, 0, 4], num_to_select=3, selected={2, 3, 4}, max_num=5):
    """
    selected = set()
    new_select = elder_select([5, 2, 1, 3, 0, 4], selected=selected)
    assert(new_select == [5, 2, 1])
    selected.update(set(new_select))
    new_select = elder_select([5, 2, 0, 3, 1, 4], selected=selected)
    assert(new_select == [5, 2, 0])
    selected.update(set(new_select))
    new_select = elder_select([3, 4, 0, 1, 2, 5], selected=selected)
    assert(new_select == [3, 0, 1])
    """
    ret = []
    total_num = len(selected)
    for ele in ordered_idxs:
        if len(ret) == num_to_select:
            break
        if ele in selected:
            ret.append(ele)
        else:
            if total_num == max_num:
                continue
            else:
                ret.append(ele)
                total_num += 1
    return ret


class _Tags:

    _UNI_TAG = defaultdict(lambda: 'X', {
        "!": ".",
        "#": ".",
        "$": ".",
        "''": ".",
        "(": ".",
        ")": ".",
        ",": ".",
        "-LRB-": ".",
        "-RRB-": ".",
        ".": ".",
        ":": ".",
        "?": ".",
        "CC": "CONJ",
        "CD": "NUM",
        "DT": "DET",
        "EX": "DET",
        "FW": "X",
        "IN": "ADP",
        "JJ": "ADJ",
        "JJR": "ADJ",
        "JJS": "ADJ",
        "LS": "X",
        "MD": "VERB",
        "NN": "NOUN",
        "NNP": "NOUN",
        "NNPS": "NOUN",
        "NNS": "NOUN",
        "NP": "NOUN",
        "PDT": "DET",
        "POS": "PRT",
        "PRP": "PRON",
        "PRP$": "PRON",
        "PRT": "PRT",
        "RB": "ADV",
        "RBR": "ADV",
        "RBS": "ADV",
        "RN": "X",
        "RP": "PRT",
        "SYM": "X",
        "TO": "PRT",
        "UH": "X",
        "VB": "VERB",
        "VBD": "VERB",
        "VBG": "VERB",
        "VBN": "VERB",
        "VBP": "VERB",
        "VBZ": "VERB",
        "VP": "VERB",
        "WDT": "DET",
        "WH": "X",
        "WP": "PRON",
        "WP$": "PRON",
        "WRB": "ADV",
        "``": "."
    })

    def __getitem__(self, k):
        assert k
        ret = []
        if 'n' in k:
            ret += ['NN', 'NNS', 'NNP', 'NNPS']
        if 'j' in k:
            ret += ['JJ', 'JJR', 'JJS']
        if 'v' in k:
            ret += ['VB', 'VBD', 'VBG', 'VBN', 'VBZ', 'VBP']
        if 'i' in k:
            ret += ['IN']
        if 'r' in k:
            ret += ['RB', 'RBR', 'RBS']
        if 'x' in k:
            ret += [
                "#", "$", "''", ",", "-LRB-", "-RRB-", ".", ":", "CC", "CD", "DT", "EX", "FW", "LS",
                "MD", "PDT", "POS", "PRP", "PRP$ ", "RP", "SYM", "TO", "UH", "WDT", "WP", "WP$",
                "WRB", "``"
            ]
        return tuple(ret)

    def ptb2uni(self, k):
        return self._UNI_TAG[k]
        
    @lru_cache(maxsize=None)
    def uni2ptb(self, k):
        ret = []
        for ele in self._UNI_TAG:
            if self._UNI_TAG[ele] == k:
                ret.append(ele)
        return tuple(ret)

    def ptb2njvri(self, k):
        if k[:2] == 'NN':
            return 'n'
        elif k[:2] == 'JJ':
            return 'j'
        elif k[:2] == 'RB':
            return 'r'
        elif k[:2] == 'VB':
            return 'v'
        elif k[:2] == 'IN':
            return 'i'
        else:
            return 'x'


HACK_TAGS = _Tags()

# Code for fucking VSCode debug console


class V:
    def __sub__(self, tsr):
        for ele in tsr.__repr__().split('\n'):
            print(ele)


v = V()

# ===================================================================
#   Below are helper functions for blackbox settings
# ===================================================================

# legal_words = list(range(30000))


def gen_idxs_to_substitute(idxs, repl_num, cand_num):
    return np.stack([np.random.choice(idxs, replace=False, size=repl_num) for _ in range(cand_num)])


def subsitute_by_idxs(words, idxs, vocab_idxs):
    ret = []
    cand_num = len(idxs)
    repl_num = len(idxs[0])
    to_repl = np.random.choice(vocab_idxs, size=cand_num * repl_num)
    to_repl_id = 0

    for i in range(cand_num):
        tmp = words.copy()
        for j in range(repl_num):
            tmp[idxs[i][j]] = to_repl[to_repl_id]
            to_repl_id += 1
        ret.append(tmp)
    return ret


def subsitute_by_idxs_2(words, idxs, start_idx, vocab_idxs_lst):
    ret = []
    cand_num = len(idxs)
    repl_num = len(idxs[0])
    to_repl_lst = []

    for vocab_idxs in vocab_idxs_lst:
        to_repl_lst.append(np.random.choice(vocab_idxs, size=cand_num))

    for i in range(cand_num):
        tmp = words.copy()
        for j in range(repl_num):
            tmp[idxs[i][j]] = to_repl_lst[idxs[i][j] - start_idx][i]
        ret.append(tmp)
    return ret
