import numpy as np


def young_select(ordered_idxs=[5, 2, 1, 3, 0, 4],
                 num_to_select=3,
                 selected={2, 3, 4},
                 max_num=4):
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


def elder_select(ordered_idxs=[5, 2, 1, 3, 0, 4],
                 num_to_select=3,
                 selected={2, 3, 4},
                 max_num=5):
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
        return tuple(ret)

    def get_coarse(self, k):
        if k[:2] == 'NN':
            return self['n']
        elif k[:2] == 'JJ':
            return self['j']
        elif k[:2] == 'RB':
            return self['r']
        elif k[:2] == 'VB':
            return self['v']
        elif k[:2] == 'IN':
            return self['i']


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


def gen_idxs_to_substitute(start, end, repl_num, cand_num):
    return np.stack([
        np.random.choice(list(range(start, end)), replace=False, size=repl_num)
        for _ in range(cand_num)
    ])


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

