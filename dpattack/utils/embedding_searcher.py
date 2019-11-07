import torch
from tabulate import tabulate

from typing import Union


class EmbeddingSearcher:
    def __init__(self,
                 embed: torch.Tensor,
                 word2idx: callable,
                 idx2word: callable):
        self.embed = embed
        self.word2idx = word2idx
        self.idx2word = idx2word

    def show_embedding_info(self):
        show_mean_std(self.embed, 'params of embed')
        show_mean_std(torch.norm(self.embed, p=2, dim=1), 'norm of embed')

    def find_neighbours(self,
                        element: Union[int, str, torch.Tensor],
                        topk,
                        measure,
                        verbose=False):
        assert measure in ['euc', 'cos']
        measure_fn = cos_dist if measure == 'cos' else euc_dist
        if isinstance(element, int):
            idx = element
            query_vector = self.embed[idx]
        elif isinstance(element, str):
            idx = self.word2idx(element)
            query_vector = self.embed[idx]
        elif isinstance(element, torch.Tensor):
            query_vector = element
        else:
            raise TypeError('You passed a {}, int/str/torch.Tensor required'.
                            format(type(element)))

        dists = measure_fn(query_vector, self.embed)
        tk_vals, tk_idxs = torch.topk(dists, topk, largest=False)
        if verbose:
            table = []
            print('Top-{} closest word measured by {}'.format(
                topk, measure))
            for i in tk_idxs:
                i = i.item()
                table.append([i, self.idx2word(i), dists[i].item()])
            print(tabulate(table))
        return tk_idxs, tk_vals


def cos_dist(qry, mem):
    return - torch.nn.functional.cosine_similarity(
        mem, qry.repeat(mem.size(0), 1), dim=1)


def euc_dist(qry, mem):
    return torch.sqrt((qry - mem).pow(2).sum(dim=1))


if __name__ == '__main__':

    def compare_idxes(nbr1, nbr2):
        nbr1 = set(cast_list(nbr1))
        nbr2 = set(cast_list(nbr2))
        inter = nbr1.intersection(nbr2)
        return len(inter)


    from dpattack.libs.luna import fetch_best_ckpt_name, cast_list, show_mean_std
    from dpattack.utils.parser_helper import load_parser
    from dpattack.utils.vocab import Vocab

    vocab = torch.load("/disks/sdb/zjiehang/zhou_data/ptb/vocab")  # type: Vocab
    parser = load_parser(fetch_best_ckpt_name("/disks/sdb/zjiehang/zhou_data/saved_models/word_tag/lzynb"))
    # print(type(vocab))
    esglv = EmbeddingSearcher(embed=vocab.embeddings,
                              idx2word=lambda x: vocab.words[x],
                              word2idx=lambda x: vocab.word_dict[x])
    esmdl = EmbeddingSearcher(embed=parser.embed.weight,
                              idx2word=lambda x: vocab.words[x],
                              word2idx=lambda x: vocab.word_dict[x])

    # esglv.show_embedding_info()
    # es.show_embedding_info()
    #
    # embed = es.embed[es.word2idx('red')]
    # idxs, vals = es.find_neighbours(embed, 20, 'euc', True)
    #
    # embed += 10 * torch.sign(embed)
    # es.find_neighbours(embed, 20, 'euc', True)

    for word in ['company', 'red', 'happy', 'play', 'down']:
        euc_idxes = esmdl.find_neighbours(word, 100, 'euc', verbose=True)
    exit()

    total = 0
    for ele in cast_list(torch.randint(len(vocab.words), (100,))):
        euc_idxes, _ = esmdl.find_neighbours(ele, 20, 'euc', verbose=False)
        cos_idxes, _ = esglv.find_neighbours(ele, 20, 'cos', verbose=False)
        print(vocab.words[ele], '\t\t', compare_idxes(euc_idxes, cos_idxes))
        total += compare_idxes(euc_idxes, cos_idxes)
    print(total, '\n')

    total = 0
    for ele in cast_list(torch.randint(len(vocab.words), (100,))):
        mdl_idxes, _ = esmdl.find_neighbours(ele, 20, 'euc', verbose=False)
        glv_idxes, _ = esglv.find_neighbours(ele, 20, 'euc', verbose=False)
        print(vocab.words[ele], '\t\t', compare_idxes(mdl_idxes, glv_idxes))
        total += compare_idxes(mdl_idxes, glv_idxes)
    print(total, '\n')
