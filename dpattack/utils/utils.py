from dpattack.libs import nlpaug as naw
from dpattack.utils.aug import RandomTagAug

def get_blackbox_augmentor(method, path, revised_rate, vocab = None, ftrain=None):
    aug = None
    #print("Load the revised method")
    if method == 'bert':
        aug = naw.ContextualWordEmbsAug(model_path=path, aug_p=revised_rate, aug_min=1)
    elif method == 'glove':
        aug = naw.WordEmbsAug(model_type=method, model_path=path, aug_p=revised_rate, aug_min=1)
    elif method == 'wordnet':
        aug = naw.SynonymAug(aug_src='wordnet',aug_min=1,aug_p=revised_rate)
    elif method == 'tag':
        aug = RandomTagAug(vocab, ftrain, revised_rate)
    else:
        print("Unsupporting augmentation method. Program exits!")
        exit()
    return aug


def merge_tag(tag):
    if tag.startswith("JJ"):
        return "JJ"
    elif tag.startswith("NN"):
        return "NN"
    elif tag.startswith("PRP"):
        return "PRP"
    elif tag.startswith("RB"):
        return "RB"
    elif tag.startswith("VB"):
        return "VB"
    elif tag.startswith("W"):
        return "W"
    else:
        return tag

def merge_tag_seq(tags):
    return [merge_tag(tag) for tag in tags]