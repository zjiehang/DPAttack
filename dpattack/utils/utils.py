from dpattack.libs.nlpaug.augmenter import word as naw
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


def is_chars_judger(model, tags = None, chars = None):
    if hasattr(model, 'char_lstm'):
        return True if chars is None else chars
    else:
        return False if tags is None else tags
