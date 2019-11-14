from dpattack.cmds.zeng.whitebox.base import WhiteBoxAttackBase
from dpattack.utils.metric import ParserMetric as Metric

class WholeSentenceAttack(WhiteBoxAttackBase):
    def __init__(self):
        super(WholeSentenceAttack, self).__init__()

        self.grads = {}
        self.attack_try_times = 100

    def extract_embed_grad(self, module, grad_in, grad_out):
        self.grads['embed'] = grad_out[0]

    def __call__(self, config):
        loader = self.pre_attack(config)

        raw_metric = Metric()
        attack_metric = Metric()

        for sid, (words, tags, arcs, rels) in enumerate(loader):

            raw_words = words.clone()
            words_text = self.get_seqs_name(words)
            tags_text = self.get_tags_name(tags)

            for i in range(self.attack_try_times):
                pass