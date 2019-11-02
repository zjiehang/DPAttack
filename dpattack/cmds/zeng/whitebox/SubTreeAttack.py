from dpattack.cmds.zeng.whitebox import WhiteBoxAttackBase

class SubTreeAttack(WhiteBoxAttackBase):
    def __init__(self):
        super(SubTreeAttack, self).__init__()

    def __call__(self, config):
        corpus, loader = self.pre_attack(config)

        for sid, (words, tags, arcs, rels) in enumerate(loader):
            raw_words = words.clone()
            words_text = self.get_seqs_name(words)
            tags_text = self.get_tags_name(tags)