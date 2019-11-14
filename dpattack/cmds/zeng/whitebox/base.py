from dpattack.cmds.zeng.attack import Attack

class WhiteBoxAttackBase(Attack):
    def __init__(self):
        super(WhiteBoxAttackBase, self).__init__()

    def attack(self, *args, **kwargs):
        pass