from dpattack.cmds.zeng.attack import Attack

class WhiteBoxAttackBase(Attack):
    def __init__(self):
        super(WhiteBoxAttackBase, self).__init__()
        self.extract_grad = []

    def extract_embed_grad(self, module, grad_in, grad_out):
         self.extract_grad.append(grad_out[0])

    def pre_attack(self, config):
        corpus, loader = super().pre_attack()
        # 为了兼容
        if self.parser.embed is None:
            self.parser.pretrained.register_backward_hook(self.extract_embed_grad)
        else:
            self.parser.embed.register_backward_hook(self.extract_embed_grad)
        return corpus, loader

    def attack(self, *args, **kwargs):
        pass