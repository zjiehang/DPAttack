# -*- coding: utf-8 -*-

import argparse
import os
from dpattack.cmds import Evaluate, Predict, Train
# from dpattack.cmds import Evaluate, Predict, Train, BlackBoxAttack, WholeSentenceAttack, SubTreeAttack, Augmentation
from config import Config
import torch

from dpattack.cmds.zhou.hacksubtree import HackSubtree
from dpattack.cmds.zhou.hackwhole import HackWhole

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create the Biaffine Parser models.'
    )

    parser.add_argument('--mode', default='hacksubtree',help='running mode choice')
    parser.add_argument('--conf', default='config.ini',help='the path of config file')
    args = parser.parse_args()

    subcommands = {
        'evaluate': Evaluate(),
        'predict': Predict(),
        'train': Train(),
        # zhou
        'hackwhole': HackWhole(),
        'hacksubtree': HackSubtree(),
        # zeng
        # 'blackbox': BlackBoxAttack(),
        # 'subtree': SubTreeAttack(),
        # 'sentencew':WholeSentenceAttack(),
        # 'augmentation':Augmentation()
    }

    print(f"Override the default configs with parsed arguments")
    config = Config(args.conf)
    config.update(vars(args))
    print(config)

    torch.set_num_threads(config.threads)
    torch.manual_seed(config.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = config.device

    print(f"Run the subcommand in mode {config.mode}")
    cmd = subcommands[config.mode]
    cmd(config)
