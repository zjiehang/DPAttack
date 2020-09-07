# -*- coding: utf-8 -*-

import argparse
import os
from dpattack.cmds import Evaluate, Predict, Train
# from dpattack.cmds import Evaluate, Predict, Train, BlackBoxAttack, WholeSentenceAttack, SubTreeAttack, Augmentation
from config import Config
import torch

from dpattack.cmds.zhou.hackoutside import HackOutside
from dpattack.cmds.zhou.hackwhole import HackWhole
from dpattack.cmds.zhou.hacksubtree import HackSubtree
from dpattack.cmds.zhou.hackchar import HackChar

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create the Biaffine Parser models.')

    # mode: hackwhole/hacksubtree/hackchar
    #   hackwhole: to decrease the accuracy of the parser (word level)
    #   hackchar:  to decrease the accuracy of the parser (char level)
    #   hacksubtree: to decrease the accuracy of a subtree by modifying another one
    parser.add_argument('--mode', default='hackchar', help='running mode choice')
    parser.add_argument('--conf', default='config.ini', help='the path of config file')
    # data for training
    parser.add_argument('--ftrain',
                        default='/disks/sdb/zjiehang/zhou_data_new/ptb/aug_word.sd')
    # data for attacking
    parser.add_argument('--fdata',
                        default='/disks/sdb/zjiehang/zhou_data_new/ptb/ptb_test_3.3.0.sd')
                        # default='/disks/sdb/zjiehang/DependencyParsing/result/char/black_typo_unk_0.15.conllx')
                        # default='char.json.logs/char.conll')
                        # default='/disks/sdb/zjiehang/DependencyParsing/result/word/black_substitute_unk_0.15.conllx')
                        # default='bertag.json.logs/word.conll')
                        # default='/disks/sdb/zjiehang/DependencyParsing/result/word_tag/black_substitute_unk_0.15.conllx')
                        # default='bertag.json.logs/wordtag.conll')
    parser.add_argument('--input', default='<placeholder>', type=str)
    parser.add_argument('--parser_model',
                        # default="/disks/sdb/zjiehang/zhou_data_new/saved_models/word/lzynb")
                        default="/disks/sdb/zjiehang/zhou_data_new/saved_models/char/aug")
                        # default="/disks/sdb/zjiehang/zhou_data_new/saved_models/word_tag/aug")
                        # default="/disks/sdb/zjiehang/zhou_data_new/saved_models/word/aug")
                        # default="/disks/sdb/zjiehang/zhou_data_new/saved_models/word_tag/aug12")
                        # default="/disks/sdb/zjiehang/zhou_data_new/saved_models/word/aug12")
    parser.add_argument('--logf', default="off", type=str)
    parser.add_argument('--device', default='0', type=str)
    parser.add_argument('--alchemist', default='off')
    parser.add_argument('--seed', default=1, type=int)

    # --------------------------
    # settings for hacking a parser at char-level
    # --------------------------
    parser.add_argument('--hkc_max_change', default=3, type=float)
    parser.add_argument('--hkc_iter_change', default=2, type=int)
    parser.add_argument('--hkc_steps', default=30)
    parser.add_argument('--hkc_selection', default='elder')
    parser.add_argument('--hkc_dist_measure', default='euc')
    parser.add_argument('--hkc_step_size', default=100, type=int)
    parser.add_argument('--hkc_mst', default='on')

    # --------------------------
    # settings for hacking a parser at subtree-level
    # --------------------------
    parser.add_argument('--hks_eps', default=0.3, type=float)
    parser.add_argument('--hks_dist_measure', default='euc', type=str)
    parser.add_argument('--hks_step_size', default=15, type=int)
    parser.add_argument('--hks_max_change', default=3, type=float)
    parser.add_argument('--hks_iter_change', default=8)
    parser.add_argument('--hks_steps', default=30, type=int)
    parser.add_argument('--hks_constraint', default='any')
    parser.add_argument('--hks_min_span_len', default=4, type=int)
    parser.add_argument('--hks_max_span_len', default=12, type=int)
    parser.add_argument('--hks_span_gap', default=1, type=int)
    parser.add_argument('--hks_span_selection', default='jacobian1')
    parser.add_argument('--hks_word_random', default='off')
    parser.add_argument('--hks_mst', default='on')
    parser.add_argument('--hks_loss', default='sum')
    parser.add_argument('--hks_color', default='white')
    parser.add_argument('--hks_cand_num', default=256, type=int)
    parser.add_argument('--hks_topk_pair', default=100, type=int)
    parser.add_argument('--hks_blk_repl_tag', default='any', type=str)


    # --------------------------
    # settings for hacking a parser at word-level
    # --------------------------
    parser.add_argument('--hkw_step_size', default=10, type=int)
    parser.add_argument('--hkw_steps', default=40, type=int)
    parser.add_argument('--hkw_eps', default=0.3, type=float)
    parser.add_argument('--hkw_dist_measure', default='euc')
    parser.add_argument('--hkw_loss_based_on', default='logit')
    parser.add_argument('--hkw_max_change', default=0.15, type=float)
    parser.add_argument('--hkw_iter_change', default=2, type=float)
    parser.add_argument('--hkw_mst', default='on')
    parser.add_argument('--hkw_tag_type', default='njvr', type=str)
    parser.add_argument('--hkw_repl_method', default='bertag')
    # parser.add_argument('--hkw_selection', default="twin", type=str)

    # general settings
    parser.add_argument('--hk_training_set', default='off', type=str)
    parser.add_argument('--hk_use_worker', default='off')
    parser.add_argument('--hk_num_worker', default=9, type=int)
    parser.add_argument('--hk_worker_id', default=8, type=int)

    args = parser.parse_args()

    subcommands = {
        'evaluate': Evaluate(),
        'predict': Predict(),
        'train': Train(),
        # zhou
        'hackwhole': HackWhole(),
        'hackoutside': HackOutside(),
        "hacksubtree": HackSubtree(),
        "hackchar": HackChar()
        # zeng
        # 'blackbox': BlackBoxAttack(),
        # 'subtree': SubTreeAttack(),
        # 'sentencew':WholeSentenceAttack(),
        # 'augmentation':Augmentation()
    }

    print(f"Override the default configs with parsed arguments")
    config = Config(args.conf)
    config.update(vars(args))

    if config.mode in ['train', 'evaluate', 'test']:
        print(config)

    torch.set_num_threads(config.threads)
    torch.manual_seed(config.seed)
    if config.alchemist == 'off':
        os.environ['CUDA_VISIBLE_DEVICES'] = config.device

    print(f"Run the subcommand in mode {config.mode}")
    cmd = subcommands[config.mode]


    cmd(config)
