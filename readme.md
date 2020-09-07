# DPAttack

*TODO: Code refactoring*

Pytorch implementation for ACL 2020 Paper: "Evaluating and Enhancing the Robustness of Neural Network-based Dependency Parsing Models with Adversarial Examples", [URL](https://acl2020.org/).

You may cite our paper by:

```
@inproceedings{zheng-etal-2020-evaluating,
    title = "Evaluating and Enhancing the Robustness of Neural Network-based Dependency Parsing Models with Adversarial Examples",
    author = "Zheng, Xiaoqing  and
      Zeng, Jiehang  and
      Zhou, Yi  and
      Hsieh, Cho-Jui  and
      Cheng, Minhao  and
      Huang, Xuanjing",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    year = "2020",
    publisher = "Association for Computational Linguistics"
}
```

# Requirements:

`Python`: 3.7

`PyTorch`: 1.4.0

# Usage

### Train
```sh
$ python run.py -mode=train -conf=config.ini
```

You can configure the model in `config.ini`

### Evaluate
```sh
$ python run.py -mode=evaluate -conf=config.ini
```

### Attack

#### Blackbox

```sh
$ python run.py -mode=attack_type -conf=config.ini
```

where `attack_type` includes `blackbox`,  `blackbox_phrase`, `whitebox_phrase`

`blackbox`: black-box on sentence-level attack

`blackbox_phrase`: black-box on phrase-level attack

`whitebox_phrase`: white-box on phrase-level attack

#### Whitebox

```sh
$ python run_whitebox.py -mode=hackwhole|hackchar|hacksubtree
```

where `hackwhole` denotes attack a parser at word level, `hackchar` denotes attack a parser at char level, and `hacksubtree` denotes attack a subtree in a parser by modifying another one.

The settings for attacking can be found in `run_whitebox.py`, some basic configuration for the parser can be found in `config.ini`. If there exists some conflictions, the config in `*.py` will override that in `*.ini`.


<!-- # Result

Results of Black-box attack and White-box attack on 10% (maximum percentage of words that are allowed to be modified) are listed belowed:

|  | | UAS  | LAS | Success Rate% | #avg. change | 
| :----: |  :----:|  :----:|  :----: |  :----:|  :----: |
| char | clean |  |  |  |  | 
|  | black-box  |  |  |  |  | 
|  | white-box  |  |  |  |  | 
| word | clean |  |  |  |  | 
|  | black-box  |  |  |  |  | 
|  | white-box  |  |  |  |  | 
| word + tag | clean |  |  |  |  | 
|  | black-box  |  |  |  |  | 
|  | white-box  |  |  |  |  | 


More details can be seen in paper. -->

# Acknowledgement

The implementation is based on [yzhangcs](https://github.com/yzhangcs)'s code in [biaffine-parser](https://github.com/yzhangcs/biaffine-parser). Thanks for his implementation of "Deep Biaffine Attention for Neural Dependency Parsing". 

The nlpaug library is based on [makcedward](https://github.com/makcedward)'s libary in [nlpaug](https://github.com/makcedward/nlpaug), a python library for data augmentation in NLP. Thanks for their awesome library.

 