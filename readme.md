# DPAttack (TODO: Reconstruction code)

Pytorch implementation for ACL 2020 Paper: "Evaluating and Enhancing the Robustness of Neural Network-based Dependency Parsing Models with Adversarial Examples", [URL](https://acl2020.org/).

# Requirement:
Python: 3.7

PyTorch: 1.4.0

# Usage

### Train
```sh
$ python run.py -mode=train -conf=config.ini
```

More details about the model can be adjusted in config.ini

### Evaluate
```sh
$ python run.py -mode=evaluate -conf=config.ini
```

### Attack
```sh
$ python run.py -mode=attack_type -conf=config.ini
```

where attack_type includes blackbox, whitebox, blackbox_phrase, whitebox_phrase

blackbox: black-box on sentence-level attack

whitebox: white-box on sentence-level attack

blackbox_phrase: black-box on phrase-level attack

whitebox_phrase: white-box on phrase-level attack



# Result

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


More details can be seen in paper.

# Acknowledgement

The implementation is based on [yzhangcs](https://github.com/yzhangcs)'s code in [biaffine-parser](https://github.com/yzhangcs/biaffine-parser). Thanks for his implementation of "Deep Biaffine Attention for Neural Dependency Parsing". 

The nlpaug library is based on [makcedward](https://github.com/makcedward)'s libary in [nlpaug](https://github.com/makcedward/nlpaug), a python library for data augmentation in NLP. Thanks for their awesome library.

# Cite:
 