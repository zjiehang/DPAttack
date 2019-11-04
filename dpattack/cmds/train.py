# -*- coding: utf-8 -*-

import os
from datetime import datetime, timedelta
from dpattack.utils.parser_helper import init_parser, load_parser
from dpattack.utils.metric import ParserMetric, TaggerMetric
from dpattack.utils.corpus import Corpus
from dpattack.utils.pretrained import Pretrained
from dpattack.utils.vocab import Vocab
from dpattack.models import WordParser, WordTagParser, WordCharParser, CharParser, PosTagger
from dpattack.utils.data import TextDataset, batchify
from dpattack.task import ParserTask, TaggerTask
from shutil import copyfile

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR


class Train(object):
    def __call__(self, config):
        print("Preprocess the data")
        train = Corpus.load(config.ftrain)
        dev = Corpus.load(config.fdev)
        test = Corpus.load(config.ftest)
        if os.path.exists(config.vocab):
            vocab = torch.load(config.vocab)
        else:
            vocab = Vocab.from_corpus(corpus=train, min_freq=2)
            vocab.read_embeddings(Pretrained.load(config.fembed, config.unk))
            torch.save(vocab, config.vocab)
        config.update({
            'n_words': vocab.n_train_words,
            'n_tags': vocab.n_tags,
            'n_rels': vocab.n_rels,
            'n_chars': vocab.n_chars,
            'pad_index': vocab.pad_index,
            'unk_index': vocab.unk_index
        })
        print(vocab)

        print("Load the dataset")
        trainset = TextDataset(vocab.numericalize(train))
        devset = TextDataset(vocab.numericalize(dev))
        testset = TextDataset(vocab.numericalize(test))
        # set the data loaders
        train_loader = batchify(dataset=trainset,
                                batch_size=config.batch_size,
                                n_buckets=config.buckets,
                                shuffle=True)
        dev_loader = batchify(dataset=devset,
                              batch_size=config.batch_size,
                              n_buckets=config.buckets)
        test_loader = batchify(dataset=testset,
                               batch_size=config.batch_size,
                               n_buckets=config.buckets)
        print(f"{'train:':6} {len(trainset):5} sentences in total, "
              f"{len(train_loader):3} batches provided")
        print(f"{'dev:':6} {len(devset):5} sentences in total, "
              f"{len(dev_loader):3} batches provided")
        print(f"{'test:':6} {len(testset):5} sentences in total, "
              f"{len(test_loader):3} batches provided")

        print("Create the models")
        assert config.train_task in ['parser', 'tagger']
        is_training_parser = config.train_task == 'parser'

        if is_training_parser:
            model = init_parser(config, vocab.embeddings)
            task = ParserTask(vocab, model)
            best_e, best_metric = 1, ParserMetric()
        else:
            model = PosTagger(config, vocab.embeddings)
            task = TaggerTask(vocab, model)
            best_e, best_metric = 1, TaggerMetric()

        if torch.cuda.is_available():
            model = model.cuda()
        print(f"{model}\n")

        total_time = timedelta()
        # best_e, best_metric = 1, TaggerMetric()
        task.optimizer = Adam(
            task.model.parameters(),
            config.lr,
            (config.beta_1, config.beta_2),
            config.epsilon
        )
        task.scheduler = ExponentialLR(
            task.optimizer,
            config.decay ** (1 / config.steps)
        )

        for epoch in range(1, config.epochs + 1):
            start = datetime.now()
            # train one epoch and update the parameters
            task.train(train_loader)

            print(f"Epoch {epoch} / {config.epochs}:")
            loss, train_metric = task.evaluate(train_loader, config.punct)
            print(f"{'train:':6} Loss: {loss:.4f} {train_metric}")
            loss, dev_metric = task.evaluate(dev_loader, config.punct)
            print(f"{'dev:':6} Loss: {loss:.4f} {dev_metric}")
            loss, test_metric = task.evaluate(test_loader, config.punct)
            print(f"{'test:':6} Loss: {loss:.4f} {test_metric}")

            t = datetime.now() - start
            # save the models if it is the best so far
            # if dev_metric > best_metric:
            #     best_e, best_metric = epoch, dev_metric
            #     models.dpattack.save(config.parser_model + f".{best_e}")
            #     print(f"{t}s elapsed (saved)\n")
            # else:
            #     print(f"{t}s elapsed\n")
            # total_time += t
            # # if epoch - best_e >= config.patience:
            # #     break
            if dev_metric > best_metric and epoch > config.patience:
                best_e, best_metric = epoch, dev_metric
                if is_training_parser:
                    task.model.save(config.parser_model + f".{best_e}")
                else:
                    task.model.save(config.tagger_model + f".{best_e}")
                print(f"{t}s elapsed (saved)\n")
            else:
                print(f"{t}s elapsed\n")
            total_time += t
            if epoch - best_e >= config.patience:
                break

        if is_training_parser:
            copyfile(config.parser_model + f'.{best_e}',
                     config.parser_model + '.best')
            task.model = load_parser(config.parser_model + f".{best_e}")
        else:
            copyfile(config.tagger_model + f'.{best_e}',
                     config.tagger_model + '.best')
            task.model = PosTagger.load(config.tagger_model + f".{best_e}")
        loss, metric = task.evaluate(test_loader, config.punct)

        print(f"max score of dev is {best_metric.score:.2%} at epoch {best_e}")
        print(f"the score of test at epoch {best_e} is {metric.score:.2%}")
        print(f"average time of each epoch is {total_time / epoch}s")
        print(f"{total_time}s elapsed")
