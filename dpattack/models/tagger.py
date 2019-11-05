# -*- coding: utf-8 -*-

from dpattack.models.modules import (BiLSTM, SharedDropout, EmbeddingDropout)

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from dpattack.libs.luna import fetch_best_ckpt_name, create_folder_for_file


class PosTagger(nn.Module):

    def __init__(self, config, embeddings):
        super(PosTagger, self).__init__()


        self.config = config
        # the embedding layer
        self.embed = nn.Embedding.from_pretrained(embeddings[0:config.n_words], freeze=False)

        self.embed_dropout = EmbeddingDropout(p=config.embed_dropout)

        # the word-lstm layer
        self.lstm = BiLSTM(input_size=config.n_embed,
                           hidden_size=config.tag_n_lstm_hidden,
                           num_layers=config.tag_n_lstm_layers)
        self.lstm_dropout = SharedDropout(p=config.lstm_dropout)

        self.tagger_mlp = nn.Linear(config.tag_n_lstm_hidden*2,config.n_tags)
        self.mlp_dropout = nn.Dropout(0.3)

        self.pad_index = config.pad_index
        self.unk_index = config.unk_index


    def forward(self, words):
        # get the mask and lengths of given batch
        mask = words.ne(self.pad_index)
        lens = mask.sum(dim=1)
        # set the indices larger than num_embeddings to unk_index
        ext_mask = words.ge(self.embed.num_embeddings)
        ext_words = words.masked_fill(ext_mask, self.unk_index)

        # get outputs from embedding layers
        embed = self.embed(ext_words)
        x = self.embed_dropout(embed)

        sorted_lens, indices = torch.sort(lens, descending=True)
        inverse_indices = indices.argsort()
        x = pack_padded_sequence(x[indices], sorted_lens, True)
        x = self.lstm(x)
        x, _ = pad_packed_sequence(x, True)
        x = self.lstm_dropout(x)[inverse_indices]

        s_tag = self.tagger_mlp(x)
        s_tag = self.mlp_dropout(s_tag)
        return s_tag

    def decorator_forward(self, words, root, return_device='cuda'):
        squeeze_needed = False
        if len(words.shape) == 1:
            squeeze_needed = True
            words = words.unsqueeze(0)
        if torch.cuda.is_available():
            words = words.cuda()
        s_tag = self.forward(words)
        pred_tag = s_tag[:, 1:].argmax(-1)
        pred_tag = torch.cat([torch.zeros_like(pred_tag[:, :1]).fill_(root), pred_tag], dim=1)
        if squeeze_needed:
            pred_tag = pred_tag.squeeze()
        if return_device == 'cpu':
            return pred_tag.cpu()
        else:
            return pred_tag

    @classmethod
    def load(cls, fname):
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        state = torch.load(fname, map_location=device)
        parser = cls(state['config'], state['embeddings'])
        parser.load_state_dict(state['state_dict'])
        parser.to(device)

        return parser

    def save(self, fname):
        state = {
            'config': self.config,
            'embeddings': self.embed.weight,
            'state_dict': self.state_dict()
        }
        # create_folder_for_file(fname)
        torch.save(state, fname)
