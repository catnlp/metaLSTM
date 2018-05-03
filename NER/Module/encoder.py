# encoding:utf-8
'''
@Author: catnlp
@Email: wk_nlp@163.com
@Time: 2018/5/2 15:01
'''
from RNNs import RNN, LSTM
from MetaRNNs import MetaRNN, MetaLSTM

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        print('---build batched Encoder---')
        self.gpu = config.gpu
        self.use_char = config.use_char
        self.batch_size = config.batch_size

        self.embedding_dim = config.word_emb_dim
        self.hidden_dim = config.hidden_dim
        self.hyper_hidden_dim = config.hyper_hidden_dim
        self.hyper_embedding_dim = config.hyper_embedding_dim
        self.layers = config.layers
        self.drop = nn.Dropout(config.dropout)
        self.word_embeddings = nn.Embedding(config.word_dict.size(), self.embedding_dim)
        if config.pretain_word_embedding is not None:
            self.word_embeddings.weight.data.copy_(torch.from_numpy(config.pretain_word_embedding))
        else:
            self.word_embeddings.weight.data.copy_(torch.from_numpy(self.random_embedding(config.word_dict.size(), self.embedding_dim)))

        mode = config.word_features
        if mode == 'BaseRNN':
            self.encoder = nn.RNN(self.embedding_dim, self.hidden_dim, num_layers=self.layers, batch_first=True)
        elif mode == 'RNN':
            self.encoder = RNN(self.embedding_dim, self.hidden_dim, num_layers=self.layers)
        elif mode == 'MetaRNN':
            self.encoder = MetaRNN(self.embedding_dim, self.hidden_dim, self.hyper_hidden_dim, self.hyper_embedding_dim, num_layers=self.layers)
        elif mode == 'BaseLSTM':
            self.encoder = nn.LSTM(self.embedding_dim, self.hidden_dim, num_layers=self.layers, batch_first=True)
        elif mode == 'LSTM':
            self.encoder = LSTM(self.embedding_dim, self.hidden_dim, num_layers=self.layers)
        elif mode == 'MetaLSTM':
            self.encoder = MetaLSTM(self.embedding_dim, self.hidden_dim, self.hyper_hidden_dim, self.hyper_embedding_dim, num_layers=self.layers)
        else:
            print('Error word feature selection, please check config.word_features.')
            exit(0)

        self.hidden2tag = nn.Linear(self.hidden_dim, config.label_dict_size)

    def random_embedding(self, vocab_size, embedding_dim):
        initial_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for i in range(vocab_size):
            initial_emb[i, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return initial_emb

    def get_word_features(self, word_inputs, word_seq_lengths):
        word_embs = self.word_embeddings(word_inputs)
        word_embs = self.drop(word_embs)

        packed_words = pack_padded_sequence(word_embs, word_seq_lengths.cpu().numpy(), True)
        out, _ = self.encoder(packed_words)
        out, _ = pad_packed_sequence(out)

        return out

    def get_output_score(self, word_inputs, word_seq_lengths):
        out = self.get_word_features(word_inputs, word_seq_lengths)
        out = self.hidden2tag(out)
        return out

    def neg_log_likelihood_loss(self, word_inputs, word_seq_lengths, batch_label):
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        total_words = batch_size * seq_len
        loss_function = nn.NLLLoss(ignore_index=0, size_average=False)
        out = self.get_output_score(word_inputs, word_seq_lengths)
        out = out.view(total_words, -1)
        score = F.log_softmax(out, 1)
        loss = loss_function(score, batch_label.view(total_words))
        _, tag_seq = torch.max(score, 1)
        tag_seq = tag_seq.view(batch_size, seq_len)
        return loss, tag_seq

    def forward(self, word_inputs, word_seq_lengths, mask):
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        total_words = batch_size * seq_len
        out = self.get_output_score(word_inputs, word_seq_lengths)
        out = out.view(total_words, -1)
        _, tag_seq = torch.max(out, 1)
        tag_seq = tag_seq.view(batch_size, seq_len)
        decode_seq = mask.long() * tag_seq
        return decode_seq






