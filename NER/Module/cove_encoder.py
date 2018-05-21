# encoding:utf-8
'''
@Author: catnlp
@Email: wk_nlp@163.com
@Time: 2018/5/21 12:38
'''
from Modules.RNNs import RNN, LSTM
from Modules.MetaRNNs import MetaRNN, MetaLSTM
from Modules.NormLSTM import NormLSTM, BNLSTMCell
from Modules.MetaNormLSTM import MetaNormLSTM, MetaLSTMCell
from NER.Module.char import Char

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.utils.model_zoo as model_zoo

model_urls = {
    'wmt-lstm' : 'https://s3.amazonaws.com/research.metamind.io/cove/wmtlstm-b142a7f2.pth'
}

model_cache = os.path.join(os.path.dirname(os.path.realpath(__file__)), '.torch')

class CoVeEncoder(nn.Module):
    def __init__(self, config):
        super(CoVeEncoder, self).__init__()
        print('---build batched CoVeEncoder---')
        self.gpu = config.gpu
        self.bidirectional = config.bid_flag
        self.batch_size = config.batch_size
        self.char_hidden_dim = 0

        self.use_char = config.use_char
        if self.use_char:
            self.char_hidden_dim = config.char_hidden_dim
            self.char_embedding_dim = config.char_emb_dim
            self.char = Char(config.char_features, config.char_alphabet.size(), self.char_embedding_dim, self.char_hidden_dim, config.dropout, self.gpu)

        self.rnn = nn.LSTM(300, 300, num_layers=2, bidirectional=True, batch_first=True)
        self.rnn.load_state_dict(model_zoo.load_url(model_urls['wmt-lstm'], model_dir=model_cache))

        self.embedding_dim = config.word_emb_dim  # catnlp
        self.hidden_dim = config.hidden_dim
        self.hyper_hidden_dim = config.hyper_hidden_dim
        self.hyper_embedding_dim = config.hyper_embedding_dim
        self.layers = config.layers
        self.drop = nn.Dropout(config.dropout)
        self.word_embeddings = nn.Embedding(config.word_alphabet.size(), self.embedding_dim)
        if config.pretrain_word_embedding is not None:
            self.word_embeddings.weight.data.copy_(torch.from_numpy(config.pretrain_word_embedding))
        else:
            self.word_embeddings.weight.data.copy_(torch.from_numpy(self.random_embedding(config.word_alphabet.size(), self.embedding_dim)))

        self.mode = config.word_features
        self.embedding_dim = 600
        if self.mode == 'BaseRNN':
            self.encoder = nn.RNN(self.char_hidden_dim+self.embedding_dim, self.hidden_dim, num_layers=self.layers, batch_first=True)
        elif self.mode == 'RNN':
            self.encoder = RNN(self.char_hidden_dim+self.embedding_dim, self.hidden_dim, num_layers=self.layers, gpu=self.gpu)
        elif self.mode == 'MetaRNN':
            self.encoder = MetaRNN(self.char_hidden_dim+self.embedding_dim, self.hidden_dim, self.hyper_hidden_dim, self.hyper_embedding_dim, num_layers=self.layers, gpu=self.gpu)
        elif self.mode == 'BaseLSTM':
            self.encoder = nn.LSTM(self.char_hidden_dim+self.embedding_dim, self.hidden_dim//2, num_layers=self.layers, batch_first=True, bidirectional=True)
        elif self.mode == 'LSTM':
            self.encoder = LSTM(self.char_hidden_dim+self.embedding_dim, self.hidden_dim//2, num_layers=self.layers, gpu=self.gpu, bidirectional=self.bidirectional)
        elif self.mode == 'MetaLSTM':
            self.encoder = MetaLSTM(self.char_hidden_dim+self.embedding_dim, self.hidden_dim//2, self.hyper_hidden_dim, self.hyper_embedding_dim, num_layers=self.layers, gpu=self.gpu, bidirectional=self.bidirectional)
        elif self.mode == 'NormLSTM':
            self.encoder = NormLSTM(BNLSTMCell, self.char_hidden_dim+self.embedding_dim, self.hidden_dim, num_layers=self.layers, batch_first=True, max_length=config.MAX_SENTENCE_LENGTH)
        elif self.mode == 'MetaNormLSTM':
            self.encoder = MetaNormLSTM(MetaLSTMCell, self.char_hidden_dim+self.embedding_dim, self.hidden_dim, num_layers=self.layers, batch_first=True, max_length=config.MAX_SENTENCE_LENGTH)
        else:
            print('Error word feature selection, please check config.word_features.')
            exit(0)

        if config.gpu:
            self.rnn = self.rnn.cuda()
            self.encoder = self.encoder.cuda()

        self.hidden2tag = nn.Linear(self.hidden_dim, config.label_alphabet_size)

    def random_embedding(self, vocab_size, embedding_dim):
        initial_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for i in range(vocab_size):
            initial_emb[i, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return initial_emb

    def get_word_features(self, word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover):
        batch_size = word_inputs.size(0)
        sent_len = word_inputs.size(1)
        word_embs = self.word_embeddings(word_inputs)

        word_embs = self.drop(word_embs)
        packed_embs = pack_padded_sequence(word_embs, word_seq_lengths.cpu().numpy(), True)
        outputs, _ = self.rnn(packed_embs)
        word_embs, _ = pad_packed_sequence(outputs, batch_first=True)
        word_embs.contiguous()
        word_embs = word_embs.view(batch_size, sent_len, -1)

        if self.use_char:
            char_features = self.char.get_last_hiddens(char_inputs, char_seq_lengths.cpu().numpy())
            char_features = char_features[char_seq_recover]
            char_features = char_features.view(batch_size, sent_len, -1)
            word_embs = torch.cat([word_embs, char_features], 2)

        word_embs = self.drop(word_embs)
        if self.mode.startswith('Base'):
            packed_words = pack_padded_sequence(word_embs, word_seq_lengths.cpu().numpy(), True)
            out, _ = self.encoder(packed_words)
            out, _ = pad_packed_sequence(out)
            out = out.transpose(1, 0)
        else:
            out, _ = self.encoder(word_embs)
        out = self.drop(out) ## catnlp
        return out

    def get_output_score(self, word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover):
        out = self.get_word_features(word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)
        out = self.hidden2tag(out)
        return out

    def neg_log_likelihood_loss(self, word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, batch_label):
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        total_words = batch_size * seq_len
        loss_function = nn.NLLLoss(ignore_index=0, size_average=False)
        out = self.get_output_score(word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)
        out = out.view(total_words, -1)
        score = F.log_softmax(out, 1)
        loss = loss_function(score, batch_label.view(total_words))
        _, tag_seq = torch.max(score, 1)
        tag_seq = tag_seq.view(batch_size, seq_len)
        return loss, tag_seq

    def forward(self, word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, mask):
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        total_words = batch_size * seq_len
        out = self.get_output_score(word_inputs, word_seq_lengths, char_seq_lengths, char_seq_recover)
        out = out.view(total_words, -1)
        _, tag_seq = torch.max(out, 1)
        tag_seq = tag_seq.view(batch_size, seq_len)
        decode_seq = mask.long() * tag_seq
        return decode_seq
