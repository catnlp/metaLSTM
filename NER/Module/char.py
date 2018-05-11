# encoding:utf-8
'''
@Author: catnlp
@Email: wk_nlp@163.com
@Time: 2018/5/2 15:02
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np

class Char(nn.Module):
    def __init__(self, mode, alphabet_size, embedding_dim, hidden_dim, dropout, gpu):
        super(Char, self).__init__()
        print('---build batched char---')
        self.mode = mode
        self.gpu = gpu
        self.dropout = nn.Dropout(dropout)
        self.embeddings = nn.Embedding(alphabet_size, embedding_dim)
        self.embeddings.weight.data.copy_(torch.from_numpy(self.random_embedding(alphabet_size, embedding_dim)))

        if self.mode == 'LSTM':
            self.hidden_dim = hidden_dim // 2
            self.char = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        elif self.mode == 'CNN':
            self.hidden_dim = hidden_dim
            self.char = nn.Conv1d(embedding_dim, self.hidden_dim, kernel_size=3, padding=1)
        else:
            print('Error char feature selection, please check parameter data.char_features.')
            exit(0)

        if self.gpu:
            self.dropout = self.dropout.cuda()
            self.embeddings = self.embeddings.cuda()
            self.char = self.char.cuda()

    def random_embedding(self, vocab_size, embedding_dim):
        emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            emb[index, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return emb

    def get_last_hiddens(self, input, seq_lengths):
        batch_size = input.size(0)
        char_embeds = self.dropout(self.embeddings(input))
        if self.mode == 'LSTM':
            pack_input = pack_padded_sequence(char_embeds, seq_lengths, True)
            char_rnn_out, char_hidden = self.char(pack_input)
            char_rnn_out, _ = pad_packed_sequence(char_rnn_out)
            last_hiddens = char_hidden[0].transpose(1, 0).contiguous().view(batch_size, -1)
        elif self.mode == 'CNN':
            char_embeds = char_embeds.transpose(2, 1).contiguous()
            char_cnn_out = self.char(char_embeds)
            last_hiddens = F.max_pool1d(char_cnn_out, char_cnn_out.size(2)).view(batch_size, -1)
        else:
            last_hiddens = None
        return last_hiddens

    def get_all_hiddens(self, input, seq_lengths):
        char_embeds = self.dropout(self.embeddings(input))
        if self.mode == 'LSTM':
            pack_input = pack_padded_sequence(char_embeds, seq_lengths, True)
            char_rnn_out, char_hidden = self.char(pack_input)
            char_rnn_out, _ = pad_packed_sequence(char_rnn_out)
            all_hiddens = char_rnn_out.transpose(1, 0)
        elif self.mode == 'CNN':
            char_embeds = char_embeds.transpose(2, 1).contiguous()
            all_hiddens = self.char(char_embeds).transpose(2, 1).contiguous()
        else:
            all_hiddens = None
        return all_hiddens

    def forward(self, input, seq_lengths):
        return self.get_all_hiddens(input, seq_lengths)
