# encoding:utf-8
'''
@Author: catnlp
@Email: wk_nlp@163.com
@Time: 2018/5/2 15:06
'''
from NER.Module.encoder import Encoder

import torch.nn as nn

class NER(nn.Module):
    def __init__(self, config):
        super(NER, self).__init__()
        print('---build batched NER---')
        self.encoder = Encoder(config)

    def neg_log_likelihood_loss(self, word_inputs, word_seq_lengths, batch_label):
        loss, tag_seq = self.encoder.neg_log_likelihood_loss(word_inputs, word_seq_lengths, batch_label)
        return loss, tag_seq

    def forward(self, word_inputs, word_seq_lengths, mask):
        decode_seq = self.encoder(word_inputs, word_seq_lengths, mask)
        return decode_seq

    def get_word_features(self, word_inputs, word_seq_lengths):
        return self.encoder.get_word_features(word_inputs, word_seq_lengths)
