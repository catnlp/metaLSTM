# encoding:utf-8
'''
@Author: catnlp
@Email: wk_nlp@163.com
@Time: 2018/5/2 15:06
'''
from NER.Module.encoder import Encoder
from NER.Module.cove_encoder import CoVeEncoder
from NER.Module.crf import CRF
import torch.nn as nn

class NER(nn.Module):
    def __init__(self, config, cove_flag=False):
        super(NER, self).__init__()
        print('---build batched NER---')
        label_size = config.label_alphabet_size
        config.label_alphabet_size += 2
        if cove_flag:
            self.encoder = CoVeEncoder(config)
        else:
            self.encoder = Encoder(config)
        self.crf = CRF(label_size, config.gpu)

    def neg_log_likelihood_loss(self, word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, batch_label, mask):
        # loss, tag_seq = self.encoder.neg_log_likelihood_loss(word_inputs, word_seq_lengths, batch_label)
        # return loss, tag_seq
        outs = self.encoder.get_output_score(word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)
        total_loss = self.crf.neg_log_likelihood_loss(outs, batch_label, mask)
        scores, tag_seq = self.crf.viterbi_decode(outs, mask)
        return total_loss, tag_seq

    def forward(self, word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, mask):
        # decode_seq = self.encoder(word_inputs, word_seq_lengths, mask)
        # return decode_seq
        outs = self.encoder.get_output_score(word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)
        scores, tag_seq = self.crf.viterbi_decode(outs, mask)
        return tag_seq

    def get_word_features(self, word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover):
        return self.encoder.get_word_features(word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)
