# encoding:utf-8
'''
@Author: catnlp
@Email: wk_nlp@163.com
@Time: 2018/5/2 19:37
'''

from NER.utils.alphabet import Alphabet
from NER.utils.functions import *

import sys

START = '</s>'
UNKNOWN = '</unk>'
PADDING = '</pad>'

class Config:
    def __init__(self):
        self.MAX_SENTENCE_LENGTH = 250
        self.MAX_WORD_LENGTH = -1
        self.number_normalized = True
        self.norm_word_emb = False
        self.norm_char_emb = False
        self.word_alphabet = Alphabet('word')
        self.char_alphabet = Alphabet('character')
        self.label_alphabet = Alphabet('label', True)
        self.tagScheme = 'NoSeg'
        self.word_features = 'BaseRNN' # 'RNN' / 'MetaRNN'
        self.char_features = 'LSTM'
        self.optim = 'SGD'

        self.train_texts = []
        self.dev_texts = []
        self.test_texts = []
        self.raw_texts = []

        self.train_ids = []
        self.dev_ids = []
        self.test_ids = []
        self.raw_ids = []

        self.word_emb_dim = 100
        self.char_emb_dim = 30
        self.pretrain_word_embedding = None
        self.pretrain_char_embedding = None
        self.label_size = 0
        self.word_alphabet_size = 0
        self.char_alphabet_size = 0
        self.label_alphabet_size = 0

        # hyper parameters
        self.iteration = 50
        self.batch_size = 10
        self.hidden_dim = 100
        self.hyper_hidden_dim = 100
        self.hyper_embedding_dim = 16
        self.char_hidden_dim = 50
        self.use_char = True
        self.dropout = 0.5
        self.layers = 1
        self.bid_flag = False
        self.gpu = False
        self.lr = 0.0015
        self.lr_decay = 0.05
        self.clip = False
        self.momentum = 0

    def show_data_summary(self):
        print("DATA SUMMARY START:")
        print("\tTag scheme: %s" % (self.tagScheme))
        print("\tMAX SENTENCE LENGTH: %s" % (self.MAX_SENTENCE_LENGTH))
        print("\tMAX WORD LENGTH: %s" % (self.MAX_WORD_LENGTH))
        print("\tNumber normalized: %s" % (self.number_normalized))
        print("\tWord alphabet size: %s" % (self.word_alphabet_size))
        print("\tChar alphabet size: %s" % (self.char_alphabet_size))
        print("\tLabel alphabet size: %s" % (self.label_alphabet_size))
        print("\tWord embedding size: %s" % (self.word_emb_dim))
        print("\tChar embedding size: %s" % (self.char_emb_dim))
        print("\tNorm word emb: %s" % (self.norm_word_emb))
        print("\tNorm char emb: %s" % (self.norm_char_emb))
        print("\tTrain instance number: %s" % (len(self.train_texts)))
        print("\tDev instance number: %s" % (len(self.dev_texts)))
        print("\tTest instance number: %s" % (len(self.test_texts)))
        print("\tRaw instance number: %s" % (len(self.raw_texts)))
        print("\tHyper iteration: %s" % (self.iteration))
        print("\tHyper batch size: %s" % (self.batch_size))
        print("\tHyper optimizer: %s" % (self.optim))
        print("\tHyper lr: %s" % (self.lr))
        print("\tHyper lr_decay: %s" % (self.lr_decay))
        print("\tHyper clip: %s" % (self.clip))
        print("\tHyper momentum: %s" % (self.momentum))
        print("\tHyper hidden_dim: %s" % (self.hidden_dim))
        print("\tHyper bid_flag: %s" % (self.bid_flag))
        print("\tHyper hyper_hidden_dim: %s" % (self.hyper_hidden_dim))
        print("\tHyper hyper_embedding_dim: %s" % (self.hyper_embedding_dim))
        print("\tHyper dropout: %s" % (self.dropout))
        print("\tHyper layers: %s" % (self.layers))
        print("\tHyper GPU: %s" % (self.gpu))
        print("\tHyper user_char: %s" % (self.use_char))
        if self.use_char:
            print("\tChar features: %s" % (self.char_features))
        print("DATA SUMMARY END.")
        sys.stdout.flush()

    def refresh_label_alphabet(self, input_file):
        old_size = self.label_alphabet_size
        self.label_alphabet.clear(True)
        in_lines = open(input_file, 'r').readlines()
        for line in in_lines:
            if len(line) > 2:
                pairs = line.strip().split()
                label = pairs[-1]
                self.label_alphabet.add(label)
        self.label_alphabet_size = self.label_alphabet.size()
        startS = False
        startB = False
        for label, _ in self.label_alphabet.iteritems():
            if 'S-' in label.upper():
                startS = True
            elif 'B-' in label.upper():
                startB = True
        if startB:
            if startS:
                self.tagScheme = 'BMES'
            else:
                self.tagScheme = 'BIO'
        self.fix_alphabet()
        print('Refresh label alphabet finished: old:%s -> new:%s' % (old_size, self.label_alphabet_size))

    def extend_word_char_alphabet(self, input_file_list):
        old_word_size = self.word_alphabet_size
        old_char_size = self.char_alphabet_size
        for input_file in input_file_list:
            in_lines = open(input_file, 'r').readlines()
            for line in in_lines:
                if len(line) > 2:
                    pairs = line.strip().split()
                    word = pairs[0]
                    if self.number_normalized:
                        word = normalize_word(word)
                    self.word_alphabet.add(word)
                    for char in word:
                        self.char_alphabet.add(char)
            self.word_alphabet_size = self.word_alphabet.size()
            self.char_alphabet_size = self.char_alphabet.size()
            print('Extend word/char alphabet finished!')
            print('\told word:%s -> new word:%s' % (old_word_size, self.word_alphabet_size))
            print('\told char:%s -> new char:%s' % (old_char_size, self.char_alphabet_size))
            for input_file in input_file_list:
                print('\tfrom file:%s' % input_file)

    def build_alphabet(self, input_file):
        in_lines = open(input_file, 'r').readlines()
        for line in in_lines:
            if len(line) > 2:
                pairs = line.strip().split()
                word = pairs[0] # catnlp
                if self.number_normalized:
                    word = normalize_word(word)
                label = pairs[-1]
                self.label_alphabet.add(label)
                self.word_alphabet.add(word)
                for char in word:
                    self.char_alphabet.add(char)
        self.word_alphabet_size = self.word_alphabet.size()
        self.char_alphabet_size = self.char_alphabet.size()
        self.label_alphabet_size = self.label_alphabet.size()
        startS = False
        startB = False
        for label, _ in self.label_alphabet.iteritems():
            if 'S-' in label.upper():
                startS = True
            elif 'B-' in label.upper():
                startB = True
        if startB:
            if startS:
                self.tagScheme = 'BMES'
            else:
                self.tagScheme = 'BIO'

    def fix_alphabet(self):
        self.word_alphabet.close()
        self.label_alphabet.close()
        self.char_alphabet.close()

    def build_word_pretain_emb(self, emb_path):
        self.pretrain_word_embedding, self.word_emb_dim = build_pretrain_embedding(emb_path, self.word_alphabet, self.word_emb_dim, self.norm_word_emb)

    def build_char_pretrain_emb(self, emb_path):
        self.pretrain_char_embedding, self.char_emb_dim = build_pretrain_embedding(emb_path, self.char_alphabet, self.char_emb_dim, self.norm_char_emb)

    def generate_instance(self, input_file, name):
        self.fix_alphabet()
        if name == 'train':
            self.train_texts, self.train_ids = read_instance(input_file, self.word_alphabet, self.char_alphabet, self.label_alphabet, self.number_normalized, self.MAX_SENTENCE_LENGTH)
        elif name == 'dev':
            self.dev_texts, self.dev_ids = read_instance(input_file, self.word_alphabet, self.char_alphabet, self.label_alphabet, self.number_normalized, self.MAX_SENTENCE_LENGTH)
        elif name == 'test':
            self.test_texts, self.test_ids = read_instance(input_file, self.word_alphabet, self.char_alphabet, self.label_alphabet, self.number_normalized, self.MAX_SENTENCE_LENGTH)
        elif name == 'raw':
            self.raw_texts, self.raw_ids = read_instance(input_file, self.word_alphabet, self.char_alphabet, self.label_alphabet, self.number_normalized, self.MAX_SENTENCE_LENGTH)
        else:
            print('Error: you can only generate train/dev/test/raw instance! Illegal input:%s' % name)

    def write_decoded_result(self, output_file, predict_results, name):
        fout = open(output_file, 'w')
        sent_num = len(predict_results)
        content_list = []
        if name == 'raw':
            content_list = self.raw_texts
        elif name == 'test':
            content_list = self.test_texts
        elif name == 'dev':
            content_list = self.dev_texts
        elif name == 'train':
            content_list = self.train_texts
        else:
            print('Error: illegal name during prdict resutl, name should be within train/dev/test/raw')

        assert(sent_num == len(content_list))
        for idx in range(sent_num):
            sent_length = len(predict_results[idx])
            for idy in range(sent_length):
                fout.write(content_list[idx][0][idy].encode('utf-8') + ' ' + predict_results[idx][idy] + '\n')
            fout.write('\n')
        fout.close()
        print('Predict %s result has been written into file. %s' % (name, output_file))