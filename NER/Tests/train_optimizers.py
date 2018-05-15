# encoding:utf-8
'''
@Author: catnlp
@Email: wk_nlp@163.com
@Time: 2018/5/15 21:24
'''
from NER.utils.config import Config
from NER.utils.helpers import *

import sys
import argparse
import random
import torch
import numpy as np

seed_num = 100
random.seed(seed_num)
torch.manual_seed(seed_num)
np.random.seed(seed_num)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tuning with NER')
    parser.add_argument('--wordemb', help='Embedding for words', default='glove')
    parser.add_argument('--charemb', help='Embedding for chars', default='None')
    parser.add_argument('--train', default='../../data/conll2003/train.bmes')
    parser.add_argument('--dev', default='../../data/conll2003/dev.bmes')
    parser.add_argument('--test', default='../../data/conll2003/test.bmes')
    parser.add_argument('--gpu', default='True')
    args = parser.parse_args()

    train_file = args.train
    dev_file = args.dev
    test_file = args.test

    if args.gpu.lower() == 'false':
        gpu = False
    else:
        gpu = torch.cuda.is_available()

    print('Seed num: ', seed_num)
    print('GPU available: ', gpu)
    print('Train file: ', train_file)
    print('Dev file: ', dev_file)
    print('Test file: ', test_file)
    sys.stdout.flush()

    emb = args.wordemb.lower()
    print('Word Embedding: ', emb)
    if emb == 'glove':
        emb_file = '../../data/embedding/glove.6B.100d.txt'
    else:
        emb_file = None
    char_emb_file = args.charemb.lower()
    print('Char Embedding: ', char_emb_file)

    name = 'BaseLSTM'  # catnlp
    config = Config()
    config.lr = 0.015
    config.hidden_dim = 200
    config.number_normalized = True
    data_initialization(config, train_file, dev_file, test_file)
    config.gpu = gpu
    config.word_features = name
    print('Word features: ', config.word_features)
    config.generate_instance(train_file, 'train')
    config.generate_instance(dev_file, 'dev')
    config.generate_instance(test_file, 'test')
    if emb_file:
        print('load word emb file...norm: ', config.norm_word_emb)
        config.build_word_pretain_emb(emb_file)
    if char_emb_file != 'none':
        print('load char emb file...norm: ', config.norm_char_emb)
        config.build_char_pretrain_emb(char_emb_file)
    test_optimizer(config)