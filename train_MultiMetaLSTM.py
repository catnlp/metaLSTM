# encoding:utf-8
'''
@Author: catnlp
@Email: wk_nlp@163.com
@Time: 2018/12/7 15:56
'''

# encoding:utf-8
'''
@Author: catnlp
@Email: wk_nlp@163.com
@Time: 2018/5/2 14:14
'''
from NER.utils.config import Config
from NER.utils.helpers import *

import os
import sys
import argparse
import random
import torch
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = '3'

seed_num = 100
random.seed(seed_num)
torch.manual_seed(seed_num)
np.random.seed(seed_num)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tuning with NER')
    parser.add_argument('--wordemb', help='Embedding for words', default='glove')
    parser.add_argument('--charemb', help='Embedding for chars', default='None')
    parser.add_argument('--status', choices=['train', 'test', 'decode'], help='update algorithm', default='train')
    parser.add_argument('--savemodel', default='../models/cc/MultiMetaLSTM') # catnlp
    parser.add_argument('--savedset', help='Dir of saved data setting', default='../models/cc/MultiMetaLSTM.dset') # catnlp
    parser.add_argument('--dataset', help='Dir of datset', default='../data/cc/data_path.txt')
    parser.add_argument('--train', default='train.tsv')
    parser.add_argument('--devel', default='devel.tsv')
    parser.add_argument('--test', default='test.tsv')
    parser.add_argument('--gpu', default='True')
    parser.add_argument('--seg', default='True')
    parser.add_argument('--extendalphabet', default='True')
    # parser.add_argument('--raw', default='data/E_group/conll2003/test.bmes')
    # parser.add_argument('--loadmodel', default='E_model/conll2003/train_bmes/conll2003_train_bmes')
    # parser.add_argument('--output', default='data/E_group/conll2003/decode_train_bmes.txt')
    args = parser.parse_args()

    # train_file = args.train
    # dev_file = args.dev
    # test_file = args.test
    # raw_file = args.raw
    # model_dir = args.loadmodel
    dset_dir = args.savedset
    # output_file = args.output
    if args.seg.lower() == 'true':
        seg = True
    else:
        seg = False
    status = args.status.lower()

    save_model_dir = args.savemodel
    if args.gpu.lower() == 'false':
        gpu = False
    else:
        gpu = torch.cuda.is_available()

    print('Seed num: ', seed_num)
    print('GPU available: ', gpu)
    print('Status: ', status)

    print('Seg: ', seg)

    # print('Raw file: ', raw_file)
    if status == 'train':
        print('Model saved to: ', save_model_dir)
    sys.stdout.flush()

    if status == 'train':
        emb = args.wordemb.lower()
        print('Word Embedding: ', emb)
        if emb == 'glove':
            emb_file = 'data/embedding/glove.6B.100d.txt'
        else:
            emb_file = None
        char_emb_file = args.charemb.lower()
        print('Char Embedding: ', char_emb_file)

        name = 'BaseLSTM'  # catnlp
        config = Config()
        config.optim = 'SGD'
        config.lr = 0.015
        config.iteration = 200
        config.hidden_dim = 200
        # config.clip = True
        config.number_normalized = True
        config.gpu = gpu
        config.word_features = name
        print('Word features: ', config.word_features)
        count = 0
        with open(args.dataset, 'r') as f:
            for line in f:
                print(line)
                count += 1
                line = line.strip()
                if not line:
                    break
                train_file = os.path.join(line, args.train)
                devel_file = os.path.join(line, args.devel)
                test_file = os.path.join(line, args.test)

                print('Train file: ', train_file)
                print('Dev file: ', devel_file)
                print('Test file: ', test_file)
                data_initialization(config, train_file, devel_file, test_file)
            # config.fix_alphabet()
        config.num_corpus = count

        with open(args.dataset, 'r') as f:
            for line in f:
                line = line.strip()
                train_file = os.path.join(line, args.train)
                devel_file = os.path.join(line, args.devel)
                test_file = os.path.join(line, args.test)
                config.generate_instance(train_file, 'train')
                config.generate_instance(devel_file, 'dev')
                config.generate_instance(test_file, 'test')

        if emb_file:
            print('load word emb file...norm: ', config.norm_word_emb)
            config.build_word_pretain_emb(emb_file)
        if char_emb_file != 'none':
            print('load char emb file...norm: ', config.norm_char_emb)
            config.build_char_pretrain_emb(char_emb_file)

        for label in config.label_alphabet.instances:
            print(label)
        name = 'MultiMetaLSTM'
        train(config, name, dset_dir, save_model_dir, seg)
    # elif status == 'test':
    #     data = load_data_setting(dset_dir)
    #     data.generate_instance(dev_file, 'dev')
    #     load_model_decode(model_dir, data, 'dev', gpu, seg)
    #     data.generate_instance(test_file, 'test')
    #     load_model_decode(model_dir, data, 'test', gpu, seg)
    # elif status == 'decode':
    #     data = load_data_setting(dset_dir)
    #     data.generate_instance(raw_file, 'raw')
    #     decode_results = load_model_decode(model_dir, data, 'raw', gpu, seg)
    #     data.write_decoded_results(output_file, decode_results, 'raw')
    else:
        print('Invalid argument! Please use valid arguments! (train/test/decode)')
