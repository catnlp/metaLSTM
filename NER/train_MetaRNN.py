# encoding:utf-8
'''
@Author: catnlp
@Email: wk_nlp@163.com
@Time: 2018/5/2 14:14
'''
from NER.utils.metric import get_ner_fmeasure
from NER.Module.ner import NER
from NER.utils.config import Config

import time
import sys
import argparse
import random
import copy
import torch
import gc
import pickle
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import numpy as np
import visdom

seed_num = 100
random.seed(seed_num)
torch.manual_seed(seed_num)
np.random.seed(seed_num)

def data_initialization(data, train_file, dev_file, test_file):
    data.build_alphabet(train_file)
    data.build_alphabet(dev_file)
    data.build_alphabet(test_file)
    data.fix_alphabet()

def predict_check(pred_variable, gold_variable, mask_variable):
    pred = pred_variable.cpu().data.numpy()
    gold = gold_variable.cpu().data.numpy()
    mask = mask_variable.cpu().data.numpy()
    overlaped = (pred == gold)
    right_token = np.sum(overlaped * mask)
    total_token = mask.sum()
    return right_token, total_token

def recover_label(pred_variable, gold_variable, mask_variable, label_alphabet, word_recover):
    pred_variable = pred_variable[word_recover]
    gold_variable = gold_variable[word_recover]
    mask_variable = mask_variable[word_recover]
    batch_size = gold_variable.size(0)
    seq_len = gold_variable.size(1)
    mask = mask_variable.cpu().data.numpy()
    pred_tag = pred_variable.cpu().data.numpy()
    gold_tag = gold_variable.cpu().data.numpy()
    # batch_size = mask.shape[0]
    pred_label = []
    gold_label = []
    for idx in range(batch_size):
        pred = [label_alphabet.get_instance(pred_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
        gold = [label_alphabet.get_instance(gold_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
        assert(len(pred) == len(gold))
        pred_label.append(pred)
        gold_label.append(gold)

    return pred_label, gold_label

def save_data_setting(data, save_file):
    new_data = copy.deepcopy(data)
    new_data.train_texts = []
    new_data.dev_texts = []
    new_data.test_texts = []
    new_data.raw_texts = []

    new_data.train_ids = []
    new_data.dev_ids = []
    new_data.test_ids = []
    new_data.raw_ids = []

    with open(save_file, 'wb+') as fp:
        pickle.dump(new_data, fp)
    print('Data setting saved to file: ', save_file)

def load_data_setting(save_file):
    with open(save_file, 'r') as fp:
        data = pickle.load(fp)
    print('Data setting loaded from file: ', save_file)
    data.show_data_summary()
    return data

def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr / (1+decay_rate*epoch)
    print('Learning rate is setted as : ', lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def evaluate(data, model, name):
    if name == 'train':
        instances = data.train_ids
    elif name == 'dev':
        instances = data.dev_ids
    elif name == 'test':
        instances = data.test_ids
    elif name == 'raw':
        instances = data.raw_ids
    else:
        print('Error: wrong evaluate name, ', name)

    right_token = 0
    whole_token = 0
    pred_results = []
    gold_results = []

    model.eval()
    batch_size = data.batch_size
    start_time = time.time()
    train_num = len(instances)
    total_batch = train_num // batch_size + 1
    for batch_id in range(total_batch):
        start = batch_id * batch_size
        end = (batch_id+1) * batch_size
        if end > train_num:
            end = train_num
        instance = instances[start: end]
        if not instance:
            continue
        batch_word, batch_wordlen, batch_wordrecover, batch_label, mask = batchify_with_label(instance, data.gpu, True)
        tag_seq = model(batch_word, batch_wordlen, mask)
        pred_label, gold_label = recover_label(tag_seq, batch_label, mask, data.label_alphabet, batch_wordrecover)
        pred_results += pred_label
        gold_results += gold_label
    decode_time = time.time() - start_time
    speed = len(instances) / decode_time
    acc, p, r, f = get_ner_fmeasure(gold_results, pred_results, data.tagScheme)
    return speed, acc, p, r, f, pred_results

def batchify_with_label(input_batch_list, gpu, volatile_flag=False):
    batch_size = len(input_batch_list)
    words = [sent[0] for sent in input_batch_list]
    labels = [sent[-1] for sent in input_batch_list]
    word_seq_lengths = torch.LongTensor(list(map(len, words)))
    max_seq_len = word_seq_lengths.max()
    word_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len)), volatile=volatile_flag).long()
    label_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len)), volatile=volatile_flag).long()
    mask = autograd.Variable(torch.zeros((batch_size, max_seq_len)), volatile=volatile_flag).byte()
    for idx, (seq, label,seqlen) in enumerate(zip(words, labels, word_seq_lengths)):
        word_seq_tensor[idx, : seqlen] = torch.LongTensor(seq)
        label_seq_tensor[idx, : seqlen] = torch.LongTensor(label)
        mask[idx, : seqlen] = torch.Tensor([1]*seqlen)
    word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending=True)
    word_seq_tensor = word_seq_tensor[word_perm_idx]
    label_seq_tensor = label_seq_tensor[word_perm_idx]
    mask = mask[word_perm_idx]

    _, word_seq_recover = word_perm_idx.sort(0, descending=False)
    if gpu:
        word_seq_tensor = word_seq_tensor.cuda()
        word_seq_lengths = word_seq_lengths.cuda()
        word_seq_recover = word_seq_recover.cuda()
        label_seq_tensor = label_seq_tensor.cuda()
        mask = mask.cuda()
    return word_seq_tensor, word_seq_lengths, word_seq_recover, label_seq_tensor, mask

def train(data, name, save_dset, save_model_dir, seg=True):
    print('---Training model---')
    data.show_data_summary()
    save_data_name = save_dset
    save_data_setting(data, save_data_name)
    model = NER(data)
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=data.lr, momentum=data.momentum)
    best_test = -1
    epoch = data.iteration
    vis = visdom.Visdom()
    losses = []
    all_F = [[0., 0., 0.]]
    for idx in range(epoch):
        epoch_start = time.time()
        tmp_start = epoch_start
        print('Epoch: %s/%s' % (idx, epoch))
        optimizer = lr_decay(optimizer, idx, data.lr_decay, data.lr)
        instance_count = 0
        sample_id = 0
        sample_loss = 0
        total_loss = 0
        right_token = 0
        whole_token = 0
        random.shuffle(data.train_ids)
        model.train()
        model.zero_grad()
        batch_size = data.batch_size
        train_num = len(data.train_ids)
        total_batch = train_num // batch_size + 1
        for batch_id in range(total_batch):
            start = batch_id * batch_size
            end = (batch_id+1) * batch_size
            if end > train_num:
                end = train_num
            instance = data.train_ids[start: end]
            if not instance:
                continue
            batch_word, batch_wordlen, batch_wordrecover, batch_label, mask = batchify_with_label(instance, data.gpu)
            instance_count += 1
            loss, tag_seq = model.neg_log_likelihood_loss(batch_word, batch_wordlen, batch_label)
            right, whole = predict_check(tag_seq, batch_label, mask)
            right_token += right
            whole_token += whole
            sample_loss += loss.data[0]
            total_loss += loss.data[0]

            if end % 500 == 0:
                tmp_time = time.time()
                tmp_cost = tmp_time - tmp_start
                tmp_start = tmp_time
                print('\tInstance: %s; Time: %.2fs; loss: %.4f; acc: %s/%s=%.4f'
                      % ( end, tmp_cost, sample_loss, right_token, whole_token, (right_token+0.0) / whole_token))
                sys.stdout.flush()
                losses.append(sample_loss / 500.0)
                Lwin = 'Loss of ' + name
                vis.line(np.array(losses), X=np.array([i for i in range(len(losses))]),
                         win=Lwin, opts={'title': Lwin, 'legend': ['loss']})
                sample_loss = 0
            loss.backward()
            if data.clip:
                torch.nn.utils.clip_grad_norm(model.parameters(), 10.0)
            optimizer.step()
            model.zero_grad()
        tmp_time = time.time()
        tmp_cost = tmp_time - tmp_start
        print('\tInstance: %s; Time: %.2fs; loss: %.4f; acc: %s/%s=%.4f'
              % (end, tmp_cost, sample_loss, right_token, whole_token, (right_token+0.0) / whole_token))
        epoch_finish = time.time()
        epoch_cost = epoch_finish - epoch_start
        print('Epoch: %s training finished. Time: %.2fs, speed: %.2ft/s, total_loss: %s'
              % (idx, epoch_cost, train_num/epoch_cost, total_loss))
        speed, acc, p, r, f_train, _ = evaluate(data, model, 'train')
        speed, acc, p, r, f_dev, _ = evaluate(data, model, 'dev')
        speed, acc, p, r, f_test, _ = evaluate(data, model, 'test')
        test_finish = time.time()
        test_cost = test_finish - epoch_finish

        if seg:
            current_score = f_test
            print('Test: time: %.2fs, speed: %.2ft/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f'
                  % (test_cost, speed, acc, p, r, f_test))
        else:
            current_score = acc
            print('Test: time: %.2fs, speed: %.2ft/s; acc: %.4f'
                  % (test_cost, speed, acc))
        if current_score > best_test:
            if seg:
                print('Exceed previous best f score: ', best_test)
            else:
                print('Exceed previous best acc score: ', best_test)
            model_name = save_model_dir + '/' + name + '_model_' + str(idx)
            torch.save(model.state_dict(), model_name)
            best_test = current_score
            with open(save_model_dir + '/' + name + '_eval_' + str(idx) + '.txt', 'w') as f:
                if seg:
                    f.write('acc: %.4f, p: %.4f, r: %.4f, f: %.4f' % (acc, p, r, best_test))
                    f.write('acc: %.4f, p: %.4f' % (acc, p))
                else:
                    f.write('acc: %.4f' % acc)

        if seg:
            print('Current best f score: ', best_test)
        else:
            print('Current best acc score: ', best_test)

        all_F.append([f_train*100.0, f_dev*100.0, f_test*100.0])
        Fwin = 'F-score of ' + name + ' {train, dev, test}'
        vis.line(np.array(all_F), X=np.array([i for i in range(len(all_F))]),
                 win=Fwin, opts={'title': Fwin, 'legend': ['train', 'dev', 'test']})
        gc.collect()

def load_model_decode(model_dir, data, name, gpu, seg=True):
    data.gpu = gpu
    print('Load Model from file: ', model_dir)
    model = NER(data)
    model.load_state_dict(torch.load(model_dir))
    print('Decode % data ... ' % name)
    start_time = time.time()
    speed, acc, p, r, f, pred_results = evaluate(data, model, name)
    end_time = time.time()
    time_cost = end_time - start_time
    if seg:
        print('%s: time: %.2f, speed:%.2ft/s; acc: %.4f, r: %.4f, f: %.4f'
              % (name, time_cost, speed, acc, p, r, f))
    else:
        print('%s: time: %.2f, speed: %.2ft/s; acc: %.4f' % (name, time_cost, speed, acc))
    return pred_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tuning with NER')
    parser.add_argument('--wordemb', help='Embedding for words', default='glove')
    parser.add_argument('--status', choices=['train', 'test', 'decode'], help='update algorithm', default='train')
    parser.add_argument('--savemodel', default='../models/conll2003/MetaRNN') # catnlp
    parser.add_argument('--savedset', help='Dir of saved data setting', default='../models/conll2003/MetaRNN.dset') # catnlp
    parser.add_argument('--train', default='../data/conll2003/train.bmes')
    parser.add_argument('--dev', default='../data/conll2003/dev.bmes')
    parser.add_argument('--test', default='../data/conll2003/test.bmes')
    parser.add_argument('--gpu', default='False')
    parser.add_argument('--seg', default='True')
    parser.add_argument('--extendalphabet', default='True')
    parser.add_argument('--raw')
    parser.add_argument('--loadmodel')
    parser.add_argument('--output')
    args = parser.parse_args()

    train_file = args.train
    dev_file = args.dev
    test_file = args.test
    raw_file = args.raw
    model_dir = args.loadmodel
    dset_dir = args.savedset
    output_file = args.output
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
    print('Train file: ', train_file)
    print('Dev file: ', dev_file)
    print('Test file: ', test_file)
    print('Raw file: ', raw_file)
    if status == 'train':
        print('Model saved to: ', save_model_dir)
    sys.stdout.flush()

    if status == 'train':
        emb = args.wordemb.lower()
        print('Word Embedding: ', emb)
        if emb == 'glove':
            emb_file = '../data/embedding/glove.6B.100d.txt'
        else:
            emb_file = None
        name = 'MetaRNN'  # catnlp
        config = Config()
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

        train(config, name, dset_dir, save_model_dir, seg)
    elif status == 'test':
        data = load_data_setting(dset_dir)
        data.generate_instance(dev_file, 'dev')
        load_model_decode(model_dir, data, 'dev', gpu, seg)
        data.generate_instance(test_file, 'test')
        load_model_decode(model_dir, data, 'test', gpu, seg)
    elif status == 'decode':
        data = load_data_setting(dset_dir)
        data.generate_instance(raw_file, 'raw')
        decode_results = load_model_decode(model_dir, data, 'raw', gpu, seg)
        data.write_decoded_results(output_file, decode_results, 'raw')
    else:
        print('Invalid argument! Please use valid arguments! (train/test/decode)')
