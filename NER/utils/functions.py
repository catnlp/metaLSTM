# encoding:utf-8
'''
@Author: catnlp
@Email: wk_nlp@163.com
@Time: 2018/5/2 21:04
'''
import numpy as np

def normalize_word(word):
    new_word = ''
    for char in word:
        if char.isdigit():
            new_word += '0'
        else:
            new_word += char
    return new_word

def read_instance(input_file, word_alphabet, char_alphabet, label_alphabet, number_normalized, max_sent_length, char_padding_size=-1, char_padding_symbol='</pad>'):
    in_lines = open(input_file, 'r').readlines()
    instance_texts = []
    instance_ids = []
    words = []
    chars = []
    labels = []
    word_ids = []
    char_ids = []
    labels_ids = []
    for line in in_lines:
        if len(line) > 2:
            pairs = line.strip().split()
            word = pairs[0] # catnlp
            if number_normalized:
                word = normalize_word(word)
            label = pairs[-1]
            words.append(word)
            labels.append(label)
            word_ids.append(word_alphabet.get_index(word))
            labels_ids.append(label_alphabet.get_index(label))
            char_list = []
            char_id = []
            for char in word:
                char_list.append(char)
            if char_padding_size > 0:
                char_number = len(char_list)
                if char_number < char_padding_size:
                    char_list = char_list + [char_padding_symbol] * (char_padding_size - char_number)
                assert (len(char_list) == char_padding_size)
            for char in char_list:
                char_id.append(char_alphabet.get_index(char))
            chars.append(char_list)
            char_ids.append(char_id)
        else:
            if (max_sent_length < 0) or (len(words) < max_sent_length):
                instance_texts.append([words, chars, labels])
                instance_ids.append([word_ids, char_ids, labels_ids])

            words = []
            chars = []
            labels = []
            word_ids = []
            char_ids = []
            labels_ids = []
    return instance_texts, instance_ids

def build_pretrain_embedding(embedding_path, word_alphabet, embed_dim=100, norm=True):
    embed_dict = dict()
    if embedding_path != None:
        embed_dict, embed_dim = load_pretrain_emb(embedding_path)
    alphabet_size = word_alphabet.size()
    scale = np.sqrt(3.0 / embed_dim)
    pretrain_emb = np.empty([word_alphabet.size(), embed_dim])
    perfect_match = 0
    case_match = 0
    not_match = 0
    for word, index in word_alphabet.iteritems():
        if word in embed_dict:
            if norm:
                pretrain_emb[index, :] = norm2one(embed_dict[word])
            else:
                pretrain_emb[index, :] = embed_dict[word]
            perfect_match += 1
        elif word.lower() in embed_dict:
            if norm:
                pretrain_emb[index, :] = norm2one(embed_dict[word.lower()])
            else:
                pretrain_emb[index, :] = embed_dict[word.lower()]
            case_match += 1
        else:
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embed_dim])
            not_match += 1
    pretrain_size = len(embed_dict)
    print('Embedding:\n\tpretrain word:%s, perfect match:%s, case_match:%s, oov:%s, oov%%:%s'
          % (pretrain_size, perfect_match, case_match, not_match, (not_match+0.)/alphabet_size))
    return pretrain_emb, embed_dim

def norm2one(vec):
    root_sum_square = np.sqrt(np.sum(np.square(vec)))
    return vec/root_sum_square

def load_pretrain_emb(embedding_path):
    embed_dim = -1
    embed_dict = dict()
    with open(embedding_path, 'r') as file:
        for line in file:
            line = line.strip()
            if len(line) == 0:
                continue
            tokens = line.split()
            if embed_dim < 0:
                embed_dim = len(tokens) - 1
            # else:
            #     if(len(tokens) != embed_dim + 1):
            #         print(tokens)
            #     assert(embed_dim + 1 == len(tokens))
            embed = np.empty([1, embed_dim])
            embed[:] = tokens[len(tokens) - embed_dim: ]
            name = tokens[0]
            size = len(tokens) - embed_dim - 1
            for i in range(size):
                name += ' ' + tokens[i+1]
            embed_dict[name] = embed # catnlp
    return embed_dict, embed_dim
