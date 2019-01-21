#!/usr/bin/python
#-*- coding:utf-8 -*-
############################
#File Name: run_cnn_bigruatt.py
#Created Time: 2018-08-10 17:28:39
############################
from __future__ import print_function
import os
import sys
import time
from datetime import timedelta

import numpy as np
import tensorflow as tf
from sklearn import metrics
from tqdm import tqdm

import skopt
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args

from cnn_bigruatt_model import CnnBiGruAttConfig, CnnBiGruAttModel

import pandas as pd
import pdb

base_dir = 'data/'
train_dir = os.path.join(base_dir, 'train.txt')
val_dir = os.path.join(base_dir, 'val.txt')
test_dir = os.path.join(base_dir, 'test.txt')
vocab_dir = os.path.join(base_dir, 'vocab.txt')

save_dir = 'model/cnn_bigruatt'
encoder_save_path = os.path.join(save_dir, 'encoder.weight')  # 最佳验证结果保存路径
model_save_path = os.path.join(save_dir, 'model.weight')  # 最佳验证结果保存路径

global_encoder_save_path = os.path.join(save_dir, 'global.encoder.weight')  # 最佳验证结果保存路径
global_model_save_path = os.path.join(save_dir, 'global.model.weight')  # 最佳验证结果保存路径

best_accuracy = 0.0
def strQ2B(ustring):#全角转半角
    """全角转半角"""
    rstring = ''
    for uchar in ustring:
        inside_code=ord(uchar)
        if inside_code == 12288: # 全角空格直接转换
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374): # 全角字符（除空格）根据关系转化
            inside_code -= 65248
        rstring += chr(inside_code)
    return rstring

train_data = pd.read_csv(train_dir, encoding='utf-8', header=None, delimiter='\t')
train_data[0] = train_data[0].apply(strQ2B)
train_data[0] = train_data[0].str.lower()
train_data[1] = train_data[1].apply(strQ2B)
train_data[1] = train_data[1].str.lower()

def build_vocab(vocab_dir, min_count = 2, vocab_size = 5000):
    chars = {}
    for s in tqdm(iter(train_data[1])):#迭代 字的个数
        for c in s:
            if c not in chars:
                chars[c] = 0
            chars[c] += 1

    # 0: padding标记
    # 1: unk标记
    chars = {i:j for i,j in sorted(chars.items(), key=lambda x:x[1]) if j >= min_count}#词频限制 得到词典
    id2char = {i+2:j for i,j in enumerate(chars) if (i+2) <= vocab_size}#id to word
    id2char[0] = '<PAD>'
    id2char[1] = '<UNK>'

    char2id = {j:i for i,j in id2char.items()}#word to id

    open(vocab_dir, 'w').write('\n'.join(char2id.keys()) + '\n')

    return id2char, char2id

def read_category():
    cates = set([])
    for s in tqdm(iter(train_data[0])):
        if s in cates:
            continue
        else:
            cates.add(s)

    id2cate = {i:j for i,j in enumerate(cates)}
    cate2id = {j:i for i,j in id2cate.items()}
    #pdb.set_trace()

    return cates, id2cate, cate2id

def read_vocab(vocab_dir):
    """读取词汇表"""
    with open(vocab_dir, 'r') as fp:
        # 如果是py2 则每个值都转化为unicode
        words = [_.strip() for _ in fp.readlines()]
    word2id = dict(zip(words, range(len(words))))
    id2word = dict(zip(range(len(words)), words))
    return words, word2id, id2word

def string2id(s):
    _ = [char2id.get(i, 1) for i in s[:config.seq_len]]#最大长度maxlen  如果不存在的话unknow word 
    _ = _ + [0] * (config.seq_len - len(_))#不够的填充
    return _


def category2id(cate):
    #pdb.set_trace()
    return cate2id[cate]

def overfit(dev_acc):
    n = len(dev_acc)
    if n < 5:
        return False
    for i in range(n-4, n):
        if dev_acc[i] > dev_acc[i-1]:
            return False
    return True

def train():
    print("Loading training and validation data...")
    # 载入训练集与验证集
    global train_data
    train_data[2] = train_data[0].apply(category2id)
    train_data[3] = train_data[1].apply(string2id)#训练数据
    train_data = train_data.sample(frac=1)
    x_train = np.array(list(train_data[3]))
    y_train = np.array(list(train_data[2])).reshape((-1,1))
    val_acc = []
    best_acc = 0
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    while True:
        model.model.fit(x_train, y_train, batch_size=config.batch_size)
        top1_right, _, _ = test(val_dir, model.encoder)
        if top1_right > best_acc:
            best_acc = top1_right
            model.model.save_weights(model_save_path)
            model.encoder.save_weights(encoder_save_path)

        val_acc.append(top1_right)
        print("\nRecently accuracy:")
        print(val_acc[-10:])
        if overfit(val_acc):
            print('Overfit!!')
            break
        print("")

    print('best acc:' + str(best_acc))
    pass

def test(test_dir, encoder):
    """评测函数
    如果按相似度排序后的前n个句子中出现了输入句子的同义句，那么topn的命中数就加1
    """
    test_data = pd.read_csv(test_dir, encoding='utf-8', header=None, delimiter='\t')
    test_data[1] = test_data[1].apply(strQ2B)
    test_data[1] = test_data[1].str.lower()
    test_data[2] = test_data[1].apply(string2id)#训练与验证数据

    num = len(test_data)

    print ('测试总数：%d' % num)

    test_vec = encoder.predict(np.array(list(test_data[2])),
                                verbose=True,
                                batch_size=config.batch_size)#编码
    total = 0.
    top1_right = 0.
    top5_right = 0.
    top10_right = 0.
    with open(test_dir + '.knowledge', 'w') as file_w_knowledge, open(test_dir + '.query', 'w') as file_w_query:
        for k in tqdm(iter(range(num))):
            total += 1
            max_sim_indexs = np.dot(test_vec, test_vec[k]).argsort()[-11:][::-1]#每个句子找相近的11个句子并排序
            #pdb.set_trace()
            max_sim_knowledges = [test_data.iloc[i][0] for i in max_sim_indexs]#知识点
            max_sim_queries = [test_data.iloc[i][1] for i in max_sim_indexs]#问题
            input_knowledge = max_sim_knowledges[0]#最相似的 也就是它本身
            input_query = max_sim_queries[0] 
            file_w_knowledge.write(input_knowledge + '\t' + test_data[1][k] + '\t' + max_sim_knowledges[1] + '\t' + max_sim_knowledges[2] + '\t' + max_sim_knowledges[3]  + '\n')
            file_w_query.write(input_knowledge + '\t' + test_data[1][k] + '\t' + max_sim_queries[1] + '\t' + max_sim_queries[2] + '\t' + max_sim_queries[3]  + '\n')
            max_sim_knowledges = max_sim_knowledges[1:]
            if input_knowledge == max_sim_knowledges[0]:
                top1_right += 1
                top5_right += 1
                top10_right += 1
            elif input_knowledge in max_sim_knowledges[:5]:
                top5_right += 1
                top10_right += 1
            elif input_knowledge in max_sim_knowledges[:10]:
                top10_right += 1
    return top1_right/total, top5_right/total, top10_right/total

def tune_hyper():
    hidden_size = Integer(low=64, high=512, name='hidden_size') 
    low_kernel_size = Integer(low=3, high=5, name='low_kernel_size')
    high_kernel_size = Integer(low=5, high=7, name='high_kernel_size')
    num_filters = Integer(low=64, high=512, name='num_filters')
    activation_func = Categorical(categories=['relu', 'sigmoid', 'tanh'], name='activation_func')
    embedding_dim = Integer(low=64, high=256, name='embedding_dim')

    dimensions = [hidden_size, low_kernel_size, high_kernel_size, num_filters, activation_func, embedding_dim]
    default_parameters = [128 , 3, 5, 256, 'relu', 128]

    global train_data
    train_data[2] = train_data[0].apply(category2id)
    train_data[3] = train_data[1].apply(string2id)#训练数据
    train_data = train_data.sample(frac=1)
    x_train = np.array(list(train_data[3]))
    y_train = np.array(list(train_data[2])).reshape((-1,1))

    @use_named_args(dimensions=dimensions)
    def fitness(hidden_size, low_kernel_size, high_kernel_size, num_filters, activation_func, embedding_dim):
        # Print the hyper-parameters.
        print('hidden_size:', hidden_size)
        print('low_kernel_size:', low_kernel_size)
        print('high_kernel_size:', high_kernel_size)
        print('num_filters:', num_filters)
        print('activation_func:', activation_func)
        print('embedding_dim:', embedding_dim)

        config.hidden_size = int(hidden_size)
        config.low_kernel_size = low_kernel_size
        config.high_kernel_size = high_kernel_size
        config.num_filters = num_filters
        config.activation_func = activation_func
        config.embedding_dim = embedding_dim

        model = CnnBiGruAttModel(config)

        val_acc = []
        best_acc = 0
        while True:
            model.model.fit(x_train, y_train, batch_size=config.batch_size)
            top1_right, _, _ = test(val_dir, model.encoder)
            if top1_right > best_acc:
                best_acc = top1_right
                model.model.save_weights(model_save_path)
                model.encoder.save_weights(encoder_save_path)

            val_acc.append(top1_right)
            print("\nRecently accuracy:")
            print(val_acc[-10:])
            if overfit(val_acc):
                print('Overfit!!')
                break
            print("")

        print('Accuracy:{0:.2%}'.format(best_acc))

        global best_accuracy
        if best_acc > best_accuracy:
            model.model.save_weights(global_model_save_path)
            model.encoder.save_weights(global_encoder_save_path)
            best_accuracy = best_acc

        del model

        return -best_acc

    search_result = gp_minimize(func=fitness,
                                dimensions=dimensions,
                                acq_func='EI', # Expected Improvement.
                                n_calls=100,
                                x0=default_parameters)
    space = search_result.space
    print('sorted validation result:')
    print(sorted(zip(search_result.func_vals, search_result.x_iters)))
    print('best parameter:')
    print(space.point_to_dict(search_result.x))
    pass


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in ['train', 'test', 'tune_hyper']:
        raise ValueError("""usage: python run_cnn_bigruatt.py [train / test / tune_hyper]""")

    print('Configuring CnnBiGruAtt model...')
    config = CnnBiGruAttConfig()
    if not os.path.exists(vocab_dir):  # 如果不存在词汇表，重建
        build_vocab(vocab_dir, min_count = config.min_count, vocab_size = config.vocab_size)
    categories, id2cate, cate2id = read_category()
    #pdb.set_trace()
    chars, char2id, id2char = read_vocab(vocab_dir)
    config.vocab_size = len(chars)#避免词的数量达不到设置的
    config.id2char = id2char
    config.num_classes = len(categories)
    model = CnnBiGruAttModel(config)

    if sys.argv[1] == 'train':
        train()
    elif sys.argv[1] == 'test':
        if not os.path.exists(encoder_save_path):
            raise Exception(encoder_save_path + ' not found.')
        model.encoder.load_weights(encoder_save_path)
        print(test(test_dir, model.encoder))
    else:
        tune_hyper()
