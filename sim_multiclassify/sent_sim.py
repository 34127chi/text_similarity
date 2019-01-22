#! -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from tqdm import tqdm
import pdb
from gensim.models import Word2Vec as word2vec

from imblearn.over_sampling import RandomOverSampler 


maxlen = 32
batch_size = 64 
min_count = 2 
word_size = 128
#epochs = 1000 # amsoftmax需要25个epoch，其它需要20个epoch


train_data = pd.read_csv('./data/train.txt', encoding='utf-8', header=None, delimiter='\t')

#pdb.set_trace()

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


train_data[1] = train_data[1].apply(strQ2B)
train_data[1] = train_data[1].str.lower()

chars = {}
for s in tqdm(iter(train_data[1])):#迭代 字的个数
    for c in s:
        if c not in chars:
            chars[c] = 0
        chars[c] += 1

# 0: padding标记
# 1: unk标记
chars = {i:j for i,j in chars.items() if j >= min_count}#词频限制 得到词典
id2char = {i+2:j for i,j in enumerate(chars)}#id to word
id2char[0] = '<PAD>'
id2char[1] = '<UNK>'
char2id = {j:i for i,j in id2char.items()}#word to id


#unknow word, pad word.
def string2id(s):
    _ = [char2id.get(i, 1) for i in s[:maxlen]]#最大长度maxlen  如果不存在的话unknow word 
    _ = _ + [0] * (maxlen - len(_))#不够的填充
    return _

cates = set([])
for s in tqdm(iter(train_data[0])):
    if s in cates:
        continue
    else:
        cates.add(s)

id2cate = {i:j for i,j in enumerate(cates)}
cate2id = {j:i for i,j in id2cate.items()}
#pdb.set_trace()

def category2id(s):
    return cate2id[s]

train_data[2] = train_data[0].apply(category2id)
train_data[3] = train_data[1].apply(string2id)#训练数据
#pdb.set_trace()

train_data = train_data.sample(frac=1)
x_train = np.array(list(train_data[3]))
y_train = np.array(list(train_data[2])).reshape((-1,1))
#ros = RandomOverSampler(random_state=42)
#x_train, y_train = ros.fit_sample(x_train, y_train)
#pdb.set_trace()

from keras.models import Model
from keras.layers import *
from keras.constraints import unit_norm
from margin_softmax import *

#load pretrained embedding
word2vec_model_path = './data/word2vec_model_char_10000'
word2vec_model = word2vec.load(word2vec_model_path)

word2vec_dict = {}
for idx, word in enumerate(word2vec_model.wv.vocab):
    word2vec_dict[word] = word2vec_model.wv[word]

word_embedding_2dlist = [[]] * (len(id2char)) 

word_embedding_2dlist[0] = np.zeros(word_size)
bound = np.sqrt(6.0) / np.sqrt(len(id2char))  # bound for random variables.
count_exist = 0
count_not_exist = 0

for i in range(1, len(id2char)):  # loop each word
    word = id2char[i]  # get a word
    embedding = None
    try:
        embedding = word2vec_dict[word][:word_size]  # try to get vector:it is an array.
    except Exception:
        embedding = None
    if embedding is not None:  # the 'word' exist a embedding
        word_embedding_2dlist[i] = embedding
        count_exist = count_exist + 1  # assign array to this word.
    else:  # no embedding for this word
        word_embedding_2dlist[i] = np.random.uniform(-bound, bound, word_size);
        count_not_exist = count_not_exist + 1  # init a random value for the word.
word_embedding_final = np.array(word_embedding_2dlist)  # covert to 2d array.
print("word. exists embedding:", count_exist, " ;word not exist embedding:", count_not_exist)



x_in = Input(shape=(maxlen,))
#x_embedded = Embedding(len(chars)+2,
#                       word_size)(x_in)#embedding的大小
x_embedded = Embedding(len(id2char), word_size, weights=[word_embedding_final])(x_in)

pdb.set_trace()
x = Bidirectional(GRU(word_size))(x_embedded)
#pdb.set_trace()
#x = GRU(word_size)(x_embedded)
x = Lambda(lambda x: K.l2_normalize(x, 1))(x)
#pdb.set_trace()

pred = Dense(len(cate2id),
             use_bias=False,
             kernel_constraint=unit_norm())(x)

#pdb.set_trace()
encoder = Model(x_in, x) # 最终的目的是要得到一个编码器  获得对应的编码器
model = Model(x_in, pred) # 用分类问题做训练 分类训练使用

model.compile(loss=sparse_amsoftmax_loss,
              optimizer='adam',
              metrics=['sparse_categorical_accuracy'])

def overfit(dev_acc):
    n = len(dev_acc)
    if n < 5:
        return False
    for i in range(n-4, n):
        if dev_acc[i] > dev_acc[i-1]:
            return False
    return True

def test(data_file, encoder):
    """评测函数
    如果按相似度排序后的前n个句子中出现了输入句子的同义句，那么topn的命中数就加1
    """
    data_file = data_file + '.preprocessed'
    test_data = pd.read_csv(data_file, encoding='utf-8', header=None, delimiter='\t')
    test_data[1] = test_data[1].apply(strQ2B)
    test_data[1] = test_data[1].str.lower()
    test_data[2] = test_data[1].apply(string2id)#训练与验证数据

    num = len(test_data)

    print ('测试总数：%d' % num)

    test_vec = encoder.predict(np.array(list(test_data[2])),
                                verbose=True,
                                batch_size=1000)#编码
    total = 0.
    top1_right = 0.
    top5_right = 0.
    top10_right = 0.
    with open(data_file + '.knowledge', 'w') as file_w_knowledge, open(data_file + '.query', 'w') as file_w_query:
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

dev_data_file = './data/dev.txt'
dev_acc = []
best_acc = 0
while True:
    model.fit(x_train, y_train, batch_size=batch_size)
    top1_right, _, _ = test(dev_data_file, encoder)
    if top1_right > best_acc:
        best_acc = top1_right
        model.save_weights('sent_sim_amsoftmax_bi.weights')
        encoder.save_weights('sent_sim_encoder_bi.weights')
        best_encoder = encoder

    dev_acc.append(top1_right)
    print("\nRecently accuracy:")
    print(dev_acc[-10:])
    if overfit(dev_acc):
        print('Overfit!!')
        break
    print("")

print("best_acc:" + str(best_acc))

train_data_file = './data/train.txt'
print(test(train_data_file, best_encoder))
dev_data_file = './data/dev.txt'
print(test(dev_data_file, best_encoder))

