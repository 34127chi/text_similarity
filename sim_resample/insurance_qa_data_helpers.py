#encoding=utf-8
import pdb
import numpy as np
import random
import tensorflow as tf
import jieba
import pdb

UNKNOWN_WORD="<UNKNOWN>"
PAD_WORD="<PAD/>"
BOW="<"
EOW=">"
CHAR_GRAM_TOKEN="ngm_"

empty_vector = []
for i in range(0, 100):
    empty_vector.append(float(0.0))
onevector = []
for i in range(0, 10):
    onevector.append(float(1))
zerovector = []
for i in range(0, 10):
    zerovector.append(float(0))

def build_vocab(file_name):
    #构建基于词的vocab
    #标签分词\t问句分词
    '''
    desc:根据样本文件构建字典
    input:
        filename:样本文件
    return:
        vocab:map 字典
    '''
    code = int(0)
    vocab = {}
    vocab[UNKNOWN_WORD] = code
    code += 1
    vocab[PAD_WORD] = code
    code += 1
    for line in open(file_name):
        tmp_list = line.strip().split('\t')
        if len(tmp_list) < 2:continue
        for i in range(len(tmp_list)):
            words = tmp_list[i].split(' ')
            for word in words:
                if not word in vocab:
                    vocab[word] = code
                    code += 1
    return vocab

def build_vocab_char(file_name):
    #构建基于字的vocab
    #标签分词\t问句分词
    '''
    desc:根据样本文件构建字典
    input:
        filename:样本文件名称
    return:
        vocab:map 基于字的字典
    '''
    code = int(0)
    vocab = {}
    vocab[UNKNOWN_WORD] = code
    code += 1
    vocab[PAD_WORD] = code
    code += 1
    for line in open(file_name):
        tmp_list = line.strip().split('\t')
        if len(tmp_list) < 2:continue
        for i in range(len(tmp_list)):
            words = tmp_list[i].split(' ')
            for word in words:
                uni_word = word.decode("utf-8")
                for uni_char in uni_word:
                    char = uni_char.encode("utf-8")
                    if not char in vocab:
                        vocab[char] = code
                        code += 1
    return vocab

def build_vocab_subword(file_name, max_wordlen=6):
    #构建基于subword的vocab
    #标签分词\t问句分词
    '''
    desc:根据样本文件构建字典库

    input:
        filename:样本文件名
        max_wordlen:subword构造时最大限制数
    return:
        vocab:基于subword的字典
        
    '''
    code = int(0)
    vocab = {}
    vocab[UNKNOWN_WORD] = code
    code += 1
    vocab[PAD_WORD] = code
    code += 1
    for line in open(file_name):
        tmp_list = line.strip().split('\t')
        if len(tmp_list) < 2:continue
        for i in range(len(tmp_list)):
            words = tmp_list[i].split(' ')
            for word in words:
                if not word in vocab:
                    vocab[word] = code
                    code += 1
                    subword_set = get_subword(word, max_wordlen)#构建subword
                    for subword in subword_set:
                        if subword in vocab or subword == "":continue
                        vocab[subword] = code
                        code += 1
    return vocab

def get_subword(word, max_wordlen=6):
    '''
    desc:根据单词构建subword
    input:
        word:单词
        max_wordlen:subword最大长度
    return:
        sub_word:
    '''
    #提取subword
    uni_word = (BOW+word+EOW).decode("utf-8")
    sub_word = set()
    for inx in xrange(len(uni_word)-1):
        if inx >= max_wordlen:break
        sw = CHAR_GRAM_TOKEN + uni_word[inx:inx+2].encode("utf-8")
        sub_word.add(sw)
    return sub_word

def rand_qa(qalist, target):
    #随机找一个与正例不同的反例
    index = random.randint(0, len(qalist) - 1)
    while qalist[index] == target:
        index = random.randint(0, len(qalist) - 1)
    return qalist[index]

def read_alist(file_name, word_vocab, max_seq_len, embedding_type):
    #读取true answer及其对应的Word id
    '''
    desc:读取答案以及encode答案

    input:
        filename:样本文件
        word_vocab:字典
        max_seq_len:最大长度
        embedding_type:词向量类型

    return:
        alist:答案 list [ans1, ans2...]
        avec_list:答案对应的编码结果 list=[ans1_vec, ans2_vec]  ans1_vec=[ans1_id,indics1]
    '''
    alist = []
    avec_list = []
    check_set = set([])
    for line in open(file_name):
        items = line.strip().split('\t')
        if items[0] in check_set:continue
        check_set.add(items[0])
        alist.append(items[0])
        avec_list.append(encode_sent(word_vocab,items[0], max_seq_len, 0, embedding_type))
    print('read_alist done ......')
    return alist,avec_list

def load_test(file_name, ans_list):
    #读取测试语料，标签\t问句
    #并准备负样本，目前是把所有其余标签都作为负样本
    #后面需要改进用触发词筛选
    '''
    desc:读取测试语料 准备负样本

    input:
        filename:样本文件
        ans_list:答案list
    return:
        testList:测试样本 list
    '''
    testList = []
    for line in open(file_name):
        tmp_list = line.strip().split("\t")
        tmp_str = str(1) + "\t" + tmp_list[0]+ "\t" + tmp_list[1] #tag\tlabel\tquery
        testList.append(tmp_str)
        for e in ans_list:
            if e == tmp_list[0]:continue
            tmp_str = str(0) + "\t" + e + "\t" + tmp_list[1]
            testList.append(tmp_str)
    return testList

def read_raw(file_name):
    '''
    desc:读取样本文件

    input:
        filename:文件名称
    return:
        raw:文件内容 list=[ele1,ele2...] ele1=[label,query]
        label_map:query与label、label与label的映射 map
    '''
    raw = []
    label_map = {}
    for line in open(file_name):
        items = line.strip().split('\t')
        if len(items) < 2:continue
        raw.append(items)
        label = items[0]
        query = items[1]
        label_map[query] = label
        label_map[label] = label
    return raw, label_map


def encode_sent(vocab, string, size, indic_start=0, embedding_type="word_embedding", max_wordlen=6):
    '''
    desc:根据词向量类型编码文本
    
    input:
        vocab:字典
        string:输入文本
        size:最大长度
        indic_start:subword时用到 文本起始位置
        embedding_type:词向量类型
        max_wordlen:subword时用到 subword最大的长度
    return:
        x:文本向量化的结果
        indics:词对应的位置
    '''
    words = string.split(' ')#词 list
    if embedding_type == "char_embedding":
        x = char_based_encode(vocab, words, size)
        return x, None
    elif embedding_type == "subword_embedding":
        x, indics = subword_based_encode(vocab, words, size, indic_start, max_wordlen)
        return x, indics
    else:
        x = word_based_encode(vocab, words, size)
        return x, None

def word_based_encode(vocab, word_list, size):
    #基于词的encode
    '''
    desc:基于词的编码

    input:
        vocab:字典
        word_list:文本的词
        size:文本的最大长度
    return:
        x:文本的编码结果
    '''
    x = []
    for i in range(0, size):
        if i >= len(word_list):
            x.append(vocab[PAD_WORD])
        elif word_list[i] in vocab:
            x.append(vocab[word_list[i]])
        else:
            x.append(vocab[UNKNOWN_WORD])
    return x

def char_based_encode(vocab, word_list, size):
    #基于字的encode
    '''
    desc:基于字的编码

    input:
        vocab:字典
        word_list:文本的词
        size:文本的最大长度
    return:
        x:文本的编码结果
    '''
    x = []
    string = "".join(word_list)
    uni_string = string.decode("utf-8")
    for i in range(0, size):
        if i >= len(uni_string):
            x.append(vocab[PAD_WORD])
        elif uni_string[i].encode("utf-8") in vocab:
            x.append(vocab[uni_string[i].encode("utf-8")])
        else:
            x.append(vocab[UNKNOWN_WORD])
    return x

def subword_based_encode(vocab, word_list, size, indic_start=0, max_wordlen=6):
    #基于subword的encode
    #indic_start:每个batch的indic_start都是从0开始
    #return：x 词id 1,2,3, indices：词坐标 (0,0),(1,0),(2,0),(2,1),(2,2)
    '''
    desc:基于subword的编码

    input:
        vocab:字典
        word_list:文本的词
        size:文本的最大长度
        indic_start:subword的起始坐标信息
        max_wordlen:subword的最大长度
    return:
        x:文本的编码结果
        indics:subword的坐标信息
    '''
    
    x = []
    indics = []
    for i in range(0, size):
        if i >= len(word_list):
            x.append(vocab[PAD_WORD])
            indics.append((i+indic_start,0))
        else:
            indic_start_y = 0
            if word_list[i] in vocab:
                x.append(vocab[word_list[i]])
                indics.append((i+indic_start,indic_start_y))
                indic_start_y += 1
            #add subword
            uni_word = (BOW+word_list[i]+EOW).decode("utf-8")
            for inx in xrange(len(uni_word)-1):
                if indic_start_y >= max_wordlen:break
                char = CHAR_GRAM_TOKEN + uni_word[inx:inx+2].encode("utf-8")
                if char in vocab:
                    x.append(vocab[char])
                else:
                    x.append(vocab[UNKNOWN_WORD])
                indics.append((i+indic_start,indic_start_y))
                indic_start_y += 1
    return x,indics

def change_indic_start(old_indic_start,new_indic_start,data_list):
    new_list = []
    for item in data_list:
        new_list.append((item[0]-old_indic_start+new_indic_start,item[1]))
    return new_list

def rand_sample(vocab, alist, raw, batch_size, max_seq_len, embedding_type="word_embedding"):
    '''
    desc:随机采样

    input:
        vocab:字典
        alist:答案列表
        raw:样本集合
        batch_size:批次大小
        max_seq_len:最大的长度
        embedding_type:词向量类型
    return:
        x_train_1:query
        x_train_2:答案正样本
        x_train_3:答案负样本
        indices_train_1:
        indices_train_2:
        indices_train_3:
    '''
    x_train_1 = []
    x_train_2 = []
    x_train_3 = []
    indices_train_1 = []
    indices_train_2 = []
    indices_train_3 = []
    for i in range(0, batch_size):
        items = raw[random.randint(0, len(raw) - 1)]
        nega = rand_qa(alist, items[0])
        indic_start = max_seq_len * i
        encode_res_1 = encode_sent(vocab, items[1], max_seq_len, indic_start, embedding_type)
        encode_res_2 = encode_sent(vocab, items[0], max_seq_len, indic_start, embedding_type)
        encode_res_3 = encode_sent(vocab, nega, max_seq_len, indic_start, embedding_type)
        if embedding_type == "subword_embedding":
            x_train_1 += encode_res_1[0]
            x_train_2 += encode_res_2[0]
            x_train_3 += encode_res_3[0]
            indices_train_1 += encode_res_1[1]
            indices_train_2 += encode_res_2[1]
            indices_train_3 += encode_res_3[1]
        else:
            x_train_1.append(encode_res_1[0])
            x_train_2.append(encode_res_2[0])
            x_train_3.append(encode_res_3[0])
    return np.array(x_train_1), np.array(x_train_2), np.array(x_train_3),\
        np.array(indices_train_1), np.array(indices_train_2), np.array(indices_train_3)

def semihard_sample(sess, model_obj, vocab, raw, batch_size, max_seq_len, avec_list, label_map, alist, macro_size, min_margin, max_margin, embedding_type="word_embedding", max_wordlen=6):
    '''
    desc:semihard采样

    input:
        sess:会话
        model_obj:模型对象
        vocab:字典
        bacth_size:批次大小
        max_seq_len: 文本最大长度
        avec_list:文本id化的结果
        label_map:query到label的映射关系
        alist:答案列表
        macro_size:
        min_margin:
        max_margin:
        embedding_type:词向量类型
        max_wordlen:subword的最大长度
    return:
        
    '''
    macro_x_train_1 = []
    macro_x_train_2 = []
    macro_indices_train_1 = []
    macro_indices_train_2 = []
    raw_x_1_res = []
    raw_x_2_res = []
    raw_query2label = []
    #准备Marco问句和正例
    for i in range(0, macro_size):
        items = raw[random.randint(0, len(raw) - 1)]
        indic_start = max_seq_len * i
        encode_res_1 = encode_sent(vocab, items[1], max_seq_len, indic_start, embedding_type)#query
        encode_res_2 = encode_sent(vocab, items[0], max_seq_len, indic_start, embedding_type)#ans
        raw_query2label.append(label_map[items[1]])#query 到 ans
        if embedding_type == "subword_embedding":
            macro_x_train_1 += encode_res_1[0]
            macro_x_train_2 += encode_res_2[0]
            macro_indices_train_1 += encode_res_1[1]
            macro_indices_train_2 += encode_res_2[1]
            raw_x_1_res.append((encode_res_1[0],encode_res_1[1], indic_start))
            raw_x_2_res.append((encode_res_2[0],encode_res_2[1], indic_start))
        else:
            macro_x_train_1.append(encode_res_1[0])
            macro_x_train_2.append(encode_res_2[0])
        
    ans_cnt = 0
    max_ans_cnt = 500#最大的采样数
    macro_x_train_3 = []
    macro_indices_train_3 = []
    if len(avec_list) <= max_ans_cnt:
        ans_inx_list = range(len(avec_list))#ans id list
    else:
        ans_inx_list = random.sample(range(len(avec_list)), max_ans_cnt)#随机采样多少

    for inx in ans_inx_list:
        if embedding_type == "subword_embedding":
            macro_x_train_3 += avec_list[inx][0]
            macro_indices_train_3 += change_indic_start(0, ans_cnt*max_seq_len, avec_list[inx][1])
        else:
            macro_x_train_3.append(avec_list[inx][0])
        ans_cnt += 1
    
    feed_dict = {
      model_obj.input_x_1: macro_x_train_1,
      model_obj.input_x_2: macro_x_train_2,
      model_obj.input_x_3: macro_x_train_3,
      model_obj.dropout_keep_prob: 1.0
    }
    if embedding_type == "subword_embedding":
        feed_dict[model_obj.indices_x_1] = macro_indices_train_1 
        feed_dict[model_obj.indices_x_2] = macro_indices_train_2
        feed_dict[model_obj.indices_x_3] = macro_indices_train_3
        feed_dict[model_obj.subword_shape_1] = np.array([macro_size*max_seq_len,max_wordlen]) 
        feed_dict[model_obj.subword_shape_2] = np.array([macro_size*max_seq_len,max_wordlen])
        feed_dict[model_obj.subword_shape_3] =np.array([macro_size*max_seq_len,max_wordlen])
    pos_sim, pair_sim = sess.run([model_obj.cos_12, model_obj.pair_sim_13], feed_dict)
    pdb.set_trace()
    
    x_train_1 = []
    x_train_2 = []
    x_train_3 = []
    indices_train_1 = []
    indices_train_2 = []
    indices_train_3 = []
    rand_list = []
    semi_cnt = 0
    for inx in range(len(pos_sim)):
        if semi_cnt >= batch_size:break
        pos_score = pos_sim[inx]
        pair_score = pair_sim[inx]
        sorted_list = np.argsort(pair_score)
        semi_iny = -1
        for tmp_i in xrange(len(sorted_list) - 1, -1, -1):#角标 从大到小排列
            iny = sorted_list[tmp_i]
            cur_margin = pos_score - pair_score[iny]
            tmp_iny = ans_inx_list[iny]
            tmp_label = label_map[alist[tmp_iny]]#相似的ans
            if tmp_label == raw_query2label[inx]:continue
            if cur_margin > min_margin and cur_margin < max_margin:
                semi_iny = iny    
                break
        if semi_iny == -1:
            rand_list.append(inx)
            continue
        semi_iny = ans_inx_list[semi_iny]#第几个ans
        items = avec_list[semi_iny]#ans对应的items
        if embedding_type == "subword_embedding":
            x_train_1 += raw_x_1_res[inx][0]
            x_train_2 += raw_x_2_res[inx][0]
            x_train_3 += items[0]
            indices_train_1 += change_indic_start(raw_x_1_res[inx][2],semi_cnt*max_seq_len,raw_x_1_res[inx][1]) 
            indices_train_2 += change_indic_start(raw_x_2_res[inx][2],semi_cnt*max_seq_len,raw_x_2_res[inx][1])
            indices_train_3 += change_indic_start(0,semi_cnt*max_seq_len,items[1])
        else:
            x_train_1.append(macro_x_train_1[inx])
            x_train_2.append(macro_x_train_2[inx])
            x_train_3.append(items[0])
        semi_cnt += 1
    #需要semihard采样的个数和需要随机采样的个数
    print "Semi-hard negative samples",semi_cnt,"rand negative samples",batch_size-semi_cnt

    rand_inx = 0
    while semi_cnt < batch_size and rand_inx < len(rand_list):
        inx = rand_list[rand_inx]
        tmp_iny = random.randint(0, len(avec_list) - 1)
        tmp_label = label_map[alist[tmp_iny]]
        while tmp_label == raw_query2label[inx]:
            tmp_iny = random.randint(0, len(avec_list) - 1)
            tmp_label = label_map[alist[tmp_iny]]
        items = avec_list[tmp_iny]
        if embedding_type == "subword_embedding":
            x_train_1 += raw_x_1_res[inx][0]
            x_train_2 += raw_x_2_res[inx][0]
            x_train_3 += items[0]
            indices_train_1 += change_indic_start(raw_x_1_res[inx][2],semi_cnt*max_seq_len,raw_x_1_res[inx][1])
            indices_train_2 += change_indic_start(raw_x_2_res[inx][2],semi_cnt*max_seq_len,raw_x_2_res[inx][1])
            indices_train_3 += change_indic_start(0,semi_cnt*max_seq_len,items[1])
        else:
            x_train_1.append(macro_x_train_1[inx])
            x_train_2.append(macro_x_train_2[inx])
            x_train_3.append(items[0])
        rand_inx += 1
        semi_cnt += 1
    return np.array(x_train_1), np.array(x_train_2), np.array(x_train_3), \
        np.array(indices_train_1), np.array(indices_train_2), np.array(indices_train_3)

def load_data_val(testList, vocab, index, batch, max_seq_len, embedding_type="word_embedding"):
    '''
    desc:测试集批次化

    input:
        testList:测试集数据
        vocab:字典
        index:起始值
        batch:批次大小
        max_seq_len:最大长度
        embedding_type:词向量类型
    return:
        x_train_1:query问题
        x_train_2:ans答案
        indic_train_1:query问题的坐标信息
        indic_train_2:ans答案的坐标信息
        tag_list:标签
    '''
    x_train_1 = []
    x_train_2 = []
    indic_train_1 = []
    indic_train_2 = []
    tag_list = []
    for i in range(0, batch):
        true_index = index + i
        if (true_index >= len(testList)):
            true_index = len(testList) - 1
            break
        indic_start = max_seq_len * i
        items = testList[true_index].split('\t')
        tag_list.append(items[0])#答案
        encode_res_1 = encode_sent(vocab, items[2], max_seq_len, indic_start, embedding_type)#正样例
        encode_res_2 = encode_sent(vocab, items[1], max_seq_len, indic_start, embedding_type)#负样例
        if embedding_type == "subword_embedding":
            x_train_1 += encode_res_1[0]
            x_train_2 += encode_res_2[0]
            indic_train_1 += encode_res_1[1]
            indic_train_2 += encode_res_2[1]
        else:
            x_train_1.append(encode_res_1[0])
            x_train_2.append(encode_res_2[0])
    return np.array(x_train_1), np.array(x_train_2), np.array(indic_train_1), np.array(indic_train_2), tag_list

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def get_seg_list(query):
    seg_list = jieba.cut(query)
    return seg_list

def get_input_seg(file_name):
    #prepare train data for segment
    fw = open(file_name+".seg","w")
    with open(file_name) as f:
        for line in f:
            tmp_list = line.strip().split('\t')
            label = tmp_list[0]
            query = tmp_list[1]
            label_seg_list = get_seg_list(label)
            query_seg_list = get_seg_list(query)
            label_seg = ' '.join(label_seg_list).encode('utf-8')
            query_seg = ' '.join(query_seg_list).encode('utf-8')
            fw.write(label_seg + "\t" + query_seg+'\n')
    fw.close()
    return

if __name__=='__main__':
    get_input_seg('./data/cnews.test.txt')
