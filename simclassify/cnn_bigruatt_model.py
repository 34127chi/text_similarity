#!/usr/bin/python
#-*- coding:utf-8 -*-
############################
#File Name: cnn_bigruatt_model.py
#Created Time: 2018-08-10 17:20:59
############################
from keras.models import Model
from keras.layers import *
from keras.constraints import unit_norm
from margin_softmax import *
from gensim.models import Word2Vec as word2vec

class CnnBiGruAttConfig(object):
    '''
    配置
    '''
    #507, 3, 5, 148, 'tanh', 255
    def __init__(self, seq_len = 32, batch_size = 64, embedding_dim = 300, min_count = 2, low_kernel_size = 3, high_kernel_size = 5, hidden_size = 512, num_filters = 128, vocab_size = 5000, num_classes = 12, use_bias = False, use_pretrained_embedding = True, activation_func = 'tanh', word2vec_model_path = './data/word2vec_model_char_10000', id2char = {}):
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.min_count = min_count
        self.low_kernel_size = low_kernel_size
        self.high_kernel_size = high_kernel_size
        self.hidden_size = hidden_size
        self.num_filters = num_filters
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.use_bias = use_bias
        self.use_pretrained_embedding = use_pretrained_embedding
        self.activation_func = activation_func
        self.word2vec_model_path = word2vec_model_path
        self.id2char = id2char
        pass


class CnnBiGruAttModel(object):
    '''
    多分类模型
    '''
    def __init__(self, config):
        self.config = config
        self.cnn_bigruatt()

    def cnn_bigruatt(self):
        x_in = Input(shape=(self.config.seq_len,))
        if self.config.use_pretrained_embedding == False:
            x_embedded = Embedding(self.config.vocab_size, self.config.embedding_dim
                                   )(x_in)#embedding的大小
        else:
            word2vec_model = word2vec.load(self.config.word2vec_model_path)
            word2vec_dict = {}
            for idx, word in enumerate(word2vec_model.wv.vocab):
                word2vec_dict[word] = word2vec_model.wv[word]
            word_embedding_2dlist = [[]] * (self.config.vocab_size) 
            word_embedding_2dlist[0] = np.zeros(self.config.embedding_dim)
            bound = np.sqrt(6.0) / np.sqrt(self.config.vocab_size)  # bound for random variables.
            count_exist = 0
            count_not_exist = 0
            for i in range(1, self.config.vocab_size):  # loop each word
                word = self.config.id2char[i]  # get a word
                embedding = None
                try:
                    embedding = word2vec_dict[word][:self.config.embedding_dim]  # try to get vector:it is an array.
                except Exception:
                    embedding = None
                if embedding is not None:  # the 'word' exist a embedding
                    word_embedding_2dlist[i] = embedding
                    count_exist = count_exist + 1  # assign array to this word.
                else:  # no embedding for this word
                    word_embedding_2dlist[i] = np.random.uniform(-bound, bound, self.config.embedding_dim);
                    count_not_exist = count_not_exist + 1  # init a random value for the word.
            word_embedding_final = np.array(word_embedding_2dlist)  # covert to 2d array.
            print("word. exists embedding:", count_exist, " ;word not exist embedding:", count_not_exist)

            #pdb.set_trace()
            x_embedded = Embedding(self.config.vocab_size, self.config.embedding_dim, weights=[word_embedding_final])(x_in)

        reshape = Reshape((self.config.seq_len, self.config.embedding_dim, 1))(x_embedded)

        maxpool_list = []
        for filter_size in range(self.config.low_kernel_size, self.config.high_kernel_size + 1):
            conv = Conv2D(self.config.num_filters, kernel_size=(filter_size, self.config.embedding_dim), padding='valid', kernel_initializer='normal', activation=self.config.activation_func)(reshape)
            maxpool = MaxPool2D(pool_size=(self.config.seq_len - filter_size + 1, 1), strides=(1,1), padding='valid')(conv)
            maxpool_list.append(maxpool)
        
        if len(maxpool_list) == 1:
            concatenated_tensor = maxpool_list[0]
        else:
            concatenated_tensor = Concatenate(axis=1)(maxpool_list)

        flatten = Flatten()(concatenated_tensor)
        #pdb.set_trace()
        x = Lambda(lambda x: K.l2_normalize(x, 1))(flatten)

        y = Bidirectional(GRU(self.config.hidden_size, activation = self.config.activation_func, return_sequences=True))(x_embedded)

        attention_weight = TimeDistributed(Dense(1, activation=self.config.activation_func))(y) 
        attention_weight = Flatten()(attention_weight)
        attention_weight = Activation('softmax')(attention_weight)
        attention_weight = RepeatVector(2*self.config.hidden_size)(attention_weight)
        attention_weight = Permute([2, 1])(attention_weight)
        sent_representation = multiply([y, attention_weight])
        sent_representation = Lambda(lambda xin: K.sum(xin, axis=1))(sent_representation)
        y = sent_representation
        y = Lambda(lambda y: K.l2_normalize(y, 1))(y)

        merged_representation = Concatenate(axis=1)([x, y])

        #pdb.set_trace()

        pred = Dense(self.config.num_classes,
                     use_bias=self.config.use_bias,
                     kernel_constraint=unit_norm())(merged_representation)

        self.encoder = Model(x_in, merged_representation) # 最终的目的是要得到一个编码器  获得对应的编码器
        self.model = Model(x_in, pred) # 用分类问题做训练 分类训练使用

        self.model.compile(loss=sparse_amsoftmax_loss, optimizer='adam', metrics=['sparse_categorical_accuracy'])

