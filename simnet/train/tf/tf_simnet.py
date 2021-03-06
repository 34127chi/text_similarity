#coding=utf-8

# Copyright (c) 2018 Baidu, Inc. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import json
import sys
import os

import tensorflow as tf
from tensorflow.python.framework import graph_util

from utils import datafeeds
from utils import controler
from utils import utility
from utils import converter

_WORK_DIR = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(os.path.join(_WORK_DIR, '../../common'))
import log 


def load_config(config_file):
    """
    load config
    """
    with open(config_file, "r") as f:
        try:
            conf = json.load(f)
        except Exception:
            logging.error("load json file %s error" % config_file)
    conf_dict = {}
    unused = [conf_dict.update(conf[k]) for k in conf]
    logging.debug("\n".join(
        ["%s=%s" % (u, conf_dict[u]) for u in conf_dict]))
    return conf_dict


def train(conf_dict):
    """
    train

    定义网络所需要的基础组件=>定义输入流=>定义网络预测/训练结构=>定义损失函数=>定义优化算法
    """
    training_mode = conf_dict["training_mode"]
    net = utility.import_object(
        conf_dict["net_py"], conf_dict["net_class"])(conf_dict)#初始化网络结构
    if training_mode == "pointwise":
        datafeed = datafeeds.TFPointwisePaddingData(conf_dict)#输入数据的类
        input_l, input_r, label_y = datafeed.ops()#初始化输入流数据
        pred = net.predict(input_l, input_r)#初始化预测网络结果
        output_prob = tf.nn.softmax(pred, -1, name="output_prob")
        loss_layer = utility.import_object(
            conf_dict["loss_py"], conf_dict["loss_class"])()
        loss = loss_layer.ops(pred, label_y)
    elif training_mode == "pairwise":
        datafeed = datafeeds.TFPairwisePaddingData(conf_dict)
        input_l, input_r, neg_input = datafeed.ops()
        pos_score = net.predict(input_l, input_r)
        output_prob = tf.identity(pos_score, name="output_preb")
        neg_score = net.predict(input_l, neg_input)
        loss_layer = utility.import_object(
            conf_dict["loss_py"], conf_dict["loss_class"])(conf_dict)
        loss = loss_layer.ops(pos_score, neg_score)
    else:
        print >> sys.stderr, "training mode not supported"
        sys.exit(1)
    # define optimizer
    lr = float(conf_dict["learning_rate"])
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

    # run_trainer
    controler.run_trainer(loss, optimizer, conf_dict)


def predict(conf_dict):
    """
    predict
    """
    net = utility.import_object(
        conf_dict["net_py"], conf_dict["net_class"])(conf_dict)#初始化网络
    conf_dict.update({"num_epochs": "1", "batch_size": "1",
                      "shuffle": "0", "train_file": conf_dict["test_file"]})
    test_datafeed = datafeeds.TFPointwisePaddingData(conf_dict)
    test_l, test_r, test_y = test_datafeed.ops()
    # test network
    pred = net.predict(test_l, test_r)
    controler.run_predict(pred, test_y, conf_dict)


def freeze(conf_dict):
    """
    freeze net for c api predict
    """
    model_path = conf_dict["save_path"]
    freeze_path = conf_dict["freeze_path"]
    saver = tf.train.import_meta_graph(model_path + '.meta')
    with tf.Session() as sess:
        saver.restore(sess, model_path)
        var_graph_def = tf.get_default_graph().as_graph_def()
        const_graph_def = graph_util.convert_variables_to_constants(sess, var_graph_def, ["output_prob"])
        with tf.gfile.GFile(freeze_path, "wb") as f:
            f.write(const_graph_def.SerializeToString())

def convert(conf_dict):
    """
    convert
    """
    converter.run_convert(conf_dict)


if __name__ == "__main__": 
    log.init_log("./log/tensorflow")
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='train',
                        help='task: train/predict/freeze/convert, the default value is train.')
    parser.add_argument('--task_conf', default='./examples/cnn-pointwise.json',
                        help='task_conf: config file for this task')
    args = parser.parse_args()
    task_conf = args.task_conf
    config = load_config(task_conf)#加载配置文件
    task = args.task
    if args.task == 'train':
        train(config)#训练
    elif args.task == 'predict':
        predict(config)#预测
    elif args.task == 'freeze':
        freeze(config)#存储模型的参数
    elif args.task == 'convert':
        convert(config)#格式转换
    else:
        print >> sys.stderr, 'task type error.'
