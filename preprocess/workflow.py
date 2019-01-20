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

import glob
import os

import operation_unit
import op_out

class WorkFlow(object):
    """
    work flow
    """
    def __init__(self, **configs):
        """
        Initialize instance
        实例化
        """
        self.feature_name = configs.get('name')#名称
        self.flow = configs.get('flow')#流程
        self.model_type = configs.get('model_type')#模型类型
        self.src_data = configs.get('data')#输入源
        self.platform = configs.get('platform').split(',')#平台
        self.output_dir = configs.get('output_dir')#输出文件
        self.src_sep = configs.get('src_data_seg_sep')#分词之间的间隔
        self.ops = None

    def __rglob(self, path, filelist):
        """
        Recursively search files, support matching
        递归搜索文件
        """
        for fn in glob.glob(path):
            if os.path.isdir(fn):
                for f in os.listdir(fn):
                    self.__rglob(os.path.join(fn, f), filelist)
            else:
                filelist.append(fn)

    def read(self):
        """
        Load data
        加载数据
        """
        filelist = []
        self.__rglob(self.src_data, filelist)#文件夹下的文件名称
        if self.model_type == 'pointwise':
            dataset = self.read_pointwise(filelist)#数据 list 每个元素是个样本
        else:
            dataset = self.read_pairwise(filelist)
        return dataset

    def read_pointwise(self, filelist):
        """
        Load pointwise data
        加载pointwise数据文件
        """
        dataset = []
        for file in filelist:
            with open(file) as f:
                for line in f:
                    tpl = line.rstrip('\n').split('\t')
                    if 0 in map(lambda t:len(t), tpl):#长度为空即跳过
                        continue
                    oo = op_out.OperationOut()#每个对象即是一条记录
                    oo.set_pointwise_data(label=tpl[0], query=tpl[1], \
                            title=tpl[2], src_sep=self.src_sep)
                    dataset.append(oo)
        return dataset

    def read_pairwise(self, filelist):
        """
        Load pairwise data
        加载pairwise数据
        """
        inputdata = []
        for file in filelist:
            with open(file) as f:
                for line in f:
                    tpl = line.rstrip('\n').split('\t')
                    if 0 in map(lambda t:len(t), tpl):
                        continue
                    inputdata.append(tpl)
        dataset = []
        qid = 0
        prev_query = ''
        sorted_data = sorted(inputdata, key=lambda t:t[1])#按照query排序
        for (label, query, title) in sorted_data:#相同的query 同一个qid
            if query != prev_query:
                qid += 1
                prev_query = query
            oo = op_out.OperationOut()
            oo.set_pairwise_data(qid=qid, label=label, query=query, \
                    title=title, src_sep=self.src_sep)
            dataset.append(oo)
        return dataset

    def check_topological(self):
        """
        Check the workflow topology
        检测自定义工作流的拓扑结构
        """
        standard = [
            operation_unit.OpGenerateDict,
            operation_unit.OpTerm2ID,
            operation_unit.OpPartition,
            operation_unit.OpWriteData
            ]
        control = [type(op) for op in self.ops]
        intersec = list(set(standard) & set(control))#标准工作流与自定义工作流之间的交集
        intersec.sort(key=standard.index)
        for i, op in enumerate(self.ops):
            if not isinstance(op, intersec[i]):
                raise Exception("Wrong workflow topology, please check.")

    def initialize(self, **configs):
        """
        Initialize work flow
        初始化工作流
        """
        self.ops = []
        get_termdict_method = None
        for op in self.flow.split(','):#操作
            if op == 'gendict':
                op_generate_dict = operation_unit.OpGenerateDict(**configs)
                self.ops.append(op_generate_dict)
                get_termdict_method = op_generate_dict.get_termdict
            elif op == 'convertid':
                conf = dict(**configs)
                conf['get_termdict'] = get_termdict_method
                op_term2id = operation_unit.OpTerm2ID(**conf)
                self.ops.append(op_term2id)
            elif op == 'partition':
                op_partition = operation_unit.OpPartition(**configs)
                self.ops.append(op_partition)
            elif op == 'write':
                if operation_unit.OpPartition not in map(type, self.ops):
                    op_partition = \
                    operation_unit.OpPartition(passive=True, **configs)
                    self.ops.append(op_partition)
                op_write = operation_unit.OpWriteData(**configs)
                self.ops.append(op_write)

        self.check_topological()#check拓扑结构

        if 'write' in self.flow or 'gendict' in self.flow:#存在写入、生成字典文件的话就生成文件夹
            outdir = os.path.join(self.output_dir, self.feature_name)
            if not os.path.exists(outdir):
                os.makedirs(outdir)

    def start_flow(self):
        """
        start workflow
        执行工作流
        """
        prev_op_out = self.read()#读取数据
        for op in self.ops:
            prev_op_out = op.operate(prev_op_out)#执行每一个操作
        return prev_op_out

# vim: set ts=4 sw=4 sts=4 tw=100:
