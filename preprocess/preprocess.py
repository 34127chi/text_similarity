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

import ConfigParser
import glob
import sys
import os

import workflow

# @todo: commandline tools

class Preprocessor(object):
    """
    querysim trainning preprocess module
    """
    def __init__(self, conf_file):
        self.configs = None
        self.workflows = None
        self.conf_file = conf_file

    def initialize(self):
        """
        initialize config and workflow
        初始化配置和工作流
        """
        self.configs = self.load_configs()
        self.workflows = self.load_workflow()

    def load_configs(self):
        """
        load configs file
        加载配置文件
        """
        configs = {}
        cf = ConfigParser.ConfigParser()
        cf.read(self.conf_file)
        configs = dict(cf.items('GLOBAL'))#global配置信息
        configs['feature'] = []
        subconf = dict()
        sep_stat = configs.get('src_data_seg_sep')
        assert sep_stat != 'None', \
        "please set the delimiter of the input data in config, set parameter src_data_seg_sep"
        subconf['src_data_seg_sep'] = configs.get('src_data_seg_sep').strip('\'')
        subconf['flow'] = configs['flow']#工作流
        subconf['data'] = configs['src_data']#输入文件夹
        subconf['output_dir'] = configs['output_dir']#输出文件夹
        subconf['platform'] = configs.get('platform', 'paddle')#平台选择
        subconf['model_type'] = configs['model_type']#模型类别
        subconf['partition_ratio'] = configs.get('partition_ratio', '10:0')#样本分配比例
        for sec in cf.sections():
            if sec.startswith('FEATURE'):#自定义feature
                feature_conf = dict(cf.items(sec))
                feature_conf.update(subconf)
                configs['feature'].append(feature_conf)
        return configs

    def load_workflow(self):
        """
        read operation
        初始化工作流
        """
        workflows = []
        flow = self.configs.get('flow', '')
        for feature in self.configs['feature']:
            param = dict(feature)
            wf = workflow.WorkFlow(**param)
            wf.initialize(**param)
            workflows.append(wf)
        return workflows

    def start(self):
        """
        run
        执行工作流
        """
        for wf in self.workflows:
            print 'start workflow for %s' % wf.feature_name
            wf.start_flow()

if __name__ == '__main__':
    conf_file = sys.argv[1]#预处理配置文件
    p = Preprocessor(conf_file)
    p.initialize()#预处理对象初始化
    p.start()

# vim: set ts=4 sw=4 sts=4 tw=100:
