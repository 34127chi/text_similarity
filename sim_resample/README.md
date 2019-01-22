# text-similarity
nlp for short text similarity calculation


================description==================

CNN model for Q&A(Question and Answering), tensorflow code implementation
计算输入问句与标准问句之间的相似度
支持两种负采样方式
支持多种embedding方式


================dataset================

实验用的是清华大学的新闻分类数据集，实验效果来看
rand负采样的效果会差一些

=================run=====================

训练集和测试集的格式：标签\t问句

数据集demo在：knn-classification/knn-classification/data/

进行分词处理：python insurance_qa_data_helpers.py

训练：
python train.py --model_verion=xxx

model_verion:模型版本号


pairwise计算相似度时，需要找每个正样例的负样例，即每个问题query不仅要有正确答案corr_ans，还要错误答案fault_ans

rand采样:
随机采样,即随机选择错误答案fault_ans


semihard采样:
当fault_ans与corr_ans相差较大时，模型类似于pointwise 
当fault_ans与corr_ans很接近时，模型开始训练时容易陷入局部最优
当fault_ans与corr_ans介于上述两者之间时，效果最好

semihard采样的解读:http://www.shuang0420.com/2018/03/17/%E6%89%AF%E6%89%AF%20Semi-hard%20Negative%20Samples/
