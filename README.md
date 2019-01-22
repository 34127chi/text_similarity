文本相似度(语义):
1) simnet模块是对百度文本相似度计算模块simnet的梳理，主要特点是介绍基于pointwise、pairwise的排序模型，以及里面的基础组件;
2) sim_multiclassify模块是将文本相似度计算看作多分类，主要特点在amsoft-max损失函数;@苏剑林(Jianlin Su)
3) sim_resample,主要特点是在构建基于pairwise的负样本时采用semihard采样机制; @weiyujian
4) 基于pairwise的排序模型中，正负样例之间的距离度量--metric learning、dot、cos等;
5) 考虑多个损失函数、共现词组特征;
