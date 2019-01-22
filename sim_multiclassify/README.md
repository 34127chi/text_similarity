网络结构:
1) cnn + (birnn + att) concat;
2) cnn ;
3) birnn;
4) birnn + att;

损失函数:
1) 普通的交叉熵函数;
2) 稀疏版的am-softmax函数;

其中2)的直观理解是训练时减少样本真实label对应的概率，使得预测时增加样本真实label对应的概率，起到类间分离，而类内是聚合的


