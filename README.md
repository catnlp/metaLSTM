<p align="center"><img width="100%" src="images/catnlp_logo.png" /></p>

--------------------------------------------------------------------------------

# metaLSTM

Meta Learning for LSTM.

## 1 参考资料

- [x] [(1) pytorch_workplace/rnn](https://github.com/DingKe/pytorch_workplace/tree/master/rnn)
- [x] [(2) Pytorch Doc](http://pytorch.org/docs/0.3.1/)
- [x] [(3) HyperNetworks](https://arxiv.org/pdf/1609.09106.pdf)
- [x] [(4) supercell](https://github.com/hardmaru/supercell)
- [x] [(5) Meta Multi-Task Learning for Sequence Modeling](https://arxiv.org/pdf/1802.08969.pdf)
- [x] [(6) Optimization As a Model For Few-Shot Learning ](https://openreview.net/pdf?id=rJY0-Kcll)
- [x] [(7) NCRF++](https://github.com/jiesutd/NCRFpp)
- [x] [(8) Recurrent Batch Normalization](https://arxiv.org/pdf/1603.09025.pdf)
- [x] [(9) batch_normalized_LSTM](https://github.com/sysuNie/batch_normalized_LSTM)

## 2 环境

```
pip install -r requirements.txt
```

## 3 进展

- [x] metaRNNs
- [x] 简单测试RNNs和MetaRNNs
- [x] 在MNIST上测试RNNs
- [x] 在MNIST上测试MetaRNNs
- [x] 在CoNLL-2003上测试RNNs
- [x] 在CoNLL-2003上测试MetaRNNs
- [ ] 冲刺state of the art

## 4 实验

### 4.1 测试集MNIST

[MNIST官网](http://yann.lecun.com/exdb/mnist/)

MNIST是一个手写数字数据集，训练集有60，000个例子，测试集有10，000个例子。

#### 4.1.1 标准RNN和RNN

- [x] 实验结果

<p align="center"><img width="100%" src="images/base_RNN_MNIST.PNG" /></p>

#### 4.1.2 标准LSTM和LSTM

- [x] 实验结果

<p align="center"><img width="100%" src="images/base_LSTM_MNIST.PNG" /></p>

#### 4.1.3 MetaRNN和MetaLSTM

- [x] 实验结果

<p align="center"><img width="100%" src="images/meta_RNN_LSTM_MNIST.PNG" /></p>

### 4.2 测试集CoNLL-2003

[CoNLL-2003官网](https://www.clips.uantwerpen.be/conll2003/ner/)

CoNLL-2003是一个命名实体识别数据集，包含4类实体：PER, LOC, ORG, MISC

#### 4.2.1 标准RNN和RNN

- [x] 实验结果

<p align="center"><img width="100%" src="images/base_RNN_CoNLL-2003.PNG" /></p>

*注：RNN图很快停止是因为训练时出现了NAN*

#### 4.2.2 标准LSTM和LSTM

- [x] 实验结果

<p align="center"><img width="100%" src="images/base_LSTM_CoNLL-2003.PNG" /></p>

#### 4.2.3 MetaRNN和MetaLSTM

- [x] 实验结果

<p align="center"><img width="100%" src="images/meta_RNN_LSTM_CoNLL-2003.PNG" /></p>

*注：MetaRNN图很快停止是因为训练时出现了NAN*

### 4.3 冲刺state of the art

- [ ] 梯度更新方法（SGD, Adagrad, Adadelta, Adam, Nadam ...）
- [ ] 归一化方法（Dropout, Batch, Layer）
- [ ] 词向量（cove）
- [ ] 注意力机制（待学习）
- [ ] 多任务学习（加标签）
- [ ] 元学习（学习率更新）

#### 4.3.1 模型最优

- 双向
- 超参数

- [ ] 实验结果

#### 4.3.2 梯度更新方法

- [ ] 实验结果

#### 4.3.3 归一化方法

- [ ] 实验结果

#### 4.3.4 词向量

- [ ] 实验结果

#### 4.3.5 注意力机制

- [ ] 实验结果

#### 4.3.6 多任务学习

- [ ] 实验结果

#### 4.3.7 元学习

- [ ] 实验结果

## 5 体会

- [x] SGD训练，学习率lr设置很重要，过大容易训练不了
- [x] LSTM比RNN及其变种更容易训练，即使学习率lr设置过大
- [x] RNN在训练过程中，Loss容易变成NAN，而无法进一步训练
- [x] 对于NER任务，加CRF，双向LSTM效果显著（2~3个点），hidden_emb有些许的提升（1个点）