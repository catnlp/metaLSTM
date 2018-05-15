<p align="center"><img width="100%" src="images/catnlp_logo.png" /></p>

--------------------------------------------------------------------------------

# metaLSTM

Meta Learning for LSTM

*我用PyCharm运行项目，远程连接服务器。如果直接通过命令行运行文件可能不行，会出现找不到包的问题。*

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
- [x] [(10) Optimal Hyperparameters for Deep LSTM-Networks for Sequence Labeling Tasks](https://arxiv.org/pdf/1707.06799.pdf)

## 2 环境

```
pip install -r requirements.txt
```

## 3 目录

本地目录还包括data/，models/，完整目录如下：
    
    metaLSTM
        ----data
            ----conll2003（保存CoNLL-2003，用BMES标注）
            ----embedding（词向量目录）
        ----images (图片)
        ----MNIST（在MNIST数据集上测试RNNs）
        ----models（保存训练好的模型）
        ----Modules（RNNs, MetaRNNs, NormLSTM...）
        ----NER（主要实验的目录）
            ----Module（char, encoder, crf, ner）
            ----utils（配置文件，功能函数）

## 4 进展

- [x] metaRNNs
- [x] 简单测试RNNs和MetaRNNs
- [x] 在MNIST上测试RNNs
- [x] 在MNIST上测试MetaRNNs
- [x] 在CoNLL-2003上测试RNNs
- [x] 在CoNLL-2003上测试MetaRNNs
- [ ] 冲刺state of the art

## 5 实验

*5.1和5.2用于测试RNNs的性能，没有调整超参，所以效果并不好，等5.3完成后，再更新图片*

### 5.1 测试集MNIST

[MNIST官网](http://yann.lecun.com/exdb/mnist/)

MNIST是一个手写数字数据集，训练集有60，000个例子，测试集有10，000个例子。

#### 5.1.1 标准RNN和RNN

- [x] 实验结果

<p align="center"><img width="100%" src="images/base_RNN_MNIST.PNG" /></p>

#### 5.1.2 标准LSTM和LSTM

- [x] 实验结果

<p align="center"><img width="100%" src="images/base_LSTM_MNIST.PNG" /></p>

#### 5.1.3 MetaRNN和MetaLSTM

- [x] 实验结果

<p align="center"><img width="100%" src="images/meta_RNN_LSTM_MNIST.PNG" /></p>

### 5.2 测试集CoNLL-2003

[CoNLL-2003官网](https://www.clips.uantwerpen.be/conll2003/ner/)

CoNLL-2003是一个命名实体识别数据集，包含4类实体：PER, LOC, ORG, MISC

#### 5.2.1 标准RNN和RNN

- [x] 实验结果

<p align="center"><img width="100%" src="images/base_RNN_CoNLL-2003.PNG" /></p>

*注：RNN图很快停止是因为训练时出现了NAN*

#### 5.2.2 标准LSTM和LSTM

- [x] 实验结果

<p align="center"><img width="100%" src="images/base_LSTM_CoNLL-2003.PNG" /></p>

#### 5.2.3 MetaRNN和MetaLSTM

- [x] 实验结果

<p align="center"><img width="100%" src="images/meta_RNN_LSTM_CoNLL-2003.PNG" /></p>

*注：MetaRNN图很快停止是因为训练时出现了NAN*

### 5.3 冲刺state of the art

- [ ] 梯度更新方法（SGD, Adagrad, Adadelta, Adam, Nadam ...）
- [ ] 归一化方法（Dropout, Batch, Layer）
- [ ] 词向量（cove）
- [ ] 注意力机制（待学习）
- [ ] 多任务学习（加标签）
- [ ] 元学习（学习率更新）

#### 5.3.1 模型最优

- 双向
- 超参数

Model | Hidden_size | LR Method | Bidirectional | F1
:-: | :-: | :-: | :-: | :-:
BaseLSTM | 200| SGD(0.005) | True | 91.23
LSTM | 200 | SGD(0.015) | True | 91.01
MetaLSTM | 200 | SGD(0.015) | True | 90.42

#### 5.3.2 梯度更新方法

- [ ] 实验结果

#### 5.3.3 归一化方法

- [ ] 实验结果

#### 5.3.4 词向量

- [ ] 实验结果

#### 5.3.5 注意力机制

- [ ] 实验结果

#### 5.3.6 多任务学习

- [ ] 实验结果

#### 5.3.7 元学习

- [ ] 实验结果

## 6 体会

- [x] SGD训练，学习率lr设置很重要，过大容易训练不了
- [x] LSTM比RNN及其变种更容易训练，即使学习率lr设置过大
- [x] RNN在训练过程中，Loss容易变成NAN，而无法进一步训练
- [x] 对于NER任务，加CRF，双向LSTM效果显著（2~3个点），hidden_emb有些许的提升（1个点）