# encoding:utf-8
'''
@Author: catnlp
@Email: wk_nlp@163.com
@Time: 2018/4/24 19:35
'''
import math

import torch
from torch.nn import Module, Parameter
import torch.nn.functional as F

class RNNCellBase(Module):
    def __repr__(selfs):
        s = '{name}({input_size}, {hidden_size}'
        if 'bias' in self.__dict__ and self.bias is not True:
            s += ', bias={bias}'
        if 'nonlinearity' in self.__dict__ and self.nonlinearity != 'tanh':
            s += ', nonlinearity={nonlinearity}'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

'''
@Math: h' = tanh(w_{ih}x  + w_{hh}h + b)
'''
class RNNCell(RNNCellBase):
    def __init__(self, input_size, hidden_size, bias=True):
        super(RNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.weight_ih = Parameter(torch.Tensor(hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(hidden_size, hidden_size))
        if bias:
            self.bias = Parameter(torch.Tensor(hidden_size))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, h):
        output = F.linear(input, self.weight_ih) + F.linear(h, self.weight_hh) + self.bias
        output = F.relu(output)

        return output

'''
i = sigmoid(W_{ii}x + W_{hi}h + b_i)
f = sigmoid(W_{if}x + W_{hf}h + b_f)
g = tanh(W_{ig}x + W_{hg}h + b_g)
o = sigmoid(W_{io}x + W_{ho}h + b_o)
c' = f * c + i * g
h' = o * tanh(c')
'''
class LSTMCell(RNNCellBase):
    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.weight_ih = Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        if bias:
            self.bias = Parameter(torch.Tensor(4 * hidden_size))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, hx):
        h, c = hx

        pre = F.linear(input, self.weight_ih) + F.linear(h, self.weight_hh) + self.bias

        i = F.sigmoid(pre[:, : self.hidden_size])
        f = F.sigmoid(pre[:, self.hidden_size: self.hidden_size * 2])
        g = F.tanh(pre[:, self.hidden_size * 2: self.hidden_size * 3])
        o = F.sigmoid(pre[:, self.hidden_size * 3: ])
        c = f * c + i * g
        h = o * F.tanh(c)
        return h, c