# encoding:utf-8
'''
@Author: catnlp
@Email: wk_nlp@163.com
@Time: 2018/4/24 19:36
'''
import RNNCells

import torch
from torch.nn import Module
from torch.autograd import Variable

class RNNBase(Module):
    def __init__(self, mode, input_size, hidden_size, num_layers, recurrent_size=None, bias=True, grad_clip=None):
        super(RNNBase, self).__init__()
        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.recurrent_size = recurrent_size
        self.num_layers = num_layers
        self.bias = bias
        self.grad_clip = grad_clip

        mode2cell = {'RNN': RNNCells.RNNCell,
                     'LSTM': RNNCells.LSTMCell}

        Cell = mode2cell[mode]

        kwargs = {'input_size': input_size,
                  'hidden_size': hidden_size,
                  'bias': bias,
                  'grad_clip': grad_clip}

        self.cell0 = Cell(**kwargs)
        for i in range(1, num_layers):
            kwargs['input_size'] = hidden_size
            cell = Cell(**kwargs)
            setattr(self, 'cell{}'.format(i), cell)

    def forward(self, input, initial_states=None):
        if initial_states is None:
            zeros = Variable(torch.zeros(input.size(0), self.hidden_size))
            if self.mode == 'LSTM':
                initial_states = [(zeros, zeros), ] * self.num_layers
            else:
                initial_states = [zeros] * self.num_layers
        assert len(initial_states) == self.num_layers

        states = initial_states
        outputs = []

        time_steps = input.size(1)
        for t in range(time_steps):
            x = input[:, t, :]
            for num in range(self.num_layers):
                hx = getattr(self, 'cell{}'.format(num))(x, states[num])
                states[num] = hx
                if self.mode.startswith('LSTM'):
                    x = hx[0]
                else:
                    x = hx
            outputs.append(hx)

        if self.mode.startswith('LSTM'):
            hs, cs = zip(*outputs)
            h = torch.stack(hs).transpose(0, 1)
            output = h, (outputs[-1][0], outputs[-1][1])
        else:
            output = torch.stack(outputs).transpose(0, 1), outputs[-1]
        return output

class RNN(RNNBase):
    def __init__(self, *args, **kwargs):
        super(RNN, self).__init__('RNN', *args, **kwargs)

class LSTM(RNNBase):
    def __init__(self, *args, **kwargs):
        super(LSTM, self).__init__('LSTM', *args, **kwargs)