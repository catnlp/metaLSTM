# encoding:utf-8
'''
@Author: catnlp
@Email: wk_nlp@163.com
@Time: 2018/4/25 21:19
'''
import MetaRNNCells

import torch
from torch.nn import Module
from torch.autograd import Variable

class MetaRNNBase(Module):
    def __init__(self, mode, input_size, hidden_size, hyper_hidden_size, hyper_embedding_size, num_layers, recurrent_size=None, bias=True, bias_hyper=True, grad_clip=None):
        super(MetaRNNBase, self).__init__()
        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hyper_hidden_size = hyper_hidden_size
        self.hyper_embedding_size = hyper_embedding_size
        self.recurrent_size = recurrent_size
        self.num_layers = num_layers
        self.bias = bias
        self.bias_hyper = bias_hyper
        self.grad_clip = grad_clip

        mode2cell = {'MetaRNN': MetaRNNCells.MetaRNNCell,
                     'MetaLSTM': MetaRNNCells.MetaLSTMCell}

        Cell = mode2cell[mode]

        kwargs = {'input_size': input_size,
                  'hidden_size': hidden_size,
                  'hyper_hidden_size': hyper_hidden_size,
                  'hyper_embedding_size': hyper_embedding_size,
                  'bias': bias,
                  'bias_hyper': bias_hyper,
                  'grad_clip': grad_clip}

        self.cell0 = Cell(**kwargs)
        for i in range(1, num_layers):
            kwargs['input_size'] = hidden_size
            cell = Cell(**kwargs)
            setattr(self, 'cell{}'.format(i), cell)

    def forward(self, input, initial_states=None):
        if initial_states is None:
            main_zeros = Variable(torch.zeros(input.size(0), self.hidden_size))
            meta_zeros = Variable(torch.zeros(input.size(0), self.hyper_hidden_size))
            zeros = (main_zeros, meta_zeros)
            if self.mode == 'MetaLSTM':
                initial_states = [((main_zeros, main_zeros), (meta_zeros, meta_zeros)), ] * self.num_layers
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
                if self.mode.startswith('MetaLSTM'):
                    x = hx[0][0]
                else:
                    x = hx[0]
            outputs.append(hx[0])

        if self.mode.startswith('MetaLSTM'):
            hs, cs = zip(*outputs)
            h = torch.stack(hs).transpose(0, 1)
            output = h, (outputs[-1][0], outputs[-1][1])
        else:
            output = torch.stack(outputs).transpose(0, 1), outputs[-1]
        return output

class MetaRNN(MetaRNNBase):
    def __init__(self, *args, **kwargs):
        super(MetaRNN, self).__init__('MetaRNN', *args, **kwargs)

class MetaLSTM(MetaRNNBase):
    def __init__(self, *args, **kwargs):
        super(MetaLSTM, self).__init__('MetaLSTM', *args, **kwargs)