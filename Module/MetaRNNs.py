# encoding:utf-8
'''
@Author: catnlp
@Email: wk_nlp@163.com
@Time: 2018/4/25 21:19
'''
from Module import MetaRNNCells

import torch
from torch.nn import Module
from torch.autograd import Variable

class MetaRNNBase(Module):
    def __init__(self, mode, input_size, hidden_size, hyper_hidden_size, hyper_embedding_size, num_layers, recurrent_size=None, bias=True, bias_hyper=True, grad_clip=None, gpu=False, bidirectional=False):
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
        self.gpu = gpu
        self.bidirectional=bidirectional

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

        if self.bidirectional:
            self.cellb0 = Cell(**kwargs)
            for i in range(1, num_layers):
                kwargs['input_size'] = hidden_size
                cell = Cell(**kwargs)
                setattr(self, 'cellb{}'.format(i), cell)

    def _initial_states(self, inputSize):
        main_zeros = Variable(torch.zeros(inputSize, self.hidden_size))
        meta_zeros = Variable(torch.zeros(inputSize, self.hyper_hidden_size))
        if self.gpu:
            main_zeros = main_zeros.cuda()
            meta_zeros = meta_zeros.cuda()
        zeros = (main_zeros, meta_zeros)
        if self.mode == 'MetaLSTM':
            states = [((main_zeros, main_zeros), (meta_zeros, meta_zeros)), ] * self.num_layers
        else:
            states = [zeros] * self.num_layers
        return states

    def forward(self, input):
        states = self._initial_states(input.size(0))
        outputs = []
        time_steps = input.size(1)

        if self.bidirectional:
            states_b = self._initial_states(input.size(0))
            outputs_f = []
            outputs_b = []
            hx = None

            for num in range(self.num_layers):
                for t in range(time_steps):
                    x = input[:, t, :]
                    hx = getattr(self, 'cell{}'.format(num))(x, states[num])
                    states[num] = hx
                    if self.mode.startswith('MetaLSTM'):
                        outputs_f.append(hx[0])
                    else:
                        outputs_f.append(hx)
                for t in range(time_steps)[::-1]:
                    x = input[:, t, :]
                    hx = getattr(self, 'cellb{}'.format(num))(x, states_b[num])
                    states_b[num] = hx
                    if self.mode.startswith('MetaLSTM'):
                        outputs_b.append(hx[0][0])
                    else:
                        outputs_b.append(hx[0])
                input = torch.cat([torch.stack(outputs_f).transpose(0, 1), torch.stack(outputs_b).transpose(0, 1)], 2)
                outputs_f = []
                outputs_b = []
            output = input, hx[0]
        else:
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