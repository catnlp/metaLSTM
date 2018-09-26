# encoding:utf-8
'''
@Author: catnlp
@Email: wk_nlp@163.com
@Time: 2018/4/24 19:36
'''
from Modules import RNNCells

import torch
from torch.nn import Module
from torch.autograd import Variable

class RNNBase(Module):
    def __init__(self, mode, input_size, hidden_size, num_layers, bias=True, gpu=False, bidirectional=False):
        super(RNNBase, self).__init__()
        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.gpu = gpu
        self.bidirectional = bidirectional

        mode2cell = {'RNN': RNNCells.RNNCell,
                     'LSTM': RNNCells.LSTMCell}

        Cell = mode2cell[mode]

        kwargs = {'input_size': input_size,
                  'hidden_size': hidden_size,
                  'bias': bias}

        if self.bidirectional:
            self.cell0 = Cell(**kwargs)
            for i in range(1, num_layers):
                kwargs['input_size'] = hidden_size * 2
                cell = Cell(**kwargs)
                setattr(self, 'cell{}'.format(i), cell)

            kwargs['input_size'] = input_size
            self.cellb0 = Cell(**kwargs)
            for i in range(1, num_layers):
                kwargs['input_size'] = hidden_size * 2
                cell = Cell(**kwargs)
                setattr(self, 'cellb{}'.format(i), cell)
        else:
            self.cell0 = Cell(**kwargs)
            for i in range(1, num_layers):
                kwargs['input_size'] = hidden_size
                cell = Cell(**kwargs)
                setattr(self, 'cell{}'.format(i), cell)

    def _initial_states(self, inputSize):
        zeros = Variable(torch.zeros(inputSize, self.hidden_size))
        if self.gpu:
            zeros = zeros.cuda()
        if self.mode == 'LSTM':
            states = [(zeros, zeros), ] * self.num_layers
        else:
            states = [zeros] * self.num_layers
        return states

    def forward(self, input, length=None):
        states = self._initial_states(input.size(0))
        time_steps = input.size(1)

        if length is None:
            length = Variable(torch.LongTensor([time_steps] * input.size(0)))
            if self.gpu:
                length = length.cuda()

        if self.bidirectional:
            states_b = self._initial_states(input.size(0))
            outputs_f = []
            outputs_b = []

            for num in range(self.num_layers):
                for t in range(time_steps):
                    x = input[:, t, :]
                    # hx = getattr(self, 'cell{}'.format(num))(x, states[num])
                    # mask = (t < length).float().unsqueeze(1).expand_as(hx)
                    # hx = hx * mask + states[0] * (1 - mask)
                    # states[num] = hx
                    if self.mode.startswith('LSTM'):
                        h, c = getattr(self, 'cell{}'.format(num))(x, states[num])
                        mask_h = (t < length).float().unsqueeze(1).expand_as(h)
                        h = h * mask_h + states[0][0] * (1 - mask_h)
                        mask_c = (t < length).float().unsqueeze(1).expand_as(c)
                        c = c * mask_c + states[0][1] * (1 - mask_c)
                        states[num] = h, c
                        outputs_f.append(h)
                    else:
                        hx = getattr(self, 'cell{}'.format(num))(x, states[num])
                        mask = (t < length).float().unsqueeze(1).expand_as(hx)
                        hx = hx * mask + states[0] * (1 - mask)
                        states[num] = hx
                        outputs_f.append(hx)
                for t in range(time_steps)[::-1]:
                    x = input[:, t, :]
                    if self.mode.startswith('LSTM'):
                        h, c = getattr(self, 'cell{}'.format(num))(x, states_b[num])
                        mask_h = (t < length).float().unsqueeze(1).expand_as(h)
                        h = h * mask_h + states_b[0][0] * (1 - mask_h)
                        mask_c = (t < length).float().unsqueeze(1).expand_as(c)
                        c = c * mask_c + states_b[0][1] * (1 - mask_c)
                        states_b[num] = h, c
                        outputs_b.append(h)
                    else:
                        hx = getattr(self, 'cell{}'.format(num))(x, states_b[num])
                        mask = (t < length).float().unsqueeze(1).expand_as(hx)
                        hx = hx * mask + states_b[0] * (1 - mask)
                        states_b[num] = hx
                        outputs_b.append(hx)
                    # hx = getattr(self, 'cellb{}'.format(num))(x, states_b[num])
                    # mask = (t < length).float().unsqueeze(1).expand_as(hx)
                    # hx = hx * mask + states_b[0] * (1 - mask)
                    # states_b[num] = hx
                    # if self.mode.startswith('LSTM'):
                    #     outputs_b.append(hx[0])
                    # else:
                    #     outputs_b.append(hx)
                outputs_b.reverse()
                input = torch.cat([torch.stack(outputs_f).transpose(0, 1), torch.stack(outputs_b).transpose(0, 1)], 2)
                outputs_f = []
                outputs_b = []
        else:
            # outputs = []
            # for t in range(time_steps):
            #     x = input[:, t, :]
            #     for num in range(self.num_layers):
            #         hx = getattr(self, 'cell{}'.format(num))(x, states[num])
            #         states[num] = hx
            #         if self.mode.startswith('LSTM'):
            #             x = hx[0]
            #         else:
            #             x = hx
            #     outputs.append(hx)
            #  if self.mode.startswith('LSTM'):
            #         hs, cs = zip(*outputs)
            #         h = torch.stack(hs).transpose(0, 1)
            #         output = h, (outputs[-1][0], outputs[-1][1])
            #  else:
            #         output = torch.stack(outputs).transpose(0, 1), outputs[-1]
            outputs_f = []
            for num in range(self.num_layers):
                for t in range(time_steps):
                    x = input[:, t, :]
                    if self.mode.startswith('LSTM'):
                        h, c = getattr(self, 'cell{}'.format(num))(x, states[num])
                        mask_h = (t < length).float().unsqueeze(1).expand_as(h)
                        h = h * mask_h + states[0][0] * (1 - mask_h)
                        mask_c = (t < length).float().unsqueeze(1).expand_as(c)
                        c = c * mask_c + states[0][1] * (1 - mask_c)
                        states[num] = h, c
                        outputs_f.append(h)
                    else:
                        hx = getattr(self, 'cell{}'.format(num))(x, states[num])
                        mask = (t < length).float().unsqueeze(1).expand_as(hx)
                        hx = hx * mask + states[0] * (1 - mask)
                        states[num] = hx
                        outputs_f.append(hx)
                    # hx = getattr(self, 'cell{}'.format(num))(x, states[num])
                    # mask = (t < length).float().unsqueeze(1).expand_as(hx)
                    # hx = hx * mask + states[0] * (1 - mask)
                    # states[num] = hx
                    # if self.mode.startswith('LSTM'):
                    #     outputs_f.append(hx[0])
                    # else:
                    #     outputs_f.append(hx)
                input = torch.stack(outputs_f).transpose(0, 1)
                outputs_f = []

        output = input, input[-1]
        return output

class RNN(RNNBase):
    def __init__(self, *args, **kwargs):
        super(RNN, self).__init__('RNN', *args, **kwargs)

class LSTM(RNNBase):
    def __init__(self, *args, **kwargs):
        super(LSTM, self).__init__('LSTM', *args, **kwargs)