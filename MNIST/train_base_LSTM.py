# encoding:utf-8
'''
@Author: catnlp
@Email: wk_nlp@163.com
@Time: 2018/4/27 22:01
'''
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import visdom
import math

torch.manual_seed(100)

# Hyper Parameters
sequence_length = 28
input_size = 28
hidden_size = 128
num_layers = 2
num_classes = 10
batch_size = 100
num_epochs = 100
learning_rate = 0.01

# MNIST Dataset
train_dataset = dsets.MNIST(root='../data/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = dsets.MNIST(root='../data/', train=False, transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# RNN Model (Many-to-One)
class base_LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, bias=True):
        super(base_LSTMModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=num_layers, bias=bias, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x):
        # set initial states
        # initial_states = [Variable(torch.zeros(x.size(0), self.hidden_size)) for _ in range(self.num_layers)]

        # forward propagate RNN
        out, _ = self.rnn(x)
        # print('out0-------')
        # print(out.size())
        out = out[:, -1, :]
        # print('out1------')
        # print(out.size())
        out.view(-1, self.hidden_size)
        # print('out2----------')
        # print(out.size())
        out = self.fc(out)
        # print('out3--------')
        # print(out.size())
        out = out.view(-1, self.num_classes)
        # print('out4----------')
        # print(out.size())
        return out

base_model = base_LSTMModel(input_size, hidden_size, num_layers, num_classes, bias=True)

criterion = nn.CrossEntropyLoss()

# Test the Model
def evaluate(model):
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = Variable(images.view(-1, sequence_length, input_size))
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    accuracy = 100.0 * correct / total
    print('Test Accuracy of the model on the 10000 test images: %.2f %%' % accuracy)
    return accuracy

# Train the Model
def train(model, model_name, save_path):
    vis = visdom.Visdom()
    best_accuracy = 0
    losses = []
    accuracy = []
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        model.train(True)
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images.view(-1, sequence_length, input_size))
            labels = Variable(labels)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            sample_loss = loss.data[0]
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                # draw the loss line
                losses.append(sample_loss)
                vis.line(np.array(losses), X=np.array([i for i in range(len(losses))]),
                         win=model_name+'_loss', opts={'title': model_name+'_loss', 'legend': ['loss']})
                print('Epoch [%d], Step [%d], Loss: %.4f' % (epoch+1, i+1, sample_loss))
        model.train(False)
        current_accuracy = evaluate(model)

        # draw the accuracy line
        accuracy.append(current_accuracy)
        vis.line(np.array(accuracy), X=np.array([i for i in range(len(accuracy))]),
                 win=model_name+'_accuracy', opts={'title': model_name+'_accuracy', 'legend': ['accuracy']})
        if(current_accuracy > best_accuracy):
            best_accuracy = current_accuracy
            torch.save(model.state_dict(), save_path)
    print('Best Accuracy of the model on the 10000 test images: %.2f %%' % best_accuracy)

train(base_model, 'base_LSTM', '../models/base_LSTM.pkl')