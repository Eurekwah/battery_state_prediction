import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import torch 
from torch import nn
from torch.autograd import Variable
import time as tm
import os

INPUT_SIZE  = 7
HIDDEN_SIZE = 48
OUTPUT_SIZE = 1
NUM_LAYERS  = 1
RATE        = 8e-3


# 建立模型
class LSTM_CONV(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE, num_layers=NUM_LAYERS):
        super(LSTM_CONV, self).__init__()

        self.conv = nn.Conv1d(in_channels=1, out_channels=6, kernel_size=3, stride=1)
        self.rnn = nn.LSTM(5, hidden_size, num_layers)#, dropout=0.5)
        self.reg_1 = nn.Linear(hidden_size, output_size)
        self.reg_2 = nn.Linear(6, output_size)
        
    def forward(self, x):
        x = self.conv(x)
        #torch.nn.ReLU()
        #print(x.shape)
        x, _ = self.rnn(x)
        #print(x.shape)
        s,b,h = x.shape
        x = x.view(s*b, h)
        x = self.reg_1(x)
        #torch.nn.ReLU()
        x = x.view(s, -1)
        #print(x.shape)
        x = self.reg_2(x)
        x = x.view(s,1,1)
        return x


criterion = nn.MSELoss()
def test(data_X, data_Y):

    seq = len(data_X[:,0])
    time = np.arange(0,seq/0.5,2)
    net = LSTM_CONV().cuda()
    net.load_state_dict(torch.load('net_params.pkl'))

    data_X = data_X.reshape(-1,1,7)
    #data_X = torch.from_numpy(data_X)

    data_Y = data_Y.reshape(-1)
    #data_Y = torch.from_numpy(data_Y)

    net.eval()

    var_data = Variable(data_X.cuda())
    pred_test = net(var_data)
    loss = criterion(pred_test.reshape(-1), data_Y.cuda())
    pred_test = pred_test.view(-1).cpu().data#.numpy()

    test_loss = []
    for i in range(len(data_Y)):
        test_loss.append(criterion(pred_test[i], data_Y[i]))


    print(loss)

    plt.subplot(211)
    plt.plot(time, pred_test, 'r', label='prediction')
    plt.plot(time, data_Y, 'b', label='real')
    plt.legend(loc='best')
    plt.title('result')
    plt.subplot(212)
    plt.plot(time, test_loss)
    plt.title('MSELoss')
    plt.show()
