import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import torch 
from torch import nn
from torch.autograd import Variable
import time as tm
import os
import Soc_test_v2 as test
import math


INPUT_SIZE  = 7
HIDDEN_SIZE = 48
OUTPUT_SIZE = 1
NUM_LAYERS  = 1
RATE        = 5e-3
EPOCH       = 1500

path = os.getcwd()
#dir = os.listdir(path + '/smallNewModelData')
dir = os.listdir(path + '/smallNewModelData')

def get_value(filename):
    data_csv = pd.read_csv(path + '/smallNewModelData/' + filename)#, index_col='Time')
    # 去掉NA 提取数值
    data_csv = data_csv.dropna() 
    dataset = data_csv.values
    dataset = np.array(dataset.astype('float32'))
    return dataset

def get_pre():
    datalist = []
    for i in dir:
        datalist.append(i)
    datalist = sorted(datalist, key=lambda x: int(x[3:5]))

    pre_data = []
    for i in datalist:
        pre_data.append(get_value(i))
    print(datalist[-1])
    test_data = pre_data[-1]
    del pre_data[-1]
    return pre_data, test_data



# 获取时序步长

#seq = len(dataset[:,0])
#time = np.arange(0,seq/10,0.1)
# 在处理数据是进行标准化，因此训练程序中无需进行标准化

# 获得训练数据

def get_Data(dataset):
    data_X = dataset[:,1:]
    data_Y = dataset[:,0].reshape(-1,1)
    data_X = torch.from_numpy(data_X.reshape(-1, 1, 7))
    data_Y = torch.from_numpy(data_Y.reshape(-1, 1, 1))
    return data_X, data_Y

#print(train_x)
#pause

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
        torch.nn.ReLU()
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

def rest_time(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    print ('rest time:', "%02d:%02d:%02d" % (h, m, s))

# 训练
net = LSTM_CONV().cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr = RATE, betas=(0.9, 0.999), eps=1e-08)
l = []
def train(pre_data):
    print('begin')
    time_start=tm.time()
    for e in range(EPOCH):
        for i in pre_data:
            train_x, train_y = get_Data(i)
            var_x = Variable(train_x.cuda())
            var_y = Variable(train_y.cuda())
            out = net(var_x)
            loss = criterion(out, var_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            

        l.append(loss)
        if (e + 1) % 50 == 0:    
            time_end=tm.time()    
            rest = (time_end - time_start) * (EPOCH - e) / 50
            rest_time(rest)
            time_start=tm.time()
            print('Epoch:{}, Loss:{:.5f}'.format(e+1, math.sqrt(loss.item())))
    torch.save(net.state_dict(), 'net_params.pkl')    
    print('end')
    plt.plot(l)
    plt.show()
'''
def train(pre_data):
    
    train_step(train_x, train_y)
    torch.save(net.state_dict(), 'net_params.pkl')
    plt.plot(l)
    plt.show()
'''