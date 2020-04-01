import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import torch
from torch import nn
from torch.autograd import Variable

fileName = 'SOC.txt'
soc = np.loadtxt(fileName)

fileName = 'time.txt'
time = np.loadtxt(fileName)

filename = 'input.txt'
input = np.loadtxt(filename)


plt.plot(time, soc)
plt.plot(time,input[:,0])
plt.plot(time,input[:,1])
plt.plot(time,input[:,2])
plt.show()

input = input.reshape(17011,1,5)
input = torch.from_numpy(input)
input = torch.tensor(input, dtype=torch.float32)

soc = soc.reshape(17011,1,1)
soc = torch.from_numpy(soc)
soc = torch.tensor(soc, dtype=torch.float32)

class lstm_cnn(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=2):
        super(lstm_cnn, self).__init__()
        self.convolution = nn.Conv1d(in_channels=1, out_channels=6, kernel_size=3, stride=2)
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers) # rnn
        self.reg = nn.Linear(hidden_size, output_size) # 回归
        
    def forward(self, x):
        x, _ = self.rnn(x) # (seq, batch, hidden)
        s, b, h = x.shape

        x = x.view(s*b, h) # 转换成线性层的输入格式
        x = self.reg(x)

        x = x.view(s, b, -1)

        return x

net = lstm_cnn(1, 20)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
# 开始训练
for e in range(200):
    var_x = Variable(input)
    var_y = Variable(soc)
    # 前向传播
    out = net(var_x)
    loss = criterion(out, var_y)
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (e + 1) % 100 == 0: # 每 100 次输出结果
        plt.plot(out.view(-1).data, 'r', label = 'prediction')
        plt.plot(var_y.view(-1).data, 'b', label = 'real')
        plt.legend(loc = 'best')
        plt.show()
        print('Epoch: {}, Loss: {:.10f}'.format(e + 1, loss.data.item()))

'''
net = net.eval() # 转换成测试模式
data_X = data_X.reshape(-1, 1, 3)
data_X = torch.from_numpy(data_X)
var_data = Variable(data_X)
pred_test = net(var_data) # 测试集的预测结果
# 改变输出的格式
pred_test = pred_test.view(-1).data.numpy()
# 画出实际结果和预测的结果
plt.plot(pred_test, 'r', label='prediction')
plt.plot(dataset[2:], 'b', label='real')
plt.legend(loc='best')
print(torch.from_numpy(pred_test).shape)
print()
plt.show()
'''