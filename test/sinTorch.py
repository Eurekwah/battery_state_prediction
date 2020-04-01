import torch
import torch.nn as nn
import numpy as np
import matplotlib.animation
import math, random
from torch import optim
from torch.nn import functional as F
from matplotlib import pyplot as plt
torch.__version__

time_step = 10        # 步长
input_size = 1        # 输入维度
hiden_size = 64       # 隐藏单元数
epochs = 1000          # 训练次数
hiden_state = None    # 隐层状态
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

# 定义网络结构
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        #self.rnn = nn.RNN(input_size, hiden_size, 1, batch_first = True)
        self.rnn = nn.RNN(
        input_size=input_size,
        hidden_size=hiden_size, 
        num_layers=3, 
        batch_first=True,
        )
        self.out = nn.Linear(hiden_size, 1)
    
    def forward(self, x, hidden_state):
        r_out, hidden_state = self.rnn(x, hidden_state)
        
        outs =[]
        for time_step in range(r_out.size(1)):
            outs.append(self.out(r_out[:, time_step, :]))
        return torch.stack(outs, dim=1), hidden_state

rnn = RNN().to(device)

optimizer = torch.optim.Adam(rnn.parameters())
criterion = nn.MSELoss()    # 损失

# 训练
rnn.train()
plt.figure(2) 
for step in range(epochs):
    begin, end = (step%math.pi)* math.pi, ((step + 1)%math.pi) * math.pi
    steps = torch.linspace(begin, end, time_step)
    x = steps
    y = torch.sin(steps)

    #调整维度
    x = torch.unsqueeze(x, 1)
    x = torch.unsqueeze(x, 0)
    y = torch.unsqueeze(y, 1)
    y = torch.unsqueeze(y, 0)

    # 使用gpu
    x = x.to(device)
    y = y.to(device)
    prediction, hiden_state = rnn(x, hiden_state)

    hiden_state = hiden_state.data
    loss = criterion(prediction, y) 

    optimizer.zero_grad() 
    loss.backward() 
    optimizer.step() 

    if (step+1)%100==0: 
        print("EPOCHS: {},Loss:{:4f}".format(step,loss))
        plt.plot(steps, y.flatten().cpu(), 'r-')
        plt.plot(steps, prediction.data.cpu().numpy().flatten(), 'b-')
        plt.draw()
        plt.pause(0.8)
        plt.close()
plt.pause(0)   