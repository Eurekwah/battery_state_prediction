import numpy as np
import pandas as pd
import matplotlib.pyplot as plt#matplotlib inline
import math
import torch
from torch import nn
from torch.autograd import Variable

#生成数据集
dataset = []

for point in torch.arange(0, 4 * math.pi, 0.1):
    point = math.sin(point)
    dataset.append(point)
dataset = torch.tensor(dataset, dtype=torch.float32)
plt.plot(torch.arange(0, 4 * math.pi, 0.1),dataset)
print(dataset.size())