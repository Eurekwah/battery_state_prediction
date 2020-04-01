import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import torch
from torch import nn
from torch.autograd import Variable

filename = 'input.txt'
input = np.loadtxt(filename)
input = input.reshape(17011,1,5)
input = torch.from_numpy(input)
input = torch.tensor(input, dtype=torch.float32)
print(input.shape)
conv1 = nn.Conv1d(in_channels=1, out_channels=6, kernel_size=3, stride=2)
output = conv1(input)
print(output.shape)
