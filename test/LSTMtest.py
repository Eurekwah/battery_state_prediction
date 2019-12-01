import numpy as np
import pandas as pd
import matplotlib.pyplot as plt#matplotlib inline
import math
import torch
from torch import nn
from torch.autograd import Variable

lstm = nn.LSTM(10, 20, 2)
input = Variable(torch.randn(5, 3, 10))
print(input)
h0 = Variable(torch.randn(2, 3, 20))
c0 = Variable(torch.randn(2, 3, 20))
output, hn = lstm(input, (h0, c0))
print(output)

#print(output)  