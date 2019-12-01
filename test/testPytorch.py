from __future__ import print_function
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation
from torch import autograd
from torch.autograd import Variable, Function

torch.__version__
a = torch.linspace(0, 10, 21)
a = list(a)
b = a[-3:]
b = np.array(b)
print(a)
print(b)
b = torch.from_numpy(b)
print(b)