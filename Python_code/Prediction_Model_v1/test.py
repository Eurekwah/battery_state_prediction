import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import torch 
from torch import nn
from torch.autograd import Variable
import os


'''
path = os.getcwd()
dir = os.listdir(path + '/ModelData')
datalist = []
for i in dir:
    datalist.append(i)

datalist = sorted(datalist, key=lambda x: int(x[3:5]))
def tran(filename):
    a = pd.read_excel(path + '/ModelData/'+ filename, index_col=0)
    a.to_csv(path + '/modelData/'+ filename[0:5] + '.csv')

for i in datalist:
    tran(i)
'''
def logic(index):
    if index == 0:
        return False
    elif (index-1)%10 == 0:
        return False
    else:
        return True
path = os.getcwd()
dir = os.listdir(path + '/newModelData')
datalist = []
for i in dir:
    datalist.append(i)

datalist = sorted(datalist, key=lambda x: int(x[3:5]))
def select(filename):
    a = pd.read_csv(path + '/newModelData/'+ filename, skiprows=lambda x: logic(x), index_col=0)
    a.to_csv(path + '/smallNewModelData/'+ filename)
for i in datalist:
    select(i)

