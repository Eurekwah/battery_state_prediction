import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import torch
from torch import nn
from torch.autograd import Variable

def avg(num):
    nsum = 0
    for i in range(len(num)):
        nsum += num[i]
    return nsum / len(num)

fileName = 'I.txt'
I = np.loadtxt(fileName)

fileName = 'SOC_Timeseries.txt'
soc_timeseries = np.loadtxt(fileName)

soc = soc_timeseries[:,1]
print(soc.shape)

doc = open('SOC.txt', 'r+')
for i in range(17011):
    print(soc[i], file=doc)
doc.close()

fileName = 'temperature.txt'
temperature = np.loadtxt(fileName)

time = soc_timeseries[:,0]
doc = open('time.txt', 'r+')
for i in range(17011):
    print(time[i], file=doc)
doc.close()


fileName = 'Uoc.txt'
Uoc = np.loadtxt(fileName)
print(Uoc.shape)
'''
doc = open('input.txt','r+')

Ia = []
Ua = []

for i in range(17011):
    temp=[]
    print(I[i], file=doc, end=" ")
    print(Uoc[i], file=doc, end=" ")
    print(temperature[i], file=doc, end=" ")
    if i >= 50:  
        del Ia[0]
        del Ua[0]
    Ia.append(I[i])
    print(avg(Ia), file=doc, end=" ")
    Ua.append(Uoc[i])
    print(avg(Ua), file=doc)

    



doc.close()
'''

