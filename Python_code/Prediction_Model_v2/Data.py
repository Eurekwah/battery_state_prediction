import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import torch 
from torch import nn
from torch.autograd import Variable
import os

# 获取数据名称列表
path = os.getcwd()
dir = os.listdir(path + '/modelData')
datalist = []
for i in dir:
    datalist.append(i)

datalist = sorted(datalist, key=lambda x: int(x[3:5]))

# 标准化
def nor(str):
    max = np.max(str)
    min = np.min(str)
    str = (str - min) / (max - min)
    return str, max, min

# 计算平均值
def create_average(voltage, current, speed, length):
    avgVoltage, avgCurrent, avgSpeed = [], [], []
    for i in range(len(voltage)):
        if i < length:
            avgVoltage.append(np.mean(voltage[:i+1]))
            avgCurrent.append(np.mean(current[:i+1]))
            avgSpeed.append(np.mean(speed[:i+1]))
        else:
            avgVoltage.append(np.mean(voltage[i-(length-1):i+1]))
            avgCurrent.append(np.mean(current[i-(length-1):i+1]))
            avgSpeed.append(np.mean(speed[i-(length-1):i+1]))
    return np.array(avgVoltage), np.array(avgCurrent), np.array(avgSpeed)
 
def create_newData(filename):
    data_csv = pd.read_csv(path + '/modelData/' + filename)

    #time = list(data_csv.values[:,0])
    soc = list(data_csv.values[:,1])
    speed = list(data_csv.values[:,5])
    voltage = list(data_csv.values[:,4])#.reshape(-1,1)
    current = list(data_csv.values[:,3])#.reshape(-1,1)
    temperature = list(data_csv.values[:,2])#.reshape(-1,1)

    soc, soc_max, soc_min = nor(soc)
    voltage, v_max, v_min = nor(voltage)
    current, c_max, c_min = nor(current)
    temperature, t_max, t_min = nor(temperature)
    speed, s_max, s_min = nor(speed)

    avgVoltage, avgCurrent, avgSpeed = create_average(voltage, current, speed, 50)

    data = np.vstack((soc, voltage, current, temperature, speed, avgVoltage, avgCurrent, avgSpeed)).T
    df = pd.DataFrame(data, columns=['Soc','Voltage', 'Current', 'Temperature', 'Speed', 'AvgVoltage', 'AvgCurrent', 'AvgSpeed'], dtype='double')
    df.to_csv(path + '/newModelData/' + 'soc' + filename[3:5] + '.csv', index=False)

def get_data():
    for i in datalist:
        create_newData(i)
    print('done')


