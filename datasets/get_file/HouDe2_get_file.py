# Fs = 1e4
import os
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm
from datasets.datasets_utils.generalfunciton import sig_process, add_noise
import pandas as pd
import random

import matplotlib.pyplot as plt
import utils.general_func as sigp
from scipy.fft import fft

# 随机数设置
seed = 999
np.random.seed(seed)  # seed是一个固定的整数即可
random.seed(seed)

fault_name = ['H', 'W', 'P', 'F']
n_name = ['1800I02', '1800I05', '2400I02', '2400I05']

sig_channel = 20 - 1  # 14 | 20
lab_name = ['H', 'W', 'P', 'F']


# generate Training Dataset and Testing Dataset
def get_files(root, signal_size=1024, SNR=None, signal_type='time', downsample=1):
    '''
    '''

    data, lab_f, lab_n, time_index = [], [], [], []
    for i in tqdm(range(len(fault_name))):
        for j in range(len(n_name)):
            path = os.path.join(root, fault_name[i] + 'RPM' + n_name[j] + '.mat')
            data1, ti1 = data_load(path, signal_type=signal_type, signal_size=int(signal_size), SNR=SNR,
                                   downsample=downsample)
            lab_f.append(np.ones(len(ti1)) * i)
            lab_n.append(np.ones(len(ti1)) * j)
            data.append(data1)
            time_index.append(ti1)
    return [data, lab_f, lab_n, time_index], lab_name


def data_load(filename, signal_type, signal_size, SNR=None, downsample=1):
    '''
    This function is mainly used to generate test data and training data.
    filename:Data location
    axisname:Select which channel's data,---->"_DE_time","_FE_time","_BA_time"
    '''
    func = getattr(sig_process, signal_type)
    fl = loadmat(filename)['Datas']
    fl = fl[::downsample, sig_channel]
    if SNR is not None:
        fl = add_noise(fl, SNR)
    step = signal_size
    if signal_type == 'fft':
        signal_size *= 2
        step = step // 2
    data = []
    ti = []
    start, end = 0, signal_size
    while end <= fl.shape[0]:
        x = func(fl[start:end])
        if len(x.shape) == 1:
            x = x.reshape([1] + list(x.shape))  # get channel dimension
        data.append(x)  # data[batch,channel/1,signal_size]
        ti.append(len(data))
        start += step
        end = end + step
    return np.array(data), np.array(ti)


if __name__ == '__main__':
    root = r"E:\6-数据集\0-机械故障诊断数据集\9-厚德平行轴\振动响应\数据导出\斜齿轮"
    a = get_files(root)
