import random

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.fft as fft
import scipy.signal as signal

from utils.plot_func import setdefault
from analysis.ana_utils.transform import trans_STFT, trans_De_angle, trans_FFT, trans_Series

setdefault()

matplotlib.use("Qt5Agg")

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)

def plot_signals(signals, title=None,xlim=None): # 绘制时域
    fig, ax = plt.subplots(nrows=signals.shape[0], ncols=1, sharex=True)
    if title:
        ax[0].set_title(title)
    for i in range(signals.shape[0]):
        ax[i].plot(signals[i])
        ax[i].set_ylabel(f"sub {i + 1}")
    if xlim:
        plt.xlim(xlim)
    fig.tight_layout()

def plot_signals_cycle(signals, title=None,xlim=None): # 绘制循环谱

    trans_stft = trans_STFT(10000,10, win_len=50,win_g=10,mfft=500, keep_N=True)  # keep_N=True
    trans_de_angle = trans_De_angle(de_mean=True)  # de_mean=True
    trans_fft = trans_FFT(one_side=True)  # one_side=True
    trans_series = trans_Series(trans_stft, trans_de_angle, trans_fft)
    fig, ax = plt.subplots(nrows=signals.shape[0], ncols=1, sharex=True)
    if type(ax) is not list:
        ax = [ax]
    if title:
        ax[0].set_title(title)
    for i in range(signals.shape[0]):
        temp = trans_series.forward(signals[i])
        ax[i].pcolormesh(abs(temp[-1]),cmap='viridis', shading='gouraud')
        ax[i].set_ylabel(f"sub {i + 1}")
    if xlim:
        plt.xlim(xlim)
    fig.tight_layout()

def generate_simu_signals(L=2000, f_carrier=0.3,f_fault_delta=100, phi=0, RF=None, beta = 0.04):
    t = np.arange(L)
    ImpulseResponse = np.exp(-beta * t) * np.sin(np.pi * f_carrier * t)  # 脉冲函数 2.3/beta
    start = int(phi / 2 / np.pi * f_fault_delta)
    impulse = np.zeros(L+f_fault_delta)
    for i in range(len(impulse)):
        if i % f_fault_delta == start:
            impulse[i] = 1
    impulse_result = signal.convolve(impulse, ImpulseResponse[:np.where(beta * t > 10)[0][0]], mode='full')[f_fault_delta:len(t)+f_fault_delta]
    amp = 1 + 0.2 * np.cos(np.pi * RF * t) if RF else 1
    result = amp * impulse_result
    return result


def generate_simu_signals_N(L=2000, N_sample=10, f_carrier=(0.3,), f_fault=(0.01,),
                         Alim=(0.8, 1), f_carrier_lim=(0.2, 0.8),
                          f_fault_lim=(0.004, 0.04), phi_lim=(0, 2 * np.pi),
                         N_sub=2, SNR=5, seed=None, verbose=False):
    '''
    生成N_sample个信号，每个信号包含N_sub个子信号
    :param L: 信号长度
    :param N_sample: 信号数量
    :param f_carrier: 载波频率（f/fs*2）,指定<=N_sub个子信号的载波频率，剩余子信号则随机生成
    :param f_fault: 故障频率（f/fs*2）,指定<=N_sub个子信号的载波频率，剩余子信号则随机生成
    :param Alim: 信号幅度范围
    :param f_carrier_lim: 载波频率范围
    :param f_fault_lim: 故障频率范围
    :param phi_lim: 相位范围
    :param N_sub: 单个信号中的子信号数量
    :param SNR: 信噪比
    :param seed: 随机种子
    :param verbose:  是否返回子信号信息，即：signals, subsignals, info:[N_sample,k,N_sub]
    :return:
    '''
    if seed:
        seed_everything(seed)
    signals = np.zeros((N_sample, N_sub, L),dtype=np.float32)
    # Generate signal parameters
    A = np.random.uniform(Alim[0], Alim[1], (N_sample, N_sub))
    f_c = np.random.uniform(f_carrier_lim[0], f_carrier_lim[1], (N_sample, N_sub))
    f_f = np.random.uniform(f_fault_lim[0], f_fault_lim[1], (N_sample, N_sub))
    phi = np.random.uniform(phi_lim[0], phi_lim[1], (N_sample, N_sub))
    if f_carrier is not None:
        for i, item in enumerate(f_carrier):
            f_c[:, i] = item
    if f_fault is not None:
        for i, item in enumerate(f_fault):
            f_f[:, i] = item
    f_f_delta = (2/f_f).astype(int) # translate frequency to delta delay (f_fault must be divide exactly by 2)
    # Generate signals
    for i in range(N_sample):
        for j in range(N_sub):
            temp= generate_simu_signals(L=L, f_carrier=f_c[i,j], f_fault_delta=f_f_delta[i,j],
                                        phi=phi[i,j], RF=0.002, beta = 0.04)
            signals[i, j, :] = A[i, j] * temp
    result = signals.sum(-2).astype(np.float32)
    # add noise
    if SNR is not None:
        noise = np.random.randn(*result.shape) # 生成噪声 [N_sample,L]
        result += noise * np.sqrt(np.var(result,axis=-1) / np.power(10, (SNR / 20)))[...,np.newaxis]
    if verbose:
        info = np.array([f_c, 2/f_f_delta, A, phi]).swapaxes(0,1)
        return result, signals, info  # signals, subsignals, info:[N_sample,k,N_sub]
    else:
        return result

def generate_simu_signals_all(*args, **kwargs):
    '''
    对generate_simu_signals进行封装，生成两组信号
    :param args:
    :param kwargs:
    :return:
    '''
    print('Generating signals ...(Health)')
    signals_h, _, info_h = generate_simu_signals_N(*args, f_carrier=(0.3,), f_fault=(0.01,),verbose=True, **kwargs)
    print('Generating signals ...(Fault-1)')
    signals_f1, _, info_f1 = generate_simu_signals_N(*args, f_carrier=(0.3,0.5,), f_fault=(0.01,0.02,),verbose=True, **kwargs)
    print('Generating signals ...(Fault-2)')
    signals_f2, _, info_f2 = generate_simu_signals_N(*args, f_carrier=(0.3,0.7,), f_fault=(0.01,0.025,),verbose=True, **kwargs)
    signals = np.concatenate([signals_h, signals_f1, signals_f2], axis=0)
    infos = np.concatenate([info_h, info_f1, info_f2], axis=0)
    labels = np.concatenate([np.zeros(len(signals_h)), np.ones(len(signals_f1)), 2*np.ones(len(signals_f2))], axis=0).astype(np.int16)
    labelname = ['Health', 'Fault-1','Fault-2']
    return signals, labels, labelname, infos

def test_generate_simu_signals_N():
    signals,verb,info = generate_simu_signals_N(L=2000,N_sample=3, verbose=True)
    for i in range(3):
        temp = verb[i]
        plot_signals(np.vstack([temp, signals[i][np.newaxis, :]]), title=f"Signal {i + 1}")
        plot_signals_cycle(np.vstack([temp, signals[i][np.newaxis, :]]), title=f"Signal {i + 1}")
    plt.show()


def test_generate_simu_signals_all():
    trans_stft = trans_STFT(10000,10, win_len=50,win_g=10,mfft=500, keep_N=True)  # keep_N=True
    trans_de_angle = trans_De_angle(de_mean=True)  # de_mean=True
    trans_fft = trans_FFT(one_side=True)  # one_side=True
    trans_series = trans_Series(trans_stft, trans_de_angle, trans_fft)
    def temp_plot(signals):
        n = len(signals)
        nonlocal trans_series
        fig, ax = plt.subplots(nrows=3, ncols=n,figsize=(5*n/2.54,6/2.54),dpi=300)
        for i in range(n):
            signal = signals[i]
            # 时域
            ax[0][i].plot(signal)
            ax[0][i].margins(x=0)
            # 频域
            data = np.abs(fft.fft(signal))
            f = np.arange(len(data)) / len(data) * 2
            ax[1][i].plot(f[:len(data)//2],data[:len(data)//2])
            ax[1][i].margins(x=0)
            # 循环谱
            temp = trans_series.forward(signals[i])
            ax[2][i].pcolormesh(abs(temp[-1]), cmap='viridis', shading='gouraud')
        fig.tight_layout()

    signals, labels, labelname, info = generate_simu_signals_all(L=2000, N_sample=1, SNR=-10)
    temp_plot(signals)
    plt.show()


if __name__ == '__main__':
    # test_generate_simu_signals_N()
    # a,b,c,d = generate_simu_signals_all()
    test_generate_simu_signals_all()
    print(1)
