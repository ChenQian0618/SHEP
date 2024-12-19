import numpy as np
import scipy.signal as signal
import scipy.fft as fft
import matplotlib.pyplot as plt
import random
import matplotlib

matplotlib.use("Qt5Agg")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)


def plot_signals(signals, title=None):
    fig, ax = plt.subplots(nrows=signals.shape[0], ncols=1, sharex=True)
    if title:
        ax[0].set_title(title)
    for i in range(signals.shape[0]):
        ax[i].plot(signals[i])
        ax[i].set_ylabel(f"sub {i + 1}")
    plt.xlim([0, 50])


def generate_single_cosine_signal(L, A, f, phi):
    t = np.arange(0, L, 1)
    return A * np.cos(2 * np.pi * 0.5 * f * t + phi)


def generate_periodic_signals(L=1024, N_sample: int = 10, f_fault=(0.3,),
                              Alim=(0.5, 1), f_lim=(0.05, 0.95), phi_lim=(0, 2 * np.pi),
                              N_sub=4, seed=999, verbose=False):
    seed_everything(seed)
    signals = np.zeros((N_sample, L, N_sub))
    # Generate signal parameters
    A = np.random.uniform(Alim[0], Alim[1], (N_sample, N_sub))
    f = np.random.uniform(f_lim[0], f_lim[1], (N_sample, N_sub))
    phi = np.random.uniform(phi_lim[0], phi_lim[1], (N_sample, N_sub))
    if f_fault is not None:
        for i, item in enumerate(f_fault):
            f[:, i] = item
    # Generate signals
    for i in range(N_sample):
        for j in range(N_sub):
            signal = generate_single_cosine_signal(L, A[i, j], f[i, j], phi[i, j])
            signals[i, :, j] = signal
    if verbose:
        return signals.sum(-1).astype(np.float32), signals  # signals, subsignals
    else:
        return signals.sum(-1).astype(np.float32)


def generate_CS2_signals(L=1024, N_sample=10, f_fault=(0.015,),
                         Alim=(0.5, 1), f_lim=(0.0025, 0.0475), phi_lim=(0, 2 * np.pi),
                         N_sub=4, seed=999):
    # cosine part
    cos_signals = generate_periodic_signals(L=L, N_sample=N_sample, f_fault=f_fault,
                                                   Alim=Alim, f_lim=f_lim, phi_lim=phi_lim,
                                                   N_sub=N_sub, seed=seed)
    # Gaussian noise
    seed_everything(seed)
    noise_c_origin = np.random.normal(0, 1, (N_sample, L))
    noise_z = np.random.normal(0, 1, (N_sample, L))
    # filter
    b, a = signal.butter(4, [0.5, 0.7], 'bandpass', output='ba')
    noise_c = signal.filtfilt(b, a, noise_c_origin, axis=-1)
    # combine
    temp = cos_signals - cos_signals.min(-1).reshape([-1, 1]) + 0.1
    signals = temp * noise_c + noise_z

    # show filter process
    # f = np.arange(L)/L*2
    # plt.plot(f,abs(fft.fft(noise_c_origin[0]))/L,f,abs(fft.fft(noise_c[0]))/L)
    # plt.xlim([0, 1])

    return signals.astype(np.float32)


def generate_periodic_signals_all(*args, **kwargs):
    '''
    对generate_periodic_signals进行封装，生成两组信号
    :param args:
    :param kwargs:
    :return:
    '''
    signals_h = generate_periodic_signals(*args, f_fault=None, **kwargs)
    signals_f = generate_periodic_signals(*args, f_fault=(0.3,), **kwargs)
    signals = np.concatenate([signals_h, signals_f], axis=0)
    labels = np.array([np.ones(item) * i for i, item in enumerate([signals_h.shape[0], signals_f.shape[0]])]).reshape(-1)
    labelname = ['health', 'fault']
    return signals, labels, labelname


def generate_periodic_signals_all_V2(*args, **kwargs):  # 2021.10.20
    '''
    对generate_periodic_signals进行封装，生成三组信号，
    三组信号均保留0.2的固定成分，第二组信号增加0.4的故障成分，第三组信号增加0.6的故障成分
    :param args:
    :param kwargs:
    :return:
    '''
    signals_h = generate_periodic_signals(*args, f_fault=(0.2,), **kwargs)
    signals_1 = generate_periodic_signals(*args, f_fault=(0.2, 0.4,), **kwargs)
    signals_2 = generate_periodic_signals(*args, f_fault=(0.2, 0.6,), **kwargs)
    signals = np.concatenate([signals_h, signals_1, signals_2], axis=0)
    lens = [item.shape[0] for item in (signals_h, signals_1, signals_2)]
    labels = np.array([np.ones(item) * i for i, item in enumerate(lens)]).reshape(-1)
    labelname = ['health', 'fault_1', 'fault_2']
    return signals, labels, labelname


def generate_CS2_signals_all(*args, **kwargs):
    '''
    对generate_CS2_signals进行封装，生成两组信号
    :param args:
    :param kwargs:
    :return:
    '''
    signals_h, labels_h = generate_CS2_signals(*args, f_fault=None, **kwargs)
    signals_f, labels_f = generate_CS2_signals(*args, f_fault=(0.015,), **kwargs)
    signals, labels = np.concatenate([signals_h, signals_f], axis=0), np.array(labels_h + labels_f)
    labelname = ['health', 'fault']
    return signals, labels, labelname


def test_generate_periodic_signals():
    result, labels, signals = generate_periodic_signals(f_fault=(0.3,), verbose=True)
    for i in range(3):
        temp = signals[i].swapaxes(0, 1)
        plot_signals(np.vstack([temp, temp.sum(0)[np.newaxis, :]]), title=f"Signal {i + 1}")
    plt.show()


def test_generate_CS2_signals():
    signals = generate_CS2_signals(f_fault=(0.015,), L=int(1e4))
    fig, ax = plt.subplots(nrows=3, ncols=1)
    L = signals.shape[1]
    # ax 0 = time
    ax[0].plot(np.arange(L), signals[0])
    ax[0].set_xlim([0, min(L, 5e2)])
    ax[0].set_ylabel("time")
    # ax 1 = FFT
    f = np.arange(L) / L * 2
    fft_signal = fft.fft(signals[0])
    ax[1].plot(f, abs(fft_signal) / L)
    ax[1].set_xlim([0, 1])
    ax[1].set_ylabel("FFT")
    # ax 2 = envolope
    b, a = signal.butter(4, [0.5, 0.7], 'bandpass', output='ba')
    temp = signal.filtfilt(b, a, signals[0], axis=-1)
    envelope = abs(signal.hilbert(temp))
    ax[2].plot(f, abs(fft.fft(envelope - envelope.mean())) / L)
    ax[2].set_xlim([0, 0.05])
    ax[2].set_ylabel("envolpe-FFT")


if __name__ == '__main__':
    test_generate_periodic_signals()
    test_generate_CS2_signals()
    generate_periodic_signals_all()
    print(1)
