import os
from collections import Counter

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

from analysis.ana_utils import transform
from utils.plot_func import setdefault,get_plot_params,color_modify_HSV

setdefault()

stft_params = {'SampFreq': 10000, 'hop': 10, 'win_len': 50, 'keep_N': False}
trans_stft = transform.trans_STFT(stft_params['SampFreq'], stft_params['hop'],
                                  win_len=stft_params['win_len'], keep_N=stft_params['keep_N'])


def find_all_index_N(label, Target_number=0, squeeze=True):
    '''
    :param label: (N,) int,ndarray
    :param Target_number: int
    :param squeeze: boll
    :return:
    '''
    if Target_number:
        Target_number = min(min(Counter(label).values()), Target_number)
    else:
        Target_number = min(Counter(label).values())
    Index = np.array([np.where(label == temp_l)[0][:Target_number] for temp_l in set(label)])
    if squeeze:
        return Index.reshape(-1)
    else:
        return Index  # (c,n)


def func_unpatch_base(data, shap_value=None, trans_series=None, back_N=0):
    out_data = trans_series.backward(data, N=back_N)
    if shap_value is not None:
        # [N,Feature,c] float -> [c,N,Feature] object
        new_shap_value = ShapValue_reshape(shap_value, data)

        temp = new_shap_value.reshape(-1, new_shap_value.shape[-1])

        # [c*N,Feature] object ->k * [c*N,Feature] float, k is the number of [phase, mean, values]
        out_shap_value = trans_series.backward(temp, N=back_N)

        if type(out_shap_value) == tuple:  # for special case: [phase, mean, values]
            # k * [c,N,Feature] object -> k * [c,N,d1,d2...] float
            out_shap_value = tuple(item.reshape(new_shap_value.shape[:2] + out_shap_value[i].shape[1:]) for i, item in
                                   enumerate(out_shap_value))
        else:  # for common case: values
            # [c,N,Feature] object -> [c,N,d1,d2...] float
            out_shap_value = out_shap_value.reshape(new_shap_value.shape[:2] + out_shap_value.shape[1:])
        return out_data, out_shap_value
    else:
        return out_data


def ShapValue_reshape(shap_values, data):
    '''
    :param shap_values: [N,Feature,y] float
    :param data: [N,Feature] object
    :return: [y,N,Feature] object
    '''
    shape = shap_values.shape
    out = np.zeros(shape, dtype=object)
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                out[i, j, k] = np.ones_like(abs(data[i, j])) * shap_values[i, j, k]
    out = np.moveaxis(out, -1, 0)
    return out


# ------------------------------    time   ------------------------------
def func_trans_time(patch_mode: str = '1'):
    name = 'time'
    patch_mp = dict(zip(['1', '2', '3', '4', '5'],
                        [(3,), (6,), (12,), (24,), (48,)]))  # (1,) | (3,)
    if patch_mode not in patch_mp:
        raise ValueError(f'patch_mode should be in {patch_mp.keys()}')
    p = patch_mp[str(patch_mode)]
    trans_series = transform.trans_Series(
        [transform.trans_Patch(p=p),
         transform.trans_Object_Combine()])
    func_Z = trans_series.forward
    func_Z_inv = trans_series.backward

    func_unpatch = lambda data, shap_value: func_unpatch_base(data, shap_value=shap_value, trans_series=trans_series,
                                                              back_N=2)

    return func_Z, func_Z_inv, func_unpatch, {'name': name, 'trans_series': trans_series}


# ------------------------------ frequency ------------------------------
def func_trans_frequency(patch_mode: str = '1'):
    name = 'frequency'
    patch_mp = dict(zip(['1', '2', '3', '4', '5'],
                        [(3,), (6,), (12,), (24,), (48,)]))  # (1,) | (3,)
    if patch_mode not in patch_mp:
        raise ValueError(f'patch_mode should be in {patch_mp.keys()}')
    p = patch_mp[str(patch_mode)]
    trans_series = transform.trans_Series(
        [transform.trans_FFT(one_side=True),
         transform.trans_Patch(p=p),
         transform.trans_Object_Combine()])
    func_Z = trans_series.forward
    func_Z_inv = trans_series.backward

    func_unpatch = lambda data, shap_value: func_unpatch_base(data, shap_value=shap_value, trans_series=trans_series,
                                                              back_N=2)

    return func_Z, func_Z_inv, func_unpatch, {'name': name, 'trans_series': trans_series}


def func_trans_frequency_v2(patch_mode: str = '1'):  # 2024/12/14 增加了De_angle
    name = 'frequency'
    patch_mp = dict(zip(['0','1', '2', '3', '4', '5'],
                        [(1,), (3,), (6,), (12,), (24,), (48,)]))  # (1,) | (3,)
    if patch_mode not in patch_mp:
        raise ValueError(f'patch_mode should be in {patch_mp.keys()}')
    p = patch_mp[str(patch_mode)]
    trans_series = transform.trans_Series(
        [transform.trans_FFT(one_side=True),
         transform.trans_De_angle(),
         transform.trans_Patch(p=p),
         transform.trans_Object_Combine()])
    func_Z = trans_series.forward
    func_Z_inv = trans_series.backward

    def func_unpatch(data, shap_value):
        out_data, out_value = func_unpatch_base(data, shap_value=shap_value, trans_series=trans_series, back_N=2)
        pick = lambda x: x[-1] if type(x) == tuple else x
        return pick(out_data), pick(out_value)

    return func_Z, func_Z_inv, func_unpatch, {'name': name, 'trans_series': trans_series}


# ------------------------------ envelope ------------------------------
def func_trans_envelope(patch_mode: str = '1'):
    name = 'envelope'
    patch_mp = dict(zip(['1', '2', '3', '4', '5'],
                        [(1,), (2,), (4,), (8,), (16,), ]))  # (1,) | (3,)
    if patch_mode not in patch_mp:
        raise ValueError(f'patch_mode should be in {patch_mp.keys()}')
    p = patch_mp[str(patch_mode)]
    trans_series = transform.trans_Series(
        [transform.trans_Hilbert(),
         transform.trans_De_angle(de_mean=True),
         transform.trans_FFT(one_side=True),
         transform.trans_Apart(n=0.12),
         transform.trans_Patch(p=p),
         transform.trans_Object_Combine()])
    func_Z = trans_series.forward
    func_Z_inv = trans_series.backward

    def func_unpatch(data, shap_value):
        out_data, out_value = func_unpatch_base(data, shap_value=shap_value, trans_series=trans_series, back_N=2)
        pick = lambda x: x[-1] if type(x) == tuple else x
        return pick(out_data), pick(out_value)

    return func_Z, func_Z_inv, func_unpatch, {'name': name, 'trans_series': trans_series}


def func_trans_envelope_v2(patch_mode: str = '1'):
    name = 'envelope'
    patch_mp = dict(zip(['0', '1', '2', '3', '4', '5'],
                        [(1,), (1,), (2,), (4,), (8,), (16,), ]))  # (1,) | (3,)
    if patch_mode not in patch_mp:
        raise ValueError(f'patch_mode should be in {patch_mp.keys()}')
    p = patch_mp[str(patch_mode)]
    trans_series = transform.trans_Series(
        [transform.trans_Hilbert(),
         transform.trans_De_angle(de_mean=True),
         transform.trans_FFT(one_side=True),
         transform.trans_Apart(n=0.12),
         transform.trans_De_angle(),
         transform.trans_Patch(p=p),
         transform.trans_Object_Combine()])
    func_Z = trans_series.forward
    func_Z_inv = trans_series.backward

    def func_unpatch(data, shap_value):
        out_data, out_value = func_unpatch_base(data, shap_value=shap_value, trans_series=trans_series, back_N=2)
        pick = lambda x: x[-1] if type(x) == tuple else x
        return pick(out_data), pick(out_value)

    return func_Z, func_Z_inv, func_unpatch, {'name': name, 'trans_series': trans_series}


# ------------------------------    STFT   ------------------------------
def func_trans_STFT(patch_mode: str = '1'):
    patch_mp = dict(zip(['1', '2', '3', '4', '5'],
                        [(1, 5), (2, 5), (2, 10), (4, 10), (4, 20)]))  # (1,5) | (1,10)
    if patch_mode not in patch_mp:
        raise ValueError(f'patch_mode should be in {patch_mp.keys()}')
    p = patch_mp[str(patch_mode)]

    name = 'STFT'
    trans_series = transform.trans_Series(
        [trans_stft,
         # transform.trans_De_angle(de_mean=False), # 2024/12/14 de_mean=False
         transform.trans_Patch(p=p),
         transform.trans_Object_Combine()])
    func_Z = trans_series.forward
    func_Z_inv = trans_series.backward

    def func_unpatch(data, shap_value):
        out_data, out_value = func_unpatch_base(data, shap_value=shap_value, trans_series=trans_series, back_N=2)
        pick = lambda x: x[-1] if type(x) == tuple else x
        return pick(out_data), pick(out_value)

    return func_Z, func_Z_inv, func_unpatch, {'name': name, 'trans_series': trans_series}


def func_trans_STFT_v2(patch_mode: str = '1'):
    patch_mp = dict(zip(['0', '1', '2', '3', '4', '5'],
                        [(1,2), (1, 5), (2, 5), (2, 10), (4, 10), (4, 20)]))  # (1,5) | (1,10)
    if patch_mode not in patch_mp:
        raise ValueError(f'patch_mode should be in {patch_mp.keys()}')
    p = patch_mp[str(patch_mode)]

    name = 'STFT'
    trans_series = transform.trans_Series(
        [trans_stft,
         transform.trans_De_angle(de_mean=False),  # 2024/12/14 de_mean=False
         transform.trans_Patch(p=p),
         transform.trans_Object_Combine()])
    func_Z = trans_series.forward
    func_Z_inv = trans_series.backward

    def func_unpatch(data, shap_value):
        out_data, out_value = func_unpatch_base(data, shap_value=shap_value, trans_series=trans_series, back_N=2)
        pick = lambda x: x[-1] if type(x) == tuple else x
        return pick(out_data), pick(out_value)

    return func_Z, func_Z_inv, func_unpatch, {'name': name, 'trans_series': trans_series}


def func_trans_CS(patch_mode: str = '1'):  # 2024/11/23 STFT^2 -> CSCoh | de_mean=False
    '''
    return:
    func_Z: z=func_Z(x), the transformation function
    func_Z_inv: x=func_Z_inv(z), the inverse transformation function
    func_unpatch: data [,shape] = func_unpatch(data [,shape]), unpatch the patched data to the original domain
    trans_dict: {'name':.,'trans_series':.}
    '''
    patch_mp = dict(zip(['1', '2', '3', '4', '5'],
                        [(1, 3), (2, 3), (2, 6), (4, 6), (4, 12), ]))  # (1,3)
    if patch_mode not in patch_mp:
        raise ValueError(f'patch_mode should be in {patch_mp.keys()}')
    p = patch_mp[str(patch_mode)]

    name = 'CS'
    trans_series = transform.trans_Series(
        [trans_stft,
         transform.trans_De_angle_pow(de_mean=False),
         transform.trans_FFT(one_side=True),
         transform.trans_Patch(p=p),
         transform.trans_Object_Combine()])

    func_Z = trans_series.forward
    func_Z_inv = trans_series.backward

    def func_unpatch(data, shap_value):
        out_data, out_value = func_unpatch_base(data, shap_value=shap_value, trans_series=trans_series, back_N=2)
        pick = lambda x: x[-1] if type(x) == tuple else x
        return pick(out_data), pick(out_value)

    return func_Z, func_Z_inv, func_unpatch, {'name': name, 'trans_series': trans_series}


def func_trans_CS_v2(patch_mode: str = '1'):  # 2024/11/23 STFT^2 -> CSCoh | de_mean=False
    '''
    return:
    func_Z: z=func_Z(x), the transformation function
    func_Z_inv: x=func_Z_inv(z), the inverse transformation function
    func_unpatch: data [,shape] = func_unpatch(data [,shape]), unpatch the patched data to the original domain
    trans_dict: {'name':.,'trans_series':.}
    '''
    patch_mp = dict(zip(['0', '1', '2', '3', '4', '5'],
                        [(1,1), (1, 3), (2, 3), (2, 6), (4, 6), (4, 12), ]))  # (1,3)
    if patch_mode not in patch_mp:
        raise ValueError(f'patch_mode should be in {patch_mp.keys()}')
    p = patch_mp[str(patch_mode)]

    name = 'CS'
    trans_series = transform.trans_Series(
        [trans_stft,
         transform.trans_De_angle_pow(de_mean=False),
         transform.trans_FFT(one_side=True),
         transform.trans_De_angle(de_mean=False),
         transform.trans_Patch(p=p),
         transform.trans_Object_Combine()])

    func_Z = trans_series.forward
    func_Z_inv = trans_series.backward

    def func_unpatch(data, shap_value):
        out_data, out_value = func_unpatch_base(data, shap_value=shap_value, trans_series=trans_series, back_N=2)
        pick = lambda x: x[-1] if type(x) == tuple else x
        return pick(out_data), pick(out_value)

    return func_Z, func_Z_inv, func_unpatch, {'name': name, 'trans_series': trans_series}


def attr_visualization(savedir, mode, Fs, signals, data, value,
                       label_predicts, label_name, patch_mode='1', dpi=300,
                       **kwargs):
    '''
    :param savedir: str
    :param mode: str
    :param Fs: int
    :param signals: (N,t) float
    :param data: (N,d1,d2,...) object
    :param value: (K,N,d1,d2,...) object
    :param label_predicts: [(N,), (N,K)] float
    :param label_name: list
    :param patch_mode: str
    :param dpi: int
    :param kwargs: dict: STFTs_params, attr_params
    :return:
    '''
    mode = mode.split('_')[0]
    if not os.path.exists(os.path.split(savedir)[0]):
        os.makedirs(os.path.split(savedir)[0])

    # 1) prepare value xyz
    if mode in ['CAM', 'time', 'frequency', 'envelope']:  # 1D
        dimension = 1  # 1D
        if mode in ['CAM', 'time']:  # time
            x, xlabel = np.arange(data.shape[-1]) / Fs, 'Time $t$ (s)'
        else:  # frequency
            x = np.arange(data.shape[-1]) / signals.shape[-1] * Fs
            if max(x) > 2e3:
                x *= 1e-3
                xlabel = '$f$ (kHz)'
            else:
                xlabel = '$f$ (Hz)'
        plot_params = {'x': x, 'Y_base': data, 'Ys': value, 'xlabel': xlabel,
                       'ylabel': ('Amp.', 'Value')}
    elif mode in ['STFT', 'CS']:  # 2D
        dimension = 2  # 2D
        STFT_params = kwargs['STFTs_params']
        y = (np.arange(STFT_params['mfft']) / STFT_params['mfft'])[:value.shape[-2]] * Fs * 1e-3
        ylabel = '$f$ (kHz)'
        if mode == 'STFT':
            pad = data.shape[-1] - signals.shape[-1] // STFT_params['hop']
            x = (np.arange(data.shape[-1]) - pad // 2) / Fs * STFT_params['hop']
            xlabel = 'Time $t$ (s)'
        else:  # CS
            data[..., 0:1] = data[..., 0:1] * 0.3  # remove the DC component in axis-a
            temp_len = value.shape[-1] * 2 - 1  # the length of the original time axis
            x = (np.arange(temp_len) / temp_len)[:value.shape[-1]] * Fs / STFT_params['hop']
            xlabel = r'$\alpha$ (Hz)'
        plot_params = {'x': x, 'y': y, 'Z_base': data, 'Zs': value,
                       'xlabel': xlabel, 'ylabel': ylabel}
    else:
        raise ValueError(f'mode should be in [CAM, time, frequency, envelope, STFT, CS]')

    # 2.1) define the grid
    N, K = data.shape[0], value.shape[0]
    prob = torch.nn.functional.softmax(torch.tensor(label_predicts[1]), -1).numpy()
    dimension_scale = 1.5 if dimension == 1 else 2
    figsize = ((K + 1) * 2 / 2.54, N * dimension_scale * 0.9 / 2.54)
    gs = mpl.gridspec.GridSpec(N, K + 1, hspace=1)

    plt.close('all')
    fig = plt.figure(figsize=figsize, dpi=dpi)
    Axes = [[] for _ in range(N)]
    storage = {}
    for n in range(N):
        for k in range(K + 1):
            item = fig.add_subplot(gs[n, k])
            if item not in storage:
                storage[item] = fig.add_subplot(item) if item else None
            Axes[n].append(storage[item])

    if dimension == 1:
        x, Y_base, Ys = plot_params['x'], plot_params['Y_base'], plot_params['Ys']
        xlabel, ylabel = plot_params['xlabel'], plot_params['ylabel']
        # base
        for n in range(N):
            ax = Axes[n][0]
            ax.plot(x, Y_base[n])
            ax.margins(x=0.001)
            ax.set_xlabel(xlabel + f' | n={n}')
            ax.set_ylabel(ylabel[0])

        for n in range(N):
            ys = Ys[:, n, ...]
            ylim = (1.1 * ys.min() - 0.1 * ys.max(), 1.1 * ys.max() - 0.1 * ys.min())
            for k in range(K):
                ax = Axes[n][k + 1]
                ax.plot(x, ys[k], color='C1', alpha=0.8)  # mpl.rcParams['axes.prop_cycle'].by_key()['color'][1]
                ax.margins(x=0.001)
                ax.set_xlabel(xlabel + f' | {prob[n, k] * 100:.2f}%')
                ax.set_ylim(ylim)
                ax.set_ylabel(ylabel[1])
                # value
                ax2 = ax.twinx()
                ax2.plot(x, Y_base[n], color='gray', alpha=0.25)
                ax2.set_yticks([])
                ax2.set_xticks(Axes[n][0].get_xticks())
                ax2.set_xlim(Axes[n][0].get_xlim())
                ax2.set_ylim(Axes[n][0].get_ylim())

    else:
        x, y, Z_base, Zs = plot_params['x'], plot_params['y'], plot_params['Z_base'], plot_params['Zs']
        xlabel, ylabel = plot_params['xlabel'], plot_params['ylabel']
        # base
        for n in range(N):
            ax = Axes[n][0]
            ax.pcolormesh(x, y, Z_base[n], cmap='Blues', shading='gouraud', rasterized=True)
            ax.set_xlabel(xlabel + f' | n={n}')
            ax.set_ylabel(ylabel)

        for n in range(N):
            bound = max(abs(Zs[:, n].min()), abs(Zs[:, n].max()))
            for k in range(K):
                ax = Axes[n][k + 1]
                ax.pcolormesh(x, y, Zs[k, n], cmap='bwr', shading='auto',
                              vmin=-bound, vmax=bound, rasterized=True)
                ax.set_xlabel(xlabel + f' | {prob[n, k] * 100:.2f}%')
                ax.set_ylabel(ylabel)
                ax.set_xticks(Axes[n][0].get_xticks())
                ax.set_yticks(Axes[n][0].get_yticks())
                ax.set_xlim(Axes[n][0].get_xlim())
                ax.set_ylim(Axes[n][0].get_ylim())

    gs.tight_layout(fig, pad=0.5, h_pad=0.5, w_pad=1.5, rect=None)
    fig.savefig(os.path.join(savedir, 'attribution.jpg'), dpi=dpi)
    fig.savefig(os.path.join(savedir, 'attribution.pdf'))


def data_visualization(datas, labels, labels_name, Fs, save_dir, data_name=None, dpi=600):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    traget_path = os.path.join(save_dir, 'DataShow')
    if os.path.exists(traget_path+'.jpg'): # avoid repeat
        return
    # 1) prepare data
    label = sorted(list(set(labels)))

    temp_map = {'Simulation':['H','F1','F2'],
                'CWRU':['N','I','B','O'],
                'HouDe2':['H','W','P','T']} # 'Simulation', 'CWRU', 'HouDe2'
    if data_name and data_name in temp_map:
        labels_name_new = temp_map[data_name]
    else:
        labels_name_new = labels_name
    index = [labels.tolist().index(i) for i in label]
    data = datas[index] # (N,t)
    # 2) init
    func_returns = func_trans_CS_v2()
    trans_series = func_returns[-1]['trans_series']
    out, info = trans_series.forward(data, verbose=True)
    print('\n', '-' * 10)  # print('\n'.join(info))
    func_env = func_trans_envelope_v2()[-1]['trans_series']
    out_env, info_env = func_env.forward(data, verbose=True)

    # obtain params
    mfft, hop = trans_series[0].SFT.mfft, trans_series[0].SFT.hop
    signals = data  # (N,t)
    temp_fft = lambda x: np.abs(np.fft.fft(x - x.mean(), axis=-1)[..., :(x.shape[-1] // 2 + 1)] / x.shape[-1])
    signals_freq = temp_fft(signals)  # (N,f)
    signals_STFT = np.abs(out[1][-1])  # (N,f2,t2)
    signals_CSCoh = np.abs(out[3][-1])  # (N,f2,a)
    signals_CSCoh[..., 0] = signals_CSCoh[..., 0] * 0.5  # remove the DC component in axis-a
    signals_env = np.abs(out_env[4][-1])  # (N,t)

    # prepare data
    t = np.arange(signals.shape[-1]) / Fs
    f = np.arange(signals_freq.shape[-1]) * Fs / signals.shape[-1] * 1e-3
    f1 = np.arange(signals_env.shape[1]) * Fs / signals.shape[-1]
    f2 = np.arange(signals_STFT.shape[1]) * Fs / mfft * 1e-3
    pad = signals_STFT.shape[-1] - signals.shape[-1] // hop
    t2 = (np.arange(signals_STFT.shape[-1]) - (pad + 1) // 2) * hop / Fs
    a = np.arange(signals_CSCoh.shape[-1]) / signals_STFT.shape[-1] * Fs / hop

    # plot
    setdefault(TrueType=True)
    mpl.rcParams.update({'grid.color': '0.8', 'grid.alpha': 1, })
    colors, markers, linestyles = get_plot_params()
    N = signals.shape[0]
    color_s, color_v,color_h_delta = np.linspace(0.3, 1, N), np.linspace(0.8, 0.4, N), np.linspace(-0.05, 0.08, N)
    plt_colors = [color_modify_HSV(colors[0],s=s,v=v,h_delta=hd) for s,v,hd in zip(color_s,color_v,color_h_delta)][::-1]

    # plot freq. & Env.
    plt.close('all')
    fig = plt.figure(figsize=[8.5 / 2.54, 4 / 2.54], dpi=dpi)
    gs = mpl.gridspec.GridSpec(1,2)
    for i, (x, Z, x_name) in enumerate(zip(
            [f,f1], [signals_freq, signals_env],[r'Spectral freq. $f$ (Hz)', r'Spectral freq. $f$ (Hz)'])):
        ax = fig.add_subplot(gs[i], projection='3d')
        ax.view_init(elev=30,azim=-60) # 30,-60
        ax.set_box_aspect([1, 1, 0.3])
        X, Y = np.meshgrid(x, range(Z.shape[0]))
        Lines = []
        for k in range(Z.shape[0]):
            temp = ax.plot_wireframe(X[k:k + 1], Y[k:k + 1], Z[k:k + 1],
                                     rstride=1, cstride=0, lw=0.3,
                                     color= plt_colors[k], alpha=1)
            Lines.append(temp)
        ax.tick_params('x', pad=-6.5)
        ax.tick_params('y', pad=-6)
        ax.tick_params('z', pad=-4.5)
        for item in [ax.xaxis, ax.yaxis, ax.zaxis]:
            item.set_pane_color((0.95, 0.95, 0.95, 0.2))
            item._axinfo['tick'].update({'inward_factor': 0.3, 'outward_factor': 0})
        ax.set_xlabel(x_name, labelpad=-15)
        ax.set_yticks(range(len(labels_name_new)), labels_name_new)
        ax.set_ylabel('Fault', labelpad=-15)
    gs.tight_layout(fig, rect=[0, 0.02, 1, 1.01], h_pad=0.5, w_pad=4.5, pad=0.5)
    for item in fig.get_children()[-2:]:
        zlim = item.get_zticks()
        while len(zlim)>5:
            zlim = zlim[::2]
        item.set_zticks(zlim)
    fig.savefig(traget_path + '_pre.jpg', dpi=dpi)
    fig.savefig(traget_path + '_pre.pdf', dpi=dpi)

    # plot  STFT & CSCoh
    plt.close('all')
    K = data.shape[0]
    fig = plt.figure(figsize=[8.5 / 2.54, 8.5/K*1.8 / 2.54], dpi=dpi)
    gs = mpl.gridspec.GridSpec(2,K)
    for i, (x, y, Z, x_name, y_name) in enumerate(zip(
            [t2, a], [f2,f2], [signals_STFT, signals_CSCoh],
            [r'Time $t$ (s)', r'Cyclic freq. $\alpha$ (Hz)'],
            [r'Spectral freq. $f$ (kHz)', r'Spectral freq. $f$ (kHz)'])):
        for j in range(Z.shape[0]):
            ax = fig.add_subplot(gs[i, j:j+1])
            im = ax.pcolormesh(x, y, Z[j], cmap='Blues', shading='gouraud', rasterized=True)
            ax.set_xlabel(x_name)
            ax.set_ylabel(y_name)
            if i==1:
                xlim = ax.get_xlim()
                ax.set_xlim(1.007 * xlim[0] - 0.007 * xlim[1], xlim[1])
    gs.tight_layout(fig, h_pad=1.5, w_pad=2, pad=0.5)
    fig.savefig(traget_path + '.jpg', dpi=dpi)
    fig.savefig(traget_path + '.pdf', dpi=dpi)
    plt.close('all')

if __name__ == '__main__':
    # 1) data
    inputs_ana = np.random.randn(*[2, 2000])
    pick = lambda x: x[-1] if type(x) == tuple else x

    # 2) test
    func_lists = [
        func_trans_envelope]  # func_trans_time, func_trans_frequency,func_trans_envelope, func_trans_STFT
    for func in func_lists:
        func_Z, func_Z_inv, func_unpatch, trans_dict = func()
        print('\n', '--' * 15, ' ' * 2 + trans_dict['name'] + ' ' * 2, '--' * 15)
        y, info = func_Z(inputs_ana, verbose=True)
        y_inv = func_Z_inv(y[-1], verbose=True)
        print('mean error: ', abs(y_inv[-1] - inputs_ana).mean())
        data = y[-1]
        values = np.random.randn(*data.shape, 3)
        out_data, out_value = func_unpatch(data, values)
        out_data, out_value = pick(out_data), pick(out_value)
        print(f'out_data.shape: {out_data.shape}, out_value.shape: {out_value.shape}')
