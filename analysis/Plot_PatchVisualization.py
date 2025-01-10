import os
import pandas as pd
import numpy as np
import pickle
import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
from utils.plot_func import setdefault, get_plot_params
import seaborn as sns

setdefault()


def Extract_visualization_name(filename):  # 'attribution-STFT_v2-SHAP-5.pkl'
    temp = filename.split('.')[0].split('-')
    domain, method, patch = temp[1], temp[2], temp[3]
    res = {'domain': domain, 'method': method, 'patch': patch}
    return res


def Extract_visualization_pkl(file_path):
    with open(file_path, 'rb') as f:
        save_dict = pickle.load(f)
    return save_dict

def plot_Attr1D(info, datas, values, save_dir, dpi=600, plot=False):
    '''
    :param info: dict ['mode', 'Fs', 'signals', 'label_predicts']
    :param datas: [patchs, N_sample,d1]
    :param values: [patchs, N_sample,C_prediction,d1]
    :param save_dir: str
    :return:
    '''
    # init
    mode, Fs, signals, label_predicts = info['mode'], info['Fs'], info['signals'], info['label_predicts']
    datas,values = datas.swapaxes(0,1), values.swapaxes(0,1)  # [N_sample,patchs,d1] [N_sample,patchs,C_prediction,d1]
    # if mode.split('_')[0] in ['frequency', 'envelope']:
    #     return
    # get the data
    if 'time' in mode:  # time
        x, xlabel = np.arange(datas.shape[-1]) / Fs, 'Time $t$ (s)'
    else:  # frequency and Envelope
        x = np.arange(datas.shape[-1]) / signals.shape[-1] * Fs
        if max(x) > 2e3:
            x *= 1e-3
            xlabel = 'Spectral freq. $f$ (kHz)'
        else:
            xlabel = 'Spectral freq. $f$ (Hz)'
    p_params = {'x': x, 'Y_base': datas, 'Ys': values, 'xlabel': xlabel, 'ylabel': 'Value'}
    # plot set
    if plot:
        setdefault()
        N,P,C = values.shape[0], values.shape[1], values.shape[2]
        x, Y_base, Ys = p_params['x'], p_params['Y_base'], p_params['Ys']
        xlabel, ylabel = p_params['xlabel'], p_params['ylabel']
        figsize = (8.5 / 2.54, C * 1.4 / 2.54)
        gs = mpl.gridspec.GridSpec(C, P)
        # plot
        for n in range(N):
            # init
            plt.close('all')
            fig = plt.figure(figsize=figsize, dpi=dpi)
            Axes = [[] for _ in range(C)]
            for c in range(C):
                for p in range(P):
                    item = fig.add_subplot(gs[c, p])
                    Axes[c].append(item)
            # plot
            ylim = (1.1 * Ys[n].min() - 0.1 * Ys[n].max(), 1.1 * Ys[n].max() - 0.1 * Ys[n].min())
            for p in range(P):
                y_base = Y_base[n][p]
                ys = Ys[n][p]
                for c in range(C):
                    ax = Axes[c][p]
                    ax.plot(x, ys[c], color='C1', alpha=0.8)  # mpl.rcParams['axes.prop_cycle'].by_key()['color'][1]
                    ax.margins(x=0.001)
                    ax.set_xlabel(xlabel)
                    ax.set_ylim(ylim)
                    ax.set_ylabel(ylabel)
                    # value
                    ax2 = ax.twinx()
                    ax2.plot(x, y_base, color='gray', alpha=0.25)
                    ax2.set_yticks([])
            gs.tight_layout(fig, pad=0.5, h_pad=1.5, w_pad=1.5, rect=[0, 0, 1, 1])
            fig.savefig(os.path.join(save_dir, f'SampleC{n:d}.jpg'), dpi=dpi)
            fig.savefig(os.path.join(save_dir, f'SampleC{n:d}.pdf'), dpi=dpi)
            plt.close('all')
    return p_params


def plot_Attr2D(info, datas, values, save_dir,dpi=900, plot=False):
    '''
    :param info: dict ['mode', 'Fs', 'signals', 'label_predicts', 'STFT_params']
    :param datas: [patchs, N_sample,d1, d2]
    :param values: [patchs, N_sample,C_prediction,d1, d2]
    :param save_dir: str
    :return:
    '''
    # init
    mode, Fs, signals = info['mode'], info['Fs'], info['signals']
    label_predicts, STFT_params = info['label_predicts'], info['STFTs_params']
    datas,values = datas.swapaxes(0,1), values.swapaxes(0,1)  # [N_sample,patchs,d1] [N_sample,patchs,C_prediction,d1]
    # if mode.split('_')[0] in ['STFT', 'CS']: # 'STFT', 'CS'
    #     return
    # get the data
    y = (np.arange(STFT_params['mfft']) / STFT_params['mfft'])[:values.shape[-2]] * Fs * 1e-3
    ylabel = 'Spectral freq. $f$ (kHz)'
    if 'STFT' in mode:
        pad = datas.shape[-1] - signals.shape[-1] // STFT_params['hop']
        x = (np.arange(datas.shape[-1]) - pad // 2) / Fs * STFT_params['hop']
        xlabel = 'Time $t$ (s)'
    else:  # CS
        datas[..., 0:1] = datas[..., 0:1] * 0.3  # remove the DC component in axis-a
        temp_len = values.shape[-1] * 2 - 1  # the length of the original time axis
        x = (np.arange(temp_len) / temp_len)[:values.shape[-1]] * Fs / STFT_params['hop']
        xlabel = r'Cyclic freq. $\alpha$ (Hz)'
    p_params = {'x': x, 'y': y, 'Z_base': datas, 'Zs': values,
                   'xlabel': xlabel, 'ylabel': ylabel}
    # plot set
    if plot:
        def mapped_color(data,cmap='bwr',vlim=None):
            if vlim is None:
                vlim = [data.min(), data.max()]
            cm = mpl.colormaps[cmap]
            norm = mpl.colors.Normalize(vmin=vlim[0], vmax=vlim[1])
            return cm(norm(data),bytes=True)
        setdefault()
        N,P,C = values.shape[0], values.shape[1], values.shape[2]
        x,y,Zs,Z_base = p_params['x'], p_params['y'], p_params['Zs'], p_params['Z_base']
        xlabel, ylabel = p_params['xlabel'], p_params['ylabel']
        figsize = (8.5 / 2.54, C * 2.2 / 2.54)
        gs = mpl.gridspec.GridSpec(C, P)
        # plot
        for n in range(N):
            # init
            plt.close('all')
            fig = plt.figure(figsize=figsize, dpi=dpi)
            Axes = [[] for _ in range(C)]
            for c in range(C):
                for p in range(P):
                    item = fig.add_subplot(gs[c, p])
                    Axes[c].append(item)
            # plot
            for p in range(P):
                Z = Zs[n][p]
                z_base = Z_base[n][p]
                im_base = mapped_color(z_base, cmap='Greys')
                bound = abs(Z).max()
                for c in range(C):
                    ax = Axes[c][p]
                    # ---- old version -----
                    # ax.pcolormesh(x, y, z_base, cmap='Greys',
                    #               shading='gouraud', rasterized=True,
                    #               )
                    # ax.pcolormesh(x, y, Z[c], cmap='bwr', shading='auto',alpha=0.7,
                    #               vmin=-bound, vmax=bound, rasterized=True)
                    # ---- new version -----
                    im_z = mapped_color(Z[c], cmap='bwr', vlim=[-bound, bound])
                    im_temp = np.float32(im_base*0.3+im_z)
                    im_temp = np.uint8(im_temp / im_temp.max() *255)
                    ax.imshow(im_temp,aspect='auto',origin='lower',extent=[x[0],x[-1],y[0],y[-1]])
                    ax.yaxis.set_inverted(False)
                    # ------------
                    ax.set_xlabel(xlabel)
                    ax.set_ylabel(ylabel)
                    if c > 0 or p > 0:
                        ax.set_xticks(Axes[0][0].get_xticks())
                        ax.set_yticks(Axes[0][0].get_yticks())
                        ax.set_xlim(Axes[0][0].get_xlim())
                        ax.set_ylim(Axes[0][0].get_ylim())
                    if 'CS' in mode:
                        xlim = ax.get_xlim()
                        ax.set_xlim(1.003*xlim[0]-0.003*xlim[1], xlim[1])

            gs.tight_layout(fig, pad=0.5, h_pad=1.5, w_pad=1.5, rect=[0, 0, 1, 1])
            fig.savefig(os.path.join(save_dir, f'SampleC{n:d}.jpg'), dpi=dpi)
            fig.savefig(os.path.join(save_dir, f'SampleC{n:d}.pdf'), dpi=dpi)
    return p_params

def plot_get_params(storage, df, domain, patchs, method, save_dir):
    '''
    :param storage: [N_file,C_prediction,N_sample,d1,d2]
    :param df: DataFrame ['domain', 'method', 'patch', 'StorageInd']
    :param domain: str
    :param patchs: [str]
    :param method: str
    :param save_dir: str
    :return:
    '''
    # init
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # get the data
    StoInds = [df[(df['domain'] == domain) & (df['method'] == method) & (df['patch'] == patch)]['StorageInd'].values[0]
               for patch in patchs] # [int, int, ...]
    get_pick_eachClass = lambda x: [list(x).index(i) for i in set(x)]
    Pick_eachClass =  [get_pick_eachClass(storage[i]['label_predicts'][0]) for i in StoInds]
    Attr_datas = np.array([storage[i]['data'][p] for i,p in zip(StoInds,Pick_eachClass)])  # [patchs, N_sample,d1,d2]
    Attr_values = np.array([storage[i]['value'][:,p] for i,p in zip(StoInds,Pick_eachClass)]) # [patchs, C_prediction,N_sample,d1,d2]
    Attr_values = Attr_values.swapaxes(1,2) # [patchs, N_sample,C_prediction,d1,d2]
    info = storage[StoInds[0]]
    # plot
    if info['data'].ndim == 2: # 1D
        res = plot_Attr1D(info, Attr_datas, Attr_values, save_dir)
    elif info['data'].ndim == 3: # 2D
        res = plot_Attr2D(info, Attr_datas, Attr_values, save_dir)
    else:
        raise ValueError('The dimension of data is not supported')
    return res

def plot_main(plot_params, domains, save_dir,dpi=600):
    '''
    :param plot_params:  {domain:params,...};
    1d_params = {'x': x, 'Y_base': datas, 'Ys': values, 'xlabel': xlabel, 'ylabel': ylabel}
    2d_params = {'x': x, 'y': y, 'Z_base': datas, 'Zs': values, 'xlabel': xlabel, 'ylabel': ylabel}
    Y_base: [N_sample,patchs,d1]; Ys = [N_sample,patchs,C_prediction,d1];
    :param domains:
    :param save_dir:
    :return:
    '''
    # init
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # plot prepare
    check_1D = lambda params: 'Zs' not in params
    def mapped_color(data, cmap='bwr', vlim=None):
        if vlim is None:
            vlim = [data.min(), data.max()]
        cm = mpl.colormaps[cmap]
        norm = mpl.colors.Normalize(vmin=vlim[0], vmax=vlim[1])
        return cm(norm(data), bytes=True)
    setdefault()

    # data check
    N, P, C, Ds = [],[],[],[]
    for domain in domains:
        Ds.append(int(check_1D(plot_params[domain])))
        value = plot_params[domain]['Ys'] if check_1D(plot_params[domain]) else plot_params[domain]['Zs']
        N.append(value.shape[0])
        P.append(value.shape[1])
        C.append(value.shape[2])
    if len(set(N)) != 1 or len(set(P)) != 1 or len(set(C)) != 1:
        raise ValueError('The input data is not consistent')
    N, P, C = N[0], P[0], C[0] # sample, patch, prediction_class
    D = len(Ds)
    # x, y, Zs, Z_base = p_params['x'], p_params['y'], p_params['Zs'], p_params['Z_base']
    # xlabel, ylabel = p_params['xlabel'], p_params['ylabel']
    height = [1.1 if item else 2.1 for item in Ds]
    figsize = (8.5 / 2.54, sum(height) / 2.54)
    gs = mpl.gridspec.GridSpec(D, P,height_ratios=height)

    # plot
    for n in range(N):
        # init
        plt.close('all')
        fig = plt.figure(figsize=figsize, dpi=dpi)
        Axes = [[] for _ in range(D)]
        for d in range(D):
            for p in range(P):
                item = fig.add_subplot(gs[d, p])
                Axes[d].append(item)
        # plot
        for d,domain in enumerate(domains):
            params = plot_params[domain]
            x, xlabel = params['x'], params['xlabel']
            if check_1D(params):
                Y_base, Ys = params['Y_base'], params['Ys'] # [N_sample,patchs,d1]; [N_sample,patchs,C_prediction,d1]
                ylabel = params['ylabel']
                bound = abs(Ys[n, :, n]).max()
                ylim = (-1.1 * bound, 1.1 * bound)
                for p in range(P):
                    y_base = Y_base[n][p]
                    ys = Ys[n,p,n]
                    ax = Axes[d][p]
                    ax.plot(x, ys, color='C1', alpha=0.8)
                    ax.margins(x=0.001)
                    ax.set_xlabel(xlabel)
                    ax.set_ylim(ylim)
                    ax.set_ylabel(ylabel)
                    # value
                    ax2 = ax.twinx()
                    ax2.plot(x, y_base, color='gray', alpha=0.25)
                    ax2.set_yticks([])
                    ax2.margins(y=0.001)

            else:
                y, ylabel = params['y'], params['ylabel']
                Z_base, Zs = params['Z_base'], params['Zs'] # [N_sample,patchs,d1,d2]; [N_sample,patchs,C_prediction,d1,d2]
                for p in range(P):
                    bound = abs(Zs[n, p]).max()
                    Z = Zs[n,p,n]
                    z_base = Z_base[n][p]
                    im_base = mapped_color(z_base, cmap='Greys')
                    ax = Axes[d][p]
                    im_z = mapped_color(Z, cmap='bwr', vlim=[-bound, bound])
                    im_temp = np.float32(im_base * 0.3 + im_z)
                    im_temp = np.uint8(im_temp / im_temp.max() * 255)
                    ax.imshow(im_temp, aspect='auto', origin='lower', extent=[x[0], x[-1], y[0], y[-1]])
                    ax.yaxis.set_inverted(False)
                    ax.set_xlabel(xlabel)
                    ax.set_ylabel(ylabel)
                    if p > 0:
                        ax.set_xticks(Axes[d][0].get_xticks())
                        ax.set_yticks(Axes[d][0].get_yticks())
                        ax.set_xlim(Axes[d][0].get_xlim())
                        ax.set_ylim(Axes[d][0].get_ylim())
        gs.tight_layout(fig, pad=0.5, h_pad=1.5, w_pad=0.5, rect=[0, 0, 1, 1])
        # save
        fig.savefig(os.path.join(save_dir, f'SampleC{n:d}.jpg'), dpi=dpi)
        fig.savefig(os.path.join(save_dir, f'SampleC{n:d}.pdf'), dpi=dpi)
        plt.close('all')
    pass

def main(root, filename_pre="attribution", file_end='.pkl', method='Exchange_v3'):
    '''
    :param root: file dir
    :param filename_pre: file marker
    :param file_end: file marker
    :param method: only use target method
    :return:
    '''
    # init
    save_dir = os.path.join(root, 'PatchShow')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    target_path = os.path.join(save_dir, f'storage-DfList_p{method:s}.pkl')
    # try to load data
    if os.path.exists(target_path):
        with open(target_path, 'rb') as f:
            df, storage = pickle.load(f)
    else:
        filenames = next(os.walk(root))[2]
        filenames = [name for name in filenames if filename_pre in name and name.endswith(file_end)]
        # load data
        df = []
        storage = []  # [N_file,C_prediction,N_sample,d1,d2]
        for i, filename in enumerate(filenames):
            print(f'<{len(storage):d}-{i}/{len(filenames):d}>: {filename:s}')
            filepath = os.path.join(root, filename)
            temp_dict = Extract_visualization_name(filename)
            if temp_dict['method'] != method:  # only load the method
                continue
            storage.append(Extract_visualization_pkl(filepath))  # use storage_value to store the value
            temp_dict.update({'StorageInd': len(storage) - 1})
            df.append(temp_dict)
        df = pd.DataFrame(df)
        with open(target_path, 'wb') as f:
            pickle.dump((df, storage), f)
    # generate the filename
    domains = ['frequency_v2', 'envelope_v2', 'STFT_v2', 'CS_v2']
    patchs = ['0','1', '3', '5', ]  # order
    plot_params = {}
    for domain in domains:
        print(f'Processing {domain:s}...')
        plot_params[domain] = plot_get_params(storage, df, domain, patchs, method,
                                              os.path.join(save_dir, f'{method:s}-{domain:s}'))
    plot_main(plot_params, domains, os.path.join(save_dir, f'{method:s}-all'))
    pass


if __name__ == '__main__':
    root = r'E:\OneDrive - sjtu.edu.cn\6-SoftwareFiles\GitFiles\0-个人库\03-科研\2024-PerturbationNet\checkpoint\ExpSimu\(Statistic)CNN-Simulation-time-SNR0-1211-215743\PostProcess_of_FullAnalysis\Visualization'
    # os.chdir(os.path.split(os.path.realpath(__file__))[0])  # 更改运行路径为文件所在路径
    main(root)
