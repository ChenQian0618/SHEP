import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from utils.plot_func import setdefault, get_plot_params, color_modify_HSV
import os
from scipy.io import loadmat, savemat


def extract_data(file_path):
    data = loadmat(file_path)
    return data


def plot_comparison_exchange(signal_len=2000, Fs=10e3, dpi=300,pick=2):
    # frequency
    x_len = signal_len // 2 + 1
    # init
    file_dir = r'E:\OneDrive - sjtu.edu.cn\6-SoftwareFiles\GitFiles\0-个人库\03-科研\2024-PerturbationNet\checkpoint\ExpSimu\(Combine)CNN-Simulation-time-SNR0-1211-215743\PostProcess_of_FullAnalysis'
    file_name = {'V1': 'attribution-frequency_v2-Exchange-1.mat',
                 'V2': 'attribution-frequency_v2-Exchange_v2-1.mat',
                 'V3': 'attribution-frequency_v2-Exchange_v3-1.mat',
                 'V4': 'attribution-frequency_v2-SHAP_dev-1.mat'}
    # extract data
    Extracted_data = {}
    for k, name in file_name.items():
        file_path = os.path.join(file_dir, name)
        temp = extract_data(file_path)
        Extracted_data[k] = temp['res']  # (N, L, c_b, c)
        info = temp['res_info']
    # data process
    names = list(file_name.keys())
    temp_datas = np.array(list(Extracted_data.values()))  # (Method, N, L, c_b, c)
    N_c = temp_datas.shape[-1]
    N_L = temp_datas.shape[-3]
    N_N = temp_datas.shape[-4]
    temp_datas = temp_datas[:, ::N_N // N_c, 1:, :, :]
    datas = np.zeros((temp_datas.shape[0], N_c, x_len, N_c, N_c))  # (Method, c_ana, Len, c_background, c_prediction)
    patch = int(np.ceil(x_len / (temp_datas.shape[-3])))
    for i, p in enumerate(range(0, x_len, patch)):
        datas[:, :, p:p + patch, :, :] = temp_datas[:, :, i:i + 1, :, :]
    freq = np.arange(x_len) / signal_len * Fs * 1e-3
    datas = datas[:, pick, :, :, :]  # (Method, Len, c_b, c) # only use the last sample (corresponding to last class)
    datas = np.concatenate([datas.mean(axis=-2, keepdims=True), datas], axis=-2)  # (Method, Len, 1+c_b, c)
    datas = datas.transpose([0, 3, 2, 1])  # (Method, c, 1+c_b, Len)
    class_names = ['M', 'H', 'F1', 'F2']
    # plot
    setdefault(TrueType=True)
    mpl.rcParams.update({'grid.color': '0.8', 'grid.alpha': 1, })
    colors, markers, linestyles = get_plot_params()
    N = datas.shape[2]
    color_s, color_v,color_h_delta = np.linspace(0.3, 0.6, N), np.linspace(0.9, 0.7, N), np.linspace(0.1, 0, N)
    plt_colors = [color_modify_HSV(colors[1],s=s,v=v,h_delta=hd) for s,v,hd in zip(color_s,color_v,color_h_delta)][::-1]
    plt.close('all')
    figsize = np.array([8.5 / 2.54, 8.5 / 2.54])
    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = mpl.gridspec.GridSpec(datas.shape[0], datas.shape[1])
    for i in range(datas.shape[0]):
        for j in range(datas.shape[1]):
            Z = datas[i, j, :, :]
            ax = fig.add_subplot(gs[i, j], projection='3d')
            ax.set_box_aspect([1, 1, 0.4])
            X, Y = np.meshgrid(freq, range(Z.shape[0]))
            Lines = []
            for k in range(Z.shape[0]):
                temp = ax.plot_wireframe(X[k:k + 1], Y[k:k + 1], Z[k:k + 1],
                                         rstride=1, cstride=0, lw=0.3,
                                         color=plt_colors[k], alpha=1)
                Lines.append(temp)
            ax.tick_params('x', pad=-6.5)
            ax.tick_params('y', pad=-6)
            ax.tick_params('z', pad=-5)
            for item in [ax.xaxis, ax.yaxis, ax.zaxis]:
                item.set_pane_color((0.95, 0.95, 0.95, 0.2))
                item._axinfo['tick'].update({'inward_factor': 0.3, 'outward_factor': 0})
            ax.set_xlabel('Spectral freq. $f$ (kHz)', labelpad=-15)
            ax.set_yticks(range(len(class_names)), class_names)
            ax.set_ylabel('Background class', labelpad=-15)

    gs.tight_layout(fig, rect=[-0.05, 0.02, 1.01, 1.01], h_pad=0, w_pad=0, pad=0)
    for item in fig.get_children()[1:]:
        zlim = item.get_zticks()
        while len(zlim)>4:
            zlim = zlim[::2]
        item.set_zticks(zlim)
    fig.savefig(os.path.join(file_dir,f'comparison_exchange-{pick:d}') + '.jpg', dpi=dpi)
    fig.savefig(os.path.join(file_dir,f'comparison_exchange-{pick:d}') + '.pdf', dpi=dpi)


if __name__ == '__main__':
    plot_comparison_exchange()
