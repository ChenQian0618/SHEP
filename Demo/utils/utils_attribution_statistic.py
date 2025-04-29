import sys,os
def root_path_k(x, k): return os.path.abspath(os.path.join(x, *([os.pardir] * (k + 1))))
projecht_dir = root_path_k(__file__, 2)
# add the project directory to the system path
if projecht_dir not in sys.path:
    sys.path.insert(0, projecht_dir)

import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from SHEPs.plot_func import setdefault, get_plot_params, color_modify_HSV, ticklabel_format

def Extract_info_by_name(filename):  # 'CS_#5_SHAP_values_raw.pkl'
    mps = {'SHEP_Add':'SHEP-Add', 'SHEP_Remove':'SHEP-Remove'}
    for k, v in mps.items():
        if k in filename:
            filename = filename.replace(k, v)
    temp = filename.split('.')[0].split('_')
    domain, patch, method = temp[0], temp[1], temp[2]
    res = {'domain': domain, 'patch': patch, 'method': method}
    return res


def Extract_info_by_pkl(file_path):
    with open(file_path, 'rb') as f:
        save_dict = pickle.load(f)
    ks = ['input_label', 'label_name','analyse_time']
    res = {k: save_dict[k] if k in save_dict else None for k in ks}
    return res, save_dict['domain_value']


def Calculate_Similarity(storage_value1, storage_value2,label):  # [C_prediction,N_sample,d1,d2]
    # init
    shape = storage_value1.shape
    # calculate
    func_Simi = lambda v1, v2: np.sum(v1 * v2) / np.linalg.norm(v1) / np.linalg.norm(v2)
    res = np.zeros((shape[0], shape[1]))  # [C_prediction,N_sample]
    for i in range(shape[0]):
        for j in range(shape[1]):
            res[i, j] = func_Simi(storage_value1[i,j].reshape(-1), storage_value2[i,j].reshape(-1))
    # process label
    res_new = np.zeros((shape[0],len(set(label))))  # [C_prediction,C_sample]
    for i,k in enumerate(sorted(list(set(label)))):
       res_new[:,i] = np.mean(res[:,label==k], axis=-1)
    return res_new  # [C_prediction,C_sample]

def plot_Similarity_Matrix(df, savepath, label_name=None, dpi=300):
    '''
    :param df: pd.DataFrame, ['domain', 'method', 'SimiMatrix_pred_samp']
    :param label_name: str, the name of the label )
    :param savepath: str, the path to save the figure
    :param dpi: int, the dpi of the figure
    '''
    # init
    if label_name is None:
        label_name = [f'C{i:d}' for i in range(df['SimiMatrix_pred_samp'].iloc[0].shape[0])]
    df = df.reset_index()
    col_names = df['domain'].unique().tolist()
    row_names = df['method'].unique().tolist()
    if 'SHAP' in row_names: row_names.remove('SHAP')
    # plot
    search_df = df.set_index(['method','domain'])['SimiMatrix_pred_samp']
    Rs,Cs = len(row_names),len(col_names)
    figsize = (8.5 / 2.54, 8.5*Cs/Rs*0.6 / 2.54)
    gs = mpl.gridspec.GridSpec(Rs,Cs,hspace=0.5)
    plt.close('all')
    fig = plt.figure(figsize=figsize, dpi=200)
    for i, rname in enumerate(row_names):
        for j, cname in enumerate(col_names):
            ax = fig.add_subplot(gs[i, j])
            temp = search_df.loc[rname, cname]
            sns.heatmap(temp, ax=ax, annot=True, cmap='Blues', fmt='.2f', cbar=False,
                        vmin=0, vmax=1,
                        xticklabels=label_name, yticklabels=label_name) # annot_kws={'fontsize': 4}
            ax.yaxis.set_inverted(False)
            ax.set_xlabel('Sample class')
            ax.set_ylabel('Prediction class')
            ax.tick_params(axis='both', length=0,pad=1.5)
            ax.set_title(f'{rname:s} | {cname:s}', fontsize=4, loc='center', fontweight='bold')
    gs.tight_layout(fig, rect=[0, 0, 1, 1], h_pad=1.5, w_pad=2.5, pad=0.5)
    fig.savefig(savepath + '.jpg', dpi=dpi)
    print(f'\nFigure (similarity_matrix) saved to:\n {savepath + ".jpg"}')
    plt.close('all')

def plot_Attribution_time(df, savepath, dpi=300):
    '''
    :param df: pd.DataFrame, ['domain', 'method', 'patch', 'analyse_time']
    :param savepath: str, the path to save the figure
    :param dpi: int, the dpi of the figure
    '''
    pass
    # init
    df = df.reset_index()
    domains = df['domain'].unique().tolist()
    patchs = df['patch'].unique().tolist()
    methods = df['method'].unique().tolist()
    method_names = [item if item!='SHAP' else 'SHAP (permutation=5)' for item in methods]
    # plot preparation
    def get_value(method,domain,search_column): # get the value of the search_df in the order of patchs
        temp = df.loc[(df['method'] == method) & (df['domain'] == domain)]
        order = np.argsort([patchs.index(item) for item in temp['patch'].to_list()])
        return temp.iloc[order][search_column].to_numpy()
    patch_names = patchs.copy()
    # plot
    colors, markers, linestyles = get_plot_params()
    plt.close('all')
    figsize = (8.5 / 2.54, 2.5 / 2.54 )
    gs = mpl.gridspec.GridSpec(1, 4)
    fig = plt.figure(figsize=figsize,dpi=200)
    N_m = len(methods)
    color_s, color_v,color_h_delta = np.linspace(0.25, 1, N_m), np.linspace(0.9, 0.4, N_m), np.linspace(-0.05, 0.08, N_m)
    plt_colors = [color_modify_HSV(colors[0],s=s,v=v,h_delta=hd) for s,v,hd in zip(color_s,color_v,color_h_delta)]
    for i, domain in enumerate(domains):
        ax = fig.add_subplot(gs[0,i])
        for j, method in enumerate(methods):
            temp = get_value(method, domain, 'analyse_time')
            x = np.arange(len(patch_names))
            ax.plot(x, temp, label=method_names[j],color=plt_colors[j],
                    ls = linestyles[j],marker=markers[j], markersize=1)
        ax.set_xlabel(f'Patch size | {domain}')
        ax.set_ylabel('Attribution time (s)')
        ax.set_xticks(x, patch_names)
    gs.tight_layout(fig, rect=[0, 0, 1, 0.87], h_pad=0, w_pad=1, pad=0.5)
    ax = fig.get_children()[-1]
    leg = ax.legend(bbox_to_anchor=(0.5, 0.98), loc='upper center', bbox_transform=fig.transFigure,
                    ncol=len(methods), borderaxespad=0, labelspacing=0.05,
                    handlelength=3, handletextpad=0.3,
                    columnspacing=3, )
    leg.get_frame().set(alpha=1, lw=0.5 , edgecolor=[0.5, ] * 3, linestyle='--')
    fig.savefig(savepath + '.jpg', dpi=dpi)
    print(f'\nFigure (analysis time) saved to:\n {savepath + ".jpg"}')
    plt.close('all')


def plot_Similarity_Box(df, savepath, dpi=300):
    '''
    :param df: pd.DataFrame, ['domain', 'method','patch', 'SimiMatrix_pred_samp']
    :param savepath: str, the path to save the figure
    :param dpi: int, the dpi of the figure
    '''
    # init
    df = df.reset_index()
    domains = df['domain'].unique().tolist()
    patchs = df['patch'].unique().tolist()
    methods = df['method'].unique().tolist()
    if 'SHAP' in methods: methods.remove('SHAP')

    # check
    for method in methods:
        if method not in df['method'].unique().tolist():
            raise ValueError(f"The domain {method} is not in {str(df['method'].unique().tolist()):s}")
    for domain in domains:
        if domain not in df['domain'].unique().tolist():
            raise ValueError(f"The method {domain} is not in {str(df['domain'].unique().tolist()):s}")
    # plot preparation
    def get_value(method,domain,search_column): # get the value of the search_df in the order of patchs
        temp = df.loc[(df['method'] == method) & (df['domain'] == domain)]
        order = np.argsort([patchs.index(item) for item in temp['patch'].to_list()])
        return temp.iloc[order][search_column].to_numpy()
    patch_names = patchs.copy()
    # plot
    y_min=-1.1
    setdefault()
    colors, markers, linestyles = get_plot_params()

    hatch = ['', '/' * 2, '\\' * 2, '-' * 2, '+', '-', 'x', 'O', 'o', '*', '.', '|']

    plt.close('all')
    figsize = (8.5 / 2.54, 2.5 / 2.54)
    gs = mpl.gridspec.GridSpec(1, 4)
    fig = plt.figure(figsize=figsize, dpi=200)
    N_d,N_m,width = len(domains),len(methods),0.2
    color_s, color_v,color_h_delta = np.linspace(0.3, 1, N_m), np.linspace(0.85, 0.4, N_m), np.linspace(-0.05, 0.08, N_m)
    plt_colors = [color_modify_HSV(colors[0],s=s,v=v,h_delta=hd) for s,v,hd in zip(color_s,color_v,color_h_delta)]
    for i, domain in enumerate(domains):
        x = np.arange(len(patch_names))
        ax = fig.add_subplot(gs[i])
        leg_handles = []
        for j, method in enumerate(methods):
            color = plt_colors[j]
            temp_mean = np.array(get_value(method, domain, 'SimiMatrix_pred_samp').tolist())
            temp_mean = temp_mean.reshape(temp_mean.shape[0],-1) # [patch,array]
            temp = ax.boxplot([item for item in temp_mean], positions=x + (j-N_m/2+0.5) * width,
                          widths=width*0.8, patch_artist=True, whis=1.5,
                          boxprops=dict(facecolor=color_modify_HSV(color, s_ratio=-0.95, v_ratio=0.95),
                                        edgecolor=color, lw=0.4, hatch=hatch[len(methods)-j-1]*7), # hatch=hatch[len(methods)-j-1]*5)
                          whiskerprops=dict(color=color, lw=0.4),
                          capprops=dict(color=color,lw=0.4,),
                          flierprops=dict(marker='d', markeredgecolor=color, markersize=0.5,),
                          medianprops=dict(lw=0.4,color=color),
                          showmeans=False, meanline=False, showcaps=True, showfliers=False, showbox=True,  notch=False)
            leg_handles.append(temp['boxes'][0])
            for item in temp['boxes']:
                item._hatch_color = color_modify_HSV(item._hatch_color, s_ratio=-0, v_ratio=0.2)
            #     item.set(facecolor= color_modify_HSV(item.get_edgecolor(), s_ratio=-0.5, v_ratio=0.5))
        ax.set_xlabel(f'Patch size | {domain}')
        ax.set_ylabel('Cosine similarity')
        ax.set_ylim([min(np.arange(1, y_min-0.01, -0.5)), 1])
        ax.set_yticks(np.arange(1, y_min-0.01, -0.5))
        ax.set_xticks(x, patch_names)
        xlim = [(-N_m / 2) * width, len(patch_names) - 1 + (N_m / 2) * width]
        ax.set_xlim(1.02 * xlim[0] - 0.02 * xlim[1], -0.02 * xlim[0] + 1.02 * xlim[1])
        ticklabel_format(ax, format='%g', which='y')
    gs.tight_layout(fig, rect=[0, 0, 1, 0.9], h_pad=1.5, w_pad=1, pad=0.5)
    leg = fig.legend(handles = leg_handles, labels=methods,
                     bbox_to_anchor=(0.5, 0.98), loc='upper center', bbox_transform=fig.transFigure,
                    ncol=N_m, borderaxespad=0, labelspacing=0.05,
                    handlelength=1.8, handletextpad=0.3,
                    columnspacing=3, )
    leg.get_frame().set(alpha=1, lw=0.5, edgecolor=[0.5, ] * 3, linestyle='-.')
    fig.savefig(savepath + '.jpg', dpi=dpi)
    print(f'\nFigure (similarity box) saved to:\n {savepath + ".jpg"}')
    plt.close('all')