import os
import pandas as pd
import numpy as np
import pickle
import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
from utils.plot_func import setdefault,get_plot_params
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
    ks = ['label_predicts', 'label_name']
    res = {k: save_dict[k] for k in ks}
    return res, save_dict['value']


def Extract_analysistime(filepath):
    fdir, fname = os.path.split(filepath)[0], os.path.split(filepath)[1].split('.')[0]
    new_dir = os.path.join(os.path.split(fdir)[0], fname)
    filenames = [item for item in next(os.walk(new_dir))[2] if item.endswith('.pkl')]
    with open(os.path.join(new_dir, filenames[0]), 'rb') as f:
        save_dict = pickle.load(f)
    return {'analyse_time': save_dict['analyse_time']}


def Calculate_Similarity(storage_value1, storage_value2):  # [C_prediction,N_sample,d1,d2]
    def myreshape(storage_value):  # [C_prediction,N_sample,d1]
        shape = storage_value.shape
        if shape[1] % shape[0] != 0:
            raise ValueError('The second dimension should be divided by the first dimension')
        target_shape = (shape[0], shape[0], shape[1] // shape[0], -1)
        return storage_value.reshape(target_shape)  # [C_prediction,C_sample,N_sample/C,d1]

    # init
    N_c = storage_value1.shape[0]
    value1 = myreshape(storage_value1)
    value2 = myreshape(storage_value2)
    # calculate
    func_Simi = lambda v1, v2: np.sum(v1 * v2) / np.linalg.norm(v1) / np.linalg.norm(v2)
    res = np.zeros((N_c, N_c))  # [C_prediction,C_sample]
    for i in range(N_c):
        for j in range(N_c):
            res[i, j] = func_Simi(value1[i, j], value2[i, j])
    return res  # [C_prediction,C_sample]


def plot_Similarity_Matrix(df,label_name, savepath, dpi=300):
    '''
    :param df: pd.DataFrame, ['domain', 'method', 'SimiMatrix_pred_samp']
    :param label_name: str, the name of the label )
    :param savepath: str, the path to save the figure
    :param dpi: int, the dpi of the figure
    '''
    # init
    df = df.reset_index()
    row_names = ['Mask', 'Scale','Exchange_v3']
    col_names = ['frequency_v2', 'envelope_v2','STFT_v2','CS_v2']
    label_name = ['H','F1','F2'] if len(label_name)==3 else label_name # abbreviation
    # check
    for rname in row_names:
        if rname not in df['method'].unique().tolist():
            raise ValueError(f"The domain {rname} is not in {str(df['method'].unique().tolist()):s}")
    for cname in col_names:
        if cname not in df['domain'].unique().tolist():
            raise ValueError(f"The method {cname} is not in {str(df['domain'].unique().tolist()):s}")
    # plot
    search_df = df.set_index(['method','domain'])['SimiMatrix_pred_samp']
    Rs,Cs = len(row_names),len(col_names)
    figsize = (8.5 / 2.54, 8.5*Cs/Rs*0.5 / 2.54)
    gs = mpl.gridspec.GridSpec(Rs,Cs,hspace=0.5)
    plt.close('all')
    fig = plt.figure(figsize=figsize, dpi=dpi)
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
    gs.tight_layout(fig, rect=[0, 0, 1, 1], h_pad=1.5, w_pad=2.5, pad=0.5)
    fig.savefig(savepath + '.jpg', dpi=dpi)
    fig.savefig(savepath + '.pdf', dpi=dpi)

def plot_Similarity_MeanStd(df,savepath, dpi=300):
    '''
    :param df: pd.DataFrame, ['domain', 'method', 'patch', 'Simi_mean', 'Simi_std']
    :param savepath: str, the path to save the figure
    :param dpi: int, the dpi of the figure
    '''
    # init
    df = df.reset_index()
    methods = ['Mask', 'Scale', 'Exchange_v3']
    methods_names = ['Mask', 'Scale', 'SHEP']
    domains = ['frequency_v2', 'envelope_v2', 'STFT_v2', 'CS_v2']
    patchs = df['patch'].unique().tolist()

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
    patch_names = ['#' + item for item in patchs]
    # plot
    plt_ratio = 5
    hatch = ['', '/' * 2, '\\' * 2, '-' * 2, '+', '-', 'x', 'O', 'o', '*', '.', '|']
    setdefault(ratio=plt_ratio, fontsize=4 * plt_ratio)
    plt.close('all')
    figsize = (8.5 / 2.54 *plt_ratio, 2.5 / 2.54 * plt_ratio)
    gs = mpl.gridspec.GridSpec(1, 4)
    fig = plt.figure(figsize=figsize, dpi=dpi/plt_ratio)
    N_d,N_m,width = len(domains),len(methods),0.27
    for i, domain in enumerate(domains):
        ax = fig.add_subplot(gs[0,i])
        for j, method in enumerate(methods):
            temp_mean = get_value(method, domain, 'Simi_mean')
            temp_std = get_value(method, domain, 'Simi_std')
            x = np.arange(len(patch_names))
            ax.bar(x + (j-N_m/2+0.5) * width, temp_mean, width, yerr=temp_std,
                   label=methods_names[j],
                   color=f'C{j}', edgecolor=None,ecolor=[0.6,]*3, hatch=hatch[j],
                   capsize=0.5*plt_ratio,lw=0.1*plt_ratio,
                   error_kw={'elinewidth': 0.25*plt_ratio, 'capthick': 0.25*plt_ratio})
        ax.set_xlabel('Patch size')
        ax.set_ylabel('Similarity')
        ax.set_ylim([0, 1])
        ax.set_yticks(np.arange(1, 0, -0.2))
        ax.set_xticks(x, patch_names)
    gs.tight_layout(fig, rect=[0, 0, 1, 0.87], h_pad=0, w_pad=1, pad=0.5)
    ax = fig.get_children()[-1]
    leg = ax.legend(bbox_to_anchor=(0.5, 0.98), loc='upper center', bbox_transform=fig.transFigure,
                    ncol=N_m, borderaxespad=0, labelspacing=0.05 * plt_ratio,
                    handlelength=1.8*plt_ratio, handletextpad=0.3*plt_ratio,
                    columnspacing=3*plt_ratio, )
    leg.get_frame().set(alpha=1, lw=0.5 * plt_ratio, edgecolor=[0.5, ] * 3, linestyle='-.')
    fig.savefig(savepath + '.jpg', dpi=dpi/plt_ratio)
    fig.savefig(savepath + '.pdf', dpi=dpi/plt_ratio)

def plot_Attribution_time(df, savepath, dpi=300):
    '''
    :param df: pd.DataFrame, ['domain', 'method', 'patch', 'analyse_time']
    :param savepath: str, the path to save the figure
    :param dpi: int, the dpi of the figure
    '''
    # init
    df = df.reset_index()
    methods = ['Mask', 'Scale', 'SHAP','Exchange_v3']
    methods_names = ['Mask', 'Scale','SHAP', 'SHEP']
    domains = ['frequency_v2', 'envelope_v2', 'STFT_v2', 'CS_v2']
    patchs = df['patch'].unique().tolist()
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
    patch_names = ['#' + item for item in patchs]
    # plot
    setdefault()
    _, markers, linestyles = get_plot_params()
    plt.close('all')
    figsize = (8.5 / 2.54, 2.5 / 2.54 )
    gs = mpl.gridspec.GridSpec(1, 4)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    for i, domain in enumerate(domains):
        ax = fig.add_subplot(gs[0,i])
        for j, method in enumerate(methods):
            temp = get_value(method, domain, 'analyse_time')
            x = np.arange(len(patch_names))
            ax.plot(x, temp, label=methods_names[j],
                    ls = linestyles[j],marker=markers[j], markersize=1)
        ax.set_xlabel('Patch size')
        ax.set_ylabel('Attribution time (s)')
        ax.set_xticks(x, patch_names)
    gs.tight_layout(fig, rect=[0, 0, 1, 0.87], h_pad=0, w_pad=1, pad=0.5)
    ax = fig.get_children()[-1]
    leg = ax.legend(bbox_to_anchor=(0.5, 0.98), loc='upper center', bbox_transform=fig.transFigure,
                    ncol=len(methods), borderaxespad=0, labelspacing=0.05,
                    handlelength=3, handletextpad=0.3,
                    columnspacing=3, )
    leg.get_frame().set(alpha=1, lw=0.5 , edgecolor=[0.5, ] * 3, linestyle='--')
    fig.savefig(savepath + '.jpg', dpi=dpi )
    fig.savefig(savepath + '.pdf', dpi=dpi)

def main(root, filename_pre="attribution", file_end='.pkl', flag_SaveXlsx=False):
    # init
    save_dir = os.path.join(root, 'Post')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    target_path = os.path.join(save_dir, 'storage-dataframe.pkl')
    # try to load data
    if os.path.exists(target_path):
        with open(target_path, 'rb') as f:
            df = pickle.load(f)
    else:
        filenames = next(os.walk(root))[2]
        filenames = [name for name in filenames if filename_pre in name and name.endswith(file_end)]
        # load data
        df = []
        storage_value = [None] * len(filenames)  # [N_file,C_prediction,N_sample,d1,d2]
        for i, filename in enumerate(filenames):
            print(f'<{i}|{len(filenames):d}>: {filename:s}')
            filepath = os.path.join(root, filename)
            temp_dict = Extract_visualization_name(filename)
            temp, storage_value[i] = Extract_visualization_pkl(filepath)  # use storage_value to store the value
            temp.update({'value_index': i})
            temp_dict.update(temp)
            temp_dict.update(Extract_analysistime(filepath))
            df.append(temp_dict)
        df = pd.DataFrame(df)
        # calculate similarity
        search_df = df.loc[df['method'] == 'SHAP', ['domain', 'patch', 'value_index']].set_index(['domain', 'patch'])
        def process_df(temp):
            domain, patch, index = temp['domain'], temp['patch'], temp['value_index']
            index_SHAP = search_df.loc[domain, patch].values[0]
            res = Calculate_Similarity(storage_value[index_SHAP], storage_value[index])
            return res  # [C_prediction,C_sample]
        df['SimiMatrix_pred_samp'] = df.apply(process_df, axis=1)
        df['Simi_mean'] = df['SimiMatrix_pred_samp'].apply(lambda x: x.mean())
        df['Simi_std'] = df['SimiMatrix_pred_samp'].apply(lambda x: x.std())
        with open(target_path, 'wb') as f:
            pickle.dump(df, f)
    # save Xlsx
    if flag_SaveXlsx:
        df_save = df[['domain', 'method', 'patch', 'analyse_time', 'Simi_mean', 'Simi_std']]
        df_save_1 = df_save.set_index(['domain', 'method', 'patch'])['analyse_time'].unstack()
        df_save_2 = df_save.set_index(['domain', 'method', 'patch'])['Simi_mean'].unstack()
        df_save_3 = df_save.set_index(['domain', 'method', 'patch'])['Simi_std'].unstack()
        df_saveDict = {'df': df_save, 'analyse_time': df_save_1, 'Simi_mean': df_save_2, 'Simi_std': df_save_3}
        with pd.ExcelWriter(os.path.join(save_dir, 'storage-PartialData.xlsx')) as writer:
            for k, v in df_saveDict.items():
                v.to_excel(writer, sheet_name=k)
    # plot and save figure
    plot_flags = [False, True, False,]
    # 1) plot and save the Similarity_Matrix
    if plot_flags[0]:
        plot_Similarity_Matrix(df.loc[df['patch'] == '1'][['domain', 'method', 'SimiMatrix_pred_samp']],
                               df['label_name'][0],
                               os.path.join(save_dir, 'Similarity_Matrix'), dpi=600)

    # 2) plot and save the Similarity_Matrix
    if plot_flags[1]:
        plot_Similarity_MeanStd(df[['domain', 'method', 'patch', 'Simi_mean', 'Simi_std']],
                               os.path.join(save_dir, 'Similarity_MeanStd'), dpi=600)

    # 2) plot and save the Similarity_Matrix
    if plot_flags[2]:
        plot_Attribution_time(df[['domain', 'method', 'patch', 'analyse_time']],
                              os.path.join(save_dir, 'Attribution_time'), dpi=600)


if __name__ == '__main__':
    root = r'E:\OneDrive - sjtu.edu.cn\6-SoftwareFiles\GitFiles\0-个人库\03-科研\2024-PerturbationNet\checkpoint\Experiment\CNN-Simulation-time-SNR0-1211-215743\PostProcess_of_FullAnalysis\Visualization'
    # os.chdir(os.path.split(os.path.realpath(__file__))[0])  # 更改运行路径为文件所在路径

    main(root)
