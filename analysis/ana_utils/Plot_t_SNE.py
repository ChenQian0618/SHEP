import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from sklearn import manifold
import matplotlib as mpl
import pandas as pd
from utils.plot_func import setdefault, setaxesLW

linestyle = ['--', '-', '-.', ':'] * 10
marker = ['o', '*', 's', 'v', '^', '<', '>', '8', 'p', 'h', 'H', 'D', 'd', 'P', 'X'] * 10
labelname = ['N', 'I1', 'B1', 'O1', 'I2', 'B2', 'O2', 'I3', 'B3', 'O3'] * 10


def tSNE(input: np.ndarray, label: np.ndarray, dir,maps=None,row_col=None):
    '''
    :param input: (n,p), the Data to be analyzed
    :param label: (n,) or (n,2) the label of the data, if the label is (n,2), the first column is the domain label
    :param dir: the directory to save the figure
    '''
    setdefault()
    model = manifold.TSNE(n_components=2, perplexity=30, early_exaggeration=4, learning_rate=1000, n_iter=1000)
    output = model.fit_transform(input)
    # 绘制tSNE图(考虑transfer模式)
    PlotTsne(output, label, dir,maps=maps,row_col=row_col)


def PlotTsne(data, mlabel, Dir, maps=None, row_col=None):
    if mlabel.ndim == 2:
        mlabel = mlabel[:, 0]
        clabel = mlabel[:, 1]
    else:
        clabel = None
    if row_col is None:
        row_col = (1, len(set(mlabel)))
    # -1 -> white, 0 -> original, 1 -> black
    combine_white = lambda x, ratio: \
        [(1 - abs(ratio)) * item + abs(ratio) * int(ratio > 0) for item in x]

    if clabel is None:  # for conventional task
        flag_transfer = False
        clabel = np.ones(mlabel.shape) * 2
        num_color = 2
        current_cmap = sns.color_palette(
            [combine_white(item, 0.2) for item in sns.color_palette("tab20", desat=0.8)]).as_hex()
    else:  # for domain-transfer task
        flag_transfer = True
        num_color = len(set(clabel))
        current_cmap = sns.color_palette(
            [combine_white(item, 0.2) for item in sns.color_palette("tab10", desat=0.8)]).as_hex()
    df = pd.DataFrame([data[:, 0], data[:, 1], mlabel, clabel]).transpose().set_index([2, 3])
    fig = plt.figure() # figsize=[8 / 2.54, 6 / 2.54], dpi=1000
    hs, labs = [], []
    for i, (s, c) in enumerate(df.index.value_counts().keys().sort_values()):
        num_item = df.loc[s, c].shape[0]
        # sc = plt.scatter(x=df.loc[s,c][0].to_numpy(), y=df.loc[s,c][1].to_numpy(), s=100,
        #                  c=df.loc[i,:80].index.to_numpy(),cmap=mpl.colors.ListedColormap(current_cmap,N=10),
        #                  vmin=0,vmax=9,marker=marker[i],label = labelname[i],alpha=0.98,edgecolors=[0.2,0.2,0.2],linewidths=0.8)
        sc = plt.scatter(x=df.loc[s, c][0].to_numpy(), y=df.loc[s, c][1].to_numpy(), s=2,
                         c=[int(c), ] * num_item,
                         cmap=mpl.colors.ListedColormap(sns.color_palette(f"light:{current_cmap[int(s)]}", num_color)),
                         vmin=clabel.min() - 0.1, vmax=clabel.max(),
                         marker=marker[int(s)], label=f'{int(s):d}-{int(c):d}', alpha=0.98,
                         edgecolors=[0.35, 0.35, 0.35], linewidths=0.1)
        h, _ = sc.legend_elements()
        hs.append(h[0])
    # plt.axis('on')

    # 设置基本形状和legend
    plt.margins(0.02, 0.02)
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    setaxesLW(ax, 0.5)
    reorder_func = lambda _hs, r, c: [_hs[i // r + i % r * c] for i, item in enumerate(_hs)]
    if flag_transfer:  #
        if maps is None:
            labelname = [f'{int(item[0]):d}-{int(item[1]):d}' for item in df.index.value_counts().keys().sort_values()]
        else:
            # map1 = {0: 'N', 1: 'W', 2: 'P', 3: 'F'}
            # map2 = {0: 'L0', 1: 'L1', 2: 'H0', 3: 'H1'}
            labelname = [f'{maps[0][int(item[0])]:s}-{maps[1][int(item[1])]:s}' for item in
                         df.index.value_counts().keys().sort_values()]
        ax.legend(reorder_func(hs, *row_col), reorder_func(labelname, *row_col),
                  prop={'family': 'Times New Roman', 'size': mpl.rcParams['font.size']}, bbox_to_anchor=(0.5, 1.05),
                  loc='lower center',
                  borderaxespad=0, ncol= row_col[-1], borderpad=0.5, handleheight=0.8, handlelength=2, handletextpad=0.8,
                  columnspacing=2, markerscale=0.7)
    else:  #
        if maps is None:
            labelname = [f'{int(item[0]):d}' for item in df.index.value_counts().keys().sort_values()]
        else:
            # maps = dict(zip([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            #                 ['N', 'SP', 'SC', 'Sw', 'SW', 'PC', 'PW', 'BC', 'BR', 'BI', 'BO']))
            labelname = [f'{maps[int(item[0])]:s}' for item in df.index.value_counts().keys().sort_values()]
        ax.legend(reorder_func(hs,*row_col), reorder_func(labelname,  *row_col),
                  prop={'family': 'Times New Roman', 'size': mpl.rcParams['font.size']}, bbox_to_anchor=(0.5, 1.05),
                  loc='lower center',
                  borderaxespad=0.2, ncol=row_col[-1], borderpad=0.5, handleheight=1, handlelength=2, handletextpad=0.2,
                  labelspacing=0.5, columnspacing=3, markerscale=0.6)
    ax.get_legend().get_frame().set_linewidth(0.4)

    # set transparency
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    if ax.get_legend() is not None:
        ax.get_legend().get_frame().set_facecolor('none')
    fig.tight_layout(pad=0.5)
    plt.savefig(Dir + '-withlegend.jpg')
    plt.savefig(Dir + '-withlegend.svg')

    a = ax.get_legend()
    a.set_visible(False)
    # plt.subplots_adjust(top=0.95, bottom=0.05, right=0.96, left=0.04, hspace=0, wspace=0)
    fig.set_size_inches(4 / 2.54, 3 / 2.54)
    # setaxesLW(ax, 0.5)
    fig.tight_layout(pad=0.5)
    plt.savefig(Dir + '.jpg')
    plt.savefig(Dir + '.svg')
    plt.close()


if __name__ == '__main__':
    print(1)
