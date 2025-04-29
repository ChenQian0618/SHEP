import matplotlib as mpl
import matplotlib.transforms as mtransforms
import seaborn as sns
import numpy as np

def setseaborn():
    # set color
    current_cmap = sns.color_palette("deep")
    sns.set(style="whitegrid")
    sns.set(style="ticks", context="notebook", font='Times New Roman', palette=current_cmap, font_scale=1)
    return current_cmap

def setdefault(ratio=1, fontsize=4, TrueType=True, font=['Arial']): # font=['Arial', 'simsun']
    '''
    mpl.rcParams: https://matplotlib.org/stable/users/explain/customizing.html
    {k:v for k,v in mpl.rcParams.items() if 'pad' in k}
    '''

    _ = setseaborn()
    # '#7EA4D1', '#807C7D', '#C1565E', '#DCA96A', '#82AD7F', '#79438E'
    # '#1b67ab', '#807C7D', '#C1565E', '#DCA96A', '#82AD7F', '#79438E'
    custom_colors = ['#1b67ab', '#C1565E', '#DCA96A', '#9B59B6', '#82AD7F', '#00788C', '#726A95',
                     '#20B2AA', '#839788']
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=custom_colors)

    mpl.rcParams.update({'figure.dpi': 600 / ratio,
                         'figure.figsize': [7 / 2.54 * ratio, 4 / 2.54 * ratio], })

    mpl.rcParams['axes.linewidth'] = 0.5 * ratio
    mpl.rcParams['lines.linewidth'] = 0.5 * ratio

    mpl.rcParams['font.family'] = font  # 'Times New Roman' 'Arial'
    mpl.rcParams['svg.fonttype'] = 'none'
    mpl.rcParams['pdf.fonttype'] = 42 if TrueType else 3  # 42: TrueType, 3: Type 3

    mpl.rcParams['font.size'] = fontsize
    mpl.rcParams.update({'xtick.labelsize': mpl.rcParams['font.size'],
                         'ytick.labelsize': mpl.rcParams['font.size'],
                         'axes.labelsize': mpl.rcParams['font.size'],
                         'axes.titlesize': mpl.rcParams['font.size'],
                         'legend.fontsize': mpl.rcParams['font.size'],
                         'legend.title_fontsize': mpl.rcParams['font.size'],
                         'figure.titlesize': mpl.rcParams['font.size'],
                         'figure.labelsize': mpl.rcParams['font.size'], })

    mpl.rcParams.update({'figure.subplot.bottom': 0.05,
                         'figure.subplot.hspace': 0.4,
                         'figure.subplot.left': 0.05,
                         'figure.subplot.right': 0.95,
                         'figure.subplot.top': 0.92,
                         'figure.subplot.wspace': 0.4, })

    mpl.rcParams.update({'xtick.direction': 'in',
                         'xtick.major.width': 0.5 * ratio,
                         'xtick.major.size': 2 * ratio,
                         'xtick.major.pad': 2.5 * ratio,
                         'xtick.major.top': False,
                         'ytick.direction': 'in',
                         'ytick.major.width': 0.5 * ratio,
                         'ytick.major.size': 2 * ratio,
                         'ytick.major.pad': 2.5 * ratio,
                         'ytick.major.right': False, })

    mpl.rcParams.update({'grid.color': '0.85',
                         'grid.alpha': 1,
                         'grid.linewidth': 0.3 * ratio,
                         'grid.linestyle': '--',
                         'legend.facecolor': 'none',
                         'figure.facecolor': 'none',
                         'axes.facecolor': 'none',
                         })

    mpl.rcParams.update({'axes.labelpad': 1 * ratio,
                         'axes.titlepad': 0.5 * ratio,
                         'axes.axisbelow': 'line', })  # 'line', 'true', 'false'

    mpl.rcParams.update({'hatch.linewidth': 0.25,
                         'hatch.color': 'b'})

    # mpl.rcParams['text.usetex'] = True
    mpl.rcParams["mathtext.fontset"] = 'cm'
    mpl.pyplot.switch_backend('tkagg')  # 'tkagg' 'Qt5Agg' 'pgf'

def ticklabel_format(ax,format='%g',which='both'):
    '''
    '''
    try:
        if which in ['both','x']:
            ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter(format))
        if which in ['both','y']:
            ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter(format))
    except:
        print('ticklabel_format error')

def color_modify_HSV(rgb, h_ratio=0, s_ratio=0, v_ratio=0, h=None, s=None, v=None, h_delta=None, s_delta=None, v_delta=None):
    '''
    Modify the HSV of a RGB color.
    '''
    ratio_func = lambda x, r: (1-abs(r)) * x + abs(r) * int(r > 0) # -1->0, 0->x, 1->1
    alpha=None
    rgb = np.array(rgb)
    if rgb.shape[-1] == 4:
        alpha = rgb[...,3:]
        rgb = rgb[...,:3]
    hsv = mpl.colors.rgb_to_hsv(rgb)
    # ratio
    for i,value in enumerate([h_ratio, s_ratio, v_ratio]):
        hsv[...,i] = ratio_func(hsv[...,i], value)
    # value
    for i,value in enumerate([h, s, v]):
        if value is not None:
            hsv[...,i] = value
    #delta
    for i,value in enumerate([h_delta, s_delta, v_delta]):
        if value is not None:
            hsv[...,i] += value
    rgb = mpl.colors.hsv_to_rgb(hsv)
    if alpha is not None:
        rgb = np.concatenate([rgb, alpha], axis=-1)
    return rgb

def get_plot_params():
    colors = np.array([mpl.colors.to_rgb(item) for item in mpl.rcParams['axes.prop_cycle'].by_key()['color']])
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', 'P', '*', 'h', 'H', 'X', 'd']
    linestyles = ['-', '--', '-.', ':']
    return colors, markers, linestyles

if __name__ == '__main__':
    setdefault()
