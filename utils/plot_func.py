import matplotlib as mpl
import matplotlib.transforms as mtransforms
import seaborn as sns
import numpy as np

def add_right_cax(ax, pad, width):
    '''
    在一个ax右边追加与之等高的cax.
    pad是cax与ax的间距,width是cax的宽度.
    '''
    axpos = ax.get_position()
    caxpos = mtransforms.Bbox.from_extents(
        axpos.x1 + pad,
        axpos.y0,
        axpos.x1 + pad + width,
        axpos.y1
    )
    cax = ax.figure.add_axes(caxpos)
    return cax


def adjust_ax_position(ax, dx=0, dy=0, dw=0, dh=0):
    axpos = ax.get_position()
    caxpos = mtransforms.Bbox.from_extents(
        axpos.x0 + dx,
        axpos.y0 + dy,
        axpos.x1 + dx + dw,
        axpos.y1 + dy + dh
    )
    ax.set_position(caxpos)


def setseaborn():
    # set color
    current_cmap = sns.color_palette("deep")
    sns.set(style="whitegrid")
    sns.set(style="ticks", context="notebook", font='Times New Roman', palette=current_cmap, font_scale=1)
    return current_cmap


def get_plot_params():
    colors = np.array([mpl.colors.to_rgb(item) for item in mpl.rcParams['axes.prop_cycle'].by_key()['color']])
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', 'P', '*', 'h', 'H', 'X', 'd']
    linestyles = ['-', '--', '-.', ':']
    return colors, markers, linestyles
def setdefault(ratio=1, fontsize=4, TrueType=True):
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

    mpl.rcParams.update({'figure.dpi': 500/ratio,
                         'figure.figsize': [4 / 2.54*ratio, 3 / 2.54*ratio], })

    mpl.rcParams['axes.linewidth'] = 0.5 * ratio
    mpl.rcParams['lines.linewidth'] = 0.5 * ratio

    mpl.rcParams['font.family'] = 'Arial'  # 'Times New Roman' 'Arial'
    mpl.rcParams['svg.fonttype'] = 'none'
    mpl.rcParams['pdf.fonttype'] = 42 if TrueType else 3 # 42: TrueType, 3: Type 3

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

    mpl.rcParams.update({'xtick.major.width': 0.3 * ratio,
                         'ytick.major.width': 0.3 * ratio,
                         'xtick.major.size': 1.5 * ratio,
                         'xtick.major.pad': 0.5 * ratio,
                         'xtick.direction': 'out',
                         'ytick.major.size': 1.5 * ratio,
                         'ytick.major.pad': 0.5 * ratio,
                         'ytick.direction': 'out',
                         })

    mpl.rcParams.update({'grid.color': '0.7',
                         'grid.alpha': 0.5,
                         'grid.linewidth': 0.25 * ratio,
                         'legend.facecolor': 'none',
                         'figure.facecolor': 'none',
                         'axes.facecolor': 'none',
                         })

    mpl.rcParams.update({'axes.labelpad': 0.3 * ratio,
                         'axes.titlepad': 0.5 * ratio,})

    mpl.rcParams.update({'hatch.linewidth': 0.2,
                         'hatch.color': 'b'})

    # mpl.rcParams['text.usetex'] = True
    mpl.rcParams["mathtext.fontset"] = 'cm'
    mpl.pyplot.switch_backend('tkagg')  # 'tkagg' 'Qt5Agg' 'pgf'


def setaxesLW(myaxes, axes_lw=1, tick_len=3, tick_lw=None):
    if not tick_lw:
        tick_lw = axes_lw / 3 * 2

    for item in ['top', 'left', 'bottom', 'right']:
        myaxes.spines[item].set_linewidth(axes_lw)
    myaxes.tick_params(width=tick_lw, length=tick_len)


def plot_color(color=None):
    mpl.pyplot.close('all')

    if color is None:
        color = mpl.rcParams['axes.prop_cycle'].by_key()['color']

    mpl.pyplot.bar(range(len(color)), [1, ] * len(color), color=color)
    # mpl.pyplot.show()
    mpl.pyplot.tight_layout()

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

if __name__ == '__main__':
    setdefault()
    plot_color()
