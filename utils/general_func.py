from numpy.fft import fft
import numpy as np
from sklearn import manifold
import matplotlib.pyplot as plt
# from scipy.signal import hilbert
import os

# 设置随机数种子
manualSeed = 999
np.random.seed(manualSeed)


def CatchProjectDir(iconfile='train.py'):
    cur_dir = os.path.abspath(os.path.curdir)
    count = 0
    while(count<5):
        if os.path.exists(os.path.join(cur_dir,iconfile)):
            return cur_dir
        else:
            cur_dir = os.path.split(cur_dir)[0]
        count += 1
    return 0

def tSNE(input,label,dir):
    model = manifold.TSNE(n_components=2, perplexity=30, early_exaggeration=4, learning_rate=1000, n_iter=1000)
    output = model.fit_transform(input)
    PlotTsne(output,label,dir)

def PlotTsne(data,label,Dir):
    num_class = len(set(label))
    fig = plt.figure(figsize=[10, 8], dpi=100)
    plt.scatter(data[:, 0], data[:, 1], 20, label,cmap=plt.cm.get_cmap("turbo", num_class))
    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0.02, 0.02)
    colorbar = plt.colorbar(ticks=np.arange((num_class-1)/num_class/2,num_class-1,(num_class-1)/num_class), shrink=0.9)
    colorbar.set_ticklabels(np.arange(num_class)+1)
    plt.savefig(Dir,dpi=200)
    plt.close()

def CalMutualDistance(vector,verbose = False):
    center = np.mean(vector,axis=0)
    distance = np.mean(np.sqrt(np.mean((vector-center)**2,axis=1)))
    if verbose:
        return distance,center
    else:
        return distance

def CalDistance(vectors,verbose = False):
    list_of_norms = lambda X: np.sum(np.power(X, 2), axis=1)
    XX = np.reshape(list_of_norms(vectors), newshape=(-1, 1))
    YY = np.reshape(list_of_norms(vectors), newshape=(1, -1))
    output = XX + YY - 2 * np.matmul(vectors, vectors.T)
    return np.sqrt(output.max()/vectors.shape[1]+1e-6)
def intra_inter(data,label):
    intra,centers = [],[]
    for item in set(label):
        index = np.where(label == item)[0]
        dis,center = CalMutualDistance(data[index], verbose=True)
        intra.append(dis)
        centers.append(center)
    centers = np.array(centers)
    inter = CalMutualDistance(centers)
    max_dis = CalDistance(data)

    return np.mean(intra),inter,max_dis

if __name__ == '__main__':
    Frequency_len = int(1e3)
    length = 8
    n = np.arange(-(length-1),length).reshape(1,-1)
    f = (np.arange(length*2-1)/(length*2-1)).reshape(1,-1)
    # x = np.exp(np.matmul(complex(1j)*2*np.pi*f.T,n))
    x = np.cos(np.matmul( 2 * np.pi * f.T, n))
    WinLen = 2 * length - 1
    zero_append = max(int(Frequency_len - WinLen),0)
    t_temp = np.linspace(-1, 1, WinLen)
    sigma = 0.52
    WinFun = (np.pi * sigma ** 2) ** (-1 / 4) * np.exp((-t_temp ** 2) / 2 / (sigma ** 2))
    windowed_x = np.matmul(x,np.diag(WinFun))
    # F_x = []
    # for i in range(x.shape[0]):
    #     F_x.append(np.abs(np.fft.fft(np.hstack([windowed_x[i,:],np.zeros(zero_append)])))/WinLen*2)
    # F_x = np.array(F_x)
    # f = np.arange(Frequency_len) / Frequency_len * 1
    F_x = np.fft.fft(windowed_x,n=Frequency_len)

    fig = plt.figure()
    ax1 = fig.add_subplot(4,1,1)
    ax1.plot(n.squeeze(),np.real(x[length//2-1,:]))
    ax2 = fig.add_subplot(4, 1, 2)
    ax2.plot(n.squeeze(),WinFun)

    ax4 = fig.add_subplot(4, 1, 4)
    ax4.plot(f, F_x.T)
    ax4.set_xlim(0,0.5)
    fig.tight_layout()
    plt.show()
    plt.ion()
