import numpy
import torch
import os
import pickle
from stoged_b_Simulation.utils.models import CNN
import shap
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import seaborn as sns

from stoged_b_Simulation.utils.MyExplainer import Mask_explainer, Scale_explainer, My_explainer


sns.set(context='notebook', style='ticks', font_scale=1.5)

mpl.pyplot.switch_backend('tkagg')
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# device = torch.device("cpu")

def fft_cut(x:np.ndarray):
    '''
    对时域信号进行FFT变换并放缩，只保留正频率部分，输出为复数arrray
    :param x:
    :return:
    '''
    x = x.squeeze()
    if len(x.shape) == 1: # 保证维度为2
        x = x[np.newaxis,:] # 增加一个维度
    result = np.fft.fft(x,axis=-1)/x.shape[-1]
    shape = result.shape
    return result[:,:shape[-1]//2+1]

def ifft_cut(x:np.ndarray, even=True): # 长度为偶数 如1024
    '''
    对频域信号先进行放缩，再进行IFFT变换，输出为实数array
    :param x:
    :param even:
    :return:
    '''
    x = x.squeeze()
    if len(x.shape) == 1: # 保证维度为2
        x = x[np.newaxis,:] # 增加一个维度
    if even:
        temp = np.conj(x[:,-2:0:-1])
    else:
        temp = np.conj(x[:,-1:0:-1])
    x = np.concatenate([x,temp],axis=-1)
    result = np.fft.ifft(x,axis=-1)*x.shape[-1]
    return np.real(result)

def change_superpixel(x:np.ndarray,index=False,granularity=10,flag_FFT=False):
    '''
    考虑是否需要FFT，再对数据进行超像素处理（分块处理）
    :param x:
    :param index:
    :param granularity:
    :param flag_FFT:
    :return:
    '''
    x = x.squeeze()
    if len(x.shape) == 1: # 保证维度为2
        x = x[np.newaxis,:] # 增加一个维度
    # FFT
    if flag_FFT:
        x = fft_cut(x)
    # 超像素
    shapes = x.shape
    if index == False:
        index=list(np.arange(0,shapes[-1],granularity))+[x.shape[-1]]
    result = np.zeros([shapes[0],len(index)-1],dtype=np.object_)
    for i in range(shapes[0]):
        for j in range(len(index)-1):
            result[i,j] = x[i,index[j]:index[j+1]].astype(np.object_)
    return result,index

def recover_superpixel(x:np.ndarray):
    '''
    恢复超像素
    :param x:
    :return:
    '''
    x = x.squeeze()
    if len(x.shape) == 1: # 保证维度为2
        x = x[np.newaxis,:] # 增加一个维度
    func =lambda x: np.hstack(x)
    return np.apply_along_axis(func,-1,x)

# flag_recover_superpixel: 是否需要恢复超像素
def wrapper_basic(func,flag_recover_superpixel=True, flag_FFT=False):
    '''
    包装函数：对输入数据进行预处理（恢复超像素 & 恢复FFT），再进行预测
    :param func:
    :param flag_recover_superpixel:
    :param flag_FFT:
    :return:
    '''
    def new_func(x:numpy.array, verbose=False):
            # superpixel process
            if flag_recover_superpixel:
                x = recover_superpixel(x)
            # FFT process
            if flag_FFT:
                x = ifft_cut(x)
            # shape process
            if len(x.shape) == 1:
                x = x[np.newaxis,np.newaxis,:]
            elif len(x.shape) == 2:
                x = x[:,np.newaxis,:]
            x = torch.tensor(x.astype(np.float32), dtype=torch.float32).to(device)
            # predict
            with torch.no_grad():
                result = func(x, verbose = verbose)
            return result.detach().cpu().numpy()
    return new_func

class Main(object):
    def __init__(self,dir_path):
        # 选项
        self.flag_FFT = True
        self.flag_recover_superpixel = True
        self.analysis_method = 'my' # shap | mask | scale | my
        # 初始化
        self.dir_path = dir_path
        self.postfile_path = os.path.join(self.dir_path,"shap_output")
        if not os.path.exists(self.postfile_path):
            os.mkdir(self.postfile_path)
        self.granularity = 2
        # 加载模型
        self.load_model(dir_path)
        # 分析
        self.analysis_data_init() # 加载数据集和模型
        if self.analysis_method == 'shap':
            self.analysis_shap()
        elif self.analysis_method in ['mask','scale','my']:
            self.analysis_my()
        else:
            raise ValueError('analysis_method must be shap | mask | scale | my')
        # 绘图
        if self.flag_FFT: # 绘图
            self.analysis_shap_fft_plot()
        else:
            self.analysis_shap_time_plot()
        print('Finish')

    def load_model(self, dir_path):
        # 确定路径
        data_path = os.path.join(dir_path, 'collected_data_for_analysis.pkl')
        best_name = [item for item in next(os.walk(dir_path))[2] if 'best_model.pth' in item][0]
        model_params_path = os.path.join(dir_path, best_name)
        # 加载数据
        model_params = torch.load(model_params_path)
        with open(data_path, 'rb') as f:
            self.dataset = pickle.load(f)
        # 导入模型
        # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.dataset_class_number = len(self.dataset['train'].keys())
        self.dataset_label_name = self.dataset['label_name'] if 'label_name' in self.dataset.keys() else [str(i) for i in range(self.dataset_class_number)]
        self.model = CNN(in_channel=1, out_channel=self.dataset_class_number).to(device)
        self.model.load_state_dict(model_params['state_dict'])
        self.model.eval()

    def analysis_data_init(self):
        data = np.concatenate([item[0] for item in self.dataset['train'].values()])
        label = np.concatenate([item[1] for item in self.dataset['train'].values()])
        class_label = np.sort(list(set(label)))
        # # shap-backdata 1
        # index = torch.randperm(data.shape[0])
        # self.background_data = data[index[:200]]

        # # shap-backdata 2
        self.background_data = np.vstack([data[np.where(label == item)[0][:20]] for item in class_label])

        # shap-ana data
        self.ana_data = np.vstack([data[np.where(label==item)[0][:1]] for item in class_label])


    def analysis_shap(self):
        print(f'begin analysis_shap_time\n bkg shape: {self.background_data.shape}')
        background_masker = shap.maskers.Independent(change_superpixel(self.background_data,
                                                                       granularity=self.granularity,
                                                                       flag_FFT=self.flag_FFT)[0],
                                                     max_samples=1000, fixed_background=None)
        predict_func = wrapper_basic(self.model.forward, flag_recover_superpixel=True,flag_FFT=self.flag_FFT)
        explainer = shap.Explainer(predict_func, background_masker, algorithm='permutation')
        ana_data,self.index = change_superpixel(self.ana_data, granularity=self.granularity,
                                                       flag_FFT=self.flag_FFT)
        start_time = time.time()
        self.shap_values = explainer(ana_data,max_evals=int(1e3)) # 3000
        self.output = self.shap_values.values # (batch,in_channel,out_channel)
        self.predict = self.shap_values.base_values
        analyse_time = time.time() - start_time
        print(f'end analysis_shap_time, time: {analyse_time:.4f} s')

    def analysis_my(self):
        print(f'begin analysis_my\n bkg shape: {self.background_data.shape}')
        predict_func = wrapper_basic(self.model.forward, flag_recover_superpixel=True, flag_FFT=self.flag_FFT)
        bkg = change_superpixel(self.background_data,granularity=self.granularity,flag_FFT=self.flag_FFT)[0]
        if self.analysis_method == 'mask':
            explainer = Mask_explainer(predict_func)
        elif self.analysis_method == 'scale':
            explainer = Scale_explainer(predict_func,scale=0.5)
        elif self.analysis_method == 'my':
            explainer = My_explainer(predict_func, back_ground_data=bkg)
        else:
            raise ValueError('analysis_method must be mask | scale | my')
        ana_data,self.index = change_superpixel(self.ana_data, granularity=self.granularity,
                                                       flag_FFT=self.flag_FFT)
        start_time = time.time()
        self.output = explainer(ana_data)
        self.predict = predict_func(ana_data)
        analyse_time = time.time() - start_time
        print(f'end analysis_shap_time, time: {analyse_time:.4f} s')


    def analysis_shap_time_plot(self):
        # plot
        print(f'Plotting...')
        f, Ax = mpl.pyplot.subplots(2, 1, figsize=(10, 8),sharex=True)
        node = np.arange(0, self.ana_data.shape[-1], self.granularity)
        for i in range(2):
            ax = Ax[i]
            ax.plot(self.ana_data[i])
            ax.set_xlim(0, self.ana_data.shape[-1])
            ax_t = ax.twinx()
            ax_t.bar(node+1/2*self.granularity, self.output[i,:,i], width=self.granularity, alpha=0.5, color='r')
            ax.set_ylabel('Data')
            ax_t.set_ylabel('SHAP value')
            ax.set_title(f'{i}th class - {i}th class')
        ax.set_xlabel('Node')

    def analysis_shap_fft_plot(self):
        # plot
        print(f'Plotting...')
        colors = ['r','b','c','y','g','m','k','w']
        fft_x = fft_cut(self.ana_data)
        f = np.arange(fft_x.shape[-1])/self.ana_data.shape[-1]*2
        f_delta = f[1]-f[0]
        bar_x = np.append(f[self.index[:-1]]-f_delta/2,max(f)+-f_delta/2)
        fig, Ax = mpl.pyplot.subplots(self.output.shape[-1], 1, figsize=(15, 3*self.output.shape[-1]), sharex=True)
        # fig.patch.set_alpha(0) # transparent
        for i in range(len(self.ana_data)): # sample
            ax = Ax[i]
            ax.plot(f, np.abs(fft_x[i]), color='grey')
            ax.set_xlim(f.min(),f.max())
            ax_t = ax.twinx()
            for j in range(self.output.shape[-1]): # class
                my_bar_plot(bar_x, self.output[i,:,j], ax=ax_t,color=colors[j], alpha=0.6)
            y_lim_max = np.max(np.abs(self.output[i,:,:]))*1.2
            ax_t.set_ylim(-y_lim_max, y_lim_max)
            ax.set_ylabel('Data')
            ax_t.set_ylabel('SHAP value')
            ax.set_title(f'{self.dataset_label_name[i]} sample: {self.predict[i,:].round(2)}')
            if i == len(self.ana_data)-1:
                ax.set_xlabel('Frequency')
            fig.tight_layout()
            leg = ax_t.legend([f'{self.dataset_label_name[j]}' for j in range(self.output.shape[-1])])

        fig.savefig(os.path.join(self.postfile_path,f'analysis_{self.analysis_method:s}-{self.granularity:d}-fft-plot.png'), transparent=True,dpi=300)


def my_bar_plot(x,y,ax=None,*args,**kwargs):
    '''

    :param x: (n+1,)
    :param y: (n,)
    :param ax: axes
    :param args:
    :param kwargs:
    :return:
    '''
    if ax is None:
        ax = mpl.pyplot.gca()
    x = np.repeat(x,repeats=2)[1:-1]
    y=np.repeat(y,repeats=2)
    ax.fill_between(x,y,*args,**kwargs)
    return ax

if __name__ == '__main__':
    dir_path = r'E:\OneDrive - sjtu.edu.cn\6-SoftwareFiles\GitFiles\0-个人库\03-科研\2024-PerturbationNet\b_Simulation\checkpoint\periodicV2-1024-0509-215226'
    main = Main(dir_path)
