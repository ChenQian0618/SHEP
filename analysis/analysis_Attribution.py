'''
2024/07/18
Created by ChenQian (SJTU)
This script is used to analyze the SHAP values (and CAM values) of the models under different situations
(e.g., models, transformation function, grid, etc.)
'''

import copy
import os
import pickle
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import torch
from scipy.io import savemat

import datasets
import models
from ana_utils.Plot_t_SNE import tSNE
from ana_utils.utils_SHAP import find_all_index_N
from ana_utils.utils_SHAP import func_trans_time, func_trans_frequency, func_trans_STFT, func_trans_CS, \
    func_trans_envelope
from ana_utils.utils_SHAP import func_trans_frequency_v2, func_trans_STFT_v2, func_trans_CS_v2, \
    func_trans_envelope_v2
from ana_utils.utils_SHAP import attr_visualization,data_visualization
from ana_utils.Attribution_methods import Attr_SHAP, Attr_Exchange, Attr_Exchange_v2, Attr_Exchange_v3, \
    Attr_Mask, Attr_Scale, Attr_SHAP_dev
from utils.PostProcessLib import ExtractInfo
from utils.plot_func import setdefault

mybool = lambda x: x.lower() in ['yes', 'true', 't', 'y', '1']
mylist = lambda x: [float(item) for item in x.strip(' ,').split(',')]


class Base_Analysis(object):
    def __init__(self, dir=None, flag_preload_dataset=True, fastmode=True):
        '''
        :param dir: the directory of the model
        :param Fs: the frequency of the signal, for plotting the x-axis
        '''
        setdefault()

        self.flag_load_model_weight = True  # whether load the model weight
        self.flag_preload_dataset = flag_preload_dataset  # whether load the model weight
        self.data_train_not_val = False  # whether use the training data or the validation data
        # self.selcted_index = None # 选择后续分析的数据
        self.device = torch.device('cpu')
        self.dir = dir
        self.args, Dict = ExtractInfo(os.path.join(self.dir, "training.log"))
        # args process
        self.args['SNR'] = None if self.args['SNR'] == 'None' else float(self.args['SNR'])
        self.args['test_size'] = None if self.args['test_size'] == 'None' else float(self.args['test_size'])
        # 后处理路径
        self.save_dir = os.path.join(self.dir, 'PostProcess_of_FullAnalysis')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        # 预处理
        self._get_all_data(self.flag_preload_dataset)  # self.data, self.label, self.label_name
        self._model_load()  # self.model
        if not fastmode:
            self._prediction_record()  # self.z, self.logits, self.prop, self.predict, self.misclassification, 做t-SNE分析和绘制混淆矩阵

    def _get_all_data(self, flag_preload_dataset, Target_number=100, data_plot=True):
        print('------begin data load--------')
        phase = 'train' if self.data_train_not_val else 'val'

        # (self.save_dir, f'all_dataset_{phase:s}_SavedInTraining.pkl') |  (self.save_dir, f'all_dataset_{phase:s}.pkl')
        data_path = os.path.join(self.dir, f'all_dataset_{phase:s}_SavedInTraining.pkl')

        # try to load the data
        if flag_preload_dataset and os.path.exists(data_path):
            with open(data_path, 'rb') as f:
                temp = pickle.load(f)
            print('data load done! (by preload)')
        else:
            # Load the datasets

            subargs = {k: self.args[k] for k in ['data_dir', 'data_type', 'normlizetype', 'test_size'] +
                       ['data_name', 'data_signalsize', 'SNR']}
            Dataset = getattr(datasets, subargs['data_name'])
            (train_dataset, val_dataset), label_name = Dataset(subargs).data_preprare(
                signal_size=int(subargs['data_signalsize']), SNR=subargs['SNR'], try_preload=True)

            # collect data
            phases = ['val', 'train'] if self.data_train_not_val else ['train',
                                                                       'val']  # do both, but select train/val in the end
            for temp_phase in phases:  # do both
                ToBeCollect = train_dataset if temp_phase == 'train' else val_dataset
                temp_data_path = os.path.join(os.path.split(data_path)[0], f'all_dataset_{temp_phase:s}.pkl')
                data, label, infos = [], [], []
                for i in range(len(ToBeCollect)):
                    item, label_, *info = ToBeCollect[i]
                    data.append(item)
                    label.append(label_)
                    infos.append(info)
                data, label, infos = np.array(data).squeeze(), np.array(label).squeeze(), np.array(infos)
                Index = find_all_index_N(label, Target_number)

                temp = {'data': data[Index, ...], 'label': label[Index, ...], 'infos': infos[Index, ...],
                        'label_name': [item.strip().capitalize() for item in label_name],
                        'Fs': Dataset.Fs, 'subargs': subargs}
                with open(temp_data_path, 'wb') as f:
                    pickle.dump(temp, f)
            print('data load done! (by dataset)')
        self.data, self.label, self.label_name = temp['data'], temp['label'].squeeze(), temp['label_name']
        self.infos = temp['infos']
        print('data[0,0]: ', temp['data'][0][0])
        self.Fs = temp['Fs']

        # determine the analysis data
        index2 = find_all_index_N(self.label, 5)
        print('The index of selected data for analysis', index2.tolist())
        self.inputs_ana = self.data[index2, ...]
        self.labels_ana = self.label[index2, ...]
        self.infos_ana = self.infos[index2, ...]

        # determine the background data
        Index = find_all_index_N(self.label, squeeze=False)[:, -5:].reshape(-1)
        self.inputs_bak = self.data[Index, ...]  # for SHAP analysis (as background)
        self.label_bak = self.label[Index, ...]

        # analysis data visualization
        if data_plot:
            data_visualization(self.inputs_ana, self.labels_ana, self.label_name, self.Fs,
                               os.path.join(self.save_dir, 'DataShow'),data_name=self.args['data_name'])

    def _model_load(self):
        # search
        print('------begin model load--------')
        model_name = self.args['model_name']
        # Define the models
        Dataset = getattr(datasets, self.args['data_name'])
        self.model = getattr(models, model_name)(in_channel=Dataset.inputchannel, out_channel=Dataset.num_classes)
        # Invert the models and define the loss
        self.model.to(self.device)
        if self.flag_load_model_weight:
            tempfiles = next(os.walk(self.dir))[2]
            chose_one = [i for i in tempfiles if i.endswith('.pth')]  # choose .pth
            if len(chose_one) == 0:
                raise ValueError('No model weight file found!')
            chose_one = [item for item in chose_one if 'best' in item][0]  # choose best or final
            storage = torch.load(os.path.join(self.dir, chose_one))
            self.model.load_state_dict(storage['state_dict'])
        print('model load done!')
        self.model.eval()

    def _prediction_record(self):
        # 1) get the data
        temp = self.data if len(self.data.shape) == 3 else self.data[:, np.newaxis, ...]
        input = torch.tensor(temp).to(self.device)
        label = self.label
        # 2) get the distance and the predict
        with torch.set_grad_enabled(False):
            logits, z = self.model(input, verbose=True)
        self.z = z.detach().cpu().numpy()
        self.logits = logits.detach().cpu().numpy()
        self.prop = torch.nn.functional.softmax(logits, dim=-1).detach().cpu().numpy()
        self.predict = self.logits.argmax(axis=-1)
        # 计算re_order
        temp = np.array([i[j] for i, j in zip(self.prop, label)])  # 对应label下的logit
        re_order = sorted(range(len(temp)), key=lambda k: temp[k], reverse=True)  # 高概率在前，低概率在后
        # 3) find misclassification
        misclassification_index = np.where(self.predict != label)[0]
        confusion_matrix_item = [[[] for i in range(len(self.label_name))]
                                 for j in range(len(self.label_name))]  # [Label,Predict]
        confusion_matrix = np.zeros((len(self.label_name), len(self.label_name)))
        for i, item_p, item_l in zip(re_order, self.predict[re_order], label[re_order]):
            confusion_matrix_item[item_l][item_p].append(i)
            confusion_matrix[item_l][item_p] += 1
        self.misclassification = {
            'index': misclassification_index,
            'confusion_matrix': confusion_matrix,
            'confusion_matrix_item': confusion_matrix_item}

        # t-SNE
        self._plot_tSNE()
        # confusion_matrix
        self._plot_confusion_matrix()

    def _plot_tSNE(self):
        savedir = os.path.join(self.save_dir, 't_SNE')
        # tSNE
        if not os.path.exists(savedir):
            os.mkdir(savedir)
        # 重新排序
        phase = 'train' if self.data_train_not_val else 'val'
        tSNE(self.z, self.label, os.path.join(savedir, f"tSNE-{phase}"))

    def _plot_confusion_matrix(self):
        savedir = os.path.join(self.save_dir, 'Confusion_matrix')
        # confusion_matrix
        if not os.path.exists(savedir):
            os.mkdir(savedir)
        f, ax = plt.subplots()  # figsize=[8/2.54, 6/2.54], dpi=1000
        sns.heatmap(self.misclassification['confusion_matrix'].astype(int), annot=True, ax=ax, fmt="d",
                    cmap=sns.color_palette("light:b", as_cmap=True),
                    linewidths=0.3, annot_kws={'size': mpl.rcParams['font.size']})
        ax.invert_yaxis()
        ax.set_xticklabels(self.label_name)
        ax.set_yticklabels(self.label_name)
        ax.set_xlabel('Prediction')
        ax.set_ylabel('Truth')
        f.tight_layout(pad=0.2)
        phase = 'train' if self.data_train_not_val else 'val'
        name = os.path.join(savedir, f"confusion_matrix_{phase}")
        f.savefig(name + '.svg')
        f.savefig(name + '.jpg')

    def SHAP_analysis(self, preload=True, mode='CS', patch_mode='1', attr_name='SHAP'):

        # 0) attribution domain
        mode_map = {'time': func_trans_time,
                    'frequency': func_trans_frequency,
                    'envelope': func_trans_envelope,
                    'STFT': func_trans_STFT,
                    'CS': func_trans_CS,
                    'frequency_v2': func_trans_frequency_v2,
                    'envelope_v2': func_trans_envelope_v2,
                    'STFT_v2': func_trans_STFT_v2,
                    'CS_v2': func_trans_CS_v2, }
        if mode not in mode_map.keys():
            raise ValueError('mode should be in ', mode_map.keys())
        func_trans = mode_map[mode]

        # 0) attribution model
        attrModel_map = {'SHAP': Attr_SHAP,
                         "SHAP_dev":Attr_SHAP_dev,
                         'Exchange': Attr_Exchange,
                         'Mask': Attr_Mask,
                         'Scale': Attr_Scale,
                         'Exchange_v2': Attr_Exchange_v2,
                         'Exchange_v3': Attr_Exchange_v3, }
        if attr_name not in attrModel_map.keys():
            raise ValueError('attr_name should be in ', attrModel_map.keys())
        Attr_Model = attrModel_map[attr_name]

        # 1) init the transformation
        func_Z, func_Z_inv, func_unpatch, trans_dict = func_trans(patch_mode)
        '''
        func_Z: z=func_Z(x), the transformation function
        func_Z_inv: x=func_Z_inv(z), the inverse transformation function
        func_unpatch: data [,shape] = func_unpatch(data [,shape]), unpatch the patched data to the original domain
        trans_dict: {'name':.,'trans_series':.}                
        '''

        def func_M(input, device=self.device):
            if isinstance(self.model, torch.nn.Module):
                input = input if len(input.shape) == 3 else input[:, np.newaxis, ...]
                input = torch.tensor(input, dtype=torch.float32).to(device)
                with torch.no_grad():
                    logits = self.model(input)
                res = logits.detach().cpu().numpy()
            else:
                res = self.model(input)
            return res

        func_predict = lambda x: func_M(func_Z_inv(x))

        # 2) get the data
        inputs_bak = copy.deepcopy(self.inputs_bak)  # for SHAP analysis (as background)
        inputs_ana = copy.deepcopy(self.inputs_ana)
        labels_ana = copy.deepcopy(self.labels_ana)

        # 3) test
        y,info1 = func_Z(inputs_ana, verbose=True)
        y_inv,info2 = func_Z_inv(y[-1], verbose=True)
        print('mean error: ', abs(y_inv[-1] - inputs_ana).mean())
        predict_ana = func_predict(y[-1])
        print(np.argmax(predict_ana, -1))

        # 4) SHAP
        savedir = os.path.join(self.save_dir, f'attribution-{mode:s}-{attr_name:s}-{patch_mode:s}')
        if not os.path.exists(savedir):
            os.mkdir(savedir)
        savepath = os.path.join(savedir, f'attribution_values.pkl')

        with open(os.path.join(savedir, 'verbose_info.txt'), 'w') as f: # save verbose info
            f.write('\n'.join(info1+['-'*30]+info2))
        # 4.1) try preload
        inputs_ana_Z = func_Z(inputs_ana)
        preload_OK = False
        if os.path.exists(savepath) and preload:
            with open(savepath, 'rb') as f:
                save_dict = pickle.load(f)
            attr_value, attr_data = save_dict['attr_value'], save_dict['attr_data']
            inputs_load = save_dict['inputs_ana']
            print('inputs_load[0,0]: ', inputs_load[0][0])
            preload_OK = True if inputs_load.shape == inputs_ana.shape and \
                                 np.equal(inputs_load, inputs_ana).all() and \
                                 (inputs_ana_Z.shape == attr_data.shape) else False
        print('Preload: ', preload_OK)

        if not preload_OK:
            attr_data = inputs_ana_Z
            # 4.2) SHAP analysis
            start_time = time.time()
            attr_model = Attr_Model(func_predict=func_predict, background=func_Z(inputs_bak), save_dir=savedir)
            attr_value = attr_model.attribute(inputs_ana_Z)  # (N, L, C)

            analyse_time = time.time() - start_time
            print('Analyse time: ', analyse_time)
            analyse_time /= inputs_ana.shape[0] # per sample

            # 4.3) save SHAP result
            predict_ana = func_M(inputs_ana)
            predict_ana_logit = torch.nn.functional.softmax(torch.tensor(predict_ana), -1).numpy()
            print('(saved) inputs_ana[0,0]: ', inputs_ana[0][0])
            save_dict = {'attr_value': attr_value, 'attr_data': attr_data,
                         'inputs_ana': inputs_ana, 'labels_ana': labels_ana,
                         'analyse_time': analyse_time}
            with open(savepath, 'wb') as f:
                pickle.dump(save_dict, f)  # save attribution-data

            # 4.4) save predict value
            temp_dict = {'predict_logit': predict_ana, 'predict_prob': predict_ana_logit,
                         'labels': labels_ana, 'analyse_time': analyse_time}
            savemat(
                os.path.join(savedir, f'prediction-input{int(abs(inputs_ana[0][0]*1e4)):.0f}-time{analyse_time:.0f}.mat'),
                temp_dict)  # save prediction-data

        # 5) visualization
        out_data, out_value = func_unpatch(attr_data, attr_value)  # (N, d1, d2), (C, N, d1, d2)
        # avoid complex situation

        out_data = np.abs(out_data) if 'omplex' in out_data.dtype.__class__.__name__ else out_data
        out_value = np.real(out_value) if 'omplex' in out_value.dtype.__class__.__name__ else out_value
        # 5.1) save visualization data
        visualization_dir = os.path.join(self.save_dir, 'Visualization')
        if not os.path.exists(visualization_dir):
            os.mkdir(visualization_dir)
        savepath = os.path.join(visualization_dir,
                                f'attribution-{mode:s}-{attr_name:s}-{patch_mode:s}.pkl')  # save plot-data
        save_dict = {'mode': mode, 'patch_mode': patch_mode, 'attr_name': attr_name,
                     'Fs': self.Fs, 'signals': inputs_ana, 'data': out_data, 'value': out_value,
                     'label_predicts': (labels_ana, predict_ana), 'label_name': self.label_name}
        if trans_dict['trans_series'][0].__class__.__name__ == 'trans_STFT':
            func_stft = trans_dict['trans_series'][0]
            hop, win_len, mfft = func_stft.SFT.hop, func_stft.SFT.win.shape[-1], func_stft.SFT.mfft
            win_g, keep_N = func_stft.win_g, func_stft.keep_N
            STFTs_params = {'hop': hop, 'win_len': win_len, 'mfft': mfft, 'win_g': win_g, 'keep_N': keep_N}
            save_dict['STFTs_params'] = STFTs_params
        with open(savepath, 'wb') as f:
            pickle.dump(save_dict, f)  # save plot-data
        # 5.2) visualization
        attr_visualization(savedir=savedir, **save_dict)


if __name__ == '__main__':
    temp_path = r'E:\OneDrive - sjtu.edu.cn\6-SoftwareFiles\GitFiles\0-个人库\03-科研\2024-PerturbationNet\checkpoint\ExpSimu'
    temp_names = ['(Statistic)CNN-Simulation-time-SNR0-1211-215743']

    #
    # temp_path = r'..\checkpoint\test'
    # temp_names = ['CNN-Simulation-time-SNR0-1211-215743']

    for temp_name in temp_names:
        print('\n' * 3, f"{temp_names.index(temp_name) + 1}/{len(temp_names)}: {'-' * 20} {temp_name} {'-' * 20}")

        temp_dir = os.path.join(temp_path, temp_name)
        base = Base_Analysis(dir=temp_dir, flag_preload_dataset=True, fastmode=True)

        #  ----------------------------attribution analysis------------------------------------------------
        modes = ['frequency_v2', 'envelope_v2', 'STFT_v2', 'CS_v2']  # 'time', 'frequency', 'envelope', 'STFT', 'CS', 'frequency_v2', 'envelope_v2', 'STFT_v2', 'CS_v2'
        patch_modes = ['0']  # '1','2','3','4','5'
        attr_names = ['Exchange_v3',]  #  'Exchange', 'Exchange_v2', 'Exchange_v3', 'Mask', 'Scale','SHAP'
        for i, mode in enumerate(modes):
            for j, patch_mode in enumerate(patch_modes):
                for k, attr_name in enumerate(attr_names):
                    cur = i * len(patch_modes) * len(attr_names) + j * len(attr_names) + k + 1
                    total = len(modes) * len(patch_modes) * len(attr_names)
                    print('\n' * 1, f'<{temp_names.index(temp_name) + 1:d}/{len(temp_names):d} - {temp_name:s}>:')
                    print('\n' * 1, '--' * 5 + ' ' * 2,
                          f'{cur:d}/{total:d}: {mode:s}-{patch_mode:s}-{attr_name:s}',
                          ' ' * 2 + '--' * 5)
                    # try:
                    base.SHAP_analysis(preload=True, mode=mode, patch_mode=patch_mode, attr_name=attr_name)
                    # except:
                    #     print('--'*10,'\nError!'+f'{mode:s}-{patch_mode:s}-{attr_name:s}')
                    #     continue
