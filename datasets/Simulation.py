import os
import pickle
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

import datasets.datasets_utils.sequence_aug as aug1d
from analysis.ana_utils.utils_SHAP import find_all_index_N
from datasets.datasets_utils.DatasetsBase import dataset, signal_plot
from datasets.get_file.Simulation_get_files import generate_simu_signals_all

# 随机数设置
seed = 999
np.random.seed(seed)  # seed是一个固定的整数即可
random.seed(seed)


class Simulation(object):
    num_classes = 3
    inputchannel = 1
    Fs = 1e4
    def __init__(self, args):
        self.args = args

    @staticmethod
    def _preload(pre_file_dir, args, try_preload=True):
        if os.path.exists(pre_file_dir) and try_preload:
            with open(pre_file_dir, 'rb') as f:
                data_pd, storage_args, label_name = pickle.load(f)
                if storage_args == args:
                    return data_pd, label_name

        list_data, list_labels, label_name, infos = generate_simu_signals_all \
            (L=args['signal_size'], SNR=args['SNR'], N_sample=args['N_sample'])
        # infos: [N_sample,(f_c,f_f,A.phi),N_sub]
        data_pd = pd.DataFrame({"data": [item[np.newaxis, :] for item in list_data],
                                "label": list_labels, "infos": [item for item in infos]})
        data_pd['label'] = data_pd['label'].astype(int)
        with open(pre_file_dir, 'wb') as f:
            pickle.dump((data_pd, args, label_name), f)
        return data_pd, label_name

    def data_preprare(self, signal_size=2000, SNR=None, try_preload=True, plot=False):
        test_size = self.args['test_size'] if 'test_size' in self.args.keys() else 0.3
        # print("\n"+str(SNR))
        args = {'signal_size': signal_size, 'SNR': SNR, 'N_sample': 5000}  # 用于生成数据的参数
        data_pd, label_name = self._preload \
            (os.path.join(self.args['data_dir'], 'data_buffer.pkl'), args, try_preload=try_preload)

        if plot:
            index_plot = find_all_index_N(data_pd['label'].to_numpy(), 1)
            signal_plot(data=np.vstack(data_pd['data'][index_plot].values.tolist()),
                        label=[label_name[item] for item in data_pd['label'][index_plot].values.tolist()],
                        Fs=self.Fs, save_path=os.path.join(self.args['data_dir'], 'signal_plot'))

        index_list = list(StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=0)
                          .split(np.ones(data_pd.shape[0]), data_pd["label"].to_list()))
        train_pd, val_pd = data_pd.iloc[index_list[0][0]], data_pd.iloc[index_list[0][1]]

        transform = {
            'train': aug1d.Compose([aug1d.to_numpy(), aug1d.Data_type(self.args['data_type']),
                                    aug1d.Normalize(self.args['normlizetype']), aug1d.Retype()]),
            'val': aug1d.Compose([aug1d.to_numpy(), aug1d.Data_type(self.args['data_type']),
                                  aug1d.Normalize(self.args['normlizetype']), aug1d.Retype()]), }
        train_dataset = dataset(list_data=train_pd, transform=transform['train'])
        val_dataset = dataset(list_data=val_pd, transform=transform['val'])
        return (train_dataset, val_dataset), label_name



if __name__ == '__main__':
    args = {'data_type': 'time', 'normlizetype': 'mean-std', 'test_size': 0.3,
            'data_dir': r'E:\OneDrive - sjtu.edu.cn\6-SoftwareFiles\GitFiles\0-个人库\03-科研\2024-CSCohSHAP\checkpoint\Simulation_data', }
    simulation = Simulation(args)

    a, b = simulation.data_preprare(plot=True)
    print(1)
