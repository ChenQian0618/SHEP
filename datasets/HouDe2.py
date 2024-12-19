import os
import pickle
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

import datasets.datasets_utils.sequence_aug as aug1d
from analysis.ana_utils.utils_SHAP import find_all_index_N
from datasets.datasets_utils.DatasetsBase import dataset, signal_plot
from datasets.datasets_utils.generalfunciton import balance_label
from datasets.get_file.HouDe2_get_file import get_files

# 随机数设置
seed = 999
np.random.seed(seed)  # seed是一个固定的整数即可
random.seed(seed)


class HouDe2(object):
    downsample = 2
    num_classes, inputchannel, Fs = 4, 1, 10e3/downsample
    condition = 0

    def __init__(self, args):
        '''
        :param args: dict, ['data_dir', 'normlizetype', 'test_size']
        '''
        self.args = args

    @staticmethod
    def _preload(pre_file_dir, args, try_preload=True):
        if os.path.exists(pre_file_dir) and try_preload:
            with open(pre_file_dir, 'rb') as f:
                data_pd, storage_args, label_name = pickle.load(f)
                if storage_args == args:
                    return data_pd, label_name

        list_data, label_name = get_files(args['data_dir'], signal_size=args['signal_size'], SNR=args['SNR'],
                                          downsample=HouDe2.downsample)
        data_pd = pd.DataFrame(
            {"data": [item for item in np.vstack(list_data[0])], "label_f": np.hstack(list_data[1]),
             "label_n": np.hstack(list_data[2]), 'time_index': np.hstack(list_data[-1])})
        data_pd['label_f'] = data_pd['label_f'].astype(int)
        data_pd['label_n'] = data_pd['label_n'].astype(int)
        with open(pre_file_dir, 'wb') as f:
            pickle.dump((data_pd,args,label_name), f)
        return data_pd,label_name

    def data_preprare(self, signal_size=2000, SNR=None, try_preload=True, plot=False):
        test_size = self.args['test_size'] if 'test_size' in self.args.keys() else 0.3
        # print("\n"+str(SNR))
        args = {'data_dir': self.args['data_dir'], 'signal_size': signal_size, 'SNR': SNR}
        data_pd, label_name = self._preload(os.path.join(self.args['data_dir'], 'data_buffer.pkl'),
                                            args, try_preload=try_preload)
        # preprocess
        index_list = data_pd['label_n'].apply(lambda x: x == HouDe2.condition)
        data_pd = data_pd[index_list]
        data_pd['label'] = data_pd['label_f'] # data_pd.apply(lambda x: (x['label_f'], x['label_n']), axis=1)
        data_pd.drop(['label_f','label_n'], axis=1, inplace=True)
        # balance the number of each class
        data_pd, number = balance_label(data_pd, 'label')  # balance the number of each class
        if plot:
            index_plot = find_all_index_N(data_pd['label'].to_numpy(), 1)
            signal_plot(data=np.vstack(data_pd['data'][index_plot].values.tolist()),
                        label=[label_name[item] for item in data_pd['label'][index_plot].values.tolist()],
                        Fs=self.Fs,
                        save_path=os.path.join(self.args['data_dir'],'signal_plot', f'signal_plot-{HouDe2.condition}'))

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


class Config:
    def __init__(self, param):
        for k, v in param.items():
            setattr(self, k, v)


if __name__ == '__main__':
    args = {'data_type': 'time', 'data_dir': r'E:\6-数据集\0-机械故障诊断数据集\9-厚德平行轴\振动响应\数据导出\斜齿轮',
            'normlizetype': 'mean-std', 'test_size': 0.3}
    houde = HouDe2(args)
    a, b = houde.data_preprare(try_preload=False, plot=True)
    print(1)
