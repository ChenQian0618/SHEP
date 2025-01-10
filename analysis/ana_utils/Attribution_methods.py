import os.path

import numpy as np
from tqdm import tqdm
import shap
from analysis.ana_utils.shap_MyIndependent_ValueAssign import MyIndependent
import matplotlib.pyplot as plt
from scipy.io import savemat
import copy


class Attr_SHAP(object):
    def __init__(self, func_predict, background, max_eval=0, **kwargs):
        self.func_predict = func_predict
        self.background = background
        self.background_masker = MyIndependent(background)
        self.explainer = shap.Explainer(func_predict, self.background_masker, algorithm='permutation')
        if max_eval:
            self.max_evals = max_eval
        else:
            self.max_evals = int(background.shape[-1] * 10) // 2

    def attribute(self, input):
        self.shap_values = self.explainer(input,
                                          max_evals=self.max_evals)  # max(int(1e3), int(Z.shape[-1] * 10) // 2)
        return self.shap_values.values

class Attr_SHAP_dev(object): # 用于按照background的值进行分类SHAP
    def __init__(self, func_predict, background, save_dir=None, max_eval=0, **kwargs):
        self.save_dir = save_dir
        self.func_predict = func_predict
        self.backgrounds,self.backlabels = self._background_group(func_predict, background) # (c, N, L), (c)
        self.background_maskers = [MyIndependent(item) for item in self.backgrounds]
        self.explainers = [shap.Explainer(func_predict, item, algorithm='permutation')
                           for item in self.background_maskers]
        if max_eval:
            self.max_evals = max_eval
        else:
            self.max_evals = int(background.shape[-1] * 10) // 2
    @staticmethod
    def _background_group(func_predict, background):
        label = np.argmax(func_predict(background),axis=-1)
        res_sample,res_class = [],sorted(set(label))
        for item in res_class:
            res_sample.append(background[label == item])
        return res_sample,res_class

    def attribute(self, input, save_flag=True):
        '''
        :param input: (M, L)
        :return:
        '''
        shap_values = []
        for explainer in self.explainers:
            shap_values.append(explainer(input,max_evals=self.max_evals).values)
        shap_values = np.array(shap_values) # (c_b, M, L, c_p)
        shap_values = shap_values.transpose([1,2,0,3]) # -> # (M, L, c_b, c)
        plt.close('all')
        fig, axs = plt.subplots(1, shap_values.shape[-1], figsize=[4, 1], dpi=600)
        for i, ax in enumerate(axs):
            ax.cla()
            ax.plot(shap_values[-1, 1:, :, i])
            ax.legend(['#1', '#2', '#3'])
        fig.tight_layout()
        if save_flag and self.save_dir:  # save the result for further analysis
            savemat(os.path.join(self.save_dir, 'attribute_res.mat'),
                    {'res': shap_values, 'res_info': 'res: (N, L, c_b, c)'})
        return shap_values.mean(axis=-2)  # (M, L, c)


class Attr_Exchange(object):
    def __init__(self, func_predict, background, save_dir=None, **kwargs):
        '''

        :param func_predict: function
        :param background: (N, L) or (L)
        :param kwargs:
        '''
        self.func_predict = func_predict
        self.background = background
        self.save_dir = save_dir

    def attribute(self, xs, save_flag=False):
        '''
        :param xs: (M, L) or (L)
        :param save_flag: bool (default: False) save the result for further analysis
        :return: (M, L, c)
        '''

        if xs.ndim == 1:
            xs = xs[np.newaxis, ...]
        res = []
        for i in tqdm(range(xs.shape[0])):
            res.append(self.explain_row(xs[i:i + 1, ...]))  # (L, c_b, c)
        res = np.array(res)  # (M, L, c_b, c)
        # plt.close('all')
        # fig, axs = plt.subplots(1, res.shape[-1], figsize=[4, 1], dpi=600)
        # for i, ax in enumerate(axs):
        #     ax.cla()
        #     ax.plot(res[4, 1:, :, i])
        #     ax.legend(['#1', '#2', '#3'])
        # fig.tight_layout()
        if save_flag and self.save_dir:  # save the result for further analysis
            savemat(os.path.join(self.save_dir, 'attribute_res.mat'), {'res': res, 'res_info': 'res: (N, L, c_b, c)'})
        return res.mean(axis=-2)  # (M, L, c)

    def explain_row(self, x, temp_flag=False):
        return self._exchange_row(x, mode='U', temp_flag=temp_flag)

    def _exchange_row(self, x, mode='U', temp_flag=False):
        '''
        U - U/{i}
        :param x: (1, L)
        :param mode: 'U' or 'O'
        :return: (L, c_b, c)
        '''
        N = self.background.shape[0]  # number of background samples
        L = self.background.shape[1]  # number of patchs
        if mode == 'U':
            # 1) get base value
            base_value = self.func_predict(x)  # (1, c)
            # 2) generate mask and tile input
            mask = np.diag(np.ones(L)).astype(bool)
            order = 1  # do nothing (base - pred)
        elif mode == 'O':
            # 1) get base value
            base_value = self.func_predict(self.background)  # (N, c)
            # 2) generate mask and tile input
            mask = ~np.diag(np.ones(L)).astype(bool)
            order = -1  # reverse the order (pred-base)
        else:
            raise ValueError('mode should be "U" or "O"')
        # 3) generate perturbations
        input_mask = np.tile(x, (L * N, 1))
        for i, m in enumerate(mask):
            input_mask[i * N:(i + 1) * N, m] = self.background[:, m]
        # ) check the output works well
        if temp_flag:
            a = np.zeros_like(input_mask, dtype=int)
            for i in range(a.shape[0]):
                for j in range(a.shape[1]):
                    a[i, j] = int(np.all(input_mask[i, j] == x[0, j]))
            b = np.zeros_like(input_mask, dtype=int)
            for i in range(b.shape[0]):
                for j in range(b.shape[1]):
                    b[i, j] = int(np.all(input_mask[i, j] == self.background[i % N, j]))
        # 4) predict
        pred = self.func_predict(input_mask).reshape(L, N, -1)  # (L, N, c)
        res = order * (base_value - pred)  # (L, N, c)
        res = res.reshape(L, res.shape[-1], -1, res.shape[-1])  # (L, c_b, N, c)
        res = res.mean(axis=-2)  # average over N-dim (background) (L, c_b, c): evaluate the impact of each patch based on each class background
        return res


class Attr_Exchange_v2(Attr_Exchange):
    def explain_row(self, x, temp_flag=False):
        return self._exchange_row(x, mode='O', temp_flag=temp_flag)


class Attr_Exchange_v3(Attr_Exchange):
    def explain_row(self, x, temp_flag=False):
        res = self._exchange_row(x, mode='O', temp_flag=temp_flag)
        res += self._exchange_row(x, mode='U', temp_flag=temp_flag)
        return res / 2


class Attr_Mask(object):
    def __init__(self, func_predict, **kwargs):  # background is not needed
        self.func_predict = func_predict

    def attribute(self, xs):
        if xs.ndim == 1:
            xs = xs[np.newaxis, ...]
        res = []
        for i in tqdm(range(xs.shape[0])):
            res.append(self.explain_row(xs[i:i + 1, ...]))  # ( L, N, c)
        res = np.array(res)  # (M, L, N, c)
        # plt.close('all')
        # fig, axs = plt.subplots(1, res.shape[-1], figsize=[4, 1], dpi=600)
        # for i, ax in enumerate(axs):
        #     ax.cla()
        #     ax.plot(res[::2, 1:, 2, i].T)
        #     # ax.plot(res.mean(axis=-2)[::2, 1:, i].T)
        #     ax.legend(['#1', '#2', '#3'])
        # fig.tight_layout()
        return res.mean(-2)  # (M, L, c)

    def explain_row(self, x):
        N = 1
        L = x.shape[1]  # number of patchs
        x = copy.deepcopy(x)
        # 1) get base value
        base_value = self.func_predict(x)  # (1, c)
        # 2) generate mask and tile input
        mask = np.diag(np.ones(L)).astype(bool)
        # 3) generate
        input_mask = np.tile(x, (L * N, 1))
        for i, m in enumerate(mask):
            input_mask[i * N:(i + 1) * N, m] *= 0
        # 4) predict
        pred = self.func_predict(input_mask).reshape(L, N, -1)  # (L, N, c)
        res = base_value - pred  # (L, N, c)
        return res  # (M, L, c)


class Attr_Scale(Attr_Mask):
    def explain_row(self, x, scales=(0.25,0.5,0.75)):
        N = len(scales)
        L = x.shape[1]  # number of patchs
        x = copy.deepcopy(x)
        # 1) get base value
        base_value = self.func_predict(x)  # (1, c)
        # 2) generate mask and tile input
        mask = np.diag(np.ones(L)).astype(bool)
        # 3) generate
        input_mask = np.tile(x[np.newaxis,...], (L, N, 1))
        for i, m in enumerate(mask):
            for j, scale in enumerate(scales):
                input_mask[i, j, m] *= scale
        input_mask = input_mask.reshape(L * N, -1)
        a = np.zeros_like(input_mask, dtype=int)
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                a[i, j] = int(np.all(input_mask[i, j] == x[0, j]))
        # 4) predict
        pred = self.func_predict(input_mask).reshape(L, N, -1)  # (L, N, c)
        res = base_value - pred  # (L, N, c)
        return res


if __name__ == '__main__':
    a = np.empty((3, 2), dtype=object)
    for i in range(3):
        a[i, 0] = np.array([1, 2]) * (i + 1)
        a[i, 1] = np.empty((2,), dtype=object)
        a[i, 1][0] = np.array([1, ]) * (i + 1)
        a[i, 1][1] = np.array([1, 2, 3]) * (i + 1)
    print(1)
