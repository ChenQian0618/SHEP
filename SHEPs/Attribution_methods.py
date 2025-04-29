import sys,os
def root_path_k(x, k): return os.path.abspath(
    os.path.join(x, *([os.pardir] * (k + 1))))
# add the project directory to the system path
sys.path.insert(0, root_path_k(__file__, 1))

import numpy as np
from tqdm import tqdm
import shap
from SHEPs.utils_SHAP_MyIndependent import MyIndependent
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

    def attribute(self, xs):
        '''
        :param xs: (M, L) or (L)
        :return: (M, L, c)
        '''
        self.shap_values = self.explainer(xs,
                                          max_evals=self.max_evals)  # max(int(1e3), int(Z.shape[-1] * 10) // 2)
        return self.shap_values.values


class Attr_SHEP_Remove(object):
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
        if save_flag and self.save_dir:  # save the result for further analysis
            savemat(os.path.join(self.save_dir, 'attribute_res.mat'), {'res': res, 'res_info': 'res: (N, L, c_b, c)'})
        return res.mean(axis=-2)  # (M, L, c)

    def explain_row(self, x):
        return self._exchange_row(x, mode='U')

    def _exchange_row(self, x, mode='U'):
        '''
        U - U/{i}
        :param x: (1, L)
        :param mode: 'U' or 'O'
        :return: (L, c_b, c)
        '''
        N = self.background.shape[0]  # number of background samples
        L = self.background.shape[1]  # number of patchs
        if mode == 'U': # SHEP-Remove
            # 1) get base value
            base_value = self.func_predict(x)  # (1, c)
            # 2) generate mask and tile input
            mask = np.diag(np.ones(L)).astype(bool)
            order = 1  # do nothing (base - pred)
        elif mode == 'O': # SHEP-Add
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
        # 4) predict
        pred = self.func_predict(input_mask).reshape(L, N, -1)  # (L, N, c)
        res = order * (base_value - pred)  # (L, N, c)
        res = res.reshape(L, res.shape[-1], -1, res.shape[-1])  # (L, c_b, N/c_b, c)
        res = res.mean(axis=-2)  # average over N-dim (background) (L, c_b, c): evaluate the impact of each patch based on each class background
        return res


class Attr_SHEP_Add(Attr_SHEP_Remove):
    def explain_row(self, x):
        return self._exchange_row(x, mode='O')


class Attr_SHEP(Attr_SHEP_Remove):
    def explain_row(self, x):
        res = self._exchange_row(x, mode='O')
        res += self._exchange_row(x, mode='U')
        return res / 2