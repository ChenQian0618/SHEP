import numpy as np
import pandas as pd
from tqdm import tqdm

class _basic_explainer(object):
    def __init__(self, model):
        self.model = model
    def __call__(self, input):
        '''
        :param input: np.array of shape (n_samples, n_features)
        :return:
        '''
        output = []
        for i in tqdm(range(input.shape[0])):
            output.append(self.explain_row(input[i:i+1,:]))
        return np.array(output)


    def explain_row(self, row):
        return {}


class Mask_explainer(_basic_explainer):
    def __init__(self, model):
        super().__init__(model)
    def explain_row(self, row):
        predict = self.model(row)
        output = np.zeros((row.shape[-1], predict.shape[-1]))
        for i in range(row.shape[-1]):
            row_ = row.copy()
            row_[:, i] = row_[:, i] * 0
            output[i:i+1, :] = self.model(row_)
        return predict-output

class Scale_explainer(_basic_explainer):
    def __init__(self, model, scale=0.5):
        self.scale =scale
        super().__init__(model)
    def explain_row(self, row):
        predict = self.model(row)
        output = np.zeros((row.shape[-1], predict.shape[-1]))
        for i in range(row.shape[-1]):
            row_ = row.copy()
            row_[:, i] = row_[:, i]*self.scale
            output[i:i+1, :] = self.model(row_)
        return (predict-output)/(1-self.scale+1e-6)

class My_explainer(_basic_explainer):
    def __init__(self, model,back_ground_data=None):
        self.back_ground_data = back_ground_data
        super().__init__(model)
    def explain_row(self, row):
        predict = self.model(row)
        output = np.zeros((row.shape[-1], predict.shape[-1]))
        for i in range(row.shape[-1]):
            row_ = row.copy()
            temp = []
            for j in range(self.back_ground_data.shape[0]):
                row_[0, i] = self.back_ground_data[j, i]
                temp.append(self.model(row_))
            output[i, :] = np.mean(np.vstack(temp), axis=0)
        return predict-output

