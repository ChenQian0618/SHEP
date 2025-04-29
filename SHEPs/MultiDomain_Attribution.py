'''
2025/04/25
Created by ChenQian-SJTU (chenqian2020@sjtu.edu.cn)
'''
import sys,os
def root_path_k(x, k): return os.path.abspath(
    os.path.join(x, *([os.pardir] * (k + 1))))
# add the project directory to the system path
sys.path.insert(0, root_path_k(__file__, 1))

import numpy as np
import copy, pickle, time
import torch
import shap

from SHEPs.DomainTransform import func_trans_time, func_trans_frequency, func_trans_envelope, func_trans_STFT, func_trans_CS
from SHEPs.utils_Visualization import attr_visualization
from SHEPs.Attribution_methods import Attr_SHAP, Attr_SHEP_Remove, Attr_SHEP_Add, Attr_SHEP

class MultiDomain_Attribution(object):
    def __init__(self, func_predict, background_data, save_dir):
        self.func_predict = func_predict
        self.background_data = background_data
        self.save_dir = save_dir 
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

    def explain(self, input_data, input_label, domain_mode='CS', patch_mode='1', method='SHEP', preload=True, Fs=1, plot=True):
        '''
        :param input_data: (N, L) or (L)
        :param input_label: (N, c) or (c)
        :param domain_mode: str, the mode of the transform function, e.g., 'time', 'frequency', 'envelope', 'STFT', 'CS'
        :param patch_mode: str, the mode of the patch function, i.e., '0' '1', '2', '3', '4', '5'
        :param method: str, the method of the SHAP analysis, i.e., 'SHEP', 'SHAP', 'SHEP_Remove', 'SHEP_Add'
        :param preload: bool, whether to preload the SHAP values
        :param Fs: int, the sampling frequency of the data
        :param plot: bool, whether to plot the SHAP values
        '''
        # deepcopy the input data
        input_data = copy.deepcopy(input_data) 
        # preparation
        func_Z, func_Z_inv, func_unpatch, trans_dict = self.mode_select(domain_mode,patch_mode)
        attr_method = self.method_select(method)
        func_predict_z = lambda x: self.func_predict(func_Z_inv(x))
        savedir = self.save_dir
        rawfile_savepath = os.path.join(savedir, f'{domain_mode:s}_#{patch_mode:s}_{method:s}_values_raw.pkl')

        # 1) test
        z = func_Z(input_data, verbose=True)
        print('',end='\n')
        z_inv = func_Z_inv(z[-1], verbose=True)
        print('mean error: ', abs(z_inv[-1] - input_data).mean())
        predict_ana = func_predict_z(z[-1])
        print('label of input sample:', np.argmax(predict_ana, -1))

        # 2) Try to preload the SHAP values
        preload_OK = False
        if os.path.exists(rawfile_savepath) and preload:
            with open(rawfile_savepath, 'rb') as f:
                save_dict = pickle.load(f)
            inputs_load = save_dict['input_data']
            attr_data = save_dict['attr_data']
            print('inputs_load[0,0]: ', inputs_load[0][0])
            preload_OK = True if inputs_load.shape == input_data.shape and \
                                 np.equal(inputs_load, input_data).all() and \
                                 (func_Z(inputs_load).shape == attr_data.shape) else False
            plot_params = trans_dict['get_plot_params'](input_data.shape[-1], Fs=Fs)
            save_dict['plot_params'] = plot_params
            with open(rawfile_savepath, 'wb') as f:
                pickle.dump(save_dict, f)
        print(f'Preload: True | analyze time of per sample: {save_dict["analyse_time"]/input_data.shape[0]:.1f}s'
              if preload_OK else 'Preload: False')
        # 3) if preload failed, calculate the SHAP values
        if not preload_OK:
            plot_params = trans_dict['get_plot_params'](input_data.shape[-1], Fs=Fs)
            # 3.1) conduct SHAP analysis
            attr_data = func_Z(input_data) # preprocessing the background data
            attr_model =attr_method(func_predict=func_predict_z, background=func_Z(self.background_data), save_dir=savedir)
            start_time = time.time()
            print('Start Attribution analysis, please wait...(several minutes / hours):')
            attr_value = attr_model.attribute(attr_data) # (N, L, c), the SHAP values of the input data
            analyse_time = time.time() - start_time
            print(f'Analyse time of {input_data.shape[0]:d} samples: {analyse_time:.1f}s')

            # 3.2) unpatch the SHAP values to the original domain
            domain_data, domain_value = func_unpatch(attr_data, attr_value)
            # domain_data: [N, L], domain_value: [c, N, L]
            domain_data = np.abs(
                domain_data) if 'omplex' in domain_data.dtype.__class__.__name__ else domain_data  # avoid complex situation
            domain_value = np.real(
                domain_value) if 'omplex' in domain_value.dtype.__class__.__name__ else domain_value  # avoid complex situation

            # 3.3) save SHAP result
            predict_logit = func_predict_z(func_Z(input_data))
            predict_prob = torch.nn.functional.softmax(torch.tensor(predict_logit), -1).numpy()
            print('(saved) input_data[0,0]: ', input_data[0][0])
            save_dict = {'attr_data': attr_data, 'attr_value': attr_value,
                        'domain_value': domain_value, 'domain_data': domain_data,
                          'plot_params': plot_params, 'domain_mode': domain_mode,
                          'patch_mode': patch_mode, 'method': method,
                         'input_data': input_data, 'input_label': input_label,
                         'predict_logit': predict_logit, 'predict_prob': predict_prob,
                         'analyse_time': analyse_time, }
            with open(rawfile_savepath, 'wb') as f:
                pickle.dump(save_dict, f)

        # 4) save visualization data
        if plot:
            cost_per_time = save_dict['analyse_time'] / input_data.shape[0]
            savepath = os.path.join(savedir, f'{domain_mode:s}_#{patch_mode}_{method:s}_visualization')
            attr_visualization(savepath=savepath,mode=domain_mode, data= save_dict['domain_data'],
                            value= save_dict['domain_value'],
                            plot_params=save_dict['plot_params'],
                            label= save_dict['input_label'],predict=save_dict['predict_prob'],info=f'time={cost_per_time:.1f}s',)

    def mode_select(self, domain_mode, patch_mode='1'):
        domain_mode_map = {'time': func_trans_time,
                    'frequency': func_trans_frequency,
                    'envelope': func_trans_envelope,
                    'STFT': func_trans_STFT,
                    'CS': func_trans_CS, }
        if domain_mode not in domain_mode_map.keys():
            raise ValueError('mode should be in ', domain_mode_map.keys())
        func_trans = domain_mode_map[domain_mode]
        func_Z, func_Z_inv, func_unpatch, trans_dict = func_trans(patch_mode=patch_mode)
        '''
        func_Z: z=func_Z(x), the transform function
        func_Z_inv: x=func_Z_inv(z), the inverse transform function
        func_unpatch: data [,shape] = func_unpatch(data [,shape]), unpatch the patched data to the original domain
        trans_dict: {'name':.,'trans_series':.}                
        '''
        return func_Z, func_Z_inv, func_unpatch, trans_dict
    
    def method_select(self, attr_name):
        AttrMethod_map = {'SHAP': Attr_SHAP,
                         'SHEP_Remove': Attr_SHEP_Remove,
                         'SHEP_Add': Attr_SHEP_Add,
                         'SHEP': Attr_SHEP, }
        if attr_name not in AttrMethod_map.keys():
            raise ValueError('attr_name should be in ', AttrMethod_map.keys())
        attr_method = AttrMethod_map[attr_name]
        return attr_method