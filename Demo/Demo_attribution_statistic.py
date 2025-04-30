'''
2025/04/25
Created by ChenQian-SJTU (chenqian2020@sjtu.edu.cn)
This script is used to conduct the attribution analysis of the models under different situations (e.g., model, transform function, patch etc.), and statistic the results.
'''
import sys,os
def root_path_k(x, k): return os.path.abspath(os.path.join(x, *([os.pardir] * (k + 1))))
os.chdir(root_path_k(__file__, 0)) # change the current working directory to the root path
projecht_dir = root_path_k(__file__, 1)
# add the project directory to the system path
if projecht_dir not in sys.path:
    sys.path.insert(0, projecht_dir)

import argparse,pickle
from SHEPs.MultiDomain_Attribution import MultiDomain_Attribution
from Demo.utils.DataModel_Loader import DataModel_Loader
from Demo.utils.utils_attribution_statistic import Extract_info_by_name, Extract_info_by_pkl, Calculate_Similarity
from Demo.utils.utils_attribution_statistic import plot_Similarity_Matrix, plot_Similarity_Box, plot_Attribution_time
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description='Demo_attribution_statistic')
    parser.add_argument('--checkpoint_root', type=str, default='./checkpoint', help='the path of checkpoint')
    parser.add_argument('--checkpoint_name', type=str, default='None', help='specify the checkpoint name,  e.g., "CNN-Simulation-time-SNRNone-0413-191146"')
    args = parser.parse_args()
    return args


def statistic(root, file_end='_raw.pkl', flag_SaveXlsx=False, preload=False):
    '''
    :param root: the root directory of the files
    :param file_end: the end of the file name
    :param flag_SaveXlsx: whether to save the result as xlsx file
    '''
    # init
    root = os.path.join(root, 'PostProcess_of_Attribution_Analysis')
    save_dir = os.path.join(root, 'Stat')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    target_path = os.path.join(save_dir, 'storage-dataframe.pkl')
    # try to load data
    if os.path.exists(target_path) and preload:
        with open(target_path, 'rb') as f:
            df = pickle.load(f)
        print(f'load data from {target_path:s}')
    else:
        filenames = next(os.walk(root))[2]
        filenames = [name for name in filenames if name.endswith(file_end)]
        # load data
        df = []
        storage_value = [None] * len(filenames)  # [N_file,N_sample,d1,d2,C_prediction]
        for i, filename in enumerate(filenames):
            print(f'<{i+1}|{len(filenames):d}>: {filename:s}')
            filepath = os.path.join(root, filename)
            temp_dict = Extract_info_by_name(filename)
            temp, storage_value[i] = Extract_info_by_pkl(filepath)  # use storage_value to store the value
            temp.update({'value_index': i})
            temp_dict.update(temp)
            df.append(temp_dict)
        df = pd.DataFrame(df)
        # calculate similarity
        search_df = df.loc[df['method'] == 'SHAP', ['domain', 'patch', 'value_index']].set_index(['domain', 'patch'])
        def process_df(temp):
            domain, patch, index = temp['domain'], temp['patch'], temp['value_index']
            index_SHAP = search_df.loc[domain, patch].values[0]
            res = Calculate_Similarity(storage_value[index_SHAP], storage_value[index],label=temp['input_label'])
            return res  # [C_sample,C_prediction]
        df['SimiMatrix_pred_samp'] = df.apply(process_df, axis=1) # [C_prediction,C_sample]
        df['Simi_mean'] = df['SimiMatrix_pred_samp'].apply(lambda x: x.mean())
        df['Simi_std'] = df['SimiMatrix_pred_samp'].apply(lambda x: x.std())
        with open(target_path, 'wb') as f:
            pickle.dump(df, f)
        print(f'\nsave data to:\n {target_path:s}')

    # save dataframe to excel file
    if flag_SaveXlsx:
        df_save = df[['domain', 'method', 'patch', 'analyse_time', 'Simi_mean' , 'Simi_std']]
        df_save_1 = df_save.set_index(['domain', 'method', 'patch'])['analyse_time'].unstack()
        df_save_2 = df_save.set_index(['domain', 'method', 'patch'])['Simi_mean'].unstack()
        df_save_3 = df_save.set_index(['domain', 'method', 'patch'])['Simi_std'].unstack()
        df_saveDict = {'df': df_save, 'analyse_time': df_save_1, 'Simi_mean': df_save_2, 'Simi_std': df_save_3}
        with pd.ExcelWriter(os.path.join(save_dir, 'storage-PartialData.xlsx')) as writer:
            for k, v in df_saveDict.items():
                v.to_excel(writer, sheet_name=k)
        print(f'\nsave excel file to:\n {os.path.join(save_dir, 'storage-PartialData.xlsx'):s}')

    # plot 
    for patch in df['patch'].unique():
        plot_Similarity_Matrix(df.loc[df['patch'] == patch][['domain', 'method', 'SimiMatrix_pred_samp']],
                               os.path.join(save_dir, f'Similarity_Matrix_patch{patch:s}'), dpi=600)
                               
    plot_Similarity_Box(df[['domain', 'method', 'patch','SimiMatrix_pred_samp']],
                            os.path.join(save_dir, 'Similarity_Box'), dpi=600)
    plot_Attribution_time(df[['domain', 'method', 'patch', 'analyse_time']],
                        os.path.join(save_dir, 'Attribution_time'), dpi=600)



if __name__ == '__main__':
    # ----------------------------parse args------------------------------------------------
    args = parse_args()
    checkpoint_root = args.checkpoint_root
    checkpoint_name = args.checkpoint_name
    # process checkpoint_name
    if checkpoint_name.lower() == 'none': # if None, use the first one
        checkpoint_name = next(os.walk(checkpoint_root))[1][0]
    checkpoint_dir = os.path.join(checkpoint_root, checkpoint_name)
    print(f'checkpoint_name: "{checkpoint_name:s}"')


    # ----------------------------load model and data------------------------------------------------
    loader = DataModel_Loader(dir=checkpoint_dir, flag_preload_dataset=True)
    func_predict, background_data, _, input_data, input_label = loader.get_fuc_data(n_input=1)
    save_dir = loader.save_dir

    #  ----------------------------Attribution analysis------------------------------------------------ (make sure the attrubution process is completed, before the statistic process)

    domain_modes = ['frequency', 'envelope', 'STFT', 'CS'] # ['frequency', 'envelope', 'STFT', 'CS']
    patch_modes = ['1', '2', '3','4', '5'] # ['1', '2', '3','4', '5']
    methods = ['SHEP', 'SHEP_Remove','SHEP_Add','SHAP']
    MD_Attribution = MultiDomain_Attribution(func_predict, background_data, save_dir)
    
    for i,domain_mode in enumerate(domain_modes):
        for j,patch_mode in enumerate(patch_modes):
            for k,method in enumerate(methods):
                print('\n'*2, '-'*30, '{:^30s}'.format(f'({i+1:d}/{len(domain_modes):d}) current domain: {domain_mode:s}'), '-'*20)
                print(' '*5, '-'*25, '{:^30s}'.format(f'({j+1:d}/{len(patch_modes):d}) current patch: {patch_mode:s}'), '-'*25)
                print(' '*10, '-'*20, '{:^30s}'.format(f'({k+1:d}/{len(methods):d}) current method: {method:s}'), '-'*30)
                MD_Attribution.explain(input_data, input_label, domain_mode=domain_mode, patch_mode=patch_mode, method=method, preload=True, Fs=loader.Fs, plot=True) # do not plot the figure, just save the shap values

    
    # ----------------------------statistic the results------------------------------------------------
    statistic(checkpoint_dir, flag_SaveXlsx=True)
