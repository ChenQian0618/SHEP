'''
2025/04/25
Created by ChenQian-SJTU (chenqian2020@sjtu.edu.cn)
This script is used to analyze the attribution values of the models under different situations (e.g., domain, patch, method etc.)
'''
import sys,os
def root_path_k(x, k): return os.path.abspath(os.path.join(x, *([os.pardir] * (k + 1))))
os.chdir(root_path_k(__file__, 0)) # change the current working directory to the root path
projecht_dir = root_path_k(__file__, 1)
# add the project directory to the system path
if projecht_dir not in sys.path:
    sys.path.insert(0, projecht_dir)

import argparse
from SHEPs.MultiDomain_Attribution import MultiDomain_Attribution
from Demo.utils.DataModel_Loader import DataModel_Loader


def parse_args():
    parser = argparse.ArgumentParser(description='Demo_analysis')

    parser.add_argument('--checkpoint_root', type=str, default='./checkpoint', help='the name of the data')
    parser.add_argument('--checkpoint_name', type=str, default='None', help='specify the checkpoint name,  e.g., "CNN-Simulation-time-SNRNone-0413-191146"')
    parser.add_argument('--domain_mode', type=str, default='frequency', choices=['all','time', 'frequency', 'envelope', 'STFT', 'CS'], help='the name of the data')
    parser.add_argument('--patch_mode', type=str, default='1', choices=['0', '1', '2', '3', '4', '5'], help='the name of the data')
    parser.add_argument('--method', type=str, default='SHEP', choices=['SHEP', 'SHAP', 'SHEP_Remove', 'SHEP_Add'], help='the name of the data')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # ----------------------------parse args------------------------------------------------
    args = parse_args()
    checkpoint_root = args.checkpoint_root
    checkpoint_name = args.checkpoint_name
    domain_modes = args.domain_mode
    # process checkpoint_name
    if checkpoint_name == 'None': # if None, use the first one
        checkpoint_name = next(os.walk(checkpoint_root))[1][0]
    checkpoint_dir = os.path.join(checkpoint_root, checkpoint_name)
    print(f'checkpoint_name: "{checkpoint_name:s}"')
    # process domain_mode
    if domain_modes == 'all':
        domain_modes = ['frequency', 'envelope', 'STFT', 'CS']
    else:
        domain_modes = [domain_modes]
    print(f'domain_modes: "{domain_modes}"')
    # process patch_mode
    if args.patch_mode == 'all':
        patch_modes = ['1', '3', '5']
    else:
        patch_modes = [args.patch_mode]
    print(f'patch_modes: "{patch_modes}"')
    # process method
    if args.method == 'all':
        methods = ['SHEP', 'SHAP',]
    else:
        methods = [args.method]
    print(f'methods: "{methods}"')

    # ----------------------------load model and data------------------------------------------------
    loader = DataModel_Loader(dir=checkpoint_dir, flag_preload_dataset=True)
    func_predict, background_data, _, input_data, input_label = loader.get_fuc_data(n_input=1)
    save_dir = loader.save_dir

    #  ----------------------------attribution analysis------------------------------------------------
    MD_Attribution = MultiDomain_Attribution(func_predict, background_data, save_dir)
    for i,method in enumerate(methods):
        for j,patch_mode in enumerate(patch_modes):
            for k,domain_mode in enumerate(domain_modes):
                print('\n'*2, '-'*30, f'({i+1:d}/{len(methods):d}) current method: {method:s}', '-'*30)
                print(' '*5, '-'*25, f'({j+1:d}/{len(patch_modes):d}) current patch: {patch_mode:s}', '-'*25)
                print(' '*10, '-'*20, f'({k+1:d}/{len(domain_modes):d}) current domain: {domain_mode:s}', '-'*20)
                MD_Attribution.explain(input_data, input_label, domain_mode=domain_mode, patch_mode=patch_mode, method=method, preload=True, Fs=loader.Fs)
