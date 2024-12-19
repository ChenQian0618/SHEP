import os
import sys

sys.path.append(os.path.abspath('train'))
import argparse
from datetime import datetime
from utils.logger import setlogger
import logging
from utils.train_utils import train_utils
import random
import numpy as np
import torch

# 随机数设置
seed = 999
np.random.seed(seed)  # seed是一个固定的整数即可
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

mybool = lambda x: x.lower() in ['yes', 'true', 't', 'y', '1']
mylist = lambda x: [float(item) for item in x.strip(' ,').split(',')]
mylist_int = lambda x: [int(float(item)) for item in x.strip(' ,').split(',')]


def parse_args():
    parser = argparse.ArgumentParser(description='Train')

    # datasets parameters
    # .\checkpoint\Simulation_data
    # E:\6-数据集\0-机械故障诊断数据集\1-凯斯西储大学数据
    # E:\6-数据集\0-机械故障诊断数据集\9-厚德平行轴\振动响应\数据导出\斜齿轮

    parser.add_argument('--data_name', type=str, default='Simulation', choices=['Simulation', 'CWRU', 'HouDe2'],
                        help='the name of the data')
    parser.add_argument('--data_dir', type=str,
                        default=r'.\checkpoint\Simulation_data',
                        help='the directory of the data')
    parser.add_argument('--data_type', type=str, default='time', choices=['time', ], help='the name of the data')
    parser.add_argument('--normlizetype', type=str, default='mean-std', choices=['0-1', '-1-1', 'mean-std', 'none'],
                        help='data normalization methods')
    parser.add_argument('--data_signalsize', type=int, default=2000, help='the name of the data')
    parser.add_argument('--SNR', type=float, default=0, help='')
    parser.add_argument('--test_size', type=float, default=0.3, help='')

    parser.add_argument('--batch_size', type=int, default=64, help='batchsize of the training process')
    parser.add_argument('--num_workers', type=int, default=0, help='the number of training process')

    # models parameters
    parser.add_argument('--model_name', type=str, default='CNN',
                        choices=['MLP', 'CNN', 'resnet18', 'BiLSTM', 'Transformer'], help='')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint/test',
                        help='the directory to save the models')  # ./checkpoint

    # optimization information
    parser.add_argument('--opt', type=str, choices=['sgd', 'adam'], default='adam', help='the optimizer')
    parser.add_argument('--lr', type=float, default=1e-3, help='the initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.5, help='the momentum for sgd')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='the weight decay for sgd and adam')
    parser.add_argument('--lr_scheduler', type=str, choices=['step', 'exp', 'stepLR', 'fix'], default='stepLR',
                        help='the learning rate schedule')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='learning rate scheduler parameter for step and exp')  # 0.97
    parser.add_argument('--steps', type=int, default=1, help='the learning rate decay for step and stepLR')

    # save, load and display information
    parser.add_argument('--save_model', type=mybool, default="True", help='save the models or not')
    parser.add_argument('--flag_save_feature', type=mybool, default="False", help='')
    parser.add_argument('--max_epoch', type=int, default=40, help='max number of epoch')
    parser.add_argument('--print_step', type=int, default=5, help='the interval of log training information')
    parser.add_argument('--save_N_data', type=int, default=20, help='0 means not save the data')

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()

    # 对SNR的特殊处理
    if args.SNR <= -1e2 or args.SNR >= 1e2:
        args.SNR = None

    # Prepare the saving path for the models
    sub_dir = '-'.join([args.model_name, args.data_name, args.data_type,f'SNR{str(args.SNR):s}',
                        datetime.strftime(datetime.now(), '%m%d-%H%M%S')])
    save_dir = os.path.join(args.checkpoint_dir, sub_dir).replace('\\', '/')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # set the logger
    setlogger(os.path.join(save_dir, 'training.log'))

    # save the args
    for k, v in args.__dict__.items():
        logging.info("<args> {}: {}".format(k, v))

    trainer = train_utils(args, save_dir)

    trainer.setup()

    trainer.train()

    trainer.plot_save()
