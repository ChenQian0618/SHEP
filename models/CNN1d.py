#!/usr/bin/python
# -*- coding:utf-8 -*-
from torch import nn
import warnings

# ----------------------------inputsize >=28-------------------------------------------------------------------------
class CNN(nn.Module):
    def __init__(self, in_channel=1, out_channel=10,**kwargs):
        super(CNN, self).__init__()
        base_c = 4
        channels = [in_channel, base_c*1, base_c*2, base_c*4, base_c*16, base_c*32, base_c*64, base_c*128, base_c*256]
        kernel_size = [7, 3, 3, 3, 3, 3, 3, 3]
        self.Conv_Module = nn.ModuleList()
        for i in range(len(kernel_size)):
            pool = nn.MaxPool1d(kernel_size=2, stride=2) if i != len(kernel_size)-1 else nn.AdaptiveMaxPool1d(1)
            self.Conv_Module.append(nn.Sequential(
                nn.Conv1d(channels[i], channels[i+1], kernel_size[i]),
                nn.BatchNorm1d(channels[i+1]),
                nn.ReLU(inplace=True),
                pool
            ))

        self.mlp = nn.Sequential(
            nn.Linear(channels[-1]*1, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True))
        self.fc = nn.Linear(64, out_channel)

    def forward(self, x, verbose=False):
        for i,item in enumerate(self.Conv_Module):
            x = item(x)
        z1 = x.view(x.size(0), -1)
        z2 = self.mlp(z1)
        x = self.fc(z2)
        if verbose:
            return x, z1
        else:
            return x

if __name__ == '__main__':
    import torch
    from utils.mysummary import summary

    import sys, pathlib
    # root = pathlib.Path(__file__).parent.parent
    # sys.path.insert(0,str(root))
    # from ana_utils.mysummary import summary

    model = CNN().to(torch.device('cuda'))
    summary(model, (1, 2000))
    print(1)