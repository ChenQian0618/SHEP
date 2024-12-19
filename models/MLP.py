#!/usr/bin/python
# -*- coding:utf-8 -*-
from torch import nn
import warnings


# ----------------------------inputsize >=28-------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, in_channel=1, out_channel=10, L=2000, **kwargs):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_channel*L, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64))
        self.cla = nn.Linear(64, out_channel)

    def forward(self, x, verbose=False):
        x = x.view(x.size(0), -1)
        z1 = self.mlp(x)
        x = self.cla(z1)
        if verbose:
            return x, z1
        else:
            return x


if __name__ == '__main__':
    import torch
    from utils.mysummary import summary


    for L in [1024, 2000, 5000]:
        model = MLP(L=L).to(torch.device('cuda'))
        summary(model, (1, L))