#!/usr/bin/python
# -*- coding:utf-8 -*-

from torch.utils.data import Dataset
import torch
class myDataset(Dataset):

    def __init__(self, list_data):
        self.seq_data = torch.tensor(list_data[0],dtype=torch.float32)
        self.labels = torch.tensor(list_data[1],dtype=torch.int64)

    def __len__(self):
        return len(self.seq_data)

    def __getitem__(self, item):
        return self.seq_data[item], self.labels[item]