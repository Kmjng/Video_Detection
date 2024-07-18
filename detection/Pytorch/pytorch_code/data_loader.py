# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 17:42:26 2024

@author: minjeong
"""
from torch.utils.data import Dataset
import torch

# data_loader 
class eyes_dataset(Dataset):
    def __init__(self, x_file_paths, y_file_path, transform=None):
        self.x_files = x_file_paths
        self.y_files = y_file_path
        self.transform = transform

    def __getitem__(self, idx):
        x = self.x_files[idx]
        x = torch.from_numpy(x).float()

        y = self.y_files[idx]
        y = torch.from_numpy(y).float()

        return x, y

    def __len__(self):
        return len(self.x_files)





