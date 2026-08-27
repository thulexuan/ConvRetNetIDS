from os import path

import numpy as np
import torch
import os
from torch.utils.data import Dataset
    
class CANMLDataset(Dataset):
    def __init__(self, root_dir, window_size, stride_size, is_train=True, is_test=False):
        
        self.root_dir = root_dir
        self.window_size = window_size
        self.stride_size = stride_size
        self.total_size = len(os.listdir(self.root_dir))

        if is_train:
            self.x = np.load(os.path.join(root_dir, f'xtrain-{self.window_size}-{self.stride_size}.npy')).astype(np.float32)
            self.y = np.load(os.path.join(root_dir, f'ytrain-{self.window_size}-{self.stride_size}.npy')).astype(np.int64)
        elif is_test:
            self.x = np.load(os.path.join(root_dir, f'xtest-{self.window_size}-{self.stride_size}.npy')).astype(np.float32)
            self.y = np.load(os.path.join(root_dir, f'ytest-{self.window_size}-{self.stride_size}.npy')).astype(np.int64)
        else:
            self.x = np.load(os.path.join(root_dir, f'xval-{self.window_size}-{self.stride_size}.npy')).astype(np.float32)
            self.y = np.load(os.path.join(root_dir, f'yval-{self.window_size}-{self.stride_size}.npy')).astype(np.int64)

    def __getitem__(self, idx):
        x = self.x[idx] 
        y = self.y[idx] 
        
        return torch.tensor(x), torch.tensor(y)
        
    def __len__(self):
        return len(self.x)

