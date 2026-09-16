from os import path

import numpy as np
import torch
import os
from torch.utils.data import Dataset
    
class CANMLDatasetForAETransformer(Dataset):
    """
    Args:
        data_dir: folder containing x{split}_mes.npy / x{split}_ts.npy / y{split}.npy
        split:    "train" or "test"
    """
 
    def __init__(self, data_dir: str, split: str = "train"):
        assert split in ("train", "test"), "split must be 'train' or 'test'"
        self.mes = np.load(os.path.join(data_dir, f"x{split}_mes.npy"))  # (N, T, 12)
        self.ts = np.load(os.path.join(data_dir, f"x{split}_ts.npy"))     # (N, T)
        self.labels = np.load(os.path.join(data_dir, f"y{split}.npy"))     # (N,)
 
        assert self.mes.shape[0] == self.ts.shape[0] == self.labels.shape[0], (
            f"Mismatched number of samples: mes={self.mes.shape[0]}, "
            f"ts={self.ts.shape[0]}, labels={self.labels.shape[0]}"
        )
 
    def __len__(self) -> int:
        return len(self.labels)
 
    def __getitem__(self, idx: int) -> dict:
        return {
            "mes": torch.from_numpy(self.mes[idx]).float(),        # (T, 12)
            "timestamp": torch.from_numpy(self.ts[idx]).float(),    # (T,)
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }

