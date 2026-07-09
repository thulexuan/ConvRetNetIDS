import numpy as np
import torch
import os
from torchvision import transforms
from PIL import Image
from tfrecord.torch.dataset import TFRecordDataset
from torch.utils.data import Dataset

class CANDatasetStandard(Dataset):
    def __init__(self, root_dir, window_size, is_train=True, include_data=False, transform=None):
        if is_train:
            self.root_dir = os.path.join(root_dir, 'train')
        else:
            self.root_dir = os.path.join(root_dir, 'val')
            
        self.include_data = include_data
        self.is_train = is_train
        self.transform = transform  # This will be TwoCropTransform
        self.window_size = window_size
        self.total_size = len(os.listdir(self.root_dir))
    
    def __getitem__(self, idx):
        # Load data as before
        filenames = '{}/{}.tfrec'.format(self.root_dir, idx)
        index_path = None
        description = {'id_seq': 'int', 'data_seq': 'int', 'timestamp': 'float', 'label': 'int'}
        dataset = TFRecordDataset(filenames, index_path, description)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
        data = next(iter(dataloader))
        
        id_seq, data_seq, timestamp, label = data['id_seq'], data['data_seq'], data['timestamp'], data['label']

        id_seq = id_seq.to(torch.float)
        data_seq = data_seq.to(torch.float)
        timestamp = timestamp.to(torch.float)

        id_seq[id_seq == 0] = -1
        if id_seq.numel() == 1024 and data_seq.numel() == 1024:
            id_seq = id_seq.view(32, 32)
            data_seq = data_seq.view(32, 32)
            timestamp = timestamp.view(32, 32)
        else:
            raise RuntimeError(f"Invalid tensor size for id_seq or data_seq")

        combined_tensor = torch.stack([id_seq, data_seq, timestamp], dim=-1)
        combined_tensor = combined_tensor.permute(2, 0, 1)

        # Apply transformations if provided
        if self.transform:
            combined_tensor = self.transform(combined_tensor)
        return combined_tensor, label[0][0]
        
    def __len__(self):
        return self.total_size
    
class CANDatasetEnetWithNormalizingTs(Dataset):
    def __init__(
        self,
        root_dir,
        window_size,
        is_train=True,
        is_test=False,
        include_data=False,
        transform=None,
        target_size=32,
        normalize_timestamp=True,
        fit_timestamp_scaler=False,
        timestamp_log_min=None,
        timestamp_log_max=None,
        eps=1e-8,
        timestamp_scale_factor=1e6
    ):
        self.target_size = target_size
        self.include_data = include_data
        self.is_train = is_train
        self.is_test = is_test
        self.transform = transform
        self.window_size = window_size
        self.cls_num_list = None

        self.normalize_timestamp = normalize_timestamp
        self.fit_timestamp_scaler = fit_timestamp_scaler
        self.timestamp_log_min = timestamp_log_min
        self.timestamp_log_max = timestamp_log_max
        self.eps = eps
        self.timestamp_scale_factor = timestamp_scale_factor

        if is_train:
            self.root_dir = os.path.join(root_dir, "train")
        else:
            self.root_dir = os.path.join(root_dir, "val")

        if is_test:
            self.root_dir = os.path.join(root_dir, "test")

        self.files = sorted(
            [f for f in os.listdir(self.root_dir) if f.endswith(".tfrec")],
            key=lambda x: int(os.path.splitext(x)[0])
        )

        self.total_size = len(self.files)

        if self.normalize_timestamp and self.fit_timestamp_scaler:
            self.timestamp_log_min, self.timestamp_log_max = self._fit_timestamp_log_minmax()

        if self.normalize_timestamp and not self.fit_timestamp_scaler:
            if self.timestamp_log_min is None or self.timestamp_log_max is None:
                raise ValueError(
                    "For val/test, you must pass timestamp_log_min and "
                    "timestamp_log_max fitted from the train set."
                )

    def _read_one_tfrec(self, filepath):
        index_path = None
        description = {
            "id_seq": "float",
            "data_seq": "float",
            "timestamp": "float",
            "label": "int",
        }

        dataset = TFRecordDataset(filepath, index_path, description)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
        data = next(iter(dataloader))
        return data

    def _log_transform(self, timestamp):
        timestamp = timestamp.to(torch.float)
        timestamp = torch.clamp(timestamp, min=0.0)

        # timestamp is delta time in seconds
        timestamp = timestamp * self.timestamp_scale_factor

        # paper-style log transform
        timestamp_log = torch.log1p(timestamp)

        return timestamp_log

    def _fit_timestamp_log_minmax(self):
        """
        Fit min/max of log(1 + timestamp) using TRAIN SET ONLY.
        This follows the paper's log-minmax idea, but uses train-set
        statistics for timestamp because timestamp has no fixed upper bound.
        """
        global_min = float("inf")
        global_max = float("-inf")

        for filename in self.files:
            filepath = os.path.join(self.root_dir, filename)
            data = self._read_one_tfrec(filepath)

            timestamp = data["timestamp"].to(torch.float)
            timestamp_log = self._log_transform(timestamp)

            file_min = timestamp_log.min().item()
            file_max = timestamp_log.max().item()

            global_min = min(global_min, file_min)
            global_max = max(global_max, file_max)

        print("[Timestamp log-minmax scaler fitted on TRAIN set]")
        print(f"timestamp_log_min = {global_min:.8f}")
        print(f"timestamp_log_max = {global_max:.8f}")

        return global_min, global_max

    def _normalize_timestamp_log_minmax(self, timestamp):
        """
        Apply:
            timestamp_log = ln(timestamp + 1)

            timestamp_norm =
                (timestamp_log - timestamp_log_min_train)
                /
                (timestamp_log_max_train - timestamp_log_min_train)
        """
        timestamp_log = self._log_transform(timestamp)

        timestamp_norm = (
            timestamp_log - self.timestamp_log_min
        ) / (
            self.timestamp_log_max - self.timestamp_log_min + self.eps
        )

        # Important for val/test values that may exceed train range
        timestamp_norm = torch.clamp(timestamp_norm, min=0.0, max=1.0)

        return timestamp_norm

    def __getitem__(self, idx):
        filename = self.files[idx]
        filepath = os.path.join(self.root_dir, filename)

        data = self._read_one_tfrec(filepath)

        id_seq = data["id_seq"]
        data_seq = data["data_seq"]
        timestamp = data["timestamp"]
        label = data["label"]

        id_seq = id_seq.to(torch.float).view(self.target_size, self.target_size)
        data_seq = data_seq.to(torch.float).view(self.target_size, self.target_size)
        timestamp = timestamp.to(torch.float).view(self.target_size, self.target_size)

        if self.normalize_timestamp:
            timestamp = self._normalize_timestamp_log_minmax(timestamp)

        combined_tensor = torch.stack([id_seq, data_seq, timestamp], dim=0)

        # print(id_seq)
        # print(data_seq)
        # print(timestamp)

        if self.transform:
            combined_tensor = self.transform(combined_tensor)

        return combined_tensor, label[0][0]

    def __len__(self):
        return self.total_size

class CANDatasetEnet(Dataset):
    def __init__(self, root_dir, window_size, is_train=True, is_test=False, include_data=False, transform=None, target_size=64):
        self.target_size = target_size
        if is_train:
            self.root_dir = os.path.join(root_dir, 'train')
        else:
            self.root_dir = os.path.join(root_dir, 'val')
        
        if is_test:
            self.root_dir = os.path.join(root_dir, 'test')
            
        self.include_data = include_data
        self.is_train = is_train
        self.is_test = is_test
        self.transform = transform
        self.window_size = window_size
        self.total_size = len(os.listdir(self.root_dir))
        self.cls_num_list = None
    
    def __getitem__(self, idx):
        filenames = '{}/{}.tfrec'.format(self.root_dir, idx)
        index_path = None
        description = {'id_seq': 'int', 'data_seq': 'int', 'timestamp': 'float', 'label': 'int'}
        dataset = TFRecordDataset(filenames, index_path, description)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
        data = next(iter(dataloader))
        
        id_seq, data_seq, timestamp, label = data['id_seq'], data['data_seq'], data['timestamp'], data['label']

        id_seq = id_seq.to(torch.float).view(self.target_size, self.target_size)
        data_seq = data_seq.to(torch.float).view(self.target_size, self.target_size)
        timestamp = timestamp.to(torch.float).view(self.target_size, self.target_size)

        combined_tensor = torch.stack([id_seq, data_seq, timestamp], dim=0)
        # combined_tensor = torch.stack([id_seq, data_seq], dim=0)


        if self.transform:
            combined_tensor = self.transform(combined_tensor)
        return combined_tensor, label[0][0]
        
    def __len__(self):
        return self.total_size
    
class CANDataset(Dataset):
    def __init__(self, root_dir, window_size, is_train=True, transform=None, target_size=64):
        self.root_dir = os.path.join(root_dir, 'train' if is_train else 'val')
        self.transform = transform
        self.window_size = window_size
        self.data = []
        self.labels = []
        self.class_indices = {}
        self.target_size = target_size
        # Load and group data by class
        for idx in range(len(os.listdir(self.root_dir))):
            filenames = f'{self.root_dir}/{idx}.tfrec'
            description = {'id_seq': 'int', 'data_seq': 'int', 'timestamp': 'float', 'label': 'int'}
            dataset = TFRecordDataset(filenames, None, description)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
            data = next(iter(dataloader))
            label = data['label'][0][0].item()
            self.data.append(data)
            self.labels.append(label)

            # Organize data by class
            if label not in self.class_indices:
                self.class_indices[label] = []
            self.class_indices[label].append(idx)

    def __getitem__(self, idx):
        data = self.data[idx]
        id_seq, data_seq, timestamp = data['id_seq'], data['data_seq'], data['timestamp']
        id_seq = id_seq.to(torch.float).view(self.target_size, self.target_size)
        data_seq = data_seq.to(torch.float).view(self.target_size, self.target_size)
        timestamp = timestamp.to(torch.float).view(self.target_size, self.target_size)
        combined_tensor = torch.stack([id_seq, data_seq, timestamp], dim=-1).permute(2, 0, 1)
        if self.transform:
            combined_tensor = self.transform(combined_tensor)
        return combined_tensor, self.labels[idx]

    def __len__(self):
        return len(self.data)

