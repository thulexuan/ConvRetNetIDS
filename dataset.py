from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset


class CANMLDataset(Dataset):
    """
    Dataset cho CAN-ML sliding windows, áp dụng MinMaxScaler ĐÚNG CÁCH để
    tránh data leakage - HOÀN TOÀN TRONG BỘ NHỚ (không lưu/đọc file scaler
    nào ra ngoài đĩa):

        - is_train=True  -> FIT MinMaxScaler TRÊN CHÍNH tập train, giữ lại
                             object scaler đã fit ở thuộc tính `self.scaler`.
        - is_train=False -> BẮT BUỘC truyền vào `scaler` (chính là
                             `train_ds.scaler` lấy trực tiếp từ tập train
                             vừa tạo ở trên) để transform val/test - KHÔNG
                             fit lại, và KHÔNG cần đọc/ghi bất kỳ file nào.

    Cách dùng:
        train_ds = CANMLDataset(root_dir, W, S, is_train=True)
        val_ds   = CANMLDataset(root_dir, W, S, is_train=False, is_test=False,
                                 scaler=train_ds.scaler)
        test_ds  = CANMLDataset(root_dir, W, S, is_train=False, is_test=True,
                                 scaler=train_ds.scaler)

    Yêu cầu dữ liệu đầu vào: các file .npy phải là dữ liệu CHƯA normalize
    (giá trị decimal thô: CAN ID trong [0, 2047], mỗi byte payload trong
    [0, 255]), đặt tên "x{split}_raw-{window}-{stride}.npy".
    """

    def __init__(
        self,
        root_dir: str,
        window_size: int,
        stride_size: int,
        is_train: bool = True,
        is_test: bool = False,
        scaler: Optional[MinMaxScaler] = None,
    ):
        self.root_dir = root_dir
        self.window_size = window_size
        self.stride_size = stride_size
        self.is_train = is_train
        self.is_test = is_test

        if is_train:
            split = "train"
        elif is_test:
            split = "test"
        else:
            split = "val"
        self.split = split

        x_raw_path = os.path.join(root_dir, f"x{split}_raw-{window_size}-{stride_size}.npy")
        y_path = os.path.join(root_dir, f"y{split}-{window_size}-{stride_size}.npy")

        x_raw = np.load(x_raw_path).astype(np.float32)   # (N, T, F) - GIÁ TRỊ THÔ, chưa scale
        y = np.load(y_path).astype(np.int64)

        self.x = self._apply_minmax(x_raw, scaler)
        self.y = y

    # ------------------------------------------------------------------ #
    def _apply_minmax(self, x_raw: np.ndarray, scaler: Optional[MinMaxScaler]) -> np.ndarray:
        """
        Áp MinMaxScaler lên x_raw (N, T, F), scale THEO TỪNG FEATURE độc lập.

            - is_train=True  -> fit 1 scaler MỚI trên chính x_raw này, lưu
                                 vào self.scaler (chỉ trong bộ nhớ).
            - is_train=False -> BẮT BUỘC nhận `scaler` từ bên ngoài (đã fit
                                 từ tập train), chỉ transform, KHÔNG fit lại.
        """
        N, T, F = x_raw.shape
        x_flat = x_raw.reshape(-1, F)

        if self.is_train:
            scaler = MinMaxScaler(feature_range=(0.0, 1.0))
            scaler.fit(x_flat)  # ⚠️ chỉ fit trên train
        else:
            if scaler is None:
                raise ValueError(
                    f"Tập '{self.split}' cần được truyền `scaler` đã fit từ tập train "
                    f"(vd: CANMLDataset(..., is_train=False, scaler=train_ds.scaler)). "
                    f"KHÔNG được tự fit scaler mới trên tập '{self.split}' để tránh data leakage."
                )
            # KHÔNG fit lại - chỉ transform bằng scaler đã fit từ train.

        x_scaled = scaler.transform(x_flat).reshape(N, T, F).astype(np.float32)
        self.scaler = scaler
        return x_scaled

    # ------------------------------------------------------------------ #
    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        return torch.tensor(x), torch.tensor(y)

    def __len__(self):
        return len(self.x)
