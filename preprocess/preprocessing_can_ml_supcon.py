from __future__ import annotations

import logging
import os
from pathlib import Path

from matplotlib.pylab import seed
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch import as_strided

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

WINDOW      = 128         

ID_MIN, ID_MAX = 0, 2047      
BYTE_MIN, BYTE_MAX = 0, 255    

def log_minmax(x: float, x_min: float, x_max: float) -> float:
    """
    Logarithmic Min-Max Normalization → [0, 1]
 
    Steps:
        1. x'   = ln(x + 1)
        2. x̂   = (x' - ln(min+1)) / (ln(max+1) - ln(min+1))
    """
    lx    = np.log1p(x)
    l_min = np.log1p(x_min)
    l_max = np.log1p(x_max)
    denom = l_max - l_min
    if denom == 0:
        return 0.0
    return float((lx - l_min) / denom)

def process_can_id(arb_id) -> float:
    """
    arbitration_id (decimal int or string) → log-minmax normalized float
 
    Ex:
        140 → log_minmax(140, 0, 2047) ≈ 0.621
    """
    arb_id_dec = int(arb_id)
    return log_minmax(arb_id_dec, ID_MIN, ID_MAX)
 
 
def process_data_field(data_field) -> list:
    """
    data_field hex string → list 8 giá trị float32 đã normalize
 
    Steps:
        1. Padding zfill(16) cho đủ 16 ký tự hex (= 8 bytes)
        2. Tách thành 8 cặp hex (mỗi cặp = 1 byte)
        3. Mỗi byte: hex → decimal
        4. Log-MinMax normalize về [0, 1]
 
    Ex:
        '0001240828282720'
        → ['00','01','24','08','28','28','27','20']
        → [0, 1, 36, 8, 40, 40, 39, 32]
        → [0.0, 0.198, 0.632, 0.384, 0.659, 0.659, 0.653, 0.634]
    """
    hex_str = str(data_field).strip().upper().ljust(16, "0")[:16]
 
    result = []
    for i in range(8):
        byte_hex = hex_str[i*2 : i*2 + 2]
        try:
            byte_dec = int(byte_hex, 16)
        except ValueError:
            byte_dec = 0
        result.append(log_minmax(byte_dec, BYTE_MIN, BYTE_MAX))
 
    return result   
 
def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nhận DataFrame có cột: timestamp, arbitration_id, data_field, attack
    Trả về DataFrame mới với các cột:
 
        timestamp        (giữ nguyên)
        can_id           float32  ← arbitration_id sau log-minmax
        byte_0 .. byte_7 float32  ← 8 cột từ data_field sau log-minmax
        attack           int      (giữ nguyên)
 
    Tổng: 1 + 1 + 8 + 1 = 11 cột
    """

    out = pd.DataFrame()
 
    out["timestamp"] = df["timestamp"].values
 
    # CAN ID
    out["can_id"] = df["arbitration_id"].apply(process_can_id).astype("float32")
 
    # Payload: tách thành 8 cột riêng biệt
    byte_cols = df["data_field"].apply(process_data_field)
    byte_df   = pd.DataFrame(
        byte_cols.tolist(),
        columns=[f"byte_{i}" for i in range(8)],
        dtype="float32",
    )
    out = pd.concat([out, byte_df], axis=1)
 
    out["attack"] = df["attack"].values.astype("int8")
 
    return out

def sample_10pct(X: np.ndarray, y: np.ndarray, attack_id: int, seed: int = 42):
    """
    Từ tập windows của 1 file:
        - Tách thành 2 nhóm: attack windows và normal windows
        - Lấy ngẫu nhiên 10% mỗi nhóm (ít nhất 1 sample)
        - Ghép lại và trả về
 
    Returns
    -------
    X_sampled, y_sampled
    """
    rng = np.random.default_rng(seed)
 
    idx_attack = np.where(y == attack_id)[0]
    idx_normal = np.where(y == 0)[0]
 
    def sample_group(idx):
        n = max(1, int(np.ceil(len(idx) * 0.1)))
        chosen = rng.choice(idx, size=min(n, len(idx)), replace=False)
        return chosen
 
    chosen_atk = sample_group(idx_attack) if len(idx_attack) > 0 else np.array([], dtype=int)
    chosen_nor = sample_group(idx_normal) if len(idx_normal) > 0 else np.array([], dtype=int)
 
    chosen = np.concatenate([chosen_atk, chosen_nor])
    chosen = np.sort(chosen)                    # giữ temporal order
 
    print(f"    attack windows : {len(idx_attack):5d}  → sample {len(chosen_atk):4d} (10%)")
    print(f"    normal windows : {len(idx_normal):5d}  → sample {len(chosen_nor):4d} (10%)")
 
    return X[chosen], y[chosen]

def load_attack_folder(filename: str, label, window_size=16, stride_size=16, verbose: bool = True) -> pd.DataFrame:

    try:
        df = pd.read_csv(filename)

        print("Length before removing: ", len(df))

        len_series = df["data_field"].astype(str).str.strip().str.len()
        mask_odd = len_series % 2 != 0
        
        if verbose and mask_odd.any():
            print(
                f"[preprocess_df] Loại {mask_odd.sum()} / {len(df)} hàng "
                f"({mask_odd.mean():.2%}) do data_field có độ dài lẻ."
            )
            print(df.loc[mask_odd, "data_field"].head(5).to_list())
        
        # reset_index BẮT BUỘC: sau khi lọc, index của df không còn liên tục.
        # Nếu không reset, `df[col].apply(...)` trả về Series mang index cũ,
        # trong khi `byte_df` được tạo từ list nên có RangeIndex 0..n-1
        # -> pd.concat / phép gán cột sẽ căn theo index và sinh ra NaN.
        df = df.loc[~mask_odd].reset_index(drop=True)
        
        if len(df) == 0:
            raise ValueError("Không còn hàng nào sau khi lọc data_field độ dài lẻ.")

        print("Length after removing: ", len(df))

        df = df.sort_values("timestamp", ascending=True).reset_index(drop=True)
        normalized = preprocess_df(df)
        print(normalized.head())

    except Exception as exc:
        log.error("Error reading %s", filename)
        print(f"Error occurred while reading {filename}: {exc}")


    return normalized

def build_feature_col(df: pd.DataFrame) -> list:
    """Lấy tên các cột feature (bỏ timestamp, v.v.)."""
    return [c for c in df.columns if c not in ["timestamp", "attack"]]

def sample_10pct(
    data_3d  : np.ndarray,
    labels   : np.ndarray,
    attack_id: int,
    seed     : int = 42,
):
    rng = np.random.default_rng(seed=seed)

    idx_normal = np.where(labels == 0)[0]
    idx_attack = np.where(labels == attack_id)[0]

    def _sample(idx):
        n = max(1, int(len(idx) * 0.2))
        return np.sort(rng.choice(idx, size=min(n, len(idx)), replace=False))

    chosen_normal = _sample(idx_normal)
    chosen_attack = _sample(idx_attack)
    chosen        = np.sort(np.concatenate([chosen_normal, chosen_attack]))

    print(f"  normal : {len(idx_normal):6,} → {len(chosen_normal):5,} (20%)")
    print(f"  attack : {len(idx_attack):6,} → {len(chosen_attack):5,} (20%)")
    print(f"  total  : {len(labels):6,} → {len(chosen):5,}")

    return data_3d[chosen], labels[chosen]

def run_pipeline(root_dir, output_dir, window_size=29, stride_size=15, feature_size=9, attacks=None) -> None:

    root   = Path(root_dir)
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ── load all DataFrames ───────────────────────────────────────
    log.info("Loading each .csv file of each attack")
    dfs: dict[str, pd.DataFrame] = {}

    xtrain = np.empty((0, window_size, feature_size))
    ytrain = np.empty((0))
    xtest = np.empty((0, window_size, feature_size))
    ytest = np.empty((0))

    for attack_id, attack in enumerate(attacks):
        print('Attack: {} ==============='.format(attack))
        finput = '{}/{}.csv'.format(root, attack)

        # preprocess can_id and payload
        df = load_attack_folder(finput, attack_id + 1, window_size=window_size, stride_size=stride_size)

        feature_col   = build_feature_col(df)
        df['features'] = df[feature_col].apply(lambda row: [row[col] for col in feature_col], axis=1)
        # feature_arr = df[feature_col].values.astype(np.float32)   # (N, F)
        # label_arr   = df["attack"].values.astype(np.int64)       # (N,)
 
        as_strided = np.lib.stride_tricks.as_strided
        output_shape = ((len(df) - window_size) // stride_size + 1, window_size) # (num_windows, window_size)
        features = as_strided(df.features, output_shape, (8*stride_size, 8)) # 2 trục của tham số stride_size: trục 0(nhảy giữa các window) nhảy bao nhiêu bước để đến window tiếp theo.
        # trục 1 (nhảy giữa các phần tử trong cùng 1 window): nhảy bao nhiêu bước để đến phần tử tiếp theo trong cùng 1 window
        l = as_strided(df.attack, output_shape, (1*stride_size, 1)) 
        
        dft = pd.DataFrame({
            'features': pd.Series(features.tolist()), 
            'label': pd.Series(l.tolist())
        }, index= range(len(l)))

        dft['label'] = dft['label'].apply(lambda x: (attack_id + 1) if any(x) else 0)

        ### print final preprocessed dataframe
        print(dft.head(10))

        print("Aggregating data: Done")
        print('#Normal: ', dft[dft['label'] == 0].shape[0])
        print('#Attack: ', dft[dft['label'] != 0].shape[0]) 


        # each element is a window, data_3d.len = number of windows
        # data_3d[i]: each window has window_size messages, each message has feature_size elements
        data_3d = np.array(dft['features'].tolist())
        # labels_cur.len = number of windows
        labels_cur = dft['label'].values

        print("data_3d.len:", len(data_3d), "labels_cur.len:", len(labels_cur))

        # sample 10% of attack windows and 10% of normal windows
        data_3d, labels_cur = sample_10pct(data_3d, labels_cur, attack_id + 1, seed=42)

        print("after sampling 10%, data_3d.len:", len(data_3d), "labels_cur.len:", len(labels_cur))

        attack_idx = np.where(labels_cur != 0)[0] # idx of attack windows
        benign_idx = np.where(labels_cur == 0)[0] # idx of benign windows

        # chia train/test của attack windows
        data_train_a, data_test_a, labels_train_a, labels_test_a = train_test_split(data_3d[attack_idx], labels_cur[attack_idx], test_size=0.2, random_state=42, shuffle=True)
        # chia train/test của benign windows
        data_train_b, data_test_b, labels_train_b, labels_test_b = train_test_split(data_3d[benign_idx], labels_cur[benign_idx], test_size=0.2, random_state=42, shuffle=True)

        data_train = np.concatenate((data_train_a, data_train_b))
        data_test = np.concatenate((data_test_a, data_test_b))
        labels_train = np.concatenate((labels_train_a, labels_train_b))
        labels_test = np.concatenate((labels_test_a, labels_test_b))

        xtrain = np.concatenate((xtrain, data_train))
        ytrain = np.concatenate((ytrain, labels_train))
        xtest = np.concatenate((xtest, data_test))
        ytest = np.concatenate((ytest, labels_test))

        print("Adding data of attack {} to train/test sets: Done".format(attack))

    np.save(os.path.join(output_dir, f"xtrain-{window_size}-{stride_size}.npy"), xtrain)
    np.save(os.path.join(output_dir, f"ytrain-{window_size}-{stride_size}.npy"), ytrain) 
    
    np.save(os.path.join(output_dir, f"xtest-{window_size}-{stride_size}.npy"), xtest)
    np.save(os.path.join(output_dir, f"ytest-{window_size}-{stride_size}.npy"), ytest)
    print("DONE PROCESS") 

    return xtrain, xtest, ytrain, ytest


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_folder", default="./data/can-ml/can-ml/2017-subaru-forester/090726/merged", help="Root folder with attack subfolders")
    parser.add_argument("--data_folder", default="./data/can-ml/can-ml/2017-subaru-forester/020926/preprocessed-keep-per-frame-label/", help="Root folder with attack subfolders")
    parser.add_argument("--window_size", type=int, default=32, help="window size")
    parser.add_argument("--stride_size", type=int, default=32, help="window stride")
    parser.add_argument("--rid", type=int, default=1, help="the number of rerun preprocessing")


    args = parser.parse_args()
    print("Arguments:", args)
    
    # attack_types = ["combined", "DoS", "fuzzing", "gear", "interval", "rpm", 
    #             "speed", "standstill", "systematic"]
    chd_attack_types = ["DoS", "fuzzy", "gear", "rpm"]
    if args.stride_size is None:
        args.stride_size = args.window_size
        
    outdir =  args.data_folder + 'rid{}'.format(args.rid) + '/TFRecord_w{}_s{}'.format(args.window_size, args.stride_size)
    run_pipeline( 
        root_dir    = args.root_folder,
        output_dir  = outdir,
        window_size = args.window_size,
        stride_size = args.stride_size,
        attacks = chd_attack_types   
    )