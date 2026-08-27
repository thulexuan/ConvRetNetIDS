from __future__ import annotations

import json
import logging
import os
import pickle
from pathlib import Path

from matplotlib.pylab import seed
import numpy as np
import pandas as pd
# import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch import as_strided
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
WINDOW      = 128          # messages per sample
IMG_SIZE    = 32           # 128 * 8 = 1024 = 32 * 32
N_CH        = 3
# CAN_ID_MAX  = 2047.0       # 11-bit standard CAN ID  (0x000–0x7FF)
                           # 29-bit extended: change to 536870911.0


# ----- Hằng số theo bài báo -----
ID_MIN, ID_MAX = 0, 2047       # CAN ID 11-bit - KHÔNG còn dùng cho can_id (xem build_can_id_index_map),
                                # giữ lại chỉ để tương thích ngược nếu chỗ khác còn tham chiếu.
BYTE_MIN, BYTE_MAX = 0, 255    # mỗi byte payload - KHÔNG còn dùng (xem process_data_field mới, chia thẳng /255)

# ── Core normalization (GIỮ LẠI cho tương thích ngược, không còn dùng cho can_id/payload) ──
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


# ── CAN ID: rank-based normalization (MỚI) ─────────────────────────────────────
def build_can_id_index_map(csv_paths: list[str], id_column: str = "arbitration_id") -> dict[int, float]:
    """
    Xây bảng ánh xạ can_id -> giá trị chuẩn hoá THEO THỨ HẠNG, dùng CHUNG cho
    toàn bộ dataset (đọc qua TẤT CẢ các file .csv trong `csv_paths` trước khi
    xử lý chính thức).

    Cách làm:
        1. Gom UNION mọi giá trị arbitration_id xuất hiện trong toàn bộ các
           file .csv được liệt kê.
        2. SẮP XẾP các giá trị đó tăng dần theo giá trị số (đây là quy ước
           "thứ tự trong tập" được chọn - giữ đúng ngữ nghĩa "ID nhỏ hơn = ưu
           tiên arbitration cao hơn" của CAN, thay vì thứ tự xuất hiện ngẫu
           nhiên trong file).
        3. can_id thứ hạng r (0-indexed) trong K giá trị duy nhất
           -> normalized = r / K   (nằm trong [0, 1), giá trị lớn nhất
              nhận đúng (K-1)/K, không phải 1.0, để tránh trùng với giá trị
              fallback dành cho ID lạ - xem process_can_id).

    LƯU Ý QUAN TRỌNG (đánh đổi cần biết):
    - Hàm này đọc arbitration_id từ TẤT CẢ file trong `csv_paths`, tức là
      TRƯỚC KHI chia train/test. Nghĩa là 1 ID chỉ xuất hiện trong phần dữ
      liệu sẽ rơi vào test set vẫn được tính vào bảng thứ hạng - đây LÀ một
      hình thức rò rỉ nhẹ (test-time ID được biết trước ở bước chuẩn hoá).
      Với CAN, rủi ro này thường chấp nhận được vì bản thân tập ID hợp lệ
      của 1 dòng xe gần như CỐ ĐỊNH (không phải thông tin về nhãn tấn công),
      nhưng nếu muốn nghiêm ngặt tuyệt đối, hãy build bảng này CHỈ từ các
      file/khoảng thời gian dự kiến dùng làm train, rồi dùng
      `unseen_id_fallback` cho phần còn lại.
    - PHẢI LƯU bảng ánh xạ này ra đĩa (xem `run_pipeline`) để dùng lại y hệt
      lúc suy luận trên dữ liệu mới - nếu tính lại bảng từ dữ liệu khác,
      cùng 1 can_id có thể nhận thứ hạng khác, phá vỡ tính nhất quán.
    """
    unique_ids: set[int] = set()
    for path in csv_paths:
        try:
            col = pd.read_csv(path, usecols=[id_column])[id_column]
        except Exception as exc:
            log.warning("Bo qua %s khi build can_id index map: %s", path, exc)
            continue
        unique_ids.update(int(v) for v in col.dropna().unique())

    sorted_ids = sorted(unique_ids)
    k = len(sorted_ids)
    if k == 0:
        raise ValueError("Khong tim thay can_id nao de build index map.")

    id_to_norm = {raw_id: rank / k for rank, raw_id in enumerate(sorted_ids)}
    log.info("build_can_id_index_map: %d can_id khac nhau trong dataset.", k)
    return id_to_norm


# >>> MỚI: khoảng cách thời gian với frame LIỀN TRƯỚC (inter-arrival) <<<
def compute_inter_arrival(timestamps: pd.Series) -> pd.Series:
    """
    inter_arrival_ts[i] = timestamp[i] - timestamp[i-1]
    inter_arrival_ts[0] = 0  (frame đầu tiên không có frame trước đó)
    """
    return timestamps.diff().fillna(0.0).astype("float32")
 
 
# >>> MỚI: khoảng cách thời gian với frame CÓ TIMESTAMP BÉ NHẤT (relative) <<<
def compute_relative_ts(timestamps: pd.Series) -> pd.Series:
    """
    relative_ts[i] = timestamp[i] - min(timestamp trên TOÀN BỘ dataframe)
    """
    return (timestamps - timestamps.min()).astype("float32")
 
 
# ── CAN ID (MỚI: rank-based, thay cho log-minmax) ──────────────────────────────
def process_can_id(arb_id, id_to_norm: dict[int, float], unseen_id_fallback: float = 1.0) -> float:
    """
    arbitration_id (decimal int hoặc string) → thứ hạng chuẩn hoá (rank / K),
    tra từ bảng `id_to_norm` đã build sẵn bằng build_can_id_index_map().

    unseen_id_fallback: giá trị gán cho can_id KHÔNG có trong bảng (chưa từng
    thấy lúc build vocab - ví dụ dữ liệu suy luận thực tế có ID lạ). Mặc định
    = 1.0, LỚN HƠN mọi giá trị hợp lệ trong bảng (vốn nằm trong [0, (K-1)/K]),
    để mô hình có cơ hội phân biệt "ID lạ" với "ID hợp lệ có thứ hạng cao
    nhất" - tương tự vai trò UNK đã dùng ở nhánh embedding của model.

    Ví dụ: nếu can_id=140 có thứ hạng 87 trong tổng 512 ID khác nhau
        → 87 / 512 ≈ 0.170
    """
    try:
        arb_id_dec = int(arb_id)
    except (TypeError, ValueError):
        return unseen_id_fallback
    return id_to_norm.get(arb_id_dec, unseen_id_fallback)
 
 
# ── Data field (MỚI: chia thẳng cho 255, thay cho log-minmax) ──────────────────
def process_data_field(data_field) -> list:
    """
    data_field hex string → list 8 giá trị float32 đã normalize bằng cách
    chia thẳng cho 255 (linear min-max, KHÔNG qua log nữa).
 
    Steps:
        1. Padding zfill(16) cho đủ 16 ký tự hex (= 8 bytes)
        2. Tách thành 8 cặp hex (mỗi cặp = 1 byte)
        3. Mỗi byte: hex → decimal
        4. Chia cho 255 → [0, 1]
 
    Ví dụ:
        '0001240828282720'
        → ['00','01','24','08','28','28','27','20']
        → [0, 1, 36, 8, 40, 40, 39, 32]
        → [0.0, 0.0039, 0.1412, 0.0314, 0.1569, 0.1569, 0.1529, 0.1255]
    """
    hex_str = str(data_field).strip().upper().zfill(16)[:16]
 
    result = []
    for i in range(8):
        byte_hex = hex_str[i*2 : i*2 + 2]
        try:
            byte_dec = int(byte_hex, 16)
        except ValueError:
            byte_dec = 0
        result.append(byte_dec / 255.0)
 
    return result   # list of 8 floats
 
 
# ── Apply to DataFrame ────────────────────────────────────────────────────────
def preprocess_df(df: pd.DataFrame, id_to_norm: dict[int, float]) -> pd.DataFrame:
    """
    Nhận DataFrame có cột: timestamp, arbitration_id, data_field, attack
    Trả về DataFrame mới với các cột:
 
        timestamp        (giữ nguyên)
        can_id           float32  ← arbitration_id sau rank-based normalize (rank/K)
        byte_0 .. byte_7 float32  ← 8 cột từ data_field sau /255
        attack           int      (giữ nguyên)
 
    Tổng: 1 + 1 + 8 + 1 = 11 cột

    id_to_norm: bảng {raw_can_id: rank/K} đã build SẴN cho TOÀN BỘ dataset
    bằng build_can_id_index_map() - truyền vào đây, KHÔNG build lại mỗi file.
    """
    out = pd.DataFrame()
 
    out["timestamp"] = df["timestamp"].values
 
    # CAN ID - rank-based normalization
    out["can_id"] = df["arbitration_id"].apply(
        lambda x: process_can_id(x, id_to_norm)
    ).astype("float32")
 
    # Payload: tách thành 8 cột riêng biệt
    byte_cols = df["data_field"].apply(process_data_field)
    byte_df   = pd.DataFrame(
        byte_cols.tolist(),
        columns=[f"byte_{i}" for i in range(8)],
        dtype="float32",
    )
    out = pd.concat([out, byte_df], axis=1)

    # >>> MỚI: inter_arrival_ts - khoảng cách với frame liền trước
    out["inter_arrival_ts"] = compute_inter_arrival(df["timestamp"])

    # >>> MỚI: relative_ts - khoảng cách với timestamp bé nhất toàn dataframe
    out["relative_ts"] = compute_relative_ts(df["timestamp"])
 
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

def load_attack_folder(filename: str, label, id_to_norm: dict[int, float], window_size=16, stride_size=16) -> pd.DataFrame:

    try:
        # process can id, payload
        df = pd.read_csv(filename)
        df = df.sort_values("timestamp", ascending=True).reset_index(drop=True)
        normalized = preprocess_df(df, id_to_norm)
        print(normalized.head())

    except Exception as exc:
        log.error("Error reading %s", filename)
        print(f"Error occurred while reading {filename}: {exc}")


    return normalized

def build_feature_col(df: pd.DataFrame) -> list:
    """Lấy tên các cột feature (bỏ timestamp, attack, v.v.)."""
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

def run_pipeline(root_dir, output_dir, window_size=29, stride_size=15, feature_size=11, attacks=None) -> None:

    root   = Path(root_dir)
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ── Build bảng can_id -> rank/K MỘT LẦN DUY NHẤT, từ TOÀN BỘ file
    # (trước khi xử lý từng file riêng lẻ) - đảm bảo mọi file dùng chung
    # đúng 1 bảng ánh xạ nhất quán, không build lại mỗi lần.
    log.info("Building can_id index map tu toan bo dataset")
    all_csv_paths = [str(root / f"{attack}.csv") for attack in attacks]
    id_to_norm = build_can_id_index_map(all_csv_paths, id_column="arbitration_id")

    # Lưu bảng ra đĩa - BẮT BUỘC để dùng lại y hệt lúc suy luận trên dữ liệu
    # mới, tránh 1 can_id nhận thứ hạng khác nếu build lại từ tập khác.
    id_map_path = outdir / "can_id_index_map.json"
    with open(id_map_path, "w") as f:
        json.dump({str(k): v for k, v in id_to_norm.items()}, f, indent=2)
    log.info("Da luu can_id index map (%d id) vao %s", len(id_to_norm), id_map_path)

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

        # preprocess can_id and payload (dùng CHUNG id_to_norm đã build ở trên)
        df = load_attack_folder(finput, attack_id + 1, id_to_norm, window_size=window_size, stride_size=stride_size)

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
    parser.add_argument("--data_folder", default="./data/can-ml/can-ml/2017-subaru-forester/130826/preprocessed/", help="Root folder with attack subfolders")
    parser.add_argument("--window_size", type=int, default=64, help="window size")
    parser.add_argument("--stride_size", type=int, default=64, help="window stride")
    parser.add_argument("--rid", type=int, default=1, help="the number of rerun preprocessing")


    args = parser.parse_args()
    print("Arguments:", args)
    
    attack_types = ["combined", "DoS", "fuzzing", "gear", "interval", "rpm", 
                "speed", "standstill", "systematic"]
    if args.stride_size is None:
        args.stride_size = args.window_size
        
    outdir =  args.data_folder + 'rid{}'.format(args.rid) + '/TFRecord_w{}_s{}'.format(args.window_size, args.stride_size)
    run_pipeline(
        root_dir    = args.root_folder,
        output_dir  = outdir,
        window_size = args.window_size,
        stride_size = args.stride_size,
        attacks = attack_types   
    )