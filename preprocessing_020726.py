from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
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
CAN_ID_MAX  = 2047.0       # 11-bit standard CAN ID  (0x000–0x7FF)
                           # 29-bit extended: change to 536870911.0


# ----- Hằng số theo bài báo -----
ID_MIN, ID_MAX = 0, 4095       # CAN ID 11-bit
BYTE_MIN, BYTE_MAX = 0, 255    # mỗi byte payload

def pad_or_truncate(sequence, target_length):
    """Pad the sequence with zeros or truncate to the target length."""
    if len(sequence) < target_length:
        sequence = sequence + [0] * (target_length - len(sequence))  # pad with zeros
    return sequence[:target_length]  # truncate if longer

def reshape_to_target_size(array, target_size=32):
    """Reshape or pad a flattened array into a target_size x target_size matrix."""
    padded = pad_or_truncate(array, target_size * target_size)  # Ensure target_size^2 elements
    return np.array(padded).reshape((target_size, target_size))  # Reshape into target_size x target_size

def serialize_example(x, y): 
    """converts x, y to tf.train.Example and serialize"""
    id_seq, data_seq, timestamp = x
    id_seq = tf.train.FloatList(value = np.array(id_seq).flatten())
    data_seq = tf.train.FloatList(value = np.array(data_seq).flatten())
    timestamp = tf.train.FloatList(value = np.array(timestamp).flatten())

    label = tf.train.Int64List(value = np.array([y]))

    features = tf.train.Features(
        feature = {
            "id_seq": tf.train.Feature(float_list = id_seq),
            "data_seq": tf.train.Feature(float_list = data_seq),
            "timestamp": tf.train.Feature(float_list = timestamp),
            "label" : tf.train.Feature(int64_list = label)
        }
    )
    example = tf.train.Example(features = features)
    return example.SerializeToString()

def write_tfrecord(data, filename):
    tfrecord_writer = tf.io.TFRecordWriter(filename)
    for _, row in tqdm(data.iterrows()):
        X = (row['id_seq'], row['data_seq'], row['timestamp'])
        Y = row['label']
        tfrecord_writer.write(serialize_example(X, Y))
    tfrecord_writer.close() 
    
def _log_minmax(x: np.ndarray, min_i: float, max_i: float) -> np.ndarray:
    """Eq. (1)-(2): Logarithmic Min-Max Normalization."""
    x_prime = np.log1p(x)  # ln(x_i + 1)
    denom = np.log1p(max_i) - np.log1p(min_i)
    return (x_prime - np.log1p(min_i)) / denom

def process_payload_column(data_field: pd.Series) -> pd.Series:
    """
    Xử lý RIÊNG cột `data_field`:
    - Pad bên phải bằng '0' cho đủ 16 ký tự nếu thiếu.
    - Convert hex -> dec cho từng byte.

    Returns: pd.Series, mỗi phần tử là 1 list gồm 8 giá trị thập phân (0-255)
    """
    def _parse_payload(s):
        s = str(s).strip()
        s = s.ljust(16, "0")[:16]
        return [int(s[i:i + 2], 16) for i in range(0, 16, 2)]

    return data_field.apply(_parse_payload)
 
def parse_can_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Chuyển arbitration_id (hex) -> int, data_field (hex 16 ký tự) -> 8 byte int.
    Trả về DataFrame có cột: timestamp, can_id, d1..d8, attack
    """
    df = df.copy()
 
    # arbitration_id: hex string -> int (VD: "0D4" -> 212)
    can_id = df["arbitration_id"].apply(lambda h: int(str(h), 16))
 
    # data_field: 16 ký tự hex -> 8 byte, mỗi byte 2 ký tự hex -> int (0-255)
    # Nếu data_field ngắn hơn 16 ký tự (DLC < 8), các byte còn thiếu được set = 0.
    payload = process_payload_column(df["data_field"])
 
    out = pd.DataFrame({
        "timestamp": df["timestamp"].astype(float),
        "can_id": can_id,
    })
    out["payload"] = payload.values
    out["attack"] = df["attack"].astype(int)
    return out
 
 
def normalize_can_id_column(can_id: pd.Series) -> np.ndarray:
    """Áp dụng Logarithmic Min-Max Normalization cho cột can_id (1 giá trị/message)."""
    return _log_minmax(can_id.values.astype(float), ID_MIN, ID_MAX)


def normalize_payload_column(payload: pd.Series) -> pd.Series:
    """
    Áp dụng Logarithmic Min-Max Normalization cho cột payload (list 8 giá trị/message).
    """
    payload_matrix = np.stack(payload.to_numpy()).astype(float)  # (n, 8)
    payload_normalized = _log_minmax(payload_matrix, BYTE_MIN, BYTE_MAX)
    return pd.Series(list(payload_normalized), index=payload.index)


def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Áp dụng Logarithmic Min-Max Normalization cho can_id và payload (8 byte)."""
    df = df.copy()
    df["can_id"] = normalize_can_id_column(df["can_id"])
    df["payload"] = normalize_payload_column(df["payload"])
    return df
 
 
def add_relative_time(df: pd.DataFrame) -> pd.DataFrame:
    """Chuyển timestamp tuyệt đối -> khoảng cách thời gian tương đối (inter-arrival)."""
    df = df.copy()
    dt = df["timestamp"].diff()
    dt.iloc[0] = 0.0  # frame đầu tiên không có khoảng cách trước đó
    df["delta_t"] = dt.values
    return df

def normalize_time_column(
    delta_t: pd.Series,
    min_val: float = None,
    max_val: float = None,
    unit_scale: float = 1e6,
):
    """
    Chuẩn hóa cột delta_t (khoảng thời gian tương đối) về [0,1] bằng
    Logarithmic Min-Max, GIỐNG cách làm với can_id/payload.
 
    LƯU Ý QUAN TRỌNG: delta_t tính bằng GIÂY thường rất nhỏ (VD 1e-5 đến
    1e-3). Vì ln(1+x) ≈ x khi x << 1, áp log trực tiếp lên giây gần như
    KHÔNG có tác dụng nén phân phối lệch (giống hệt Min-Max thường, không
    tận dụng được lợi ích của log-minmax như bài báo áp dụng cho can_id/
    payload). Để log thực sự phát huy tác dụng, cần đổi đơn vị delta_t lên
    đủ lớn (mặc định: NHÂN VỚI 1e6 -> micro giây) trước khi áp log.
 
    Khác với can_id (range cố định [0,4095]) hay payload byte ([0,255]),
    delta_t KHÔNG có range cố định theo lý thuyết -> min_val/max_val (tính
    trên đơn vị ĐÃ SCALE, VD micro giây) được tính TỪ CHÍNH DỮ LIỆU nếu
    không truyền vào.
 
    Parameters
    ----------
    delta_t : pd.Series giá trị delta_t thô (đơn vị giây)
    min_val, max_val : min/max tính trên đơn vị ĐÃ SCALE (VD micro giây).
        Nếu None sẽ tự tính từ delta_t truyền vào (mặc định).
        Khi xử lý tập test/production, NÊN truyền lại đúng min_val/max_val đã
        dùng lúc chuẩn hóa tập train để tránh lệch phân phối (data leakage
        ngược lại - dùng thống kê của test để tự chuẩn hóa test là sai).
    unit_scale : hệ số nhân để đổi đơn vị trước khi áp log (mặc định 1e6,
        tức giây -> micro giây). Điều chỉnh tùy tốc độ bus CAN thực tế.
 
    Returns
    -------
    normalized : np.ndarray, giá trị đã chuẩn hóa về [0,1]
    min_val, max_val : giá trị min/max thực sự đã dùng để chuẩn hóa (theo
        đơn vị đã scale) - lưu lại để tái sử dụng cho tập dữ liệu khác
    """
    vals = delta_t.values.astype(float)
    vals = np.clip(vals, a_min=0, a_max=None)  # đề phòng giá trị âm do lỗi dữ liệu
    vals = vals * unit_scale  # đổi đơn vị (VD giây -> micro giây) để log có tác dụng nén
 
    if min_val is None:
        min_val = float(vals.min())
    if max_val is None:
        max_val = float(vals.max())
 
    if max_val <= min_val:
        # toàn bộ delta_t giống nhau (VD chỉ có 1 dòng) -> tránh chia 0
        normalized = np.zeros_like(vals)
    else:
        normalized = _log_minmax(vals, min_val, max_val)
 
    return normalized, min_val, max_val


def load_csv(filename: str, attack_label, window_size: int, stride_size: int, target_size: int) -> pd.DataFrame:
    """
    Load one tab-separated CSV → clean DataFrame.

    Columns added:
        can_id   int    decimal CAN ID
        byte_0…byte_7  uint8  payload bytes
        label    str    attack class name (from folder)
    """
    df = pd.read_csv(filename)
    df = df.sort_values("timestamp", ascending=True).reset_index(drop=True)
    parsed = parse_can_frame(df)
    parsed = add_relative_time(parsed)   # tùy chọn: thêm delta_t nếu muốn dùng làm đặc trưng phụ
    print("Print before normalizing data")
    print(parsed.head())
    print(len(parsed))
    normalized = normalize_frame(parsed)

    # normalize delta timestamp column
    # normalized["delta_t"], time_min, time_max = normalize_time_column(
    #     normalized["delta_t"], min_val=time_min, max_val=time_max
    # )

    print("Print normalized data")
    print(normalized.head())
    print(len(normalized))

    # repeat can_id and timestamp 8 times
    can_id_repeated = np.repeat(normalized["can_id"].values[:, None], 8, axis=1)
    timestamp_repeated = np.repeat(normalized["delta_t"].values[:, None], 8, axis=1)
    normalized["can_id"] = list(can_id_repeated)
    normalized["timestamp"] = list(timestamp_repeated)

    # sliding window here
    as_strided = np.lib.stride_tricks.sliding_window_view
    canid = as_strided(normalized.can_id.values, window_shape=window_size)[::stride_size]
    timestamp = as_strided(normalized.timestamp.values, window_shape=window_size)[::stride_size]
    data = as_strided(normalized.payload.values, window_shape=window_size)[::stride_size]

    label = as_strided(normalized.attack.values, window_shape=window_size)[::stride_size]
    normalized = pd.DataFrame({
        'id_seq': pd.Series(canid.tolist()),
        'data_seq': pd.Series(data.tolist()),
        'timestamp': pd.Series(timestamp.tolist()),
        'label': pd.Series(label.tolist())
    }, index=range(len(canid)))

    print("After striding")
    print(normalized.head())
    normalized['label'] = normalized['label'].apply(lambda x: attack_label if any(x) else 0)

    # reshape to 128x8 rồi sang 32x32
    normalized['id_seq'] = normalized['id_seq'].apply(lambda x: reshape_to_target_size([item for sublist in x for item in sublist], target_size))
    normalized['data_seq'] = normalized['data_seq'].apply(lambda x: reshape_to_target_size([item for sublist in x for item in sublist], target_size))
    normalized['timestamp'] = normalized['timestamp'].apply(lambda x: reshape_to_target_size([item for sublist in x for item in sublist], target_size))

    return normalized[['id_seq', 'data_seq', 'timestamp', 'label']].reset_index().drop(['index'], axis=1)



def load_attack_folder(filename: str, label, verbose: bool = True, window_size: int = 128, stride_size: int = 64) -> pd.DataFrame:
    """Load and concatenate all CSVs in one attack folder."""
    try:
        df = load_csv(filename, label, window_size, stride_size, target_size=32)
    except Exception as exc:
        log.error("Error reading %s", filename)
        print(f"Error occurred while reading {filename}: {exc}")

    return df

def window_label(labels: pd.Series) -> str:
    """
    Majority-vote labelling for a window.
    Any attack class covering ≥ 30 % of frames wins over Normal.
    """
    counts   = labels.value_counts(normalize=True)
    non_norm = counts.drop("Normal", errors="ignore")
    if len(non_norm) > 0 and non_norm.iloc[0] >= 0.30:
        return non_norm.index[0]
    return "Normal"

def run_pipeline(
    root_dir:    str | Path,
    output_dir:  str | Path,
    fit_scaler:  bool = False,
    scaler_path: str | Path | None = None,
    window_size:    int = 128,
    stride_size:      int = 128,
    attacks: list[str] | None = None
) -> None:
    """
    Full pipeline entry point.

    Args:
        root_dir    folder whose subfolders are named by attack class
        output_dir  where TFRecords and scaler are written
        fit_scaler  if True, fit a new scaler on data in root_dir and save it
        scaler_path if fit_scaler=False, load a pre-fitted scaler from this path
        stride      window stride (default 64 = 50 % overlap)
    """
    root   = Path(root_dir)
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ── load all DataFrames ───────────────────────────────────────
    log.info("Loading each .csv file of each attack")
    dfs: dict[str, pd.DataFrame] = {}

    data_info = {}
    for attack_id, attack in enumerate(attacks):
        print('Attack: {} ==============='.format(attack))
        finput = '{}/{}.csv'.format(root, attack)
        df = load_attack_folder(finput, attack_id + 1)
        print(df.head())

        print("Writing...................")

        foutput_attack = '{}/{}'.format(outdir, attack)
        foutput_normal = '{}/Normal_{}'.format(outdir, attack)
        df_attack = df[df['label'] != 0].sample(frac=0.2, random_state=42)
        df_normal = df[df['label'] == 0].sample(frac=0.2, random_state=42)
        write_tfrecord(df_attack, foutput_attack)
        write_tfrecord(df_normal, foutput_normal)
        
        data_info[foutput_attack] = df_attack.shape[0]
        data_info[foutput_normal] = df_normal.shape[0]

    json.dump(data_info, open('{}/datainfo.txt'.format(outdir), 'w'))
    print("DONE!")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="./data/can-ml/can-ml/2017-subaru-forester/010726/merged", help="Root folder with attack subfolders")
    parser.add_argument("--fit_scaler", action="store_true", help="Fit and save a new scaler")
    parser.add_argument("--scaler", default=None,   help="Path to pre-fitted scaler.pkl")
    parser.add_argument("--window_size", type=int, default=128, help="window size")
    parser.add_argument("--stride_size", type=int, default=64, help="window stride")
    parser.add_argument("--rid", type=int, default=1, help="the number of rerun preprocessing")


    args = parser.parse_args()
    print("Arguments:", args)
    
    attack_types = ["combined", "DoS", "fuzzing", "gear", "interval", "rpm", 
                "speed", "standstill", "systematic"]
    if args.stride_size is None:
        args.stride_size = args.window_size
        
    outdir =  args.root + '_rid{}'.format(args.rid) + 'TFRecord_w{}_s{}'.format(args.window_size, args.stride_size)
    run_pipeline(
        root_dir    = args.root,
        output_dir  = outdir,
        fit_scaler  = args.fit_scaler,
        scaler_path = args.scaler,
        window_size = args.window_size * 4,
        stride_size = args.stride_size,
        attacks = attack_types   
    )