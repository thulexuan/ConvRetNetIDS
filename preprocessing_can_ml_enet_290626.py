import pandas as pd
# import vaex
import numpy as np
import glob
import json
import math
import csv
import time
import _warnings
import tensorflow as tf
from tqdm import tqdm
import argparse
import os

can_ml_attributes = ['timestamp', 'arbitration_id', 'data_field', 'attack']

def pad_or_truncate(sequence, target_length):
    """Pad the sequence with zeros or truncate to the target length."""
    if len(sequence) < target_length:
        sequence = sequence + [0] * (target_length - len(sequence))  # pad with zeros
    return sequence[:target_length]  # truncate if longer

def reshape_to_target_size(array, target_size=64):
    """Reshape or pad a flattened array into a target_size x target_size matrix."""
    padded = pad_or_truncate(array, target_size * target_size)  # Ensure target_size^2 elements
    return np.array(padded).reshape((target_size, target_size))  # Reshape into target_size x target_size

def split_into_list(string, type):
    res = []
    if type == 'payload':
        res = list(string)

    elif type == 'canID':
        res = list(bin(int(string))[2:].zfill(64))
    
    return [int(bit) for bit in res]

def transform_timestamp_log_delta(df, timestamp_col="timestamp", scale_factor=1e6):
    df = df.copy()

    ts = df[timestamp_col].astype(float).values

    # global delta time
    delta_t = np.diff(ts, prepend=ts[0])
    delta_t = np.maximum(delta_t, 0.0)

    # log-scaled delta time
    log_delta = np.log1p(scale_factor * delta_t)

    # min-max scale về [0, 1]
    log_delta_norm = (log_delta - log_delta.min()) / (
        log_delta.max() - log_delta.min() + 1e-8
    )

    # ghi đè lên chính cột timestamp
    df[timestamp_col] = log_delta_norm.astype(np.float32)

    return df

def parse_and_stack_timestamps(timestamps, target_size=64):
    """
    Parse and stack delta timestamps into a 2D binary array.

    Sau khi transform_timestamp_log_delta, timestamps đã là log-scaled delta
    được normalize về [0, 1]. Encode thành binary bằng cách nhân với 2^32
    rồi convert sang 64-bit binary string — giữ nguyên pipeline encode
    như CAN ID và Payload.

    Giá trị delta gần 0 (frame đến đúng chu kỳ) → binary gần 000...0
    Giá trị delta lớn (frame đến bất thường) → binary lớn hơn → dễ phân biệt
    """
    def parse_single_timestamp(timestamp):
        # timestamp đã trong [0, 1] sau log-delta normalization
        # nhân 2^32 để trải ra không gian binary 32-bit, pad thành 64 bits
        scaled = float(timestamp) * (2 ** 32)
        scaled = max(0, int(scaled))  # đảm bảo không âm
        parsed = np.array(list(bin(scaled)[2:].zfill(64))).astype(float)
        return parsed[:64]  # đảm bảo đúng 64 bits

    # Parse và stack từng timestamp
    stacked_timestamps = np.array([parse_single_timestamp(ts) for ts in timestamps])

    # Pad hoặc truncate để đủ target_size hàng
    if stacked_timestamps.shape[0] > target_size:
        stacked_timestamps = stacked_timestamps[:target_size, :]
    elif stacked_timestamps.shape[0] < target_size:
        padding = np.zeros((target_size - stacked_timestamps.shape[0], stacked_timestamps.shape[1]))
        stacked_timestamps = np.vstack((stacked_timestamps, padding))

    return stacked_timestamps

def normalize_time_zscore(time_series):
    """Normalize time using Z-score normalization."""
    mean_val = time_series.mean()
    std_val = time_series.std()
    normalized = (time_series - mean_val) / std_val
    return normalized


# def normalize_timestamp(timestamp):
#     """
#     Normalize timestamp using delta time and Min-Max scaling.

#     Args:
#         timestamp (numpy.ndarray): Array of timestamp values.

#     Returns:
#         numpy.ndarray: Normalized representation of timestamp values.
#     """
#     # Step 1: Tính delta time
#     delta_time = timestamp - timestamp[0]  # Tính khoảng thời gian so với giá trị đầu tiên
    
#     # Step 2: Min-Max Scaling
#     min_val = np.min(delta_time)
#     max_val = np.max(delta_time)
#     scaled = (delta_time - min_val) / (max_val - min_val)  # Chuẩn hóa giá trị trong khoảng [0, 1]
    
#     # Step 3: Trả về giá trị đã chuẩn hóa (float)
#     return scaled

def serialize_example(x, y): 
    """converts x, y to tf.train.Example and serialize"""
    id_seq, data_seq, timestamp, id_freq= x
    id_seq = tf.train.Int64List(value = np.array(id_seq).flatten())
    data_seq = tf.train.Int64List(value = np.array(data_seq).flatten())
    timestamp = tf.train.FloatList(value = np.array(timestamp).flatten())
    id_freq = tf.train.FloatList(value=np.array(id_freq).flatten())

    label = tf.train.Int64List(value = np.array([y]))

    features = tf.train.Features(
        feature = {
            "id_seq": tf.train.Feature(int64_list = id_seq),
            "data_seq": tf.train.Feature(int64_list = data_seq),
            "timestamp": tf.train.Feature(float_list = timestamp),
            "id_freq":  tf.train.Feature(float_list=id_freq),
            "label" : tf.train.Feature(int64_list = label)
        }
    )
    example = tf.train.Example(features = features)
    return example.SerializeToString()

def write_tfrecord(data, filename):
    tfrecord_writer = tf.io.TFRecordWriter(filename)
    for _, row in tqdm(data.iterrows()):
        X = (row['id_seq'], row['data_seq'], row['timestamp'], row['id_freq'])
        Y = row['label']
        tfrecord_writer.write(serialize_example(X, Y))
    tfrecord_writer.close() 

def split_data(file_name, attack_id, window_size, strided_size, target_size=64):
    if not os.path.exists(file_name):
        print(file_name, ' does not exist!')
        return None

    print("Window size = {}, strided = {}".format(window_size, strided_size))
    df = pd.read_csv(file_name)
    df.columns = can_ml_attributes
    print("Reading {}: done".format(file_name))

    print("The number of rows in the original CSV: ", len(df))

    df = df.sort_values('timestamp', ascending=True).reset_index(drop=True)

    print("data field", df.iloc[0]['data_field'])

    # Binary 'can data' (CAN Data)
    df['Data'] = df['data_field'].apply(lambda x: split_into_list(str(x), 'payload'))
    # Binary 'canID' (CAN ID)
    df['canID'] = df['arbitration_id'].apply(lambda x: split_into_list(x, 'canID'))

    # Normalize timestamp dùng log-scaled delta time thay vi z-score
    # Delta time nhaỵ hon voi timing attack (DoS, fuzzy thay doi inter-arrival time ro ret)
    df = transform_timestamp_log_delta(
        df,
        timestamp_col="timestamp",
        scale_factor=1e6
    )

    print(df.head())

    print("CAN-ML pre-processing: Done")

    print("Initial attack count before windowing:", df[df['attack'] != 0].shape[0])
    as_strided = np.lib.stride_tricks.sliding_window_view

    canid = as_strided(df.canID.values, window_shape=window_size)[::strided_size]
    data = as_strided(df.Data.values, window_shape=window_size)[::strided_size]
    timestamp = as_strided(df.timestamp.values, window_shape=window_size)[::strided_size]

    # raw arbitration_id windows — giữ giá trị gốc để tính frequency
    arb_id_raw = as_strided(df.arbitration_id.values, window_shape=window_size)[::strided_size]

    label = as_strided(df.attack.values, window_shape=window_size)[::strided_size]

    df = pd.DataFrame({
        'id_seq': pd.Series(canid.tolist()),
        'data_seq': pd.Series(data.tolist()),
        'timestamp': pd.Series(timestamp.tolist()),
        'arb_id_raw':  pd.Series(arb_id_raw.tolist()),
        'label': pd.Series(label.tolist())
    }, index=range(len(canid)))

    df['label'] = df['label'].apply(lambda x: attack_id if any(x) else 0)

    print("Aggregating data: Done")
    print('#Normal: ', df[df['label'] == 0].shape[0])
    print('#Attack: ', df[df['label'] != 0].shape[0])

    df['id_seq'] = df['id_seq'].apply(lambda x: reshape_to_target_size([item for sublist in x for item in sublist], target_size))
    df['data_seq'] = df['data_seq'].apply(lambda x: reshape_to_target_size([item for sublist in x for item in sublist], target_size))
    df['timestamp'] = df['timestamp'].apply(lambda x: parse_and_stack_timestamps(x, target_size))

    # Frequency map channel: mỗi window → ma trận (target_size, target_size)
    # row i = tần suất của arbitration ID ở message i, broadcast ra 64 cols
    print("Encoding ID frequency map channel...")
    df['id_freq'] = df['arb_id_raw'].apply(
        lambda window: encode_frequency_map(window, target_size)
    )

    return df[['id_seq', 'data_seq', 'timestamp', 'id_freq', 'label']].reset_index().drop(['index'], axis=1)

def encode_frequency_map(window_ids: list, target_size: int = 64) -> np.ndarray:
    """
    Encode một window gồm target_size CAN IDs thành ma trận frequency 2D.
 
    Với mỗi message trong window, tính tần suất xuất hiện của arbitration
    ID đó trong toàn bộ window (normalized về [0, 1]), rồi broadcast giá
    trị đó sang tất cả 64 columns của row tương ứng.
 
    Kết quả là ma trận (target_size, target_size) trong đó:
      - Các rows thuộc ID bị inject nhiều lần → giá trị cao (sáng)
      - Các rows thuộc ID bình thường → giá trị thấp (tối)
    Pattern này tạo ra visual signal rõ ràng để phân biệt spoofing
    attack với Normal traffic.
 
    Args:
        window_ids: list of target_size arbitration ID values (int hoặc str).
        target_size: số messages trong window = số rows/cols của output (64).
 
    Returns:
        np.ndarray shape (target_size, target_size), dtype float32,
        giá trị trong [0, 1].
    """
    n = len(window_ids)
 
    # đếm tần suất từng ID trong window
    id_counts: dict = {}
    for can_id in window_ids:
        key = str(can_id)
        id_counts[key] = id_counts.get(key, 0) + 1
 
    # normalize về [0, 1] theo số messages trong window
    id_freq = {k: v / n for k, v in id_counts.items()}
 
    # mỗi row = frequency của ID đó broadcast ra target_size columns
    rows = []
    for can_id in window_ids:
        freq = id_freq[str(can_id)]
        rows.append([freq] * target_size)
 
    matrix = np.array(rows, dtype=np.float32)  # (n, target_size)
 
    # padding nếu window ngắn hơn target_size
    if matrix.shape[0] < target_size:
        pad = np.zeros((target_size - matrix.shape[0], target_size), dtype=np.float32)
        matrix = np.vstack([matrix, pad])
 
    return matrix[:target_size]  # (target_size, target_size)

def main(indir, outdir, attacks, window_size, strided, target_size=64):
    print(outdir)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    data_info = {}

    for attack_id, attack in enumerate(attacks):
        print('Attack: {} ==============='.format(attack))
        finput = '{}/{}.csv'.format(indir, attack)
        print("Attack Id: ", attack_id)
        df = split_data(finput, attack_id + 1, window_size, strided, target_size=target_size)
        print("Writing...................")

        foutput_attack = '{}/{}'.format(outdir, attack)
        foutput_normal = '{}/Normal_{}'.format(outdir, attack)
        df_attack = df[df['label'] != 0].sample(frac=0.1, random_state=42)
        df_normal = df[df['label'] == 0].sample(frac=0.1, random_state=42)
        write_tfrecord(df_attack, foutput_attack)
        write_tfrecord(df_normal, foutput_normal)
        
        data_info[foutput_attack] = df_attack.shape[0]
        data_info[foutput_normal] = df_normal.shape[0]
    
    json.dump(data_info, open('{}/datainfo.txt'.format(outdir), 'w'))
    print("DONE!")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, default="./data/can-ml/can-ml/2017-subaru-forester/merged")
    parser.add_argument('--outdir', type=str, default="./data/can-ml/can-ml/2017-subaru-forester/290626-4-channel/TFRecord")
    parser.add_argument('--window_size', type=int, default=64)
    parser.add_argument('--strided', type=int, default=32)
    args = parser.parse_args()
    
    attack_types = ["combined", "DoS", "fuzzing", "gear", "interval", "rpm", 
                "speed", "standstill", "systematic"]
    if args.strided is None:
        args.strided = args.window_size
        
    outdir =  args.outdir + '_w{}_s{}'.format(args.window_size, args.strided)
    main(args.indir, outdir, attack_types, args.window_size, args.strided)