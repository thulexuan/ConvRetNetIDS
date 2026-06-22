import pandas as pd
import numpy as np
import json
import tensorflow as tf
from tqdm import tqdm
import argparse
import os

can_ml_attributes = ['timestamp', 'arbitration_id', 'data_field', 'attack']


def pad_or_truncate(sequence, target_length, pad_value=0):
    """Pad the sequence or truncate it to target_length."""
    if len(sequence) < target_length:
        sequence = sequence + [pad_value] * (target_length - len(sequence))
    return sequence[:target_length]


def can_id_to_64bits(value):
    """Convert CAN ID to a 64-bit binary vector."""
    s = str(value).strip()
    try:
        # Supports decimal strings and strings such as "0x123".
        v = int(s, 0)
    except ValueError:
        # Fallback for hexadecimal strings without 0x prefix.
        v = int(s, 16)
    return [int(b) for b in bin(v)[2:].zfill(64)[-64:]]


def payload_to_64bits(value):
    """
    Convert payload/data_field to a 64-bit binary vector.

    If the field is already a 64-bit binary string, keep it as bits.
    Otherwise, treat it as a hexadecimal payload, which is common for CAN data.
    """
    s = str(value).strip().replace(' ', '').replace('0x', '').replace('0X', '')

    if s == '' or s.lower() == 'nan':
        return [0] * 64

    # Already binary, e.g., "010101..." with 64 bits.
    if set(s).issubset({'0', '1'}) and len(s) == 64:
        bits = [int(b) for b in s]
        return pad_or_truncate(bits, 64, pad_value=0)

    # Hex payload, usually 8 bytes = 16 hex characters.
    try:
        s = s.zfill(16)[-16:]
        v = int(s, 16)
    except ValueError:
        # Last fallback: keep only binary-looking characters.
        bits = [int(b) for b in s if b in {'0', '1'}]
        return pad_or_truncate(bits, 64, pad_value=0)

    return [int(b) for b in bin(v)[2:].zfill(64)[-64:]]


def split_into_list(value, field_type):
    """Backward-compatible wrapper for CAN ID and payload parsing."""
    if field_type == 'payload':
        return payload_to_64bits(value)
    if field_type == 'canID':
        return can_id_to_64bits(value)
    raise ValueError(f"Unsupported field_type: {field_type}")


def reshape_binary_to_signed_image(bit_sequence, target_size=64):
    """
    Convert flattened binary bits to a target_size x target_size image
    and scale {0, 1} to {-1, +1}.
    """
    padded = pad_or_truncate(list(bit_sequence), target_size * target_size, pad_value=0)
    img = np.asarray(padded, dtype=np.float32).reshape((target_size, target_size))
    img = img * 2.0 - 1.0
    return img.astype(np.float32)


def _safe_zscore_scale(values, stats_mask=None, clip=3.0):
    """
    Standardize values using statistics from stats_mask if available,
    clip to [-clip, clip], then scale to [-1, 1].
    """
    values = np.asarray(values, dtype=np.float64)

    if stats_mask is not None and np.any(stats_mask):
        ref = values[stats_mask]
    else:
        ref = values

    mean_val = np.mean(ref)
    std_val = np.std(ref)
    if not np.isfinite(std_val) or std_val < 1e-12:
        std_val = 1.0

    z = (values - mean_val) / std_val
    z = np.clip(z, -clip, clip)
    return (z / clip).astype(np.float32), float(mean_val), float(std_val)


def compute_temporal_features(timestamps, can_ids=None, normal_mask=None, clip=3.0):
    """
    Build temporal features for the timestamp channel.

    Output features are scaled to [-1, 1]:
    1) global inter-arrival time: t_i - t_{i-1}
    2) same-ID inter-arrival time: t_i - previous_time_of_same_CAN_ID

    Statistics are estimated from normal frames if normal_mask is provided,
    otherwise from the full file.
    """
    ts = np.asarray(timestamps, dtype=np.float64)
    n = len(ts)
    if n == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    # Global inter-arrival time.
    dt_global = np.diff(ts, prepend=ts[0])
    if n > 1:
        positive_dt = dt_global[1:][dt_global[1:] > 0]
        fallback = np.median(positive_dt) if len(positive_dt) > 0 else 0.0
        dt_global[0] = fallback
    dt_global = np.maximum(dt_global, 0.0)

    # Same-CAN-ID inter-arrival time.
    if can_ids is None:
        dt_same_id = dt_global.copy()
    else:
        last_seen = {}
        dt_same_id = np.full(n, np.nan, dtype=np.float64)
        for i, (t, cid) in enumerate(zip(ts, can_ids)):
            key = str(cid)
            if key in last_seen:
                dt_same_id[i] = max(t - last_seen[key], 0.0)
            last_seen[key] = t

        finite_positive = dt_same_id[np.isfinite(dt_same_id) & (dt_same_id > 0)]
        if len(finite_positive) > 0:
            fill_value = np.median(finite_positive)
        else:
            positive_global = dt_global[dt_global > 0]
            fill_value = np.median(positive_global) if len(positive_global) > 0 else 0.0
        dt_same_id[~np.isfinite(dt_same_id)] = fill_value
        dt_same_id = np.maximum(dt_same_id, 0.0)

    # Log transform reduces the dominance of rare large time gaps.
    log_dt_global = np.log1p(dt_global)
    log_dt_same_id = np.log1p(dt_same_id)

    global_scaled, _, _ = _safe_zscore_scale(log_dt_global, stats_mask=normal_mask, clip=clip)
    same_id_scaled, _, _ = _safe_zscore_scale(log_dt_same_id, stats_mask=normal_mask, clip=clip)

    return global_scaled, same_id_scaled


def make_timestamp_map(global_dt_window, same_id_dt_window=None, target_size=64):
    """
    Convert temporal features in one CAN window to a target_size x target_size image.

    Left half columns  : global inter-arrival time.
    Right half columns : same-ID inter-arrival time.
    Values are already scaled to [-1, 1].
    """
    g = np.asarray(global_dt_window, dtype=np.float32)
    g = pad_or_truncate(g.tolist(), target_size, pad_value=0.0)
    g = np.asarray(g, dtype=np.float32)

    if same_id_dt_window is None:
        sid = g.copy()
    else:
        sid = np.asarray(same_id_dt_window, dtype=np.float32)
        sid = pad_or_truncate(sid.tolist(), target_size, pad_value=0.0)
        sid = np.asarray(sid, dtype=np.float32)

    img = np.zeros((target_size, target_size), dtype=np.float32)
    half = target_size // 2
    img[:, :half] = g[:, None]
    img[:, half:] = sid[:, None]
    return img


def serialize_example(x, y):
    """Convert x, y to tf.train.Example and serialize."""
    id_seq, data_seq, timestamp = x

    # ID and payload are now {-1, +1}; they can still be stored as int64.
    id_seq = tf.train.Int64List(value=np.asarray(id_seq, dtype=np.int64).flatten())
    data_seq = tf.train.Int64List(value=np.asarray(data_seq, dtype=np.int64).flatten())

    # Timestamp is continuous in [-1, +1], so keep it as float.
    timestamp = tf.train.FloatList(value=np.asarray(timestamp, dtype=np.float32).flatten())
    label = tf.train.Int64List(value=np.asarray([y], dtype=np.int64))

    features = tf.train.Features(
        feature={
            "id_seq": tf.train.Feature(int64_list=id_seq),
            "data_seq": tf.train.Feature(int64_list=data_seq),
            "timestamp": tf.train.Feature(float_list=timestamp),
            "label": tf.train.Feature(int64_list=label),
        }
    )
    example = tf.train.Example(features=features)
    return example.SerializeToString()


def write_tfrecord(data, filename):
    tfrecord_writer = tf.io.TFRecordWriter(filename)
    for _, row in tqdm(data.iterrows(), total=len(data)):
        X = (row['id_seq'], row['data_seq'], row['timestamp'])
        Y = row['label']
        tfrecord_writer.write(serialize_example(X, Y))
    tfrecord_writer.close()


def split_data(file_name, attack_id, window_size, strided_size, target_size=64):
    if not os.path.exists(file_name):
        print(file_name, ' does not exist!')
        return None

    print("Window size = {}, strided = {}".format(window_size, strided_size))
    df = pd.read_csv(file_name)

    print("The number of rows in the original CSV: ", len(df))

    df.columns = can_ml_attributes
    print("Reading {}: done".format(file_name))

    df = df.sort_values('timestamp', ascending=True).reset_index(drop=True)

    # 1) CAN ID channel: 64-bit binary representation.
    # df['canID'] = df['arbitration_id'].apply(lambda x: split_into_list(x, 'canID'))

    # 2) Payload channel: 64-bit binary representation.
    # df['Data'] = df['data_field'].apply(lambda x: split_into_list(x, 'payload'))

    # 3) Timestamp channel: continuous temporal features, not binary fractional bits.
    # normal_mask = (df['attack'].values == 0)
    # dt_global, dt_same_id = compute_temporal_features(
    #     timestamps=df['timestamp'].values,
    #     can_ids=df['arbitration_id'].values,
    #     normal_mask=normal_mask,
    #     clip=3.0,
    # )
    # df['dt_global'] = dt_global
    # df['dt_same_id'] = dt_same_id

    # print("CAN-ML pre-processing: Done")
    # print("Initial attack count before windowing:", df[df['attack'] != 0].shape[0])

    as_strided = np.lib.stride_tricks.sliding_window_view

    canid = as_strided(df.canID.values, window_shape=window_size)[::strided_size]
    data = as_strided(df.Data.values, window_shape=window_size)[::strided_size]
    dt_global_win = as_strided(df.dt_global.values, window_shape=window_size)[::strided_size]
    dt_same_id_win = as_strided(df.dt_same_id.values, window_shape=window_size)[::strided_size]
    label = as_strided(df.attack.values, window_shape=window_size)[::strided_size]

    out_df = pd.DataFrame({
        'id_seq': pd.Series(canid.tolist()),
        'data_seq': pd.Series(data.tolist()),
        'dt_global': pd.Series(dt_global_win.tolist()),
        'dt_same_id': pd.Series(dt_same_id_win.tolist()),
        'label': pd.Series(label.tolist()),
    }, index=range(len(canid)))

    out_df['label'] = out_df['label'].apply(lambda x: attack_id if any(x) else 0)

    print("After windowing, before scaling, the number of samples: ", len(out_df))

    print("Aggregating data: Done")
    print('#Normal: ', out_df[out_df['label'] == 0].shape[0])
    print('#Attack: ', out_df[out_df['label'] != 0].shape[0])

    # ID and payload: {0, 1} bits -> {-1, +1} images.
    out_df['id_seq'] = out_df['id_seq'].apply(
        lambda x: reshape_binary_to_signed_image([item for sublist in x for item in sublist], target_size)
    )
    out_df['data_seq'] = out_df['data_seq'].apply(
        lambda x: reshape_binary_to_signed_image([item for sublist in x for item in sublist], target_size)
    )

    # Timestamp: continuous temporal image in [-1, +1].
    out_df['timestamp'] = out_df.apply(
        lambda row: make_timestamp_map(row['dt_global'], row['dt_same_id'], target_size),
        axis=1,
    )

    print("The number of sample after windowing and scaling: ", len(out_df))

    return out_df[['id_seq', 'data_seq', 'timestamp', 'label']].reset_index(drop=True)


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
        if df is None:
            continue

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, default="./data/can-ml/can-ml/2017-subaru-forester/220626/merged")
    parser.add_argument('--outdir', type=str, default="./data/can-ml/can-ml/2017-subaru-forester/220626/TFRecord")
    parser.add_argument('--window_size', type=int, default=64)
    parser.add_argument('--strided', type=int, default=32)
    parser.add_argument('--target_size', type=int, default=64)
    args = parser.parse_args()

    attack_types = ["combined"
                    , "DoS", "fuzzing", "gear", "interval", "rpm",
                    "speed", "standstill", "systematic"
                    ]
    if args.strided is None:
        args.strided = args.window_size

    outdir = args.outdir + '_w{}_s{}'.format(args.window_size, args.strided)
    main(args.indir, outdir, attack_types, args.window_size, args.strided, target_size=args.target_size)
