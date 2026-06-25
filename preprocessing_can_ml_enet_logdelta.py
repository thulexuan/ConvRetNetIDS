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
    # log_delta_norm = (log_delta - log_delta.min()) / (
    #     log_delta.max() - log_delta.min() + 1e-8
    # )

    # ghi đè lên chính cột timestamp
    df[timestamp_col] = log_delta.astype(np.float32)

    return df

def parse_and_stack_timestamps(timestamps, target_size=64):
    """
    Stack log-delta-normalized timestamps into a 2D timestamp channel.

    Input:
        timestamps: a window of scalar timestamp features that have already been
                    transformed by transform_timestamp_log_delta(), so each value
                    is in [0, 1]. Shape: [window_size].

    Output:
        timestamp_channel: shape [target_size, target_size].
                           Each row corresponds to one CAN message. The scalar
                           timestamp feature of that message is repeated across
                           target_size columns.

    Example:
        timestamps = [0.1, 0.4, ...] with target_size=64
        output[0, :] = [0.1, 0.1, ..., 0.1]
        output[1, :] = [0.4, 0.4, ..., 0.4]
    """
    timestamps = np.asarray(timestamps, dtype=np.float32)

    # Truncate or pad the number of rows/messages to target_size
    if timestamps.shape[0] > target_size:
        timestamps = timestamps[:target_size]
    elif timestamps.shape[0] < target_size:
        pad_len = target_size - timestamps.shape[0]
        timestamps = np.pad(timestamps, (0, pad_len), mode="constant", constant_values=0.0)

    # [target_size] -> [target_size, target_size]
    # Repeat each message-level timestamp feature along the width dimension.
    timestamp_channel = np.repeat(timestamps[:, None], target_size, axis=1)

    return timestamp_channel.astype(np.float32)

def normalize_time_zscore(time_series):
    """Normalize time using Z-score normalization."""
    mean_val = time_series.mean()
    std_val = time_series.std()
    normalized = (time_series - mean_val) / std_val
    return normalized

def serialize_example(x, y): 
    """converts x, y to tf.train.Example and serialize"""
    id_seq, data_seq, timestamp = x
    id_seq = tf.train.Int64List(value = np.array(id_seq).flatten())
    data_seq = tf.train.Int64List(value = np.array(data_seq).flatten())
    timestamp = tf.train.FloatList(value = np.array(timestamp).flatten())

    label = tf.train.Int64List(value = np.array([y]))

    features = tf.train.Features(
        feature = {
            "id_seq": tf.train.Feature(int64_list = id_seq),
            "data_seq": tf.train.Feature(int64_list = data_seq),
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

    # Transform timestamp to log-scaled global delta time and normalize to [0, 1].
    # The stacked timestamp channel will be created after windowing.
    # df['timestamp'] = normalize_time_zscore(df['timestamp'].values)
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

    label = as_strided(df.attack.values, window_shape=window_size)[::strided_size]

    df = pd.DataFrame({
        'id_seq': pd.Series(canid.tolist()),
        'data_seq': pd.Series(data.tolist()),
        'timestamp': pd.Series(timestamp.tolist()),
        'label': pd.Series(label.tolist())
    }, index=range(len(canid)))

    df['label'] = df['label'].apply(lambda x: attack_id if any(x) else 0)

    print("Aggregating data: Done")
    print('#Normal: ', df[df['label'] == 0].shape[0])
    print('#Attack: ', df[df['label'] != 0].shape[0])

    df['id_seq'] = df['id_seq'].apply(lambda x: reshape_to_target_size([item for sublist in x for item in sublist], target_size))
    df['data_seq'] = df['data_seq'].apply(lambda x: reshape_to_target_size([item for sublist in x for item in sublist], target_size))
    df['timestamp'] = df['timestamp'].apply(lambda x: parse_and_stack_timestamps(x, target_size))
    return df[['id_seq', 'data_seq', 'timestamp', 'label']].reset_index().drop(['index'], axis=1)


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
    parser.add_argument('--outdir', type=str, default="./data/can-ml/can-ml/2017-subaru-forester/230626/TFRecord")
    parser.add_argument('--window_size', type=int, default=64)
    parser.add_argument('--strided', type=int, default=32)
    args = parser.parse_args()
    
    attack_types = ["combined", "DoS", "fuzzing", "gear", "interval", "rpm", 
                "speed", "standstill", "systematic"]
    if args.strided is None:
        args.strided = args.window_size
        
    outdir =  args.outdir + '_w{}_s{}'.format(args.window_size, args.strided)
    main(args.indir, outdir, attack_types, args.window_size, args.strided)