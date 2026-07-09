import argparse
import json
import os

import numpy as np
import pandas as pd
from tqdm import tqdm


RAW_COLUMNS = ['timestamp', 'arbitration_id', 'data_field', 'attack']
PAYLOAD_COLUMNS = [f'data_{i}' for i in range(8)]
PROCESSED_COLUMNS = ['timestamp', 'arbitration_id', 'attack'] + PAYLOAD_COLUMNS


def get_tensorflow():
    try:
        import tensorflow as tf
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency 'tensorflow'. Install TensorFlow in the Python "
            "environment used to run this preprocessing script before writing TFRecords."
        ) from exc
    return tf


def normalize_payload_values(values):
    values = np.asarray(values, dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = np.clip(values, 0.0, 255.0)
    if values.max(initial=0.0) > 1.0:
        values = values / 255.0
    return values.tolist()


def payload_to_8_values(value):
    """
    Convert one CAN payload to 8 normalized byte values.

    Supports the two payload forms used around this project:
    - hexadecimal text, e.g. "0200000000000002"
    - 64-bit binary text from can_ml.ipynb, e.g. "00000010..."
    """
    if pd.isna(value):
        return [0.0] * 8

    payload = str(value).strip().replace(' ', '')
    if not payload:
        return [0.0] * 8

    if set(payload).issubset({'0', '1'}) and len(payload) > 16:
        payload = payload[:64].ljust(64, '0')
        values = [int(payload[i:i + 8], 2) for i in range(0, 64, 8)]
    else:
        payload = payload.upper()
        payload = ''.join(ch for ch in payload if ch in '0123456789ABCDEF')
        payload = payload[:16].ljust(16, '0')
        values = [int(payload[i:i + 2], 16) for i in range(0, 16, 2)]

    return normalize_payload_values(values)


def load_payload_dataframe(file_name):
    df = pd.read_csv(file_name)

    if set(PROCESSED_COLUMNS).issubset(df.columns):
        df = df[PROCESSED_COLUMNS].copy()
    elif set(RAW_COLUMNS).issubset(df.columns):
        df = df[RAW_COLUMNS].copy()
    elif len(df.columns) == len(PROCESSED_COLUMNS):
        df.columns = PROCESSED_COLUMNS
    elif len(df.columns) == len(RAW_COLUMNS):
        df.columns = RAW_COLUMNS
    else:
        raise ValueError(
            f"{file_name} must contain either {RAW_COLUMNS} or {PROCESSED_COLUMNS}."
        )

    df = df.sort_values('timestamp', ascending=True).reset_index(drop=True)

    if set(PAYLOAD_COLUMNS).issubset(df.columns):
        payload_df = df[PAYLOAD_COLUMNS].apply(pd.to_numeric, errors='coerce').fillna(0.0)
        payload_df = payload_df.apply(lambda row: normalize_payload_values(row.values), axis=1)
        payload_df = pd.DataFrame(payload_df.tolist(), columns=PAYLOAD_COLUMNS, index=df.index)
    else:
        payload_df = pd.DataFrame(
            df['data_field'].apply(payload_to_8_values).tolist(),
            columns=PAYLOAD_COLUMNS,
            index=df.index,
        )

    return pd.concat([df[['timestamp', 'attack']], payload_df], axis=1).dropna()


def make_payload_image(payload_window, target_size):
    image = np.asarray(payload_window, dtype=np.float32).reshape(-1)
    expected_size = target_size * target_size

    if image.size < expected_size:
        image = np.pad(image, (0, expected_size - image.size), constant_values=0.0)
    elif image.size > expected_size:
        image = image[:expected_size]

    return image.reshape(target_size, target_size).astype(np.float32)


def serialize_example(payload_image, label):
    tf = get_tensorflow()
    payload_feature = tf.train.FloatList(value=np.asarray(payload_image, dtype=np.float32).flatten())
    label_feature = tf.train.Int64List(value=np.asarray([label], dtype=np.int64))

    features = tf.train.Features(
        feature={
            'data_seq': tf.train.Feature(float_list=payload_feature),
            'label': tf.train.Feature(int64_list=label_feature),
        }
    )
    return tf.train.Example(features=features).SerializeToString()


def write_tfrecord(data, filename):
    tf = get_tensorflow()
    with tf.io.TFRecordWriter(filename) as writer:
        for _, row in tqdm(data.iterrows(), total=len(data)):
            writer.write(serialize_example(row['data_seq'], row['label']))


def split_data(file_name, attack_id, window_size, strided_size, target_size=32):
    if not os.path.exists(file_name):
        print(file_name, 'does not exist!')
        return None

    print("Window size = {}, strided = {}".format(window_size, strided_size))
    df = load_payload_dataframe(file_name)
    print("Reading {}: done".format(file_name))
    print("The number of rows in the original CSV:", len(df))
    print("Initial attack count before windowing:", df[df['attack'] != 0].shape[0])

    payload_values = df[PAYLOAD_COLUMNS].to_numpy(dtype=np.float32)
    labels = df['attack'].to_numpy()

    if len(df) < window_size:
        print("Not enough rows for one full window.")
        return pd.DataFrame(columns=['data_seq', 'label'])

    as_strided = np.lib.stride_tricks.sliding_window_view
    payload_windows = as_strided(payload_values, window_shape=window_size, axis=0)[::strided_size]
    payload_windows = np.moveaxis(payload_windows, -1, 1)
    label_windows = as_strided(labels, window_shape=window_size)[::strided_size]

    out_df = pd.DataFrame({
        'data_seq': [make_payload_image(window, target_size) for window in payload_windows],
        'label': [attack_id if np.any(window) else 0 for window in label_windows],
    })

    print("Aggregating payload data: Done")
    print('#Normal:', out_df[out_df['label'] == 0].shape[0])
    print('#Attack:', out_df[out_df['label'] != 0].shape[0])
    return out_df


def sample_fraction(df, frac, random_state=42):
    if df.empty or frac >= 1.0:
        return df
    return df.sample(frac=frac, random_state=random_state)


def main(indir, outdir, attacks, window_size, strided, target_size=32, sample_frac=0.2):
    print(outdir)
    os.makedirs(outdir, exist_ok=True)
    data_info = {}

    for attack_id, attack in enumerate(attacks):
        print('Attack: {} ==============='.format(attack))
        finput = '{}/{}.csv'.format(indir, attack)
        print("Attack Id:", attack_id + 1)
        df = split_data(finput, attack_id + 1, window_size, strided, target_size=target_size)
        if df is None:
            continue

        print("Writing...................")
        foutput_attack = '{}/{}'.format(outdir, attack)
        foutput_normal = '{}/Normal_{}'.format(outdir, attack)

        df_attack = sample_fraction(df[df['label'] != 0], sample_frac)
        df_normal = sample_fraction(df[df['label'] == 0], sample_frac)

        write_tfrecord(df_attack, foutput_attack)
        write_tfrecord(df_normal, foutput_normal)

        data_info[foutput_attack] = df_attack.shape[0]
        data_info[foutput_normal] = df_normal.shape[0]

    json.dump(data_info, open('{}/datainfo.txt'.format(outdir), 'w'))
    print("DONE!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, default="./data/can-ml/can-ml/2017-subaru-forester/220626/merged")
    parser.add_argument('--outdir', type=str, default="./data/can-ml/can-ml/2017-subaru-forester/220626/preprocessed/payload_only/TFRecord")
    parser.add_argument('--window_size', type=int, default=128)
    parser.add_argument('--strided', type=int, default=64)
    parser.add_argument('--target_size', type=int, default=32)
    parser.add_argument('--sample_frac', type=float, default=0.2)
    args = parser.parse_args()

    attack_types = [
        "combined",
        "DoS",
        "fuzzing",
        "gear",
        "interval",
        "rpm",
        "speed",
        "standstill",
        "systematic",
    ]

    if args.strided is None:
        args.strided = args.window_size

    outdir = args.outdir + '_w{}_s{}'.format(args.window_size, args.strided)
    main(
        args.indir,
        outdir,
        attack_types,
        args.window_size,
        args.strided,
        target_size=args.target_size,
        sample_frac=args.sample_frac,
    )

