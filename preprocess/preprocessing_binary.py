"""
preprocessing_binary.py
========================

Preprocessing CAN data theo hướng BIỂU DIỄN NHỊ PHÂN (binary representation)
thay vì decimal + log-min-max/MinMax normalization:

    - CAN ID:  hex -> decimal -> biểu diễn nhị phân 29 bit
                     (mỗi bit là 1 feature, giá trị 0.0 hoặc 1.0)
    - Payload: hex -> tách 8 byte (như cũ) -> mỗi byte decimal -> biểu diễn
               nhị phân 8 bit -> 8 byte x 8 bit = 64 bit

    -> Mỗi CAN frame trở thành 1 vector (29 + 64) = 93 chiều, MỌI phần tử
       đã sẵn trong {0.0, 1.0} - KHÔNG CẦN thêm bước MinMaxScaler/log-minmax
       nào nữa (biểu diễn nhị phân vốn đã "chuẩn hoá" tự nhiên).

Ý nghĩa của cách biểu diễn này (so với decimal + MinMax):
    - Giữ được cấu trúc BIT-LEVEL của CAN ID/payload - một số tấn công thao
      túng CAN thực chất chỉ lật (flip) 1-2 bit cụ thể (ví dụ 1 flag bit),
      điều mà biểu diễn decimal liên tục có thể "làm mờ" (một thay đổi nhỏ
      ở bit cao có thể tương ứng với thay đổi giá trị decimal rất lớn, còn
      thay đổi ở bit thấp lại tương ứng thay đổi decimal rất nhỏ - không
      đồng đều, đã bàn ở phần log-min-max).
    - Với biểu diễn nhị phân, MỌI bit đều có "trọng lượng" ngữ nghĩa như
      nhau về mặt biểu diễn đầu vào (đều là 1 feature nhị phân 0/1); mô
      hình tự học được bit nào quan trọng qua trọng số của nó, thay vì bị
      áp đặt sẵn "bit cao quan trọng hơn bit thấp" như biểu diễn decimal.
    - Đánh đổi: input dim tăng mạnh (9 -> 93), và mất đi thông tin thứ tự
      "độ lớn" tự nhiên giữa các giá trị byte gần nhau (ví dụ byte giá trị
      127 và 128 chỉ khác 1 đơn vị decimal nhưng khác biểu diễn nhị phân ở
      TẤT CẢ 8 bit: 01111111 vs 10000000) - mô hình cần tự học lại mối
      quan hệ thứ tự này nếu cần, thay vì được biểu diễn tường minh.

Toàn bộ phần windowing / train-test-split / sampling giữ NGUYÊN logic như
`preprocessing_minmax.py`, chỉ thay đổi cách encode 1 CAN frame thành vector
đặc trưng.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
ID_BITS = 29      # CAN ID -> 29-bit binary representation
BYTE_BITS = 8     # mỗi payload byte -> 8-bit binary representation
NUM_PAYLOAD_BYTES = 8
PAYLOAD_BITS = BYTE_BITS * NUM_PAYLOAD_BYTES   # = 64
TOTAL_FEATURES = ID_BITS + PAYLOAD_BITS          # = 93


# ── CAN ID: hex/dec -> 29-bit binary ─────────────────────────────────────────
def process_can_id_binary(arb_id) -> list:
    """
    arbitration_id (decimal int hoặc string) -> list 29 giá trị nhị phân
    (0.0 / 1.0), bit có trọng số cao nhất (MSB) đứng đầu danh sách.

    Ví dụ:
        140 -> '0000000000000000000010001100' (29 ký tự)
             -> [0.0, 0.0, ..., 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
    """
    arb_id_dec = int(arb_id)
    # Nếu ID vượt quá 29 bit (không nên xảy ra với CAN chuẩn/mở rộng), cắt bớt
    # bit cao để tránh lỗi format, đồng thời log cảnh báo.
    if arb_id_dec >= (1 << ID_BITS):
        log.warning("CAN ID %d vượt quá %d bit, sẽ bị cắt bớt bit cao.", arb_id_dec, ID_BITS)
        arb_id_dec &= (1 << ID_BITS) - 1

    bit_str = format(arb_id_dec, f"0{ID_BITS}b")
    return [float(b) for b in bit_str]


# ── Payload: hex -> 8 byte decimal -> mỗi byte 8-bit binary -> nối lại 64 bit ─
def process_data_field_binary(data_field) -> list:
    """
    data_field (hex string) -> list 64 giá trị nhị phân (0.0 / 1.0), theo
    thứ tự byte_0 (MSB trước) -> byte_1 -> ... -> byte_7.

    Steps (giữ nguyên phần tách 8 byte như bản decimal trước đó):
        1. Padding zfill(16) cho đủ 16 ký tự hex (= 8 byte)
        2. Tách thành 8 cặp hex (mỗi cặp = 1 byte)
        3. Mỗi byte: hex -> decimal
        4. Mỗi byte decimal -> biểu diễn nhị phân 8 bit
        5. Nối 8 x 8 bit lại thành 1 vector 64 chiều

    Ví dụ:
        '0001240828282720'
        -> byte decimal: [0, 1, 36, 8, 40, 40, 39, 32]
        -> mỗi byte 8-bit: ['00000000','00000001','00100100','00001000',
                            '00101000','00101000','00100111','00100000']
        -> nối lại: 64 giá trị 0.0/1.0
    """
    hex_str = str(data_field).strip().upper().zfill(16)[:16]

    bits: list[float] = []
    for i in range(NUM_PAYLOAD_BYTES):
        byte_hex = hex_str[i * 2 : i * 2 + 2]
        try:
            byte_dec = int(byte_hex, 16)
        except ValueError:
            byte_dec = 0
        byte_bit_str = format(byte_dec, f"0{BYTE_BITS}b")
        bits.extend(float(b) for b in byte_bit_str)

    return bits  # list of 64 floats (0.0/1.0)


# ── Apply to DataFrame (binary representation) ───────────────────────────────
def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nhận DataFrame có cột: timestamp, arbitration_id, data_field, attack
    Trả về DataFrame mới với các cột:

        timestamp                    (giữ nguyên)
        id_bit_0 .. id_bit_28         float32  <- CAN ID, 29-bit binary
        payload_bit_0 .. payload_bit_63  float32  <- payload, 64-bit binary
        attack                        int     (giữ nguyên)

    Tổng: 1 + 29 + 64 + 1 = 95 cột (93 cột feature nhị phân + timestamp + attack)
    """
    out = pd.DataFrame()
    out["timestamp"] = df["timestamp"].values

    # CAN ID -> 29 cột nhị phân
    id_bits_series = df["arbitration_id"].apply(process_can_id_binary)
    id_bits_df = pd.DataFrame(
        id_bits_series.tolist(),
        columns=[f"id_bit_{i}" for i in range(ID_BITS)],
        dtype="float32",
    )

    # Payload -> 64 cột nhị phân (8 byte x 8 bit)
    payload_bits_series = df["data_field"].apply(process_data_field_binary)
    payload_bits_df = pd.DataFrame(
        payload_bits_series.tolist(),
        columns=[f"payload_bit_{i}" for i in range(PAYLOAD_BITS)],
        dtype="float32",
    )

    out = pd.concat([out, id_bits_df, payload_bits_df], axis=1)
    out["attack"] = df["attack"].values.astype("int8")

    return out


def sample_10pct(
    data_3d: np.ndarray,
    labels: np.ndarray,
    attack_id: int,
    seed: int = 42,
):
    """Lấy ngẫu nhiên 20% mỗi nhóm (attack / normal), giữ thứ tự thời gian."""
    rng = np.random.default_rng(seed=seed)

    idx_normal = np.where(labels == 0)[0]
    idx_attack = np.where(labels == attack_id)[0]

    def _sample(idx):
        n = max(1, int(len(idx) * 0.2))
        return np.sort(rng.choice(idx, size=min(n, len(idx)), replace=False))

    chosen_normal = _sample(idx_normal)
    chosen_attack = _sample(idx_attack)
    chosen = np.sort(np.concatenate([chosen_normal, chosen_attack]))

    print(f"  normal : {len(idx_normal):6,} -> {len(chosen_normal):5,} (20%)")
    print(f"  attack : {len(idx_attack):6,} -> {len(chosen_attack):5,} (20%)")
    print(f"  total  : {len(labels):6,} -> {len(chosen):5,}")

    return data_3d[chosen], labels[chosen]


def load_attack_folder(filename: str, label, window_size=16, stride_size=16) -> pd.DataFrame:
    try:
        df = pd.read_csv(filename)
        df = df.sort_values("timestamp", ascending=True).reset_index(drop=True)
        binary_df = preprocess_df(df)
        print(binary_df.head())
    except Exception as exc:
        log.error("Error reading %s", filename)
        print(f"Error occurred while reading {filename}: {exc}")
        raise

    return binary_df


def build_feature_col(df: pd.DataFrame) -> list:
    """Lấy tên các cột feature nhị phân (bỏ timestamp, attack)."""
    return [c for c in df.columns if c not in ["timestamp", "attack"]]


def run_pipeline(
    root_dir,
    output_dir,
    window_size=29,
    stride_size=15,
    feature_size=TOTAL_FEATURES,
    attacks=None,
) -> None:
    """
    Pipeline: hex -> decimal -> BIỂU DIỄN NHỊ PHÂN (93 chiều/frame) ->
    windowing -> train/test split -> lưu .npy.

    KHÔNG CẦN bước MinMaxScaler nào sau đây, vì mọi giá trị đã sẵn ở dạng
    nhị phân {0.0, 1.0} - bản thân biểu diễn này đã "chuẩn hoá" tự nhiên.
    """
    root = Path(root_dir)
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    log.info("Loading each .csv file of each attack (binary representation, %d features/frame)", feature_size)

    xtrain = np.empty((0, window_size, feature_size))
    ytrain = np.empty((0))
    xtest = np.empty((0, window_size, feature_size))
    ytest = np.empty((0))

    for attack_id, attack in enumerate(attacks):
        print("Attack: {} ===============".format(attack))
        finput = "{}/{}.csv".format(root, attack)

        df = load_attack_folder(finput, attack_id + 1, window_size=window_size, stride_size=stride_size)

        feature_col = build_feature_col(df)
        assert len(feature_col) == feature_size, (
            f"Số cột feature ({len(feature_col)}) không khớp feature_size ({feature_size})"
        )
        df["features"] = df[feature_col].apply(lambda row: [row[col] for col in feature_col], axis=1)

        as_strided = np.lib.stride_tricks.as_strided
        output_shape = ((len(df) - window_size) // stride_size + 1, window_size)
        features = as_strided(df.features, output_shape, (8 * stride_size, 8))
        l = as_strided(df.attack, output_shape, (1 * stride_size, 1))

        dft = pd.DataFrame(
            {
                "features": pd.Series(features.tolist()),
                "label": pd.Series(l.tolist()),
            },
            index=range(len(l)),
        )
        dft["label"] = dft["label"].apply(lambda x: (attack_id + 1) if any(x) else 0)

        print("Aggregating data: Done")
        print("#Normal: ", dft[dft["label"] == 0].shape[0])
        print("#Attack: ", dft[dft["label"] != 0].shape[0])

        data_3d = np.array(dft["features"].tolist())   # (num_windows, window_size, 93) - đã nhị phân
        labels_cur = dft["label"].values

        print("data_3d.len:", len(data_3d), "labels_cur.len:", len(labels_cur))

        data_3d, labels_cur = sample_10pct(data_3d, labels_cur, attack_id + 1, seed=42)

        print("after sampling 20%, data_3d.len:", len(data_3d), "labels_cur.len:", len(labels_cur))

        attack_idx = np.where(labels_cur != 0)[0]
        benign_idx = np.where(labels_cur == 0)[0]

        data_train_a, data_test_a, labels_train_a, labels_test_a = train_test_split(
            data_3d[attack_idx], labels_cur[attack_idx], test_size=0.2, random_state=42, shuffle=True
        )
        data_train_b, data_test_b, labels_train_b, labels_test_b = train_test_split(
            data_3d[benign_idx], labels_cur[benign_idx], test_size=0.2, random_state=42, shuffle=True
        )

        data_train = np.concatenate((data_train_a, data_train_b))
        data_test = np.concatenate((data_test_a, data_test_b))
        labels_train = np.concatenate((labels_train_a, labels_train_b))
        labels_test = np.concatenate((labels_test_a, labels_test_b))

        xtrain = np.concatenate((xtrain, data_train))
        ytrain = np.concatenate((ytrain, labels_train))
        xtest = np.concatenate((xtest, data_test))
        ytest = np.concatenate((ytest, labels_test))

        print("Adding data of attack {} to train/test sets: Done".format(attack))

    # Lưu trực tiếp - KHÔNG cần "_raw" + bước scale riêng, vì đã là nhị phân 0/1
    np.save(os.path.join(output_dir, f"xtrain-{window_size}-{stride_size}.npy"), xtrain.astype(np.float32))
    np.save(os.path.join(output_dir, f"ytrain-{window_size}-{stride_size}.npy"), ytrain)
    np.save(os.path.join(output_dir, f"xtest-{window_size}-{stride_size}.npy"), xtest.astype(np.float32))
    np.save(os.path.join(output_dir, f"ytest-{window_size}-{stride_size}.npy"), ytest)
    print("DONE PROCESS (binary representation, 93 features/frame)")

    return xtrain, xtest, ytrain, ytest


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root_folder", default="./data/can-ml/can-ml/2017-subaru-forester/090726/merged")
    parser.add_argument("--data_folder", default="./data/can-ml/can-ml/2017-subaru-forester/050826/preprocessed_binary/")
    parser.add_argument("--window_size", type=int, default=64)
    parser.add_argument("--stride_size", type=int, default=64)
    parser.add_argument("--rid", type=int, default=1)

    args = parser.parse_args()
    print("Arguments:", args)

    attack_types = ["combined", "DoS", "fuzzing", "gear", "interval", "rpm", "speed", "standstill", "systematic"]
    if args.stride_size is None:
        args.stride_size = args.window_size

    outdir = args.data_folder + "rid{}".format(args.rid) + "/TFRecord_w{}_s{}".format(args.window_size, args.stride_size)

    run_pipeline(
        root_dir=args.root_folder,
        output_dir=outdir,
        window_size=args.window_size,
        stride_size=args.stride_size,
        attacks=attack_types,
    )
