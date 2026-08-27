from __future__ import annotations

import json
import logging
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
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
WINDOW = 128
IMG_SIZE = 32
N_CH = 3

# Các hằng số ID_MIN/ID_MAX/BYTE_MIN/BYTE_MAX của log-min-max KHÔNG còn cần
# thiết nữa, vì bước preprocessing này CHỈ convert hex -> dec (KHÔNG normalize).
# MinMaxScaler thật sự (data-driven) được fit RIÊNG, SAU KHI đã có xtrain hoàn
# chỉnh, ở hàm `fit_transform_minmax` bên dưới - để tránh data leakage.


# ── CAN ID: CHỈ convert hex/str -> decimal, KHÔNG normalize ──────────────────
def process_can_id(arb_id) -> float:
    """
    arbitration_id (decimal int hoặc string) -> giá trị decimal thô (float32).
    KHÔNG áp dụng bất kỳ phép chuẩn hoá nào ở bước này.

    Ví dụ:
        140 -> 140.0
    """
    return float(int(arb_id))


# ── Data field: CHỈ convert hex -> decimal, KHÔNG normalize ──────────────────
def process_data_field(data_field) -> list:
    """
    data_field (hex string) -> list 8 giá trị decimal thô (float, chưa scale).

    Steps:
        1. Padding zfill(16) cho đủ 16 ký tự hex (= 8 bytes)
        2. Tách thành 8 cặp hex (mỗi cặp = 1 byte)
        3. Mỗi byte: hex -> decimal (KHÔNG normalize)

    Ví dụ:
        '0001240828282720'
        -> ['00','01','24','08','28','28','27','20']
        -> [0.0, 1.0, 36.0, 8.0, 40.0, 40.0, 39.0, 32.0]
    """
    hex_str = str(data_field).strip().upper().zfill(16)[:16]

    result = []
    for i in range(8):
        byte_hex = hex_str[i * 2 : i * 2 + 2]
        try:
            byte_dec = int(byte_hex, 16)
        except ValueError:
            byte_dec = 0
        result.append(float(byte_dec))

    return result  # list of 8 raw decimal floats


# ── Apply to DataFrame (RAW, chưa normalize) ─────────────────────────────────
def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nhận DataFrame có cột: timestamp, arbitration_id, data_field, attack
    Trả về DataFrame mới với các cột (GIÁ TRỊ THÔ, CHƯA NORMALIZE):

        timestamp        (giữ nguyên)
        can_id           float32  <- arbitration_id, decimal thô
        byte_0 .. byte_7 float32  <- 8 cột từ data_field, decimal thô
        attack           int      (giữ nguyên)

    Tổng: 1 + 1 + 8 + 1 = 11 cột

    LƯU Ý QUAN TRỌNG: Việc normalize (MinMaxScaler) được thực hiện ở BƯỚC
    RIÊNG, SAU KHI toàn bộ dữ liệu đã qua windowing + train/test split, để
    đảm bảo scaler chỉ fit trên đúng tập train cuối cùng (tránh data leakage).
    Xem hàm `fit_transform_minmax` ở phần dưới.
    """
    out = pd.DataFrame()

    out["timestamp"] = df["timestamp"].values

    # CAN ID: raw decimal, chưa normalize
    out["can_id"] = df["arbitration_id"].apply(process_can_id).astype("float32")

    # Payload: tách thành 8 cột riêng biệt, raw decimal, chưa normalize
    byte_cols = df["data_field"].apply(process_data_field)
    byte_df = pd.DataFrame(
        byte_cols.tolist(),
        columns=[f"byte_{i}" for i in range(8)],
        dtype="float32",
    )
    out = pd.concat([out, byte_df], axis=1)

    out["attack"] = df["attack"].values.astype("int8")

    return out


def sample_10pct(
    data_3d: np.ndarray,
    labels: np.ndarray,
    attack_id: int,
    seed: int = 42,
):
    """
    Từ tập windows của 1 file:
        - Tách thành 2 nhóm: attack windows và normal windows
        - Lấy ngẫu nhiên 20% mỗi nhóm (ít nhất 1 sample)
        - Ghép lại (giữ thứ tự thời gian) và trả về
    """
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
        # convert hex -> decimal, KHÔNG normalize ở bước này
        normalized = preprocess_df(df)
        print(normalized.head())
    except Exception as exc:
        log.error("Error reading %s", filename)
        print(f"Error occurred while reading {filename}: {exc}")
        raise

    return normalized


def build_feature_col(df: pd.DataFrame) -> list:
    """Lấy tên các cột feature (bỏ timestamp, attack, v.v.)."""
    return [c for c in df.columns if c not in ["timestamp", "attack"]]


# ── Pipeline: hex->dec + windowing + split, CHƯA normalize ──────────────────
def run_pipeline(root_dir, output_dir, window_size=29, stride_size=15, feature_size=9, attacks=None) -> None:
    """
    Toàn bộ pipeline dưới đây CHỈ convert hex->dec + windowing + train/test
    split. KHÔNG normalize (không MinMaxScaler) ở bước này - dữ liệu lưu ra
    (xtrain/xtest) là giá trị DECIMAL THÔ (CAN ID: 0..2047, byte: 0..255).

    Việc normalize (MinMaxScaler, fit chỉ trên train) được thực hiện ở bước
    RIÊNG SAU KHI load lại các file .npy này - xem `fit_transform_minmax`.
    """
    root = Path(root_dir)
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    log.info("Loading each .csv file of each attack (hex->dec only, chưa normalize)")

    xtrain = np.empty((0, window_size, feature_size))
    ytrain = np.empty((0))
    xtest = np.empty((0, window_size, feature_size))
    ytest = np.empty((0))

    for attack_id, attack in enumerate(attacks):
        print("Attack: {} ===============".format(attack))
        finput = "{}/{}.csv".format(root, attack)

        df = load_attack_folder(finput, attack_id + 1, window_size=window_size, stride_size=stride_size)

        feature_col = build_feature_col(df)
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

        data_3d = np.array(dft["features"].tolist())  # (num_windows, window_size, feature_size) - RAW
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

    # Lưu RAW (CHƯA normalize) - decimal thô: CAN ID 0..2047, byte 0..255
    np.save(os.path.join(output_dir, f"xtrain_raw-{window_size}-{stride_size}.npy"), xtrain)
    np.save(os.path.join(output_dir, f"ytrain-{window_size}-{stride_size}.npy"), ytrain)
    np.save(os.path.join(output_dir, f"xtest_raw-{window_size}-{stride_size}.npy"), xtest)
    np.save(os.path.join(output_dir, f"ytest-{window_size}-{stride_size}.npy"), ytest)
    print("DONE PROCESS (raw, chưa normalize)")

    return xtrain, xtest, ytrain, ytest


# ────────────────────────────────────────────────────────────────────────────
# BƯỚC 2 (RIÊNG, chạy SAU khi load lại các file .npy raw ở trên):
# Fit MinMaxScaler CHỈ trên xtrain, transform CẢ xtrain và xtest.
# ────────────────────────────────────────────────────────────────────────────
def fit_transform_minmax(
    xtrain_raw: np.ndarray,
    xtest_raw: np.ndarray,
    feature_names: list[str] | None = None,
    scaler_path: str | Path | None = None,
):
    """
    Áp dụng MinMaxScaler THƯỜNG (tuyến tính, data-driven) lên dữ liệu CAN đã
    windowing, đúng cách để KHÔNG bị data leakage:

        1. Fit scaler CHỈ trên xtrain_raw (không đụng đến xtest_raw).
        2. Dùng chính fit đó để transform CẢ xtrain_raw VÀ xtest_raw.

    xtrain_raw, xtest_raw: (N, window_size, feature_size) - giá trị decimal
        thô (CAN ID: 0..2047, mỗi byte: 0..255), lấy từ output của
        `run_pipeline` (đã lưu ở "xtrain_raw-*.npy" / "xtest_raw-*.npy").

    feature_names: tên các cột feature theo đúng thứ tự chiều cuối (mặc định
        ["can_id", "byte_0", ..., "byte_7"]) - chỉ dùng để log/kiểm tra.

    scaler_path: nếu truyền vào, lưu lại scaler đã fit (pickle) để dùng lại
        khi suy luận thực tế / streaming sau này (BẮT BUỘC dùng đúng scaler
        này, không được fit lại, để nhất quán với lúc train).

    Returns
    -------
    xtrain_scaled, xtest_scaled : cùng shape với input, giá trị trong [0, 1]
    scaler                       : sklearn MinMaxScaler đã fit (fit trên train)
    """
    assert xtrain_raw.shape[1:] == xtest_raw.shape[1:], (
        "xtrain_raw và xtest_raw phải cùng (window_size, feature_size)"
    )
    N_train, T, F = xtrain_raw.shape
    N_test = xtest_raw.shape[0]

    if feature_names is None:
        feature_names = ["can_id"] + [f"byte_{i}" for i in range(F - 1)]
    assert len(feature_names) == F, "feature_names phải khớp feature_size"

    # MinMaxScaler của sklearn scale THEO TỪNG CỘT (feature) độc lập, nên ta
    # gộp (N, T) thành 1 chiều để mỗi "hàng" là 1 timestep, mỗi "cột" là 1 field
    # -> đúng ngữ nghĩa: can_id được scale theo min/max của can_id (trên train),
    # byte_i được scale theo min/max riêng của byte_i (trên train).
    xtrain_flat = xtrain_raw.reshape(-1, F)   # (N_train*T, F)
    xtest_flat = xtest_raw.reshape(-1, F)      # (N_test*T, F)

    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    scaler.fit(xtrain_flat)  # ⚠️ CHỈ fit trên train, KHÔNG fit lại trên test

    xtrain_scaled = scaler.transform(xtrain_flat).reshape(N_train, T, F).astype("float32")
    xtest_scaled = scaler.transform(xtest_flat).reshape(N_test, T, F).astype("float32")

    log.info("Đã fit MinMaxScaler trên xtrain (%d mẫu), transform cả xtrain và xtest.", N_train)
    for i, name in enumerate(feature_names):
        log.info(
            "  %-8s: train min/max thô = [%.2f, %.2f]  -> scaler.data_min_/data_max_ = [%.2f, %.2f]",
            name, xtrain_flat[:, i].min(), xtrain_flat[:, i].max(),
            scaler.data_min_[i], scaler.data_max_[i],
        )

    if scaler_path is not None:
        scaler_path = Path(scaler_path)
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        log.info("Đã lưu scaler đã fit vào %s (dùng lại đúng scaler này khi suy luận/streaming).", scaler_path)

    return xtrain_scaled, xtest_scaled, scaler


def load_raw_and_normalize(data_dir: str | Path, window_size: int, stride_size: int, scaler_path=None):
    """
    Tiện ích: load lại các file .npy RAW đã lưu từ `run_pipeline`, rồi áp
    `fit_transform_minmax` (fit trên train, transform cả train/test).

    Đây chính là bước "load data lên rồi mới MinMaxScaler" mà bạn đã hỏi -
    hoàn toàn hợp lệ và là cách làm ĐÚNG để tránh data leakage với
    MinMaxScaler data-driven (khác hằng số cố định như log-min-max trước đó).
    """
    data_dir = Path(data_dir)
    xtrain_raw = np.load(data_dir / f"xtrain_raw-{window_size}-{stride_size}.npy")
    ytrain = np.load(data_dir / f"ytrain-{window_size}-{stride_size}.npy")
    xtest_raw = np.load(data_dir / f"xtest_raw-{window_size}-{stride_size}.npy")
    ytest = np.load(data_dir / f"ytest-{window_size}-{stride_size}.npy")

    xtrain_scaled, xtest_scaled, scaler = fit_transform_minmax(
        xtrain_raw, xtest_raw, scaler_path=scaler_path or (data_dir / f"minmax_scaler-{window_size}-{stride_size}.pkl")
    )

    np.save(data_dir / f"xtrain-{window_size}-{stride_size}.npy", xtrain_scaled)
    np.save(data_dir / f"xtest-{window_size}-{stride_size}.npy", xtest_scaled)

    return xtrain_scaled, xtest_scaled, ytrain, ytest, scaler


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root_folder", default="./data/can-ml/can-ml/2017-subaru-forester/090726/merged")
    parser.add_argument("--data_folder", default="./data/can-ml/can-ml/2017-subaru-forester/050826/preprocessed/")
    parser.add_argument("--window_size", type=int, default=64)
    parser.add_argument("--stride_size", type=int, default=64)
    parser.add_argument("--rid", type=int, default=1)
    parser.add_argument(
        "--skip_normalize", action="store_true",
        help="Nếu bật, chỉ chạy bước hex->dec + windowing (RAW), KHÔNG chạy MinMaxScaler ngay.",
    )

    args = parser.parse_args()
    print("Arguments:", args)

    attack_types = ["combined", "DoS", "fuzzing", "gear", "interval", "rpm", "speed", "standstill", "systematic"]
    if args.stride_size is None:
        args.stride_size = args.window_size

    outdir = args.data_folder + "rid{}".format(args.rid) + "/TFRecord_w{}_s{}".format(args.window_size, args.stride_size)

    # Bước 1: hex->dec + windowing + split, CHƯA normalize
    run_pipeline(
        root_dir=args.root_folder,
        output_dir=outdir,
        window_size=args.window_size,
        stride_size=args.stride_size,
        attacks=attack_types,
    )

    # # Bước 2: load lại raw .npy vừa lưu, fit MinMaxScaler CHỈ trên train,
    # # transform cả train/test, lưu bản đã scale + lưu scaler.
    # if not args.skip_normalize:
    #     load_raw_and_normalize(outdir, args.window_size, args.stride_size)
