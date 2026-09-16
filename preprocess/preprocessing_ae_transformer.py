"""
run_pipeline_can_ml_ae_transformer.py
========================================

Full pipeline combining multiple CAN-ML attack CSVs into the D_p input
expected by the AE-Transformer model, mirroring the structure of this
project's earlier `preprocessing.py` `run_pipeline()`:

    for each attack file:
        preprocess into windowed (mes, timestamp, label)   [preprocess_can_ml_csv]
        split attack/benign windows separately, 10%/10%/80% train/val/test
        accumulate across all attack files

    save xtrain/xtest .npy files (mes + timestamp arrays, separately,
    since the AE-Transformer model consumes them as two related-but-
    distinct inputs -- see preprocess_can_ml_for_ae_transformer.py's
    docstring on why AE/Time-Smooth stay OUT of preprocessing)
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from preprocess_can_ml_for_ae_transformer import preprocess_can_ml_csv


def _subsample_then_split(
    indices: np.ndarray,
    sample_frac: float = 0.20,
    train_frac: float = 0.80,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Bước A: lấy mẫu ngẫu nhiên `sample_frac` (mặc định 20%) trong số
    `indices` đưa vào.
    Bước B: chia phần đã lấy mẫu đó thành train/test theo `train_frac`
    (mặc định 80% train / 20% test).

    Dùng max(1, round(...)) để tránh lỗi làm tròn về 0 khi nhóm quá nhỏ
    (đã gặp và sửa ở _split_indices_3way trước đây).
    """
    n = len(indices)
    if n == 0:
        return np.array([], dtype=indices.dtype), np.array([], dtype=indices.dtype)

    rng = np.random.default_rng(random_state)
    shuffled = indices.copy()
    rng.shuffle(shuffled)

    # Bước A: lấy mẫu 20%
    n_sample = max(1, int(n * sample_frac))
    sampled = shuffled[:n_sample]

    if len(sampled) < 2:
        print(f"  WARNING: chỉ có {len(sampled)} mẫu sau khi lấy 20% -- "
              f"đưa hết vào train, không có test cho nhóm này.")
        return sampled, np.array([], dtype=indices.dtype)

    # Bước B: chia 80/20 trên phần đã lấy mẫu
    # n_train = max(1, round(len(sampled) * train_frac))
    # n_train = min(n_train, len(sampled) - 1)  # đảm bảo còn ít nhất 1 mẫu cho test

    # train_idx = sampled[:n_train]
    # test_idx = sampled[n_train:]
    # return train_idx, test_idx
    train_idx, test_idx = train_test_split(
        sampled,
        train_size=train_frac,
        random_state=random_state,
        shuffle=True,
    )

    return np.asarray(train_idx), np.asarray(test_idx)


def run_pipeline(
    root_dir: str,
    output_dir: str,
    attacks: list[str],
    window_size: int = 32,
    stride: int = 32,
) -> dict:
    """
    Args:
        root_dir:    folder containing one CSV per attack type, named
                     "{attack}.csv" (schema: timestamp, arbitration_id,
                     data_field, attack -- as produced/consumed by
                     preprocess_can_ml_for_ae_transformer.py)
        output_dir:  where to save the resulting .npy files
        attacks:     list of attack names, e.g. ["DoS", "Fuzzy", "RPM"];
                     attack_id = index+1 is assigned per file, matching
                     every other pipeline in this project (0 reserved
                     for Normal)
        window_size: messages per sequence (Table 2 of the AE-Transformer
                     paper: 29 for Car-Hacking-style, 15 for ROAD-style)
        stride:      sliding-window stride; defaults to window_size
                     (non-overlapping windows)

    Saves:
        xtrain_mes.npy, xtrain_ts.npy, ytrain.npy
        xtest_mes.npy,  xtest_ts.npy,  ytest.npy

    Returns:
        dict with the same 6 arrays, for immediate in-process use.
    """
    root = Path(root_dir)
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    xtrain_mes, xtrain_ts, ytrain = [], [], []
    xtest_mes, xtest_ts, ytest = [], [], []

    for attack_id, attack in enumerate(attacks):
        print(f"Attack: {attack} ===============")
        finput = str(root / f"{attack}.csv")

        result = preprocess_can_ml_csv(
            finput, attack_id=attack_id + 1, window_size=window_size, stride=stride,
        )
        mes, ts, labels = result["mes"], result["timestamp"], result["label"]

        attack_idx = np.where(labels != 0)[0]
        benign_idx = np.where(labels == 0)[0]

        train_a, test_a = _subsample_then_split(attack_idx)
        train_b, test_b = _subsample_then_split(benign_idx)

        train_idx = np.concatenate([train_a, train_b]).astype(int)
        test_idx = np.concatenate([test_a, test_b]).astype(int)

        xtrain_mes.append(mes[train_idx]); xtrain_ts.append(ts[train_idx]); ytrain.append(labels[train_idx])
        xtest_mes.append(mes[test_idx]);   xtest_ts.append(ts[test_idx]);   ytest.append(labels[test_idx])

        print(f"  -> train={len(train_idx)}, test={len(test_idx)}")

    result = {
        "xtrain_mes": np.concatenate(xtrain_mes), "xtrain_ts": np.concatenate(xtrain_ts),
        "ytrain": np.concatenate(ytrain),
        "xtest_mes": np.concatenate(xtest_mes), "xtest_ts": np.concatenate(xtest_ts),
        "ytest": np.concatenate(ytest),
    }

    for name, arr in result.items():
        np.save(outdir / f"{name}.npy", arr)

    print("\nDONE PROCESS")
    print(f"xtrain_mes: {result['xtrain_mes'].shape}, "
          f"xtest_mes: {result['xtest_mes'].shape}")

    return result


# --------------------------------------------------------------------------- #
# CLI entry point -- pass real paths as arguments instead of using tempfile
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Combine multiple CAN-ML attack CSVs into D_p (mes + "
                     "timestamp + label) for the AE-Transformer model."
    )
    parser.add_argument("--root_dir", type=str, 
                        default="./data/can-ml/can-ml/2017-subaru-forester/160926/merged-canid-in-hex",
                         help="Folder containing one CSV per attack type, "
                              "named '{attack}.csv'")
    parser.add_argument("--output_dir", type=str, 
                        default="./data/can-ml/can-ml/2017-subaru-forester/160926/preprocessed-ae-transformer-win=32-3rd",
                         help="Where to save the resulting .npy files")
    
    parser.add_argument("--window_size", type=int, default=32,
                         help="Messages per sequence (Table 2 of the paper: "
                              "29 for Car-Hacking-style, 15 for ROAD-style)")
    parser.add_argument("--stride", type=int, default=32,
                         help="Sliding-window stride; defaults to --window_size "
                              "(non-overlapping windows) if not given")

    args = parser.parse_args()
    print(args)

    attack_types = ["combined", "DoS", "fuzzing", "gear", "interval", "rpm", 
                    "speed", "standstill", "systematic"]
    
    result = run_pipeline(
        root_dir=args.root_dir,
        output_dir=args.output_dir,
        attacks=attack_types,
        window_size=args.window_size,
        stride=args.stride,
    )

    print("\n=== Shape summary ===")
    for name, arr in result.items():
        print(f"  {name}: {arr.shape}")