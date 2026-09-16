"""
preprocess_can_ml_for_ae_transformer.py
==========================================

Preprocessing for the CAN-ML dataset, producing the input D_p expected
by the AE-Transformer paper (Le, Knowledge-Based Systems 2024):

  Le, T.-D. (2024). "Multi-classification in-vehicle intrusion
  detection system using packet- and sequence-level characteristics
  from time-embedded transformer with autoencoder." Knowledge-Based
  Systems, 299, 112091.

Scope boundary (IMPORTANT -- read before assuming this does everything)
--------------------------------------------------------------------
Per the paper's own architecture (Fig. 2) and Algorithm 1, the pipeline is:

    P (data preprocessing)  ->  g (AE + Time-Smooth, TRAINABLE)  ->  f (Transformer)

Algorithm 1 explicitly lists its INPUT as D_p = {X_p^j, Y^j}: raw
per-message vectors (CAN ID + payload, concatenated, un-reduced) and
raw (re-based) timestamps -- NOT yet passed through the AE or the
Time-Smooth log formula. The AE (dimension reduction 12->10) and
Time-Smooth mechanism are component `g`, which has LEARNABLE
parameters (the AE) and therefore belongs to the MODEL, not to data
preprocessing -- exactly analogous to how, in this project's own
pipeline, `PayloadLocalEncoderPerFrame` is the model's first layer,
not something `preprocessing.py` computes ahead of time.

This script therefore stops at D_p:
    - mes:       raw Concat(CAN_ID_nibbles, payload_bytes), NOT reduced by AE
    - timestamp: raw (re-based) timestamp, NOT yet passed through the
                 Time-Smooth log formula (Eq. 9 of the paper)
    - label:     window-level label

The (untrained) AE module and Time-Smooth function are still provided
below (matching Fig. 4 / Eq. 9 of the paper) so you can see exactly
how D_p plugs into the next stage -- but they are kept as SEPARATE,
clearly-labeled pieces, not run automatically as part of preprocessing.

Input CSV schema (as given)
------------------------------
    timestamp, arbitration_id, data_field, attack
    1672531200.182267,376,FA,0

    - timestamp:      float, Unix time
    - arbitration_id: HEX STRING (e.g. "376", NOT necessarily zero-padded)
    - data_field:     HEX STRING, payload bytes concatenated (may be
                      SHORTER than 16 hex chars / 8 bytes if DLC < 8,
                      e.g. "FA" = 1 byte only)
    - attack:         0 = normal, 1 = attack (binary flag WITHIN this
                      file -- this file is assumed to contain only ONE
                      attack type, consistent with every other
                      preprocessing script built in this project; the
                      caller supplies `attack_id` to relabel window-level
                      "any attack frame present" as that specific class)

Output (D_p, ready for the AE-Transformer model's first layer)
------------------------------------------------------------------
    mes:       (num_windows, window_size, 12)  float32
               -- 4 CAN-ID nibbles (0-15) + 8 payload bytes (0-255),
                  concatenated per frame, RAW (pre-AE)
    timestamp: (num_windows, window_size)       float32
               -- re-based (t = ts - ts.min()), RAW (pre-Time-Smooth)
    label:     (num_windows,)                    int64
               -- 0 (normal) or attack_id (this file's attack type)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Per-field conversion (CAN ID -> 4 nibbles; payload -> 8 bytes)
# --------------------------------------------------------------------------- #
def hex_arbitration_id_to_nibbles(hex_str: str, n_digits: int = 4) -> list[int]:
    """
    arbitration_id (hex string, e.g. "376") -> decimal -> zero-padded to
    `n_digits` hex characters -> list of `n_digits` NIBBLES (each 0-15).

    This matches the paper's Eq. 1 (cid_p = {v_i}_{i=1}^4): the CAN ID
    is standard 11-bit (max 0x7FF = 3 hex digits), so 4 hex digits after
    zero-padding is exactly enough -- each list element is one hex
    digit's value (0-15), NOT a full byte (0-255). ("4-byte list" in
    the paper's own wording is technically imprecise -- 4 hex DIGITS,
    not 4 bytes -- confirmed by inspecting a real preprocessing
    implementation for this same family of datasets.)

    We zero-pad via the DECIMAL value re-formatted to hex (not the raw
    input string directly), so "376", "0376", and "0x376" all normalize
    to the same ["0","3","7","6"] regardless of how the CSV happened to
    format the hex string.
    """
    hex_str = str(hex_str).strip()
    dec_value = int(hex_str, 16)
    if dec_value < 0 or dec_value >= 16 ** n_digits:
        raise ValueError(
            f"arbitration_id {hex_str!r} (decimal {dec_value}) does not fit "
            f"in {n_digits} hex digits (max {16**n_digits - 1})."
        )
    padded = format(dec_value, f"0{n_digits}x")  # e.g. "0376"
    return [int(ch, 16) for ch in padded]


def hex_payload_to_bytes(hex_str: str, n_bytes: int = 8) -> list[int]:
    """
    data_field (hex string, e.g. "FA", possibly SHORTER than 16 hex
    chars if DLC < 8) -> zero-padded (trailing '00') to exactly
    `n_bytes` bytes -> list of `n_bytes` BYTE values (each 0-255).

    Per the paper (Section 4.3): "replaced all null values with '00'
    values in all messages' payloads" -- missing/short payload bytes
    are padded with 0, not dropped.
    """
    hex_str = str(hex_str).strip()
    if len(hex_str) % 2 != 0:
        raise ValueError(
            f"data_field {hex_str!r} has an odd number of hex characters "
            f"-- cannot split cleanly into whole bytes."
        )
    if len(hex_str) > n_bytes * 2:
        raise ValueError(
            f"data_field {hex_str!r} is longer than {n_bytes} bytes "
            f"({len(hex_str)} hex chars > {n_bytes*2})."
        )
    padded = hex_str.ljust(n_bytes * 2, "0")  # pad with TRAILING '00's
    return [int(padded[i:i + 2], 16) for i in range(0, n_bytes * 2, 2)]


# --------------------------------------------------------------------------- #
# Component P: Data Preprocessing -- produces D_p (this file's main output)
# --------------------------------------------------------------------------- #
def _load_csv(csv_path: str) -> pd.DataFrame:
    """Bước 1: đọc CSV, ép dtype=str cho 2 cột hex (tránh bug pandas tự
    suy luận "376" thành số decimal thay vì giữ nguyên chuỗi hex)."""
    return pd.read_csv(
        csv_path,
        header=None,
        skiprows=1,
        names=["timestamp", "arbitration_id", "data_field", "attack"],
        dtype={"arbitration_id": str, "data_field": str},
    )


def _sort_by_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """Bước 2: sắp xếp lại theo timestamp tăng dần."""
    return df.sort_values("timestamp", ascending=True).reset_index(drop=True)


def _rebase_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """Bước 3: trừ đi giá trị timestamp nhỏ nhất -> t = ts - ts.min()."""
    df = df.copy()
    df["timestamp"] = df["timestamp"].astype(float) - df["timestamp"].astype(float).min()
    return df


def _convert_can_id(df: pd.DataFrame) -> np.ndarray:
    """Bước 4: CAN ID (hex string) -> 4 nibble. Shape (N, 4)."""
    return np.stack(df["arbitration_id"].apply(hex_arbitration_id_to_nibbles).to_numpy())


def _convert_payload(df: pd.DataFrame) -> np.ndarray:
    """Bước 5: payload (hex string) -> 8 byte. Shape (N, 8)."""
    return np.stack(df["data_field"].apply(hex_payload_to_bytes).to_numpy())


def _build_mes(cid_nibbles: np.ndarray, payload_bytes: np.ndarray) -> np.ndarray:
    """Bước 6: Concat(CAN_ID_nibbles, payload_bytes) -> mes, shape (N, 12)."""
    return np.concatenate([cid_nibbles, payload_bytes], axis=1).astype(np.float32)


def _extract_flat_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Bước 7: timestamp và attack -> mảng NumPy phẳng."""
    timestamps = df["timestamp"].to_numpy(dtype=np.float32)  # (N,)
    attack_flags = df["attack"].to_numpy().astype(bool)  # (N,)
    return timestamps, attack_flags


def _compute_num_windows(n: int, window_size: int, stride: int) -> int:
    """Bước 8: tính số lượng window có thể cắt được, báo lỗi nếu không đủ dữ liệu."""
    num_windows = (n - window_size) // stride + 1
    if num_windows <= 0:
        raise ValueError(f"Not enough rows ({n}) for even one window of size {window_size}.")
    return num_windows


def _build_windows(
    mes: np.ndarray,
    timestamps: np.ndarray,
    attack_flags: np.ndarray,
    window_size: int,
    stride: int,
    attack_id: int,
    num_windows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bước 9: cắt sliding window + gán nhãn window-level (any(attack)=True
    trong window -> attack_id, ngược lại -> 0/Normal).
    """
    mes_windows = np.empty((num_windows, window_size, 12), dtype=np.float32)
    ts_windows = np.empty((num_windows, window_size), dtype=np.float32)
    labels = np.empty((num_windows,), dtype=np.int64)

    for w in range(num_windows):
        start = w * stride
        end = start + window_size
        mes_windows[w] = mes[start:end]
        ts_windows[w] = timestamps[start:end]
        labels[w] = attack_id if attack_flags[start:end].any() else 0

    return mes_windows, ts_windows, labels


def preprocess_can_ml_csv(
    csv_path: str,
    attack_id: int,
    window_size: int = 29,
    stride: int | None = None,
) -> dict:
    """
    Read one CAN-ML CSV (schema: timestamp, arbitration_id, data_field,
    attack) and produce D_p: windowed raw `mes` (12-dim per frame),
    raw re-based timestamps, and window-level labels.

    Composed of 9 sequential steps (each its own named helper function
    above, for readability/testability):
        1. _load_csv              -- đọc CSV, ép dtype=str cho cột hex
        2. _sort_by_timestamp     -- sắp xếp theo thời gian
        3. _rebase_timestamp      -- trừ đi giá trị min
        4. _convert_can_id        -- CAN ID hex -> 4 nibble
        5. _convert_payload       -- payload hex -> 8 byte
        6. _build_mes             -- Concat(CAN_ID, payload) -> 12 chiều
        7. _extract_flat_arrays   -- timestamp/attack -> mảng phẳng
        8. _compute_num_windows   -- tính số window
        9. _build_windows         -- cắt window + gán nhãn

    Args:
        csv_path:    path to the CSV file (one attack type per file,
                     matching every other preprocessing script in this
                     project)
        attack_id:   integer label to assign to windows containing ANY
                     attack frame (0 is reserved for "Normal")
        window_size: number of messages per sequence (n in the paper;
                     Table 2: 29 for Car-Hacking-style config, 15 for
                     ROAD-style -- pass the value matching your dataset)
        stride:      sliding-window stride; defaults to `window_size`
                     (non-overlapping windows) if not given

    Returns:
        {"mes": (num_windows, window_size, 12) float32,
         "timestamp": (num_windows, window_size) float32,
         "label": (num_windows,) int64}
    """
    if stride is None:
        stride = window_size

    df = _load_csv(csv_path)                       # Bước 1
    df = _sort_by_timestamp(df)                     # Bước 2
    df = _rebase_timestamp(df)                       # Bước 3

    cid_nibbles = _convert_can_id(df)                 # Bước 4
    payload_bytes = _convert_payload(df)               # Bước 5
    mes = _build_mes(cid_nibbles, payload_bytes)         # Bước 6

    timestamps, attack_flags = _extract_flat_arrays(df)   # Bước 7

    n = len(df)
    num_windows = _compute_num_windows(n, window_size, stride)  # Bước 8

    mes_windows, ts_windows, labels = _build_windows(          # Bước 9
        mes, timestamps, attack_flags, window_size, stride, attack_id, num_windows,
    )

    print(f"[{csv_path}] rows={n}, windows={num_windows}, "
          f"#Normal={int((labels == 0).sum())}, #Attack={int((labels != 0).sum())}")

    return {"mes": mes_windows, "timestamp": ts_windows, "label": labels}


# --------------------------------------------------------------------------- #
# Component g (Section 4.4): AE + Time-Smooth -- LEARNABLE, part of the
# MODEL, provided here for reference only. NOT run by preprocess_can_ml_csv().
# --------------------------------------------------------------------------- #
class SingleMessageAutoencoder:
    """
    Reference-only stub matching Fig. 4 of the paper: a small
    autoencoder reducing `mes` (12-dim) to `mes^e` (10-dim). This is a
    TRAINABLE model component (part of `g`), not a preprocessing step
    -- instantiate and train it as the first layer of your AE-Transformer
    model, feeding it the `mes` array this preprocessing script produces.

    (Left as a lightweight placeholder; swap in your actual framework's
    autoencoder -- e.g. a PyTorch nn.Module with a matching
    12 -> hidden -> 10 -> hidden -> 12 encoder/decoder -- when you build
    the full model.)
    """

    def __init__(self, input_dim: int = 12, latent_dim: int = 10):
        self.input_dim = input_dim
        self.latent_dim = latent_dim


def time_smooth(timestamps: np.ndarray, alpha: float = 1e-6) -> np.ndarray:
    """
    Time-Smooth (TS) mechanism, Eq. 9 of the paper -- provided for
    reference (this is conceptually part of the learnable component
    `g`'s pipeline, though the formula itself has no trainable
    parameters, only the fixed hyperparameter alpha):

        tse_0 = 0
        tse_i = | log2(ts_i / alpha + 1) |,  i > 0

    Args:
        timestamps: (..., window_size) RAW re-based timestamps (as
                    produced by preprocess_can_ml_csv)
        alpha:      smoothing hyperparameter (paper: 1e-6 for
                    Car-Hacking-style, 1e-8 for ROAD-style)
    Returns:
        same shape as `timestamps`
    """
    tse = np.abs(np.log2(timestamps / alpha + 1))
    tse[..., 0] = 0.0  # tse_0 = 0 by definition, overriding index 0 of each window
    return tse.astype(np.float32)


# --------------------------------------------------------------------------- #
# Quick self-test, using the exact example row format given
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import tempfile
    import os

    # --- unit tests for the two hex-conversion functions ---
    print("=== hex_arbitration_id_to_nibbles ===")
    nibbles = hex_arbitration_id_to_nibbles("376")
    print(f'"376" (hex) -> nibbles {nibbles}')
    assert nibbles == [0, 3, 7, 6]

    print("\n=== hex_payload_to_bytes ===")
    bytes_out = hex_payload_to_bytes("FA")
    print(f'"FA" (1 byte, padded) -> bytes {bytes_out}')
    assert bytes_out == [250, 0, 0, 0, 0, 0, 0, 0]

    bytes_out_full = hex_payload_to_bytes("0102030405060708")
    print(f'"0102030405060708" (already 8 bytes) -> bytes {bytes_out_full}')
    assert bytes_out_full == [1, 2, 3, 4, 5, 6, 7, 8]

    # --- end-to-end test with a synthetic CSV, matching the given example row ---
    print("\n=== End-to-end test with synthetic CSV (given row format) ===")
    rows = []
    base_ts = 1672531200.182267
    for i in range(100):
        ts = base_ts + i * 0.001
        arb_id = "376" if i % 3 != 0 else "1A2"
        data_field = "FA" if i % 5 != 0 else "0102030405060708"
        attack = 1 if 40 <= i < 45 else 0  # simulate a short attack burst
        rows.append(f"{ts},{arb_id},{data_field},{attack}")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("\n".join(rows))
        tmp_path = f.name

    result = preprocess_can_ml_csv(tmp_path, attack_id=3, window_size=29, stride=29)
    os.unlink(tmp_path)

    print(f"\nmes shape:       {result['mes'].shape}")
    print(f"timestamp shape: {result['timestamp'].shape}")
    print(f"label shape:     {result['label'].shape}")
    print(f"labels: {result['label']}")

    assert result["mes"].shape[1:] == (29, 12)
    assert result["timestamp"].shape[1:] == (29,)
    assert set(np.unique(result["label"])).issubset({0, 3})
    assert (result["label"] == 3).any(), "expected at least one attack window"
    print("\nPASSED: shapes and labels correct.")

    # --- reference-only Time-Smooth demonstration (NOT part of preprocessing output) ---
    print("\n=== Time-Smooth (component g, reference only -- NOT run above) ===")
    tse_demo = time_smooth(result["timestamp"][:1], alpha=1e-6)
    print(f"Raw timestamp (first window, first 5): {result['timestamp'][0][:5]}")
    print(f"Time-Smoothed (first window, first 5): {tse_demo[0][:5]}")
