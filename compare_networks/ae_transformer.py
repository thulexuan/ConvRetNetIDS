"""
ae_transformer.py
====================

Model implementation faithfully following the REAL reference training
script (`new_trainer.py`) for the AE-Transformer paper, rather than the
paper's own (looser) prose description. Key facts confirmed by reading
that reference code, which resolve ambiguities left open by the paper
itself:

  - d_model actually fed into the Transformer is 10 (POST-autoencoder),
    not 12 (the paper's Table 2 "Embedding input dimension (d)=12" is
    the AE's INPUT size, not the Transformer's d_model).
  - The Autoencoder1D used is ENCODER-ONLY (Linear(in,32)->ReLU->
    Linear(32,16)->ReLU->Linear(16,out), plus one extra ReLU on the
    final output) -- there is NO decoder, NO reconstruction loss
    anywhere in the reference training loop.
  - Timestamp information is added as a PRECOMPUTED tensor
    ("time_position"). Confirmed by directly reading the reference's
    `DatasetPreprocess` (dataPreprocess.py): the mechanism is a
    PRECOMPUTED SINUSOIDAL LOOKUP TABLE (`self.pe`, shape
    (max_time_position, embed)), indexed by a DISCRETIZED TIMESTAMP
    VALUE via `torch.index_select` -- NOT "position `p` + smoothed
    timestamp `tse_p`" as the paper's Eq. 12-13 prose describes. The
    timestamp-derived index REPLACES the positional index entirely
    (functions like an nn.Embedding lookup with fixed, not learned,
    rows). The discretization formula itself also differs from the
    paper's own Eq. 9 in three ways (no special-casing position 0, an
    extra inner round(), and the final value is rounded to an integer
    for use as a table index) -- see `discretize_timestamp()` below
    for the exact reproduction.
  - Pooling before the final classifier is SUM over the sequence
    (`torch.sum(out, dim=1)`), not mean.
  - Uses PyTorch's built-in `nn.TransformerEncoderLayer` /
    `nn.TransformerEncoder` directly, rather than a hand-rolled
    multi-head attention implementation.

This file implements ONLY the `mode="cb"` path (the single autoencoder
shared between CAN-ID and payload, concatenated BEFORE the AE) --
the ONLY branch that is actually functional in the reference
`new_trainer.py` (its `mode="ae"` / separate-autoencoders branch is
broken there: `Config` never sets `dout_payload`/`dout_header`, and
even if it did, `forward()`'s "ae" branch never calls
`self.header_ae`/`self.payload_ae` at all). No alternative mode is
provided here, per request -- this is a faithful, single-path
reproduction of the paper + reference code's actual working design.

Deviations from the reference code (clearly flagged, not silent):
  - The reference applies the shared per-frame autoencoder via an
    explicit Python for-loop over each window position
    (`for i in range(pad_size): ae(x[:,i,:])`). Since `nn.Linear`
    natively broadcasts over any number of leading dimensions, this
    loop is mathematically equivalent to (and much faster than) a
    single vectorized call `ae(x)` on the whole (batch, window, 12)
    tensor at once. This file uses the vectorized form.
  - This file uses `batch_first=True` in `nn.TransformerEncoderLayer`
    (available in modern PyTorch), avoiding the manual
    `.permute(1,0,2)` / `.permute(2,0,1)` juggling the reference code
    needs for the older (seq_len, batch, features) convention. The
    computation is identical; only the tensor layout bookkeeping differs.
  - PyTorch's `src_key_padding_mask` convention is `True` = PAD/IGNORE
    this position (opposite of the "1 = valid" convention used
    elsewhere in this project's other model files) -- this file follows
    PyTorch's own convention since it uses the built-in module directly.

Input:
    mes: (batch, window_size, 12)   -- ALREADY concatenated
         Concat(CAN_ID_nibbles, payload_bytes); concatenation happens
         BEFORE calling the model (matching this project's own
         preprocessing pipeline output, `xtrain_mes.npy` etc.), not
         inside forward() -- unlike an earlier version of this file
         which took `header`/`payload` separately and concatenated
         them internally.
    time_position: (batch, window_size, d_model) OR None
                                          -- precomputed additive time
                                             encoding (see
                                             compute_timestamp_encoding()
                                             below); ignored if tse=False
    padding_mask: (batch, window_size) bool, True = PAD position, or None

Output:
    logits: (batch, num_classes)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Autoencoder1D -- ENCODER ONLY, matching the reference exactly
# --------------------------------------------------------------------------- #
class Autoencoder1D(nn.Module):
    """
    Encoder-only dimension reducer, matching the reference's
    `Autoencoder1D` layer-for-layer: Linear->ReLU->Linear->ReLU->Linear,
    followed by one MORE ReLU applied to the final output in forward()
    (so the output is always non-negative) -- this extra ReLU is
    reproduced here exactly as found in the reference, not "corrected" away.

    Applied VECTORIZED over the whole (batch, window_size, input_dim)
    tensor at once (nn.Linear broadcasts over leading dims natively),
    instead of the reference's explicit per-position Python loop --
    mathematically identical, much faster.
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        return F.relu(x)  # extra ReLU, matching the reference exactly


# --------------------------------------------------------------------------- #
# Timestamp encoding: (a) EXACT reproduction of DatasetPreprocess's
# discretization + lookup-table mechanism (confirmed by reading that
# file directly -- see corrections below vs. the paper's own Eq. 9/12-13),
# and (b) the thin module that just ADDS the looked-up vector to x,
# matching the reference's `Time_Positional_Encoding` class exactly.
# --------------------------------------------------------------------------- #
def discretize_timestamp(
    raw_timestamps: torch.Tensor, gran: float = 1e-7, log_base: float = 2.0
) -> torch.Tensor:
    """
    EXACT match of DatasetPreprocess.__getitem__'s per-message timestamp
    transform (confirmed from the actual reference dataset code, NOT
    the paper's own Eq. 9 -- these differ in three ways, see below):

        value_i = round( log_{log_base}( round(ts_i / gran) + 1 ) )

    Differences from the paper's Eq. 9 (which this file's EARLIER
    version incorrectly implemented as `time_smooth()`):
      1. NO special-case "tse_0 = 0" for the first position -- EVERY
         position (including the first) goes through this same formula.
      2. An inner `round()` is applied to `ts_i / gran` BEFORE adding 1
         and taking the log (the paper's Eq. 9 has no such inner round).
      3. The FINAL log result is also rounded to the nearest integer
         (since it becomes a row index into a lookup table via
         `index_select`, which requires integer indices) -- this is a
         genuinely discrete/quantized value, not a continuous float.

    Args:
        raw_timestamps: (batch, window_size), RAW re-based timestamps
        gran:           smoothing hyperparameter (reference's own name
                        for the paper's "alpha"; new_trainer.py's
                        Config uses gran=1e-7, with a comment noting
                        the paper-style default would be 1e-6)
        log_base:       log base (reference's Config: log_e=2, matching
                        the paper's log2)
    Returns:
        (batch, window_size) LongTensor, each value in
        [0, max_time_position) -- must stay within that bound or the
        downstream table lookup will index out of range (see
        TimestampPositionTable below).
    """
    scaled = torch.round(raw_timestamps / gran)
    logged = torch.log(scaled + 1.0) / math.log(log_base)
    return torch.round(logged).long()


def pad_timestamp_indices(discretized: torch.Tensor, target_len: int) -> torch.Tensor:
    """
    Matches DatasetPreprocess's padding behavior EXACTLY: pads by
    REPEATING the last valid value (not zero-padding), per:

        for j in range(pad_size - len_timestamp):
            timestamp = np.append(timestamp, timestamp[len_timestamp - 1])

    Args:
        discretized: (batch, current_len) LongTensor
        target_len:  desired final length (>= current_len)
    Returns:
        (batch, target_len)
    """
    current_len = discretized.shape[1]
    if current_len >= target_len:
        return discretized[:, :target_len]
    last_col = discretized[:, -1:]
    pad = last_col.expand(-1, target_len - current_len)
    return torch.cat([discretized, pad], dim=1)


class TimestampPositionTable(nn.Module):
    """
    EXACT reproduction of DatasetPreprocess's `self.pe` + `get_time()`
    mechanism: a precomputed sinusoidal table, indexed by the
    DISCRETIZED TIMESTAMP value itself (via `discretize_timestamp()`
    above) -- NOT by the sequential window position `p`. This is a
    real, confirmed discrepancy from the paper's own Eq. 12-13 (which
    describes ADDING position `p` and smoothed-time `tse_p` together
    inside the sin/cos formula): the actual working reference code
    does not add them at all -- the timestamp-derived index REPLACES
    the positional index entirely, functioning like an embedding-table
    lookup (conceptually similar to `nn.Embedding`, but with fixed
    sinusoidal rather than learned rows).

    Args:
        embed:             encoding dimension (must match the AE's output_dim)
        max_time_position: table size (reference default: 10000) --
                           `discretize_timestamp()`'s output MUST stay
                           within [0, max_time_position) or `lookup()`
                           will index out of range; choose `gran` with
                           this bound in mind for your data's timescale.
    """

    def __init__(self, embed: int, max_time_position: int = 10000):
        super().__init__()
        self.embed = embed
        self.max_time_position = max_time_position

        pos = torch.arange(max_time_position, dtype=torch.float32).unsqueeze(1)  # (max_time_position, 1)
        i = torch.arange(embed, dtype=torch.float32).unsqueeze(0)  # (1, embed)
        angles = pos / (10000.0 ** ((i // 2) * 2.0 / embed))  # (max_time_position, embed)

        pe = angles.clone()
        pe[:, 0::2] = torch.sin(angles[:, 0::2])
        pe[:, 1::2] = torch.cos(angles[:, 1::2])
        self.register_buffer("pe", pe, persistent=False)  # (max_time_position, embed)

    def lookup(self, time_position_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            time_position_indices: (batch, window_size) LongTensor,
                                    each value in [0, max_time_position)
        Returns:
            (batch, window_size, embed)
        """
        if time_position_indices.max() >= self.max_time_position or time_position_indices.min() < 0:
            raise ValueError(
                f"discretized timestamp index out of range "
                f"[0, {self.max_time_position}): got min="
                f"{time_position_indices.min().item()}, max={time_position_indices.max().item()}. "
                f"Adjust `gran` (or `max_time_position`) to fit your data's timescale."
            )
        return self.pe[time_position_indices]  # fancy indexing == index_select here


def compute_timestamp_encoding(
    raw_timestamps: torch.Tensor,
    d_model: int,
    gran: float = 1e-7,
    log_base: float = 2.0,
    max_time_position: int = 10000,
) -> torch.Tensor:
    """
    Convenience wrapper: raw re-based timestamps -> discretized indices
    -> looked-up sinusoidal vectors, in one call. Equivalent to what
    DatasetPreprocess does across `discretize_timestamp` + padding +
    `get_time()`, minus the padding step (pad your `raw_timestamps`
    or the resulting encoding yourself with `pad_timestamp_indices`
    if your batch has variable-length windows).

    Args:
        raw_timestamps: (batch, window_size) RAW re-based timestamps
        d_model:        encoding dimension (must match AE output_dim)
        gran:           Time-Smooth hyperparameter
        log_base:       log base (reference: 2)
        max_time_position: lookup table size (reference: 10000)
    Returns:
        (batch, window_size, d_model)
    """
    indices = discretize_timestamp(raw_timestamps, gran=gran, log_base=log_base)
    table = TimestampPositionTable(embed=d_model, max_time_position=max_time_position)
    table = table.to(raw_timestamps.device)
    return table.lookup(indices)


class TimePositionalEncoding(nn.Module):
    """
    Matches the reference's `Time_Positional_Encoding` EXACTLY: a thin
    module that just ADDS a precomputed `time_position` tensor to x --
    no sin/cos math happens here (that lives in
    `compute_timestamp_encoding()` above, called separately).
    """

    def forward(self, x: torch.Tensor, time_position: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:             (batch, window_size, d_model)
            time_position: (batch, window_size, d_model)
        """
        return x + time_position


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding, used when tse=False."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, window_size, d_model), batch_first convention."""
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# --------------------------------------------------------------------------- #
# Full model
# --------------------------------------------------------------------------- #
class AETransformer(nn.Module):
    """
    Faithful reproduction of `TransformerPredictor`'s `mode="cb"` path
    -- the ONLY branch that is actually functional in the reference
    `new_trainer.py` (its `mode="ae"` / separate-autoencoders branch
    is broken there and is not reproduced here at all, per request).

    Architecture:
        mes (already Concat(header,payload), 12-dim) -> shared Autoencoder1D (12 -> dout_mess)
        -> + TimePositionalEncoding (or standard PositionalEncoding)
        -> nn.TransformerEncoder (num_layers x, nhead heads)
        -> sum-pool over the sequence
        -> Linear classifier

    Args:
        mes_dim:      raw per-frame mes dimension (4 CAN-ID + 8 payload = 12,
                      already concatenated BEFORE being passed to forward())
        dout_mess:    shared AE output dim, i.e. the Transformer's
                      d_model (reference: 10)
        num_heads:    attention heads (reference: 10)
        num_layers:   Transformer encoder layers (reference: 6)
        num_classes:  number of output classes
        tse:          whether to use TimePositionalEncoding (True) or
                      standard sinusoidal PositionalEncoding (False)
        dropout:      dropout rate inside the Transformer encoder layers
    """

    def __init__(
        self,
        mes_dim: int = 12,
        dout_mess: int = 10,
        num_heads: int = 10,
        num_layers: int = 6,
        num_classes: int = 5,
        tse: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.tse = tse

        self.ae = Autoencoder1D(mes_dim, dout_mess)
        d_model = dout_mess

        assert d_model % num_heads == 0, (
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        )
        self.d_model = d_model

        if tse:
            self.position_embedding = TimePositionalEncoding()
        else:
            self.position_embedding = PositionalEncoding(d_model, dropout=0.0)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dropout=dropout, batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(d_model, num_classes)

    def forward(
        self,
        mes: torch.Tensor,
        time_position: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            mes:           (batch, window_size, 12) -- ALREADY concatenated
                           Concat(CAN_ID_nibbles, payload_bytes), matching
                           this project's preprocessing output (`mes` array
                           from preprocess_can_ml_csv / xtrain_mes.npy)
                           directly -- concat happens BEFORE the model now,
                           not inside it.
            time_position: (batch, window_size, d_model), required if tse=True
                           (see compute_timestamp_encoding()); ignored if tse=False
            padding_mask:  (batch, window_size) bool, True = PAD position
                           (PyTorch's own src_key_padding_mask convention --
                           the OPPOSITE of "1=valid" used elsewhere in this
                           project's other model files, since this uses
                           nn.TransformerEncoder directly)
        Returns:
            logits: (batch, num_classes)
        """
        x = self.ae(mes)  # vectorized -- (batch, window_size, dout_mess)

        if self.tse:
            assert time_position is not None, "time_position is required when tse=True"
            out = self.position_embedding(x, time_position)
        else:
            out = self.position_embedding(x)

        out = self.transformer_encoder(out, src_key_padding_mask=padding_mask)

        out = torch.sum(out, dim=1)  # SUM pooling over the sequence, matching the reference
        logits = self.fc(out)
        return logits


# --------------------------------------------------------------------------- #
# Quick self-test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    torch.manual_seed(0)

    # --- verify discretize_timestamp matches dataPreprocess.py's exact formula ---
    print("=== discretize_timestamp (exact match check vs dataPreprocess.py) ===")
    import math as pymath
    ts_sample = torch.tensor([0.0, 0.0012, 0.0034, 0.01])
    gran, log_base = 1e-7, 2.0
    result = discretize_timestamp(ts_sample, gran=gran, log_base=log_base)
    print(f"raw timestamps: {ts_sample.tolist()}")
    print(f"discretized:    {result.tolist()}")
    # manual reference computation, mirroring the numpy/math version line-for-line
    expected = []
    for t in ts_sample.tolist():
        value = round(pymath.log(round(t / gran) + 1, log_base))
        expected.append(value)
    print(f"expected (manual, matching dataPreprocess.py line-for-line): {expected}")
    assert result.tolist() == expected
    print("PASSED: discretize_timestamp matches dataPreprocess.py exactly.\n")

    batch_size, window_size = 16, 29
    num_classes = 5

    mes = torch.rand(batch_size, window_size, 12) * 15  # rough nibble/byte-like range, ALREADY concatenated
    labels = torch.randint(0, num_classes, (batch_size,))

    print("=== AETransformer (faithful reproduction of the reference's mode='cb' path) ===")
    model = AETransformer(
        dout_mess=10, num_heads=10, num_layers=6,
        num_classes=num_classes, tse=True, dropout=0.1,
    )

    raw_timestamps = torch.cumsum(torch.rand(batch_size, window_size) * 0.001, dim=1)
    time_position = compute_timestamp_encoding(
        raw_timestamps, d_model=model.d_model, gran=1e-7, log_base=2.0, max_time_position=10000,
    )

    logits = model(mes, time_position=time_position)
    print(f"mes shape:     {tuple(mes.shape)}")
    print(f"time_position shape: {tuple(time_position.shape)}")
    print(f"logits shape:  {tuple(logits.shape)}")
    assert logits.shape == (batch_size, num_classes)

    loss = F.cross_entropy(logits, labels)
    loss.backward()
    ae_grad = model.ae.encoder[0].weight.grad
    grad_ok = ae_grad is not None and torch.isfinite(ae_grad).all() and ae_grad.abs().sum() > 0
    print(f"\nCrossEntropyLoss: {loss.item():.4f}")
    print(f"Shared AE gradient finite & nonzero: {bool(grad_ok)}")

    # padding mask sanity check -- PyTorch convention: True = PAD
    padding_mask = torch.zeros(batch_size, window_size, dtype=torch.bool)
    padding_mask[:, 25:] = True  # last 4 positions are padding
    logits_masked = model(mes, time_position=time_position, padding_mask=padding_mask)
    print(f"\n[+ padding_mask] logits shape: {tuple(logits_masked.shape)}")
    assert logits_masked.shape == (batch_size, num_classes)

    # tse=False path (standard sinusoidal PositionalEncoding instead)
    model_no_tse = AETransformer(
        dout_mess=10, num_heads=10, num_layers=2,
        num_classes=num_classes, tse=False, dropout=0.1,
    )
    logits_no_tse = model_no_tse(mes)  # tse=False -> no time_position needed
    print(f"\n[tse=False] logits shape: {tuple(logits_no_tse.shape)}")
    assert logits_no_tse.shape == (batch_size, num_classes)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {n_params:,}")
    print(f"d_model: {model.d_model}  (matches reference: 10, NOT 12)")
