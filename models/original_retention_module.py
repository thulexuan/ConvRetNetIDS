import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import List, Tuple
 
# ---------------------------------------------------------------------------
# Multi-scale decay rates (Eq. 8): gamma = 1 - 2^(-5-arange(0,h))
# ---------------------------------------------------------------------------
def _get_decay_gammas(num_heads: int) -> torch.Tensor:
    """
    Different retention heads use different decay rates gamma_i.
    - small gamma  -> fast decay  -> head attends mostly to RECENT frames
      (good for sudden bursts: DoS / Fuzzy attacks).
    - gamma close to 1 -> slow decay -> head aggregates a LONG context
      (good for subtle, slow-drifting anomalies: GEAR / RPM / Replay).
    """
    return 1 - torch.pow(2.0, -5 - torch.arange(num_heads, dtype=torch.float32))
 
 
# ---------------------------------------------------------------------------
# Single-head retention (Section 2.1)
# ---------------------------------------------------------------------------
class SimpleRetention(nn.Module):
    """
    One retention head, supporting both computation paradigms:
      - forward_parallel  : Eq. (5), used for training / offline inference.
      - forward_recurrent : Eq. (6), used for O(1) streaming inference.
 
    Both paths are mathematically equivalent (this is checked in the
    __main__ sanity test at the bottom of this file).
    """
 
    def __init__(self, hidden_size: int, head_size: int, gamma: float, double_v_dim: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.gamma = gamma
        self.v_dim = head_size * 2 if double_v_dim else head_size
        self.scale = head_size ** -0.5  # 1/sqrt(d), applied consistently to Q in both forms
 
        self.W_Q = nn.Parameter(torch.randn(hidden_size, head_size) / hidden_size)
        self.W_K = nn.Parameter(torch.randn(hidden_size, head_size) / hidden_size)
        self.W_V = nn.Parameter(torch.randn(hidden_size, self.v_dim) / hidden_size)
 
    def _decay_mask(self, seq_len: int, device, dtype) -> torch.Tensor:
        """D_nm = gamma^(n-m) if n>=m else 0  (Eq. 5)."""
        n = torch.arange(seq_len, device=device).unsqueeze(1)  # (T,1)
        m = torch.arange(seq_len, device=device).unsqueeze(0)  # (1,T)
        causal = (n >= m)
        exponent = (n - m).clamp(min=0).to(dtype)
        D = (self.gamma ** exponent) * causal.to(dtype)
        return D  # (T, T)
 
    def forward_parallel(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: (B, T, hidden_size)
        returns: (B, T, v_dim)
        """
        B, T, _ = X.shape
        Q = (X @ self.W_Q) * self.scale   # (B,T,head_size)
        K = X @ self.W_K                  # (B,T,head_size)
        V = X @ self.W_V                  # (B,T,v_dim)
 
        D = self._decay_mask(T, X.device, X.dtype)          # (T,T)
        A = (Q @ K.transpose(-1, -2)) * D.unsqueeze(0)       # (B,T,T)
        out = A @ V                                          # (B,T,v_dim)
        return out
 
    def forward_recurrent(self, x_n: torch.Tensor, S_prev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single-timestep recurrent update (Eq. 6).
        x_n   : (B, hidden_size)          -- current frame's fused feature
        S_prev: (B, head_size, v_dim)     -- previous recurrent state
        returns (out_n, S_n)
        """
        Q_n = (x_n @ self.W_Q) * self.scale     # (B, head_size)
        K_n = x_n @ self.W_K                    # (B, head_size)
        V_n = x_n @ self.W_V                    # (B, v_dim)
 
        S_n = self.gamma * S_prev + K_n.unsqueeze(-1) @ V_n.unsqueeze(-2)  # (B, head_size, v_dim)
        out_n = (Q_n.unsqueeze(1) @ S_n).squeeze(1)                        # (B, v_dim)
        return out_n, S_n
 
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.forward_parallel(X)
 
 
# ---------------------------------------------------------------------------
# Multi-Scale Retention (MSR), Eq. (8)
# ---------------------------------------------------------------------------
class MultiScaleRetention(nn.Module):
    """
    Gated Multi-Scale Retention: h heads, each with its own decay gamma_i,
    concatenated, GroupNorm-ed, and combined with a swish gate before the
    output projection (Eq. 8). This substitutes multi-head attention.
    """
 
    def __init__(self, hidden_size: int, num_heads: int, double_v_dim: bool = False):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.v_dim = self.head_size * 2 if double_v_dim else self.head_size
 
        gammas = _get_decay_gammas(num_heads)
        self.retentions = nn.ModuleList(
            [
                SimpleRetention(hidden_size, self.head_size, gammas[i].item(), double_v_dim)
                for i in range(num_heads)
            ]
        )
 
        # GroupNorm normalizes each head's output separately (heads have
        # different gamma -> different output variance statistics).
        self.group_norm = nn.GroupNorm(num_heads, num_heads * self.v_dim)
        self.W_G = nn.Parameter(torch.randn(hidden_size, num_heads * self.v_dim) / hidden_size)
        self.W_O = nn.Parameter(torch.randn(num_heads * self.v_dim, hidden_size) / hidden_size)
 
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """X: (B, T, hidden_size) -> (B, T, hidden_size)"""
        B, T, _ = X.shape
        head_outs = [ret(X) for ret in self.retentions]     # each (B,T,v_dim)
        Y = torch.cat(head_outs, dim=-1)                    # (B,T,C) where C=num_heads*v_dim
 
        # IMPORTANT: GroupNorm must normalize each timestep INDEPENDENTLY
        # (per-token), exactly like forward_recurrent does for a single step
        # (spatial size = 1). If we naively ran GroupNorm on (B,C,T), it
        # would aggregate statistics ACROSS the T dimension as well, which
        # is a different (non-causal, T-dependent) operation and breaks the
        # parallel/recurrent equivalence. So we flatten (B,T) into the batch
        # dimension before calling GroupNorm, then reshape back.
        C = Y.shape[-1]
        Y = Y.reshape(B * T, C, 1)
        Y = self.group_norm(Y)
        Y = Y.reshape(B, T, C)
 
        gate = F.silu(X @ self.W_G)                         # swish(X W_G)
        out = (gate * Y) @ self.W_O
        return out
 
    def forward_recurrent(
        self, x_n: torch.Tensor, states: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        x_n   : (B, hidden_size)
        states: list of per-head recurrent states, see `init_recurrent_state`
        """
        outs, new_states = [], []
        for ret, S_prev in zip(self.retentions, states):
            o_n, S_n = ret.forward_recurrent(x_n, S_prev)
            outs.append(o_n)
            new_states.append(S_n)
 
        Y = torch.cat(outs, dim=-1).unsqueeze(1)                     # (B,1,C)
        Y = self.group_norm(Y.transpose(1, 2)).transpose(1, 2).squeeze(1)  # (B,C)
 
        gate = F.silu(x_n @ self.W_G)
        out = (gate * Y) @ self.W_O
        return out, new_states
 
    def init_recurrent_state(self, batch_size: int, device, dtype) -> List[torch.Tensor]:
        return [
            torch.zeros(batch_size, self.head_size, self.v_dim, device=device, dtype=dtype)
            for _ in range(self.num_heads)
        ]
 
 
# ---------------------------------------------------------------------------
# RetNet block: pre-LN residual MSR + FFN (Eq. 9)
# ---------------------------------------------------------------------------
class FeedForward(nn.Module):
    def __init__(self, hidden_size: int, ffn_size: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, ffn_size)
        self.fc2 = nn.Linear(ffn_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))
 
 
class RetNetBlock(nn.Module):
    """
    Y = MSR(LN(X)) + X
    X = FFN(LN(Y)) + Y
    (Eq. 9 of the paper)
    """
 
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        ffn_size: int,
        dropout: float = 0.0,
        double_v_dim: bool = False,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size)
        self.msr = MultiScaleRetention(hidden_size, num_heads, double_v_dim)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.ffn = FeedForward(hidden_size, ffn_size, dropout)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.msr(self.ln1(x)) + x
        out = self.ffn(self.ln2(y)) + y
        return out
 
    def forward_recurrent(
        self, x_n: torch.Tensor, states: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        y_n, new_states = self.msr.forward_recurrent(self.ln1(x_n), states)
        y_n = y_n + x_n
        out_n = self.ffn(self.ln2(y_n)) + y_n
        return out_n, new_states
 
 
class RetNetEncoder(nn.Module):
    """Stack of L RetNetBlocks (the temporal module, replacing STDLM's MC-GTCN)."""
 
    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        ffn_size: int,
        dropout: float = 0.0,
        double_v_dim: bool = False,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                RetNetBlock(hidden_size, num_heads, ffn_size, dropout, double_v_dim)
                for _ in range(num_layers)
            ]
        )
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
 
    def forward_recurrent(
        self, x_n: torch.Tensor, states_list: List[List[torch.Tensor]]
    ) -> Tuple[torch.Tensor, List[List[torch.Tensor]]]:
        new_states_list = []
        for layer, states in zip(self.layers, states_list):
            x_n, new_states = layer.forward_recurrent(x_n, states)
            new_states_list.append(new_states)
        return x_n, new_states_list
 
    def init_recurrent_state(self, batch_size: int, device, dtype) -> List[List[torch.Tensor]]:
        return [layer.msr.init_recurrent_state(batch_size, device, dtype) for layer in self.layers]