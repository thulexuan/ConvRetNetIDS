import torch
import torch.nn as nn
import torch.nn.functional as F

 

def _get_decay_gammas(num_heads: int) -> torch.Tensor:
 
    return 1 - torch.pow(2.0, -5 - torch.arange(num_heads, dtype=torch.float32))
 
class SimpleRetention(nn.Module):
 
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
 
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.forward_parallel(X)
 
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
 
        self.group_norm = nn.GroupNorm(num_heads, num_heads * self.v_dim)
        self.W_G = nn.Parameter(torch.randn(hidden_size, num_heads * self.v_dim) / hidden_size)
        self.W_O = nn.Parameter(torch.randn(num_heads * self.v_dim, hidden_size) / hidden_size)
 
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """X: (B, T, hidden_size) -> (B, T, hidden_size)"""
        B, T, _ = X.shape
        head_outs = [ret(X) for ret in self.retentions]     # each (B,T,v_dim)
        Y = torch.cat(head_outs, dim=-1)                    # (B,T,C) where C=num_heads*v_dim
 
        C = Y.shape[-1]
        Y = Y.reshape(B * T, C, 1)
        Y = self.group_norm(Y)
        Y = Y.reshape(B, T, C)
 
        gate = F.silu(X @ self.W_G)                         # swish(X W_G)
        out = (gate * Y) @ self.W_O
        return out
 
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
 
 
 
class RetNetEncoder(nn.Module):
 
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
 