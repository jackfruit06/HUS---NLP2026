"""Các thành phần của Transformer Encoder, tự ghép từ PyTorch primitives."""

import math

import torch
from torch import nn


def positional_encoding(length, d_model, device=None):
    """Sinusoidal PE: (T, d_model), kể cả khi d_model lẻ."""
    positions = torch.arange(length, dtype=torch.float32, device=device)[:, None]
    frequencies = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32, device=device)
        * (-math.log(10000.0) / d_model)
    )
    angles = positions * frequencies
    pe = torch.zeros(length, d_model, device=device)
    pe[:, 0::2] = angles.sin()
    pe[:, 1::2] = angles[:, :d_model // 2].cos()
    return pe


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        if n_heads <= 0 or d_model % n_heads:
            raise ValueError('d_model phải chia hết cho n_heads > 0.')
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)

    def forward(self, x, valid=None):
        batch, length, dimension = x.shape
        if valid is None:
            valid = torch.ones(batch, length, dtype=torch.bool, device=x.device)
        if not valid.any(dim=1).all():
            raise ValueError('Mỗi sequence cần ít nhất một token thật.')

        def split_heads(projection):
            return projection(x).reshape(batch, length, self.n_heads, self.d_k).transpose(1, 2)

        q, k, v = [split_heads(layer) for layer in (self.wq, self.wk, self.wv)]
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.d_k)
        scores = scores.masked_fill(~valid[:, None, None, :], float('-inf'))
        weights = scores.softmax(dim=-1)
        weights = weights * valid[:, None, :, None]
        context = weights @ v
        merged = context.transpose(1, 2).contiguous().reshape(batch, length, dimension)
        output = self.wo(merged) * valid[:, :, None]
        return output, weights


class TransformerBlock(nn.Module):
    """Post-norm: Add & Norm sau attention, rồi Add & Norm sau FFN."""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadSelfAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, valid=None, return_attention=False):
        attended, weights = self.attention(x, valid)
        x = self.norm1(x + self.dropout(attended))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        if valid is not None:
            x = x * valid[:, :, None]
        return (x, weights) if return_attention else x
