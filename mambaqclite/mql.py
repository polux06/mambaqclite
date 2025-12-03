"""Core Mamba-Quaternion-Lite building blocks.

This module provides a minimal PyTorch implementation that mirrors the paper
summary:

* Scalar continuous dynamics with bilinear discretization.
* Quaternion injections (``B_t``) and projections (``C_t``).
* Real-valued internal state enabling the standard parallel scan in future
  optimized kernels; here we keep a sequential Python implementation for
  clarity.

The code is intentionally compact to make it easy to integrate with existing
Mamba-2 optimized kernels later on. Shapes follow the convention that the model
hidden dimension is divisible by four so that every group of four channels
represents a quaternion ``(r, i, j, k)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .quaternion import quat_init


@dataclass
class MQLConfig:
    """Configuration for the Mamba-Quaternion-Lite model."""

    vocab_size: int = 2048
    d_model: int = 256
    n_layers: int = 4
    state_dim: int = 16
    max_seq_len: int = 512
    dropout: float = 0.0
    dt_init: float = 0.1

    def __post_init__(self) -> None:
        if self.d_model % 4 != 0:
            raise ValueError("d_model must be divisible by 4 to represent quaternions")


class MambaQuaternionLiteBlock(nn.Module):
    """Single Mamba-Quaternion-Lite block with scalar dynamics."""

    def __init__(self, d_model: int, state_dim: int, dt_init: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim
        self.q_channels = d_model // 4

        # Content-dependent components
        self.gate = nn.Linear(d_model, d_model)
        self.delta = nn.Linear(d_model, self.q_channels)
        self.skip = nn.Linear(d_model, d_model)
        self.B_proj = nn.Linear(d_model, self.q_channels * state_dim * 4)
        self.C_proj = nn.Linear(d_model, self.q_channels * state_dim * 4)

        # Continuous-time eigenvalues (negative for stability)
        self.alpha = nn.Parameter(torch.randn(self.q_channels, state_dim) * 0.02)
        self.dt_bias = math.log(math.exp(dt_init) - 1.0)

        self.layer_norm = nn.LayerNorm(d_model)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.gate.weight)
        nn.init.xavier_uniform_(self.delta.weight)
        nn.init.xavier_uniform_(self.skip.weight)
        nn.init.xavier_uniform_(self.B_proj.weight)
        nn.init.xavier_uniform_(self.C_proj.weight)
        for linear in (self.gate, self.delta, self.skip, self.B_proj, self.C_proj):
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for a single sequence batch.

        Args:
            x: Tensor of shape ``(batch, seq_len, d_model)``.
        Returns:
            Tensor with the same shape representing the transformed sequence.
        """

        batch, seq_len, _ = x.shape
        x = self.layer_norm(x)
        q_channels = self.q_channels

        # Compute gating and gated signal in quaternion form
        gate = torch.sigmoid(self.gate(x)).view(batch, seq_len, q_channels, 4)
        x_quat = x.view(batch, seq_len, q_channels, 4)
        S = gate * x_quat

        # Scalar step size per quaternion channel
        delta = F.softplus(self.delta(x) + self.dt_bias)
        Lambda = -F.softplus(self.alpha)
        z = delta.unsqueeze(-1) * Lambda  # (B, T, q_channels, state_dim)
        A = (1 + 0.5 * z) / (1 - 0.5 * z)

        # Quaternion injections and projections
        B_t = self.B_proj(x).view(batch, seq_len, q_channels, self.state_dim, 4)
        C_t = self.C_proj(x).view(batch, seq_len, q_channels, self.state_dim, 4)

        # Internal real state
        h = torch.zeros(batch, q_channels, self.state_dim, device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(seq_len):
            # Inject gated signal into the real state via quaternion dot-product
            b_term = (B_t[:, t] * S[:, t].unsqueeze(-2)).sum(dim=-1)
            h = A[:, t] * h + b_term

            # Quaternion projection of the real state
            state_out = (C_t[:, t] * h.unsqueeze(-1)).sum(dim=-2)
            skip_out = self.skip(x[:, t]).view(batch, q_channels, 4) * S[:, t]
            y_t = state_out + skip_out
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1).view(batch, seq_len, self.d_model)
        return y


class MambaQuaternionLiteModel(nn.Module):
    """Tiny sequence model configured for TinyStories-scale experiments."""

    def __init__(self, config: MQLConfig) -> None:
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.d_model)
        self.layers = nn.ModuleList(
            [
                MambaQuaternionLiteBlock(
                    d_model=config.d_model,
                    state_dim=config.state_dim,
                    dt_init=config.dt_init,
                )
                for _ in range(config.n_layers)
            ]
        )
        self.norm = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: Tensor, *, positions: Optional[Tensor] = None) -> Tensor:
        """Compute logits for a batch of token ids.

        Args:
            input_ids: Tensor of shape ``(batch, seq_len)``.
            positions: Optional precomputed position indices; if omitted a
                consecutive range starting at zero is used.
        Returns:
            Logits tensor of shape ``(batch, seq_len, vocab_size)``.
        """

        batch, seq_len = input_ids.shape
        if positions is None:
            positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        pos_emb = self.pos_embed(positions)
        x = self.embed(input_ids) + pos_emb
        x = self.dropout(x)

        for layer in self.layers:
            x = x + layer(x)

        x = self.norm(x)
        logits = self.head(x)
        return logits

    @torch.no_grad()
    def generate(self, input_ids: Tensor, max_new_tokens: int) -> Tensor:
        """Autoregressive generation helper."""

        for _ in range(max_new_tokens):
            seq_len = input_ids.shape[1]
            positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
            logits = self.forward(input_ids, positions=positions)
            next_token = torch.argmax(logits[:, -1], dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids


__all__ = ["MQLConfig", "MambaQuaternionLiteBlock", "MambaQuaternionLiteModel"]
