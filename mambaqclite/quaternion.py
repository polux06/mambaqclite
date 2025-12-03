"""Quaternion helper utilities for Mamba-Quaternion-Lite.

The implementation keeps the quaternion representation explicit in four
real-valued channels ordered as (r, i, j, k). These helpers focus on the
minimum operations needed by the model: multiplication, normalization, and a
compact initialization routine.
"""

from __future__ import annotations

import torch
from torch import Tensor


def quat_mul(a: Tensor, b: Tensor) -> Tensor:
    """Multiply two quaternions ``a`` and ``b``.

    Both ``a`` and ``b`` are expected to share a broadcastable shape ending in
    ``(..., 4)`` where the trailing dimension corresponds to ``(r, i, j, k)``.
    The function returns the quaternion product with the same broadcasted
    leading shape.
    """

    ar, ai, aj, ak = torch.unbind(a, dim=-1)
    br, bi, bj, bk = torch.unbind(b, dim=-1)

    return torch.stack(
        (
            ar * br - ai * bi - aj * bj - ak * bk,
            ar * bi + ai * br + aj * bk - ak * bj,
            ar * bj - ai * bk + aj * br + ak * bi,
            ar * bk + ai * bj - aj * bi + ak * br,
        ),
        dim=-1,
    )


def quat_norm(q: Tensor, eps: float = 1e-6) -> Tensor:
    """Return the L2 norm of a quaternion tensor.

    Args:
        q: Tensor with shape ``(..., 4)``.
        eps: Small constant to avoid division by zero.
    """

    return torch.sqrt(torch.sum(q * q, dim=-1) + eps)


def quat_init(shape: tuple[int, ...], scale: float = 0.02, device=None, dtype=None) -> Tensor:
    """Isotropic quaternion initializer.

    The initializer samples each component from a normal distribution and
    re-scales the quaternion to have the requested norm.
    """

    q = torch.randn(*shape, 4, device=device, dtype=dtype)
    norm = quat_norm(q).unsqueeze(-1)
    return scale * q / norm
