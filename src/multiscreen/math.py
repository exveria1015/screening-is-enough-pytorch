"""Core math utilities for the Multiscreen architecture."""

from __future__ import annotations

import math

import torch
from torch import nn


def normalize_unit(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Normalize vectors along the last dimension."""

    norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(eps)
    return x / norm


class TanhNorm(nn.Module):
    """Norm cap defined in Equation 20."""

    def __init__(self, eps: float = 1e-12) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
        return x * (torch.tanh(norm) / norm.clamp_min(self.eps))


def window_from_parameter(s_w: torch.Tensor) -> torch.Tensor:
    """Equation 7."""

    return torch.exp(s_w) + 1.0


def relevance_width_from_parameter(s_r: torch.Tensor) -> torch.Tensor:
    """Equation 7."""

    return torch.exp(s_r) + 1.0


def mipe_gamma(window: torch.Tensor, mipe_threshold: float) -> torch.Tensor:
    """Equation 12."""

    threshold = torch.as_tensor(mipe_threshold, dtype=window.dtype, device=window.device)
    active = window < threshold
    cosine = 0.5 * (torch.cos(torch.pi * window / threshold) + 1.0)
    return torch.where(active, cosine, torch.zeros_like(window))


def apply_mipe(
    x: torch.Tensor,
    positions: torch.Tensor,
    window: torch.Tensor,
    mipe_threshold: float,
) -> torch.Tensor:
    """Apply the first-two-dimension RoPE-like rotation from Equations 9-12."""

    if x.shape[-1] < 2:
        raise ValueError("MiPE requires the last dimension to be at least 2")
    gamma = mipe_gamma(window, mipe_threshold)
    angle = torch.pi * positions.to(dtype=x.dtype, device=x.device) * gamma.to(dtype=x.dtype) / window.to(dtype=x.dtype)
    view_shape = [1] * (x.ndim - 2) + [positions.numel(), 1]
    cos = torch.cos(angle).view(*view_shape)
    sin = torch.sin(angle).view(*view_shape)
    first = x[..., 0:1]
    second = x[..., 1:2]
    rotated_first = first * cos - second * sin
    rotated_second = first * sin + second * cos
    return torch.cat((rotated_first, rotated_second, x[..., 2:]), dim=-1)


def trim_and_square(similarity: torch.Tensor, s_r: torch.Tensor) -> torch.Tensor:
    """Equation 16."""

    r = relevance_width_from_parameter(s_r).to(dtype=similarity.dtype)
    return torch.clamp(1.0 - r * (1.0 - similarity), min=0.0).square()


def build_softmask(
    query_positions: torch.Tensor,
    key_positions: torch.Tensor,
    window: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Equation 17."""

    offsets = key_positions.view(1, -1) - query_positions.view(-1, 1)
    if torch.isinf(window).item():
        return (offsets <= 0).to(dtype=dtype)
    scaled_offsets = offsets.to(dtype=dtype) / window.to(dtype=dtype)
    cosine = 0.5 * (torch.cos(torch.pi * scaled_offsets) + 1.0)
    inside = (offsets <= 0) & (offsets > -window)
    return torch.where(inside, cosine, torch.zeros_like(cosine))
