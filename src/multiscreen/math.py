"""Core math utilities for the Multiscreen architecture."""

from __future__ import annotations

import math

import torch
from torch import nn


def normalize_unit(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Normalize vectors along the last dimension."""

    norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(eps)
    return x / norm


def tanh_norm(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Apply the TanhNorm transform from Equation 20."""

    norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    return x * (torch.tanh(norm) / norm.clamp_min(eps))


class TanhNorm(nn.Module):
    """Norm cap defined in Equation 20."""

    def __init__(self, eps: float = 1e-12) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return tanh_norm(x, eps=self.eps)


def window_from_parameter(s_w: torch.Tensor) -> torch.Tensor:
    """Equation 7."""

    return torch.exp(s_w) + 1.0


def relevance_width_from_parameter(s_r: torch.Tensor) -> torch.Tensor:
    """Equation 7."""

    return torch.exp(s_r) + 1.0


def mipe_gamma(window: torch.Tensor, mipe_threshold: float) -> torch.Tensor:
    """Equation 12."""

    threshold = torch.as_tensor(mipe_threshold, dtype=window.dtype, device=window.device)
    flat_window = window.reshape(-1)
    flat_gamma = torch.zeros_like(flat_window)
    active = flat_window < threshold
    if bool(active.any()):
        flat_gamma[active] = 0.5 * (torch.cos(torch.pi * flat_window[active] / threshold) + 1.0)
    return flat_gamma.reshape(window.shape)


def build_mipe_rotation(
    positions: torch.Tensor,
    window: torch.Tensor,
    mipe_threshold: float,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Build broadcastable MiPE rotation factors shared by q and k."""

    gamma = mipe_gamma(window, mipe_threshold).to(dtype=dtype, device=device)
    if not bool(gamma.ne(0).any()):
        return None

    positions_f = positions.to(dtype=dtype, device=device)
    window_f = window.to(dtype=dtype, device=device)
    if window_f.ndim == 0:
        angle = torch.pi * positions_f * gamma / window_f
        cos = torch.cos(angle).view(positions.numel(), 1)
        sin = torch.sin(angle).view(positions.numel(), 1)
        return cos, sin

    angle = torch.pi * positions_f.view(1, -1) * gamma.view(-1, 1) / window_f.view(-1, 1)
    cos = torch.cos(angle).view(window_f.shape[0], positions.numel(), 1)
    sin = torch.sin(angle).view(window_f.shape[0], positions.numel(), 1)
    return cos, sin


def _apply_mipe_rotation(x: torch.Tensor, rotation: tuple[torch.Tensor, torch.Tensor] | None) -> torch.Tensor:
    """Rotate the first two coordinates using precomputed MiPE factors."""

    if rotation is None:
        return x
    cos, sin = rotation
    first = x[..., 0:1]
    second = x[..., 1:2]
    rotated_first = first * cos - second * sin
    rotated_second = first * sin + second * cos
    return torch.cat((rotated_first, rotated_second, x[..., 2:]), dim=-1)


def apply_mipe(
    x: torch.Tensor,
    positions: torch.Tensor,
    window: torch.Tensor,
    mipe_threshold: float,
) -> torch.Tensor:
    """Apply the first-two-dimension RoPE-like rotation from Equations 9-12."""

    if x.shape[-1] < 2:
        raise ValueError("MiPE requires the last dimension to be at least 2")
    rotation = build_mipe_rotation(
        positions,
        window,
        mipe_threshold,
        dtype=x.dtype,
        device=x.device,
    )
    if window.ndim != 0 and (x.ndim < 3 or window.shape != (x.shape[-3],)):
        raise ValueError("vector windows must match the head dimension of x")
    return _apply_mipe_rotation(x, rotation)


def normalize_and_apply_mipe(
    x: torch.Tensor,
    positions: torch.Tensor,
    window: torch.Tensor,
    mipe_threshold: float,
    *,
    eps: float = 1e-12,
    rotation: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Normalize along the last dimension, then apply MiPE without materializing the normalized tensor."""

    if x.shape[-1] < 2:
        raise ValueError("MiPE requires the last dimension to be at least 2")
    if window.ndim != 0 and (x.ndim < 3 or window.shape != (x.shape[-3],)):
        raise ValueError("vector windows must match the head dimension of x")

    norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(eps)
    first = x[..., 0:1] / norm
    second = x[..., 1:2] / norm
    rest = x[..., 2:] / norm
    effective_rotation = rotation
    if effective_rotation is None:
        effective_rotation = build_mipe_rotation(
            positions,
            window,
            mipe_threshold,
            dtype=x.dtype,
            device=x.device,
        )
    if effective_rotation is None:
        return torch.cat((first, second, rest), dim=-1)
    cos, sin = effective_rotation
    rotated_first = first * cos - second * sin
    rotated_second = first * sin + second * cos
    return torch.cat((rotated_first, rotated_second, rest), dim=-1)


def trim_and_square(similarity: torch.Tensor, s_r: torch.Tensor) -> torch.Tensor:
    """Equation 16."""

    r = relevance_width_from_parameter(s_r).to(dtype=similarity.dtype, device=similarity.device)
    if r.ndim != 0:
        if similarity.ndim < 3 or r.shape != (similarity.shape[-3],):
            raise ValueError("vector s_r must match the head dimension of similarity")
        r = r.view(*([1] * (similarity.ndim - 3)), r.shape[0], 1, 1)
    return torch.clamp(1.0 - r * (1.0 - similarity), min=0.0).square()


def build_softmask(
    query_positions: torch.Tensor,
    key_positions: torch.Tensor,
    window: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Equation 17."""

    offsets = key_positions.view(1, -1) - query_positions.view(-1, 1)
    offsets_f = offsets.to(dtype=dtype)
    window_f = window.to(dtype=dtype, device=offsets.device)
    if window_f.ndim == 0:
        scaled_offsets = offsets_f / window_f
        cosine = 0.5 * (torch.cos(torch.pi * scaled_offsets) + 1.0)
        inside = (offsets <= 0) & (offsets_f > -window_f)
        return torch.where(inside, cosine, torch.zeros_like(cosine))

    offsets_f = offsets_f.unsqueeze(0)
    window_view = window_f.view(-1, 1, 1)
    cosine = 0.5 * (torch.cos(torch.pi * offsets_f / window_view) + 1.0)
    inside = (offsets.unsqueeze(0) <= 0) & (offsets_f > -window_view)
    return torch.where(inside, cosine, torch.zeros_like(cosine))
