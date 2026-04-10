"""Optional Triton kernels for exact-form Multiscreen preprocessing and aggregation.

This module preserves the paper equations exactly, but because it changes the
floating-point reduction schedule, it should be treated as numerically close to
the reference PyTorch path rather than bitwise-identical.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice
except ImportError:  # pragma: no cover - optional dependency
    triton = None
    tl = None
    libdevice = None


_SUPPORTED_DTYPES: Final = {torch.float32}
_MAX_D_KEY: Final = 64
_MAX_D_VALUE: Final = 128


@dataclass(frozen=True, slots=True)
class TritonScreeningSupport:
    """Availability verdict for the Triton screening kernel."""

    supported: bool
    reason: str | None = None


def triton_is_available() -> bool:
    """Return whether Triton was importable in the current environment."""

    return triton is not None and tl is not None


def _check_triton_tensor_support(
    x: torch.Tensor,
    *,
    name: str,
    max_dim: int,
) -> TritonScreeningSupport:
    if not triton_is_available():
        return TritonScreeningSupport(False, "triton is not installed")
    if x.ndim != 4:
        return TritonScreeningSupport(False, f"{name} must have shape [batch, head, seq, dim]")
    if not x.is_cuda:
        return TritonScreeningSupport(False, "triton screening requires CUDA tensors")
    if x.dtype not in _SUPPORTED_DTYPES:
        return TritonScreeningSupport(False, "triton screening currently supports float32 tensors only")
    if x.shape[-1] > max_dim:
        return TritonScreeningSupport(False, f"{name} dim={x.shape[-1]} exceeds the Triton kernel limit {max_dim}")
    if not x.is_contiguous():
        return TritonScreeningSupport(False, "triton screening requires contiguous tensors")
    return TritonScreeningSupport(True)


def check_triton_screening_support(
    q_mipe: torch.Tensor,
    k_mipe: torch.Tensor,
    v_unit: torch.Tensor,
) -> TritonScreeningSupport:
    """Return whether the current tensors are compatible with the Triton path."""

    q_support = _check_triton_tensor_support(q_mipe, name="q_mipe", max_dim=_MAX_D_KEY)
    if not q_support.supported:
        return q_support
    k_support = _check_triton_tensor_support(k_mipe, name="k_mipe", max_dim=_MAX_D_KEY)
    if not k_support.supported:
        return k_support
    v_support = _check_triton_tensor_support(v_unit, name="v_unit", max_dim=_MAX_D_VALUE)
    if not v_support.supported:
        return v_support
    if q_mipe.shape != k_mipe.shape:
        return TritonScreeningSupport(False, "q_mipe and k_mipe must have identical shapes")
    if q_mipe.shape[:-1] != v_unit.shape[:-1]:
        return TritonScreeningSupport(False, "v_unit must match q_mipe on batch, head, and seq dimensions")
    return TritonScreeningSupport(True)


if triton_is_available():

    @triton.jit
    def _normalize_block(x, eps):
        norm = tl.sqrt(tl.sum(x * x, axis=1))
        return x / tl.maximum(norm, eps)[:, None]


    @triton.jit
    def _tanh_norm_block(x, eps):
        norm = libdevice.sqrt(tl.sum(x * x, axis=1))
        scale = libdevice.tanh(norm) / tl.maximum(norm, eps)
        return x * scale[:, None]


    @triton.jit
    def _apply_mipe_block(normalized, positions, window, mipe_threshold, d_offsets):
        gamma = tl.where(
            window < mipe_threshold,
            0.5 * (tl.cos(math.pi * window / mipe_threshold) + 1.0),
            0.0,
        )
        safe_window = tl.where(gamma != 0.0, window, 1.0)
        angles = math.pi * positions.to(tl.float32) * gamma / safe_window
        cos = tl.cos(angles)
        sin = tl.sin(angles)
        first = tl.sum(tl.where(d_offsets[None, :] == 0, normalized, 0.0), axis=1)
        second = tl.sum(tl.where(d_offsets[None, :] == 1, normalized, 0.0), axis=1)
        rotated_first = first * cos - second * sin
        rotated_second = first * sin + second * cos
        normalized = tl.where(d_offsets[None, :] == 0, rotated_first[:, None], normalized)
        normalized = tl.where(d_offsets[None, :] == 1, rotated_second[:, None], normalized)
        return normalized

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_T": 16}, num_warps=2, num_stages=2),
            triton.Config({"BLOCK_T": 32}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_T": 64}, num_warps=4, num_stages=2),
        ],
        key=["T", "D"],
    )
    @triton.heuristics(values={"BLOCK_D": lambda args: triton.next_power_of_2(args["D"])})
    @triton.jit
    def _normalize_and_mipe_kernel(
        x_ptr,
        window_ptr,
        out_ptr,
        stride_xb,
        stride_xh,
        stride_xt,
        stride_xd,
        stride_out_b,
        stride_out_h,
        stride_out_t,
        stride_out_d,
        H,
        T,
        D,
        eps,
        mipe_threshold,
        BLOCK_T: tl.constexpr,
        BLOCK_D: tl.constexpr,
        USE_MIPE: tl.constexpr,
    ) -> None:
        pid_bh = tl.program_id(axis=0)
        pid_t = tl.program_id(axis=1)

        b = pid_bh // H
        h = pid_bh % H

        t_offsets = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
        d_offsets = tl.arange(0, BLOCK_D)

        t_mask = t_offsets < T
        d_mask = d_offsets < D

        x_ptrs = (
            x_ptr
            + b * stride_xb
            + h * stride_xh
            + t_offsets[:, None] * stride_xt
            + d_offsets[None, :] * stride_xd
        )
        x = tl.load(x_ptrs, mask=t_mask[:, None] & d_mask[None, :], other=0.0)

        normalized = _normalize_block(x, eps)

        if USE_MIPE:
            window = tl.load(window_ptr + h)
            normalized = _apply_mipe_block(normalized, t_offsets, window, mipe_threshold, d_offsets)

        out_ptrs = (
            out_ptr
            + b * stride_out_b
            + h * stride_out_h
            + t_offsets[:, None] * stride_out_t
            + d_offsets[None, :] * stride_out_d
        )
        tl.store(out_ptrs, normalized, mask=t_mask[:, None] & d_mask[None, :])

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_Q": 16, "BLOCK_K": 32}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_Q": 16, "BLOCK_K": 64}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_Q": 32, "BLOCK_K": 32}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_Q": 32, "BLOCK_K": 64}, num_warps=8, num_stages=2),
            triton.Config({"BLOCK_Q": 64, "BLOCK_K": 32}, num_warps=8, num_stages=2),
        ],
        key=["T", "DK", "DV"],
    )
    @triton.heuristics(
        values={
            "BLOCK_DK": lambda args: triton.next_power_of_2(args["DK"]),
            "BLOCK_DV": lambda args: triton.next_power_of_2(args["DV"]),
        }
    )
    @triton.jit
    def _screening_aggregate_q_fused_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        r_ptr,
        window_ptr,
        full_causal_ptr,
        out_ptr,
        stride_qb,
        stride_qh,
        stride_qt,
        stride_qd,
        stride_kb,
        stride_kh,
        stride_kt,
        stride_kd,
        stride_vb,
        stride_vh,
        stride_vt,
        stride_vd,
        stride_out_b,
        stride_out_h,
        stride_out_t,
        stride_out_d,
        H,
        T,
        DK,
        DV,
        eps,
        mipe_threshold,
        BLOCK_Q: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_DK: tl.constexpr,
        BLOCK_DV: tl.constexpr,
        APPLY_TANH_NORM: tl.constexpr,
    ) -> None:
        pid_bh = tl.program_id(axis=0)
        pid_q = tl.program_id(axis=1)

        b = pid_bh // H
        h = pid_bh % H

        q_offsets = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
        dk_offsets = tl.arange(0, BLOCK_DK)
        dv_offsets = tl.arange(0, BLOCK_DV)

        q_mask = q_offsets < T
        dk_mask = dk_offsets < DK
        dv_mask = dv_offsets < DV

        q_ptrs = (
            q_ptr
            + b * stride_qb
            + h * stride_qh
            + q_offsets[:, None] * stride_qt
            + dk_offsets[None, :] * stride_qd
        )
        q_raw = tl.load(q_ptrs, mask=q_mask[:, None] & dk_mask[None, :], other=0.0)

        r = tl.load(r_ptr + h)
        window = tl.load(window_ptr + h)
        full_causal = tl.load(full_causal_ptr + h) != 0

        q_unit = _normalize_block(q_raw, eps)
        q_mipe = _apply_mipe_block(q_unit, q_offsets, window, mipe_threshold, dk_offsets)

        acc = tl.zeros((BLOCK_Q, BLOCK_DV), dtype=tl.float32)

        for k_block in tl.range(0, tl.cdiv(T, BLOCK_K)):
            k_offsets = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
            k_mask = k_offsets < T

            k_ptrs = (
                k_ptr
                + b * stride_kb
                + h * stride_kh
                + k_offsets[:, None] * stride_kt
                + dk_offsets[None, :] * stride_kd
            )
            v_ptrs = (
                v_ptr
                + b * stride_vb
                + h * stride_vh
                + k_offsets[:, None] * stride_vt
                + dv_offsets[None, :] * stride_vd
            )

            k_mipe = tl.load(k_ptrs, mask=k_mask[:, None] & dk_mask[None, :], other=0.0)
            v_unit = tl.load(v_ptrs, mask=k_mask[:, None] & dv_mask[None, :], other=0.0)

            similarity = tl.dot(q_mipe, tl.trans(k_mipe), input_precision="ieee", out_dtype=tl.float32)
            similarity = tl.maximum(tl.minimum(similarity, 1.0), -1.0)

            relevance = 1.0 - r * (1.0 - similarity)
            relevance = tl.maximum(relevance, 0.0)
            relevance = relevance * relevance

            offsets = k_offsets[None, :] - q_offsets[:, None]
            causal = offsets <= 0
            causal_softmask = causal.to(tl.float32)
            offsets_f = offsets.to(tl.float32)
            inside = causal & (offsets_f > -window)
            cosine = 0.5 * (tl.cos(math.pi * offsets_f / window) + 1.0)
            finite_softmask = tl.where(inside, cosine, 0.0)
            softmask = tl.where(full_causal, causal_softmask, finite_softmask)
            alpha = relevance * softmask
            acc = acc + tl.dot(alpha, v_unit, input_precision="ieee", out_dtype=tl.float32)

        if APPLY_TANH_NORM:
            acc = _tanh_norm_block(acc, eps)

        out_ptrs = (
            out_ptr
            + b * stride_out_b
            + h * stride_out_h
            + q_offsets[:, None] * stride_out_t
            + dv_offsets[None, :] * stride_out_d
        )
        tl.store(out_ptrs, acc, mask=q_mask[:, None] & dv_mask[None, :])

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_Q": 16, "BLOCK_K": 32}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_Q": 16, "BLOCK_K": 64}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_Q": 32, "BLOCK_K": 32}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_Q": 32, "BLOCK_K": 64}, num_warps=8, num_stages=2),
            triton.Config({"BLOCK_Q": 64, "BLOCK_K": 32}, num_warps=8, num_stages=2),
        ],
        key=["T", "DK", "DV"],
    )
    @triton.heuristics(
        values={
            "BLOCK_DK": lambda args: triton.next_power_of_2(args["DK"]),
            "BLOCK_DV": lambda args: triton.next_power_of_2(args["DV"]),
        }
    )
    @triton.jit
    def _screening_aggregate_fused_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        r_ptr,
        window_ptr,
        full_causal_ptr,
        out_ptr,
        stride_qb,
        stride_qh,
        stride_qt,
        stride_qd,
        stride_kb,
        stride_kh,
        stride_kt,
        stride_kd,
        stride_vb,
        stride_vh,
        stride_vt,
        stride_vd,
        stride_out_b,
        stride_out_h,
        stride_out_t,
        stride_out_d,
        H,
        T,
        DK,
        DV,
        eps,
        mipe_threshold,
        BLOCK_Q: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_DK: tl.constexpr,
        BLOCK_DV: tl.constexpr,
        APPLY_TANH_NORM: tl.constexpr,
    ) -> None:
        pid_bh = tl.program_id(axis=0)
        pid_q = tl.program_id(axis=1)

        b = pid_bh // H
        h = pid_bh % H

        q_offsets = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
        dk_offsets = tl.arange(0, BLOCK_DK)
        dv_offsets = tl.arange(0, BLOCK_DV)

        q_mask = q_offsets < T
        dk_mask = dk_offsets < DK
        dv_mask = dv_offsets < DV

        q_ptrs = (
            q_ptr
            + b * stride_qb
            + h * stride_qh
            + q_offsets[:, None] * stride_qt
            + dk_offsets[None, :] * stride_qd
        )
        q_raw = tl.load(q_ptrs, mask=q_mask[:, None] & dk_mask[None, :], other=0.0)

        r = tl.load(r_ptr + h)
        window = tl.load(window_ptr + h)
        full_causal = tl.load(full_causal_ptr + h) != 0

        q_unit = _normalize_block(q_raw, eps)
        q_mipe = _apply_mipe_block(q_unit, q_offsets, window, mipe_threshold, dk_offsets)

        acc = tl.zeros((BLOCK_Q, BLOCK_DV), dtype=tl.float32)

        for k_block in tl.range(0, tl.cdiv(T, BLOCK_K)):
            k_offsets = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
            k_mask = k_offsets < T

            k_ptrs = (
                k_ptr
                + b * stride_kb
                + h * stride_kh
                + k_offsets[:, None] * stride_kt
                + dk_offsets[None, :] * stride_kd
            )
            v_ptrs = (
                v_ptr
                + b * stride_vb
                + h * stride_vh
                + k_offsets[:, None] * stride_vt
                + dv_offsets[None, :] * stride_vd
            )

            k_raw = tl.load(k_ptrs, mask=k_mask[:, None] & dk_mask[None, :], other=0.0)
            v_raw = tl.load(v_ptrs, mask=k_mask[:, None] & dv_mask[None, :], other=0.0)

            k_unit = _normalize_block(k_raw, eps)
            k_mipe = _apply_mipe_block(k_unit, k_offsets, window, mipe_threshold, dk_offsets)
            v_unit = _normalize_block(v_raw, eps)

            similarity = tl.dot(q_mipe, tl.trans(k_mipe), input_precision="ieee", out_dtype=tl.float32)
            similarity = tl.maximum(tl.minimum(similarity, 1.0), -1.0)

            relevance = 1.0 - r * (1.0 - similarity)
            relevance = tl.maximum(relevance, 0.0)
            relevance = relevance * relevance

            offsets = k_offsets[None, :] - q_offsets[:, None]
            causal = offsets <= 0
            causal_softmask = causal.to(tl.float32)
            offsets_f = offsets.to(tl.float32)
            inside = causal & (offsets_f > -window)
            cosine = 0.5 * (tl.cos(math.pi * offsets_f / window) + 1.0)
            finite_softmask = tl.where(inside, cosine, 0.0)
            softmask = tl.where(full_causal, causal_softmask, finite_softmask)
            alpha = relevance * softmask
            acc = acc + tl.dot(alpha, v_unit, input_precision="ieee", out_dtype=tl.float32)

        if APPLY_TANH_NORM:
            acc = _tanh_norm_block(acc, eps)

        out_ptrs = (
            out_ptr
            + b * stride_out_b
            + h * stride_out_h
            + q_offsets[:, None] * stride_out_t
            + dv_offsets[None, :] * stride_out_d
        )
        tl.store(out_ptrs, acc, mask=q_mask[:, None] & dv_mask[None, :])

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_Q": 16, "BLOCK_K": 32}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_Q": 16, "BLOCK_K": 64}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_Q": 32, "BLOCK_K": 32}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_Q": 32, "BLOCK_K": 64}, num_warps=8, num_stages=2),
            triton.Config({"BLOCK_Q": 64, "BLOCK_K": 32}, num_warps=8, num_stages=2),
        ],
        key=["T", "DK", "DV"],
    )
    @triton.heuristics(
        values={
            "BLOCK_DK": lambda args: triton.next_power_of_2(args["DK"]),
            "BLOCK_DV": lambda args: triton.next_power_of_2(args["DV"]),
        }
    )
    @triton.jit
    def _screening_aggregate_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        r_ptr,
        window_ptr,
        full_causal_ptr,
        out_ptr,
        stride_qb,
        stride_qh,
        stride_qt,
        stride_qd,
        stride_kb,
        stride_kh,
        stride_kt,
        stride_kd,
        stride_vb,
        stride_vh,
        stride_vt,
        stride_vd,
        stride_out_b,
        stride_out_h,
        stride_out_t,
        stride_out_d,
        H,
        T,
        DK,
        DV,
        eps,
        BLOCK_Q: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_DK: tl.constexpr,
        BLOCK_DV: tl.constexpr,
        APPLY_TANH_NORM: tl.constexpr,
    ) -> None:
        pid_bh = tl.program_id(axis=0)
        pid_q = tl.program_id(axis=1)

        b = pid_bh // H
        h = pid_bh % H

        q_offsets = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
        k_inner_offsets = tl.arange(0, BLOCK_DK)
        v_offsets = tl.arange(0, BLOCK_DV)

        q_mask = q_offsets < T
        dk_mask = k_inner_offsets < DK
        dv_mask = v_offsets < DV

        q_ptrs = (
            q_ptr
            + b * stride_qb
            + h * stride_qh
            + q_offsets[:, None] * stride_qt
            + k_inner_offsets[None, :] * stride_qd
        )
        q = tl.load(q_ptrs, mask=q_mask[:, None] & dk_mask[None, :], other=0.0)

        r = tl.load(r_ptr + h)
        window = tl.load(window_ptr + h)
        full_causal = tl.load(full_causal_ptr + h) != 0

        acc = tl.zeros((BLOCK_Q, BLOCK_DV), dtype=tl.float32)

        for k_block in tl.range(0, tl.cdiv(T, BLOCK_K)):
            k_offsets = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
            k_mask = k_offsets < T

            k_ptrs = (
                k_ptr
                + b * stride_kb
                + h * stride_kh
                + k_offsets[:, None] * stride_kt
                + k_inner_offsets[None, :] * stride_kd
            )
            v_ptrs = (
                v_ptr
                + b * stride_vb
                + h * stride_vh
                + k_offsets[:, None] * stride_vt
                + v_offsets[None, :] * stride_vd
            )

            k = tl.load(k_ptrs, mask=k_mask[:, None] & dk_mask[None, :], other=0.0)
            v = tl.load(v_ptrs, mask=k_mask[:, None] & dv_mask[None, :], other=0.0)

            similarity = tl.dot(q, tl.trans(k), input_precision="ieee", out_dtype=tl.float32)
            similarity = tl.maximum(tl.minimum(similarity, 1.0), -1.0)

            relevance = 1.0 - r * (1.0 - similarity)
            relevance = tl.maximum(relevance, 0.0)
            relevance = relevance * relevance

            offsets = k_offsets[None, :] - q_offsets[:, None]
            causal = offsets <= 0
            causal_softmask = causal.to(tl.float32)
            offsets_f = offsets.to(tl.float32)
            inside = causal & (offsets_f > -window)
            cosine = 0.5 * (tl.cos(math.pi * offsets_f / window) + 1.0)
            finite_softmask = tl.where(inside, cosine, 0.0)
            softmask = tl.where(full_causal, causal_softmask, finite_softmask)
            alpha = relevance * softmask
            acc = acc + tl.dot(alpha, v, input_precision="ieee", out_dtype=tl.float32)

        if APPLY_TANH_NORM:
            acc = _tanh_norm_block(acc, eps)

        out_ptrs = (
            out_ptr
            + b * stride_out_b
            + h * stride_out_h
            + q_offsets[:, None] * stride_out_t
            + v_offsets[None, :] * stride_out_d
        )
        tl.store(out_ptrs, acc, mask=q_mask[:, None] & dv_mask[None, :])


def triton_normalize_unit(x: torch.Tensor, *, eps: float) -> torch.Tensor:
    """Normalize vectors with Triton along the last dimension."""

    support = _check_triton_tensor_support(x, name="x", max_dim=_MAX_D_VALUE)
    if not support.supported:
        raise RuntimeError(f"Triton normalize kernel is unavailable: {support.reason}")

    x = x.contiguous()
    output = torch.empty_like(x)
    dummy_window = torch.empty((x.shape[1],), dtype=torch.float32, device=x.device)
    batch, heads, seq_len, dim = x.shape

    grid = lambda meta: (batch * heads, triton.cdiv(seq_len, meta["BLOCK_T"]))
    _normalize_and_mipe_kernel[grid](
        x,
        dummy_window,
        output,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        heads,
        seq_len,
        dim,
        eps,
        1.0,
        USE_MIPE=False,
    )
    return output


def triton_normalize_and_apply_mipe(
    x: torch.Tensor,
    *,
    window: torch.Tensor,
    eps: float,
    mipe_threshold: float,
) -> torch.Tensor:
    """Normalize vectors, then apply MiPE rotation with Triton."""

    support = _check_triton_tensor_support(x, name="x", max_dim=_MAX_D_KEY)
    if not support.supported:
        raise RuntimeError(f"Triton normalize+MiPE kernel is unavailable: {support.reason}")
    if x.shape[-1] < 2:
        raise ValueError("MiPE requires the last dimension to be at least 2")
    if window.ndim != 1 or window.shape[0] != x.shape[1]:
        raise ValueError("window must have shape [head] for Triton MiPE")

    x = x.contiguous()
    window = window.to(device=x.device, dtype=torch.float32).contiguous()
    output = torch.empty_like(x)
    batch, heads, seq_len, dim = x.shape

    grid = lambda meta: (batch * heads, triton.cdiv(seq_len, meta["BLOCK_T"]))
    _normalize_and_mipe_kernel[grid](
        x,
        window,
        output,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        heads,
        seq_len,
        dim,
        eps,
        mipe_threshold,
        USE_MIPE=True,
    )
    return output


def triton_screening_aggregate_q_fused(
    q: torch.Tensor,
    k_mipe: torch.Tensor,
    v_unit: torch.Tensor,
    *,
    r: torch.Tensor,
    window: torch.Tensor,
    full_causal: torch.Tensor,
    eps: float,
    mipe_threshold: float,
    apply_tanh_norm: bool = False,
) -> torch.Tensor:
    """Compute exact-form screening aggregation with q preprocessing fused into the aggregate kernel."""

    q_support = _check_triton_tensor_support(q, name="q", max_dim=_MAX_D_KEY)
    if not q_support.supported:
        raise RuntimeError(f"Triton q-fused screening kernel is unavailable: {q_support.reason}")
    support = check_triton_screening_support(q, k_mipe, v_unit)
    if not support.supported:
        raise RuntimeError(f"Triton q-fused screening kernel is unavailable: {support.reason}")
    if r.ndim != 1 or window.ndim != 1 or full_causal.ndim != 1:
        raise ValueError("r, window, and full_causal must have shape [head]")
    if r.shape != window.shape or r.shape != full_causal.shape:
        raise ValueError("r, window, and full_causal must have identical shapes")
    if r.shape[0] != q.shape[1]:
        raise ValueError("per-head Triton metadata must match the head dimension")

    q = q.contiguous()
    k_mipe = k_mipe.contiguous()
    v_unit = v_unit.contiguous()
    r = r.to(device=q.device, dtype=torch.float32).contiguous()
    window = window.to(device=q.device, dtype=torch.float32).contiguous()
    full_causal = full_causal.to(device=q.device, dtype=torch.int32).contiguous()

    output = torch.empty_like(v_unit)
    batch, heads, seq_len, d_key = q.shape
    d_value = v_unit.shape[-1]

    grid = lambda meta: (batch * heads, triton.cdiv(seq_len, meta["BLOCK_Q"]))
    _screening_aggregate_q_fused_kernel[grid](
        q,
        k_mipe,
        v_unit,
        r,
        window,
        full_causal,
        output,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k_mipe.stride(0),
        k_mipe.stride(1),
        k_mipe.stride(2),
        k_mipe.stride(3),
        v_unit.stride(0),
        v_unit.stride(1),
        v_unit.stride(2),
        v_unit.stride(3),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        heads,
        seq_len,
        d_key,
        d_value,
        eps,
        mipe_threshold,
        APPLY_TANH_NORM=apply_tanh_norm,
    )
    return output


def triton_screening_aggregate_fused(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    r: torch.Tensor,
    window: torch.Tensor,
    full_causal: torch.Tensor,
    eps: float,
    mipe_threshold: float,
    apply_tanh_norm: bool = False,
) -> torch.Tensor:
    """Compute exact-form screening aggregation directly from raw q, k, v with fused preprocessing."""

    support = check_triton_screening_support(q, k, v)
    if not support.supported:
        raise RuntimeError(f"Triton fused screening kernel is unavailable: {support.reason}")
    if r.ndim != 1 or window.ndim != 1 or full_causal.ndim != 1:
        raise ValueError("r, window, and full_causal must have shape [head]")
    if r.shape != window.shape or r.shape != full_causal.shape:
        raise ValueError("r, window, and full_causal must have identical shapes")
    if r.shape[0] != q.shape[1]:
        raise ValueError("per-head Triton metadata must match the head dimension")

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    r = r.to(device=q.device, dtype=torch.float32).contiguous()
    window = window.to(device=q.device, dtype=torch.float32).contiguous()
    full_causal = full_causal.to(device=q.device, dtype=torch.int32).contiguous()

    output = torch.empty_like(v)
    batch, heads, seq_len, d_key = q.shape
    d_value = v.shape[-1]

    grid = lambda meta: (batch * heads, triton.cdiv(seq_len, meta["BLOCK_Q"]))
    _screening_aggregate_fused_kernel[grid](
        q,
        k,
        v,
        r,
        window,
        full_causal,
        output,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        heads,
        seq_len,
        d_key,
        d_value,
        eps,
        mipe_threshold,
        APPLY_TANH_NORM=apply_tanh_norm,
    )
    return output


def triton_screening_aggregate(
    q_mipe: torch.Tensor,
    k_mipe: torch.Tensor,
    v_unit: torch.Tensor,
    *,
    r: torch.Tensor,
    window: torch.Tensor,
    full_causal: torch.Tensor,
    eps: float = 1e-12,
    apply_tanh_norm: bool = False,
) -> torch.Tensor:
    """Compute exact-form screening aggregation with Triton and return raw `h_i`."""

    support = check_triton_screening_support(q_mipe, k_mipe, v_unit)
    if not support.supported:
        raise RuntimeError(f"Triton screening kernel is unavailable: {support.reason}")
    if r.ndim != 1 or window.ndim != 1 or full_causal.ndim != 1:
        raise ValueError("r, window, and full_causal must have shape [head]")
    if r.shape != window.shape or r.shape != full_causal.shape:
        raise ValueError("r, window, and full_causal must have identical shapes")
    if r.shape[0] != q_mipe.shape[1]:
        raise ValueError("per-head Triton metadata must match the head dimension")

    q_mipe = q_mipe.contiguous()
    k_mipe = k_mipe.contiguous()
    v_unit = v_unit.contiguous()
    r = r.to(device=q_mipe.device, dtype=torch.float32).contiguous()
    finite_window = torch.where(
        torch.isinf(window),
        torch.ones_like(window, dtype=torch.float32, device=q_mipe.device),
        window.to(device=q_mipe.device, dtype=torch.float32),
    ).contiguous()
    full_causal = full_causal.to(device=q_mipe.device, dtype=torch.int32).contiguous()

    output = torch.empty_like(v_unit)
    batch, heads, seq_len, d_key = q_mipe.shape
    d_value = v_unit.shape[-1]

    grid = lambda meta: (batch * heads, triton.cdiv(seq_len, meta["BLOCK_Q"]))
    _screening_aggregate_kernel[grid](
        q_mipe,
        k_mipe,
        v_unit,
        r,
        finite_window,
        full_causal,
        output,
        q_mipe.stride(0),
        q_mipe.stride(1),
        q_mipe.stride(2),
        q_mipe.stride(3),
        k_mipe.stride(0),
        k_mipe.stride(1),
        k_mipe.stride(2),
        k_mipe.stride(3),
        v_unit.stride(0),
        v_unit.stride(1),
        v_unit.stride(2),
        v_unit.stride(3),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        heads,
        seq_len,
        d_key,
        d_value,
        eps,
        APPLY_TANH_NORM=apply_tanh_norm,
    )
    return output
