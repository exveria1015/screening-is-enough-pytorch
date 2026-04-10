from __future__ import annotations

import pytest
import torch

from multiscreen.config import MultiscreenConfig
from multiscreen.math import normalize_and_apply_mipe, normalize_unit, relevance_width_from_parameter, tanh_norm
from multiscreen.model import MultiscreenLM
from multiscreen.triton_kernels import (
    triton_normalize_and_apply_mipe,
    triton_normalize_unit,
    triton_screening_aggregate,
    triton_screening_aggregate_q_fused,
    triton_screening_aggregate_fused,
)


pytest.importorskip("triton")


def _cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    return torch.device("cuda:0")


def _build_model_and_inputs(
    *,
    seq_len: int,
    max_train_seq_len: int,
    seed: int = 0,
) -> tuple[MultiscreenLM, torch.Tensor]:
    torch.manual_seed(seed)
    config = MultiscreenConfig(
        vocab_size=4096,
        d_model=64,
        n_layers=2,
        n_heads=4,
        d_key=16,
        d_value=32,
        max_seq_len=max(seq_len, 1024),
        max_train_seq_len=max_train_seq_len,
    )
    device = _cuda_device()
    model = MultiscreenLM(config).to(device)
    model.eval()
    input_ids = torch.randint(0, config.vocab_size, (2, seq_len), device=device)
    return model, input_ids


def _assert_logits_close(
    model: MultiscreenLM,
    input_ids: torch.Tensor,
    *,
    inference: bool,
    triton_fuse_preprocessing: bool = False,
) -> None:
    with torch.no_grad():
        logits_torch, _ = model(
            input_ids,
            inference=inference,
            return_relevances=False,
            query_chunk_size=10_000,
            screening_backend="torch",
        )
        logits_triton, _ = model(
            input_ids,
            inference=inference,
            return_relevances=False,
            screening_backend="triton",
            triton_fuse_preprocessing=triton_fuse_preprocessing,
        )
    assert torch.allclose(logits_triton, logits_torch, atol=1e-6, rtol=1e-6)


def test_triton_backend_matches_torch_logits_on_cuda() -> None:
    model, input_ids = _build_model_and_inputs(seq_len=256, max_train_seq_len=256)
    _assert_logits_close(model, input_ids, inference=False)


def test_triton_backend_matches_torch_logits_for_full_causal_inference() -> None:
    model, input_ids = _build_model_and_inputs(seq_len=512, max_train_seq_len=128)
    _assert_logits_close(model, input_ids, inference=True)


def test_triton_fused_backend_matches_torch_logits_on_cuda() -> None:
    model, input_ids = _build_model_and_inputs(seq_len=256, max_train_seq_len=256)
    _assert_logits_close(model, input_ids, inference=False, triton_fuse_preprocessing=True)


def test_triton_fused_backend_matches_torch_logits_for_full_causal_inference() -> None:
    model, input_ids = _build_model_and_inputs(seq_len=512, max_train_seq_len=128)
    _assert_logits_close(model, input_ids, inference=True, triton_fuse_preprocessing=True)


def test_triton_preprocess_matches_torch_preprocess_on_cuda() -> None:
    device = _cuda_device()
    torch.manual_seed(0)
    q = torch.randn(2, 4, 256, 16, device=device)
    v = torch.randn(2, 4, 256, 32, device=device)
    positions = torch.arange(256, device=device)
    window = torch.tensor([2.0, 8.0, 128.0, 512.0], device=device)

    with torch.no_grad():
        expected_q = normalize_and_apply_mipe(q, positions, window, mipe_threshold=256.0)
        actual_q = triton_normalize_and_apply_mipe(q, window=window, eps=1e-12, mipe_threshold=256.0)
        expected_v = normalize_unit(v)
        actual_v = triton_normalize_unit(v, eps=1e-12)

    assert torch.allclose(actual_q, expected_q, atol=2e-7, rtol=1e-6)
    assert torch.allclose(actual_v, expected_v, atol=2e-7, rtol=1e-6)


def test_triton_fused_aggregate_matches_preprocessed_aggregate_on_cuda() -> None:
    device = _cuda_device()
    torch.manual_seed(0)
    q = torch.randn(2, 4, 256, 16, device=device)
    k = torch.randn(2, 4, 256, 16, device=device)
    v = torch.randn(2, 4, 256, 32, device=device)
    positions = torch.arange(256, device=device)
    window = torch.tensor([2.0, 8.0, 128.0, 512.0], device=device)
    s_r = torch.zeros(4, device=device)

    with torch.no_grad():
        q_mipe = normalize_and_apply_mipe(q, positions, window, mipe_threshold=256.0)
        k_mipe = normalize_and_apply_mipe(k, positions, window, mipe_threshold=256.0)
        v_unit = normalize_unit(v)
        expected = triton_screening_aggregate(
            q_mipe,
            k_mipe,
            v_unit,
            r=relevance_width_from_parameter(s_r),
            window=window,
            full_causal=torch.isinf(window),
        )
        actual = triton_screening_aggregate_fused(
            q,
            k,
            v,
            r=relevance_width_from_parameter(s_r),
            window=window,
            full_causal=torch.isinf(window),
            eps=1e-12,
            mipe_threshold=256.0,
        )

    assert torch.allclose(actual, expected, atol=2e-5, rtol=1e-6)


def test_triton_fused_aggregate_tanh_norm_matches_reference_on_cuda() -> None:
    device = _cuda_device()
    torch.manual_seed(0)
    q = torch.randn(2, 4, 256, 16, device=device)
    k = torch.randn(2, 4, 256, 16, device=device)
    v = torch.randn(2, 4, 256, 32, device=device)
    positions = torch.arange(256, device=device)
    window = torch.tensor([2.0, 8.0, 128.0, 512.0], device=device)
    s_r = torch.zeros(4, device=device)

    with torch.no_grad():
        q_mipe = normalize_and_apply_mipe(q, positions, window, mipe_threshold=256.0)
        k_mipe = normalize_and_apply_mipe(k, positions, window, mipe_threshold=256.0)
        v_unit = normalize_unit(v)
        expected = tanh_norm(
            triton_screening_aggregate(
                q_mipe,
                k_mipe,
                v_unit,
                r=relevance_width_from_parameter(s_r),
                window=window,
                full_causal=torch.isinf(window),
            )
        )
        actual = triton_screening_aggregate_fused(
            q,
            k,
            v,
            r=relevance_width_from_parameter(s_r),
            window=window,
            full_causal=torch.isinf(window),
            eps=1e-12,
            mipe_threshold=256.0,
            apply_tanh_norm=True,
        )

    assert torch.allclose(actual, expected, atol=2e-5, rtol=1e-6)


def test_triton_q_fused_aggregate_matches_preprocessed_aggregate_on_cuda() -> None:
    device = _cuda_device()
    torch.manual_seed(0)
    q = torch.randn(2, 4, 256, 16, device=device)
    k = torch.randn(2, 4, 256, 16, device=device)
    v = torch.randn(2, 4, 256, 32, device=device)
    positions = torch.arange(256, device=device)
    window = torch.tensor([2.0, 8.0, 128.0, 512.0], device=device)
    s_r = torch.zeros(4, device=device)

    with torch.no_grad():
        q_mipe = normalize_and_apply_mipe(q, positions, window, mipe_threshold=256.0)
        k_mipe = normalize_and_apply_mipe(k, positions, window, mipe_threshold=256.0)
        v_unit = normalize_unit(v)
        expected = triton_screening_aggregate(
            q_mipe,
            k_mipe,
            v_unit,
            r=relevance_width_from_parameter(s_r),
            window=window,
            full_causal=torch.isinf(window),
        )
        actual = triton_screening_aggregate_q_fused(
            q,
            k_mipe,
            v_unit,
            r=relevance_width_from_parameter(s_r),
            window=window,
            full_causal=torch.isinf(window),
            eps=1e-12,
            mipe_threshold=256.0,
        )

    assert torch.allclose(actual, expected, atol=2e-5, rtol=1e-6)


def test_triton_q_fused_aggregate_tanh_norm_matches_reference_on_cuda() -> None:
    device = _cuda_device()
    torch.manual_seed(0)
    q = torch.randn(2, 4, 256, 16, device=device)
    k = torch.randn(2, 4, 256, 16, device=device)
    v = torch.randn(2, 4, 256, 32, device=device)
    positions = torch.arange(256, device=device)
    window = torch.tensor([2.0, 8.0, 128.0, 512.0], device=device)
    s_r = torch.zeros(4, device=device)

    with torch.no_grad():
        q_mipe = normalize_and_apply_mipe(q, positions, window, mipe_threshold=256.0)
        k_mipe = normalize_and_apply_mipe(k, positions, window, mipe_threshold=256.0)
        v_unit = normalize_unit(v)
        expected = tanh_norm(
            triton_screening_aggregate(
                q_mipe,
                k_mipe,
                v_unit,
                r=relevance_width_from_parameter(s_r),
                window=window,
                full_causal=torch.isinf(window),
            )
        )
        actual = triton_screening_aggregate_q_fused(
            q,
            k_mipe,
            v_unit,
            r=relevance_width_from_parameter(s_r),
            window=window,
            full_causal=torch.isinf(window),
            eps=1e-12,
            mipe_threshold=256.0,
            apply_tanh_norm=True,
        )

    assert torch.allclose(actual, expected, atol=2e-5, rtol=1e-6)
