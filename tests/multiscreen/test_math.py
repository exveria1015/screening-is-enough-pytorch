from __future__ import annotations

import math

import torch

from multiscreen.math import (
    TanhNorm,
    apply_mipe,
    build_mipe_rotation,
    build_softmask,
    normalize_and_apply_mipe,
    normalize_unit,
    trim_and_square,
    window_from_parameter,
)


def test_trim_and_square_matches_paper_threshold() -> None:
    similarity = torch.tensor([1.0, 0.5, 0.49, -1.0], dtype=torch.float32)
    s_r = torch.tensor(0.0, dtype=torch.float32)
    actual = trim_and_square(similarity, s_r)
    expected = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_softmask_uses_cosine_taper_inside_window() -> None:
    query_positions = torch.tensor([2], dtype=torch.long)
    key_positions = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    window = torch.tensor(4.0)
    actual = build_softmask(query_positions, key_positions, window, dtype=torch.float32)
    expected = torch.tensor(
        [[0.5 * (math.cos(-math.pi / 2.0) + 1.0), 0.5 * (math.cos(-math.pi / 4.0) + 1.0), 1.0, 0.0]],
        dtype=torch.float32,
    )
    assert torch.allclose(actual, expected, atol=1e-6)


def test_mipe_becomes_identity_at_or_above_threshold() -> None:
    x = torch.tensor([[[1.0, 2.0, 3.0]]], dtype=torch.float32)
    positions = torch.tensor([0], dtype=torch.long)
    window = torch.tensor(256.0)
    actual = apply_mipe(x, positions, window, mipe_threshold=256.0)
    assert torch.allclose(actual, x, atol=1e-6)


def test_mipe_matches_closed_form_rotation_on_first_two_coordinates() -> None:
    x = torch.tensor([[[1.0, 0.0, 7.0]]], dtype=torch.float32)
    positions = torch.tensor([1], dtype=torch.long)
    window = torch.tensor(2.0, dtype=torch.float32)
    actual = apply_mipe(x, positions, window, mipe_threshold=4.0)
    angle = math.pi / 4.0
    expected = torch.tensor([[[math.cos(angle), math.sin(angle), 7.0]]], dtype=torch.float32)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_mipe_leaves_dimensions_after_the_first_two_unchanged() -> None:
    x = torch.tensor([[[1.0, 0.0, 5.0, -3.0, 2.0]]], dtype=torch.float32)
    positions = torch.tensor([1], dtype=torch.long)
    window = torch.tensor(2.0, dtype=torch.float32)
    actual = apply_mipe(x, positions, window, mipe_threshold=4.0)
    assert torch.allclose(actual[..., 2:], x[..., 2:], atol=1e-6)


def test_normalize_and_apply_mipe_matches_explicit_composition() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 3, 5, 4, dtype=torch.float32)
    positions = torch.arange(5, dtype=torch.long)
    window = torch.tensor([2.0, 4.0, 256.0], dtype=torch.float32)
    rotation = build_mipe_rotation(positions, window, mipe_threshold=256.0, dtype=x.dtype, device=x.device)

    expected = apply_mipe(normalize_unit(x), positions, window, mipe_threshold=256.0)
    actual = normalize_and_apply_mipe(x, positions, window, mipe_threshold=256.0, rotation=rotation)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_window_parameter_matches_equation() -> None:
    s_w = torch.tensor(math.log(3.0), dtype=torch.float32)
    actual = window_from_parameter(s_w)
    assert torch.allclose(actual, torch.tensor(4.0), atol=1e-6)


def test_tanhnorm_caps_output_norm_by_one() -> None:
    x = torch.tensor([[3.0, 4.0]], dtype=torch.float32)
    y = TanhNorm()(x)
    assert float(torch.linalg.vector_norm(y, dim=-1).max()) <= 1.0


def test_normalized_dot_product_is_bounded_without_relying_on_clamp() -> None:
    torch.manual_seed(0)
    q = torch.randn(2, 3, 5, dtype=torch.float32)
    k = torch.randn(2, 3, 5, dtype=torch.float32)
    q_unit = normalize_unit(q)
    k_unit = normalize_unit(k)
    similarity = torch.einsum("bqd,bkd->bqk", q_unit, k_unit)
    assert float(similarity.max()) <= 1.0 + 1e-6
    assert float(similarity.min()) >= -1.0 - 1e-6
