from __future__ import annotations

from multiscreen.config import MultiscreenConfig
from multiscreen.model import MultiscreenLM
from multiscreen.sizing import (
    estimate_multiscreen_size_from_token_count,
    multiscreen_parameter_count,
    multiscreen_parameter_count_from_dimensions,
    multiscreen_parameter_count_from_psi,
)


def test_multiscreen_parameter_count_matches_model_parameters() -> None:
    config = MultiscreenConfig(vocab_size=32, d_model=9, n_layers=3, n_heads=3, d_key=4, d_value=5)
    model = MultiscreenLM(config)
    expected = sum(parameter.numel() for parameter in model.parameters())
    assert multiscreen_parameter_count(config) == expected
    assert (
        multiscreen_parameter_count_from_dimensions(
            vocab_size=32,
            d_model=9,
            n_layers=3,
            n_heads=3,
            d_key=4,
            d_value=5,
        )
        == expected
    )


def test_multiscreen_parameter_count_from_psi_matches_from_psi_config() -> None:
    config = MultiscreenConfig.from_psi(psi=4, vocab_size=64, d_key=16, d_value=32)
    assert multiscreen_parameter_count_from_psi(4, vocab_size=64, d_key=16, d_value=32) == multiscreen_parameter_count(config)


def test_estimate_multiscreen_size_from_token_count_selects_nearest_psi() -> None:
    psi_10_params = multiscreen_parameter_count_from_psi(10, vocab_size=50_257, d_key=16, d_value=64)
    token_count = psi_10_params * 20
    estimate = estimate_multiscreen_size_from_token_count(
        token_count,
        vocab_size=50_257,
        tokens_per_parameter=20.0,
        d_key=16,
        d_value=64,
        min_psi=1,
        max_psi=16,
    )
    assert estimate.recommended_psi == 10
    assert estimate.recommended_parameter_count == psi_10_params
    assert estimate.target_parameter_count == psi_10_params
