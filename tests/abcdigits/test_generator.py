from __future__ import annotations

import math

import pytest
import torch

from abcdigits import (
    ABCDigitsConfig,
    build_abcdigits_example,
    render_abcdigits_prompt,
    resolve_target_equation_index,
)


def test_config_requires_at_least_26_equations() -> None:
    with pytest.raises(ValueError, match="num_equations must be at least 26"):
        ABCDigitsConfig(num_equations=25, depth=0.5)


def test_config_requires_enough_digit_space_for_unique_values() -> None:
    with pytest.raises(ValueError, match="digits_per_value is too small"):
        ABCDigitsConfig(num_equations=26, depth=0.5, digits_per_value=1)


def test_resolve_target_equation_index_tracks_requested_depth() -> None:
    assert resolve_target_equation_index(26, 0.0) == 0
    assert resolve_target_equation_index(26, 1.0) == 25
    assert resolve_target_equation_index(26, 0.5) == 12


def test_render_abcdigits_prompt_appends_query_suffix() -> None:
    prompt = render_abcdigits_prompt(["A=123", "B=456"], target_letter="C", separator=" ")
    assert prompt == "A=123 B=456 C="


def test_example_has_fixed_26_keys_and_unique_target_equation() -> None:
    example = build_abcdigits_example(ABCDigitsConfig(num_equations=40, depth=0.5), generator=torch.Generator().manual_seed(0))
    assert len(example.letter_to_value) == 26
    assert len(set(example.letter_to_value)) == 26
    assert len(set(example.letter_to_value.values())) == 26
    assert example.equations.count(example.target_equation) == 1
    assert example.completion == example.target_value


def test_every_non_target_letter_appears_at_least_once() -> None:
    example = build_abcdigits_example(ABCDigitsConfig(num_equations=40, depth=0.5), generator=torch.Generator().manual_seed(1))
    counts = example.counts_by_letter
    assert counts[example.target_letter] == 1
    for letter, count in counts.items():
        if letter != example.target_letter:
            assert count >= 1


def test_non_target_weights_form_a_randomized_power_of_two_permutation() -> None:
    example = build_abcdigits_example(ABCDigitsConfig(num_equations=40, depth=0.5), generator=torch.Generator().manual_seed(2))
    weights = sorted(example.non_target_weight_by_letter.values())
    expected = [2**rank for rank in range(25)]
    assert weights == expected


def test_target_is_inserted_close_to_requested_depth() -> None:
    config = ABCDigitsConfig(num_equations=101, depth=0.7)
    example = build_abcdigits_example(config, generator=torch.Generator().manual_seed(3))
    assert example.realized_depth == pytest.approx(0.7, abs=1.0 / (config.num_equations - 1))


def test_rendered_prompt_ends_with_target_query_and_uses_separator() -> None:
    config = ABCDigitsConfig(num_equations=30, depth=0.3, separator="\n")
    example = build_abcdigits_example(config, generator=torch.Generator().manual_seed(4))
    assert example.prompt.endswith(f"\n{example.target_letter}=")
    assert example.prompt.count("\n") == config.num_equations


def test_extra_non_target_sampling_fills_requested_equation_budget() -> None:
    config = ABCDigitsConfig(num_equations=200, depth=0.9)
    example = build_abcdigits_example(config, generator=torch.Generator().manual_seed(5))
    assert len(example.equations) == config.num_equations
    assert sum(example.counts_by_letter.values()) == config.num_equations
