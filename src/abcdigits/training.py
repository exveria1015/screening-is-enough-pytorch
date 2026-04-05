"""Curriculum sampling and grid evaluation helpers for ABCDigits training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import PreTrainedTokenizerBase

from abcdigits.generator import ABCDigitsConfig
from abcdigits.task import (
    evaluate_abcdigits_exact_match,
    sample_tokenized_abcdigits_examples,
    sample_abcdigits_causal_lm_batch,
)
from abcdigits.tokenization import TokenizedABCDigitsExample, build_abcdigits_causal_lm_batch
from multiscreen.data import CausalLMBatch
from multiscreen.model import MultiscreenLM


def _randint(low: int, high: int, *, generator: torch.Generator | None) -> int:
    return int(torch.randint(low, high, (1,), generator=generator).item())


@dataclass(slots=True)
class ABCDigitsCurriculumConfig:
    """Sampling range for ABCDigits training batches."""

    min_num_equations: int = 26
    max_num_equations: int = 26
    depths: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9)
    digits_per_value: int = 6
    separator: str = " "
    add_eos: bool = True

    def __post_init__(self) -> None:
        if self.min_num_equations < 26:
            raise ValueError("min_num_equations must be at least 26")
        if self.max_num_equations < self.min_num_equations:
            raise ValueError("max_num_equations must be >= min_num_equations")
        if not self.depths:
            raise ValueError("depths must be non-empty")
        for depth in self.depths:
            if not 0.0 <= depth <= 1.0:
                raise ValueError("depths must all be in [0, 1]")
        if self.digits_per_value <= 0:
            raise ValueError("digits_per_value must be positive")
        if not self.separator:
            raise ValueError("separator must be non-empty")


@dataclass(slots=True)
class SampledABCDigitsBatch:
    """One sampled ABCDigits training batch and the config used to create it."""

    config: ABCDigitsConfig | None
    batch: CausalLMBatch
    tokenized_examples: tuple[TokenizedABCDigitsExample, ...]


@dataclass(slots=True)
class ABCDigitsTrainingPool:
    """Finite pool of tokenized ABCDigits examples for repeated sampling."""

    tokenized_examples: tuple[TokenizedABCDigitsExample, ...]

    def __post_init__(self) -> None:
        if not self.tokenized_examples:
            raise ValueError("tokenized_examples must be non-empty")


@dataclass(slots=True)
class ABCDigitsEvalCell:
    """One point on the ABCDigits evaluation grid."""

    num_equations: int
    depth: float
    tokenized_examples: tuple[TokenizedABCDigitsExample, ...]


@dataclass(slots=True)
class ABCDigitsEvalPoint:
    """Evaluated accuracy for one grid point."""

    num_equations: int
    depth: float
    accuracy: float
    digit_accuracy: float
    count: int
    unique_prediction_count: int
    unique_prediction_ratio: float


@dataclass(slots=True)
class ABCDigitsGridEvalResult:
    """Aggregate evaluation over an ABCDigits grid."""

    mean_accuracy: float
    mean_digit_accuracy: float
    mean_unique_prediction_ratio: float
    points: tuple[ABCDigitsEvalPoint, ...]


def estimate_abcdigits_max_token_length(
    *,
    num_equations: int,
    digits_per_value: int,
    separator: str = " ",
    add_eos: bool = True,
) -> int:
    """Safe upper bound for GPT-2 token length based on ASCII byte count."""

    if num_equations < 1:
        raise ValueError("num_equations must be positive")
    if digits_per_value <= 0:
        raise ValueError("digits_per_value must be positive")
    if not separator:
        raise ValueError("separator must be non-empty")

    equation_bytes = 2 + digits_per_value
    prompt_bytes = num_equations * equation_bytes + num_equations * len(separator) + 2
    completion_bytes = digits_per_value
    return prompt_bytes + completion_bytes + int(add_eos)


def sample_abcdigits_curriculum_config(
    curriculum: ABCDigitsCurriculumConfig,
    *,
    generator: torch.Generator | None = None,
) -> ABCDigitsConfig:
    """Sample one ABCDigitsConfig from the requested curriculum."""

    num_equations = _randint(curriculum.min_num_equations, curriculum.max_num_equations + 1, generator=generator)
    depth = curriculum.depths[_randint(0, len(curriculum.depths), generator=generator)]
    return ABCDigitsConfig(
        num_equations=num_equations,
        depth=depth,
        digits_per_value=curriculum.digits_per_value,
        separator=curriculum.separator,
    )


def sample_abcdigits_training_batch(
    *,
    batch_size: int,
    curriculum: ABCDigitsCurriculumConfig,
    tokenizer: PreTrainedTokenizerBase,
    generator: torch.Generator | None = None,
    supervise_completion_only: bool = True,
    ignore_index: int = -100,
) -> SampledABCDigitsBatch:
    """Sample one ABCDigits training batch from the curriculum."""

    config = sample_abcdigits_curriculum_config(curriculum, generator=generator)
    batch, tokenized_examples = sample_abcdigits_causal_lm_batch(
        batch_size=batch_size,
        config=config,
        tokenizer=tokenizer,
        generator=generator,
        add_eos=curriculum.add_eos,
        supervise_completion_only=supervise_completion_only,
        ignore_index=ignore_index,
    )
    return SampledABCDigitsBatch(
        config=config,
        batch=batch,
        tokenized_examples=tokenized_examples,
    )


def build_abcdigits_training_pool(
    *,
    pool_size: int,
    curriculum: ABCDigitsCurriculumConfig,
    tokenizer: PreTrainedTokenizerBase,
    generator: torch.Generator | None = None,
) -> ABCDigitsTrainingPool:
    """Build a finite pool of tokenized ABCDigits examples."""

    if pool_size <= 0:
        raise ValueError("pool_size must be positive")

    tokenized_examples: list[TokenizedABCDigitsExample] = []
    for _ in range(pool_size):
        config = sample_abcdigits_curriculum_config(curriculum, generator=generator)
        tokenized_examples.extend(
            sample_tokenized_abcdigits_examples(
                batch_size=1,
                config=config,
                tokenizer=tokenizer,
                generator=generator,
                add_eos=curriculum.add_eos,
            )
        )
    return ABCDigitsTrainingPool(tokenized_examples=tuple(tokenized_examples))


def sample_abcdigits_training_batch_from_pool(
    *,
    batch_size: int,
    training_pool: ABCDigitsTrainingPool,
    tokenizer: PreTrainedTokenizerBase,
    generator: torch.Generator | None = None,
    supervise_completion_only: bool = True,
    ignore_index: int = -100,
) -> SampledABCDigitsBatch:
    """Sample one training batch from a fixed pool of tokenized examples."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    indices = torch.randint(0, len(training_pool.tokenized_examples), (batch_size,), generator=generator)
    tokenized_examples = tuple(training_pool.tokenized_examples[int(index)] for index in indices.tolist())
    batch = build_abcdigits_causal_lm_batch(
        tokenized_examples,
        pad_token_id=int(tokenizer.pad_token_id),
        supervise_completion_only=supervise_completion_only,
        ignore_index=ignore_index,
    )
    first_config = tokenized_examples[0].example.config
    shared_config = first_config if all(example.example.config == first_config for example in tokenized_examples) else None
    return SampledABCDigitsBatch(
        config=shared_config,
        batch=batch,
        tokenized_examples=tokenized_examples,
    )


def build_abcdigits_eval_suite(
    *,
    num_equations_values: tuple[int, ...],
    depths: tuple[float, ...],
    examples_per_cell: int,
    tokenizer: PreTrainedTokenizerBase,
    digits_per_value: int = 6,
    separator: str = " ",
    generator: torch.Generator | None = None,
    add_eos: bool = False,
) -> tuple[ABCDigitsEvalCell, ...]:
    """Pre-sample a fixed ABCDigits evaluation grid."""

    if not num_equations_values:
        raise ValueError("num_equations_values must be non-empty")
    if not depths:
        raise ValueError("depths must be non-empty")
    if examples_per_cell <= 0:
        raise ValueError("examples_per_cell must be positive")

    cells: list[ABCDigitsEvalCell] = []
    for num_equations in num_equations_values:
        for depth in depths:
            config = ABCDigitsConfig(
                num_equations=num_equations,
                depth=depth,
                digits_per_value=digits_per_value,
                separator=separator,
            )
            _, tokenized_examples = sample_abcdigits_causal_lm_batch(
                batch_size=examples_per_cell,
                config=config,
                tokenizer=tokenizer,
                generator=generator,
                add_eos=add_eos,
                supervise_completion_only=True,
            )
            cells.append(
                ABCDigitsEvalCell(
                    num_equations=num_equations,
                    depth=depth,
                    tokenized_examples=tokenized_examples,
                )
            )
    return tuple(cells)


def evaluate_abcdigits_grid(
    model: MultiscreenLM,
    tokenizer: PreTrainedTokenizerBase,
    eval_suite: tuple[ABCDigitsEvalCell, ...] | list[ABCDigitsEvalCell],
) -> ABCDigitsGridEvalResult:
    """Evaluate exact-match accuracy over a fixed ABCDigits grid."""

    if not eval_suite:
        raise ValueError("eval_suite must be non-empty")

    points: list[ABCDigitsEvalPoint] = []
    weighted_correct = 0.0
    weighted_digit_correct = 0.0
    weighted_count = 0
    weighted_unique_prediction_ratio = 0.0
    for cell in eval_suite:
        result = evaluate_abcdigits_exact_match(model, tokenizer, cell.tokenized_examples)
        unique_prediction_ratio = result.unique_prediction_count / float(result.count)
        points.append(
            ABCDigitsEvalPoint(
                num_equations=cell.num_equations,
                depth=cell.depth,
                accuracy=result.accuracy,
                digit_accuracy=result.digit_accuracy,
                count=result.count,
                unique_prediction_count=result.unique_prediction_count,
                unique_prediction_ratio=unique_prediction_ratio,
            )
        )
        weighted_correct += result.accuracy * result.count
        weighted_digit_correct += result.digit_accuracy * result.count
        weighted_count += result.count
        weighted_unique_prediction_ratio += unique_prediction_ratio * result.count
    return ABCDigitsGridEvalResult(
        mean_accuracy=weighted_correct / float(weighted_count),
        mean_digit_accuracy=weighted_digit_correct / float(weighted_count),
        mean_unique_prediction_ratio=weighted_unique_prediction_ratio / float(weighted_count),
        points=tuple(points),
    )
