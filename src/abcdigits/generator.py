"""Generator for the ABCDigits synthetic retrieval benchmark."""

from __future__ import annotations

from dataclasses import dataclass
import string

import torch


def _randint(high: int, *, generator: torch.Generator | None) -> int:
    return int(torch.randint(0, high, (1,), generator=generator).item())


def _randperm(length: int, *, generator: torch.Generator | None) -> torch.Tensor:
    return torch.randperm(length, generator=generator)


def _sample_unique_digit_strings(
    num_values: int,
    num_digits: int,
    *,
    generator: torch.Generator | None,
) -> list[str]:
    space_size = 10**num_digits
    if num_values > space_size:
        raise ValueError("digit space is too small to assign unique values to all keys")

    values: list[str] = []
    seen: set[str] = set()
    while len(values) < num_values:
        remaining = num_values - len(values)
        sampled = torch.randint(0, space_size, (remaining * 2,), generator=generator)
        for raw_value in sampled.tolist():
            value = f"{int(raw_value):0{num_digits}d}"
            if value in seen:
                continue
            seen.add(value)
            values.append(value)
            if len(values) == num_values:
                break
    return values


def _equation(letter: str, value: str) -> str:
    return f"{letter}={value}"


@dataclass(slots=True)
class ABCDigitsConfig:
    """Configuration for generating one ABCDigits example."""

    num_equations: int
    depth: float
    digits_per_value: int = 6
    separator: str = " "
    alphabet: str = string.ascii_uppercase

    def __post_init__(self) -> None:
        if len(self.alphabet) != 26:
            raise ValueError("alphabet must contain exactly 26 uppercase letters")
        if self.num_equations < len(self.alphabet):
            raise ValueError("num_equations must be at least 26")
        if not 0.0 <= self.depth <= 1.0:
            raise ValueError("depth must be in [0, 1]")
        if self.digits_per_value <= 0:
            raise ValueError("digits_per_value must be positive")
        if 10**self.digits_per_value < len(self.alphabet):
            raise ValueError("digits_per_value is too small to assign unique values to 26 keys")
        if not self.separator:
            raise ValueError("separator must be non-empty")


@dataclass(slots=True)
class ABCDigitsExample:
    """One rendered ABCDigits retrieval example with metadata for validation."""

    config: ABCDigitsConfig
    prompt: str
    completion: str
    target_letter: str
    target_value: str
    target_equation: str
    target_equation_index: int
    equations: tuple[str, ...]
    letter_to_value: dict[str, str]
    non_target_weight_by_letter: dict[str, int]

    @property
    def realized_depth(self) -> float:
        if len(self.equations) == 1:
            return 0.0
        return self.target_equation_index / float(len(self.equations) - 1)

    @property
    def query(self) -> str:
        return f"{self.target_letter}="

    @property
    def counts_by_letter(self) -> dict[str, int]:
        counts = {letter: 0 for letter in self.letter_to_value}
        for equation in self.equations:
            counts[equation[0]] += 1
        return counts


def resolve_target_equation_index(num_equations: int, depth: float) -> int:
    """Map a requested depth in [0, 1] to a valid equation index."""

    if num_equations <= 0:
        raise ValueError("num_equations must be positive")
    if not 0.0 <= depth <= 1.0:
        raise ValueError("depth must be in [0, 1]")
    if num_equations == 1:
        return 0
    return min(int(round(depth * (num_equations - 1))), num_equations - 1)


def render_abcdigits_prompt(
    equations: list[str] | tuple[str, ...],
    *,
    target_letter: str,
    separator: str,
) -> str:
    """Render the context equations followed by the query suffix."""

    body = separator.join(equations)
    return f"{body}{separator}{target_letter}="


def build_abcdigits_example(
    config: ABCDigitsConfig,
    *,
    generator: torch.Generator | None = None,
) -> ABCDigitsExample:
    """Build a single ABCDigits example following the paper's generation recipe."""

    letters = list(config.alphabet)
    unique_values = _sample_unique_digit_strings(len(letters), config.digits_per_value, generator=generator)
    values = {letter: value for letter, value in zip(letters, unique_values)}
    target_letter = letters[_randint(len(letters), generator=generator)]
    target_value = values[target_letter]
    target_equation = _equation(target_letter, target_value)

    non_target_letters = [letter for letter in letters if letter != target_letter]
    base_non_target_equations = [_equation(letter, values[letter]) for letter in non_target_letters]

    shuffled_letters = [non_target_letters[index] for index in _randperm(len(non_target_letters), generator=generator).tolist()]
    weights = {letter: 2**rank for rank, letter in enumerate(shuffled_letters)}

    extra_non_target_count = config.num_equations - len(letters)
    sampled_non_target_equations: list[str] = []
    if extra_non_target_count > 0:
        letters_for_sampling = shuffled_letters
        probabilities = torch.tensor([float(weights[letter]) for letter in letters_for_sampling], dtype=torch.float64)
        sampled_indices = torch.multinomial(
            probabilities,
            num_samples=extra_non_target_count,
            replacement=True,
            generator=generator,
        )
        sampled_non_target_equations = [
            _equation(letters_for_sampling[int(index)], values[letters_for_sampling[int(index)]])
            for index in sampled_indices.tolist()
        ]

    non_target_equations = base_non_target_equations + sampled_non_target_equations
    shuffled_non_target_equations = [
        non_target_equations[index] for index in _randperm(len(non_target_equations), generator=generator).tolist()
    ]

    target_equation_index = resolve_target_equation_index(config.num_equations, config.depth)
    equations = list(shuffled_non_target_equations)
    equations.insert(target_equation_index, target_equation)
    prompt = render_abcdigits_prompt(equations, target_letter=target_letter, separator=config.separator)

    return ABCDigitsExample(
        config=config,
        prompt=prompt,
        completion=target_value,
        target_letter=target_letter,
        target_value=target_value,
        target_equation=target_equation,
        target_equation_index=target_equation_index,
        equations=tuple(equations),
        letter_to_value=values,
        non_target_weight_by_letter=weights,
    )
