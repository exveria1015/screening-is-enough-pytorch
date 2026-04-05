"""Model sizing helpers for Multiscreen."""

from __future__ import annotations

from dataclasses import dataclass

from multiscreen.config import MultiscreenConfig


@dataclass(slots=True)
class MultiscreenSizeEstimate:
    """Nearest paper-style psi configuration for a token budget."""

    token_count: int
    tokens_per_parameter: float
    target_parameter_count: int
    recommended_psi: int
    recommended_parameter_count: int
    recommended_tokens_per_parameter: float
    vocab_size: int
    d_key: int
    d_value: int
    smaller_psi: int | None = None
    smaller_parameter_count: int | None = None
    smaller_tokens_per_parameter: float | None = None
    larger_psi: int | None = None
    larger_parameter_count: int | None = None
    larger_tokens_per_parameter: float | None = None

    def build_config(
        self,
        *,
        max_seq_len: int,
        max_train_seq_len: int | None = None,
    ) -> MultiscreenConfig:
        """Materialize the recommended paper-style config for training."""

        train_seq_len = max_seq_len if max_train_seq_len is None else max_train_seq_len
        return MultiscreenConfig.from_psi(
            psi=self.recommended_psi,
            vocab_size=self.vocab_size,
            d_key=self.d_key,
            d_value=self.d_value,
            max_seq_len=max_seq_len,
            max_train_seq_len=train_seq_len,
        )


def multiscreen_parameter_count(config: MultiscreenConfig) -> int:
    """Return the exact parameter count for the current Multiscreen implementation."""

    return multiscreen_parameter_count_from_dimensions(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_key=config.d_key,
        d_value=config.d_value,
    )


def multiscreen_parameter_count_from_dimensions(
    *,
    vocab_size: int,
    d_model: int,
    n_layers: int,
    n_heads: int,
    d_key: int,
    d_value: int,
) -> int:
    """Return the exact parameter count for arbitrary Multiscreen dimensions."""

    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    if d_model <= 0:
        raise ValueError("d_model must be positive")
    if n_layers <= 0:
        raise ValueError("n_layers must be positive")
    if n_heads <= 0:
        raise ValueError("n_heads must be positive")
    if d_key <= 0:
        raise ValueError("d_key must be positive")
    if d_value <= 0:
        raise ValueError("d_value must be positive")

    per_tile = (2 * d_model * d_key) + (3 * d_model * d_value) + 3
    return (vocab_size * d_model) + (n_layers * n_heads * per_tile) + 2


def multiscreen_parameter_count_from_psi(
    psi: int,
    *,
    vocab_size: int = 50_257,
    d_key: int = 16,
    d_value: int = 64,
) -> int:
    """Return the exact parameter count for a paper-style psi configuration."""

    if psi <= 0:
        raise ValueError("psi must be positive")
    return multiscreen_parameter_count_from_dimensions(
        vocab_size=vocab_size,
        d_model=psi * psi,
        n_layers=psi,
        n_heads=psi,
        d_key=d_key,
        d_value=d_value,
    )


def estimate_multiscreen_size_from_token_count(
    token_count: int,
    *,
    vocab_size: int = 50_257,
    tokens_per_parameter: float = 20.0,
    d_key: int = 16,
    d_value: int = 64,
    min_psi: int = 1,
    max_psi: int = 64,
) -> MultiscreenSizeEstimate:
    """Map a corpus token budget to the nearest paper-style psi configuration."""

    if token_count <= 0:
        raise ValueError("token_count must be positive")
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    if tokens_per_parameter <= 0.0:
        raise ValueError("tokens_per_parameter must be positive")
    if min_psi <= 0:
        raise ValueError("min_psi must be positive")
    if max_psi < min_psi:
        raise ValueError("max_psi must be greater than or equal to min_psi")

    target_parameter_count = max(1, int(round(float(token_count) / tokens_per_parameter)))
    candidates = [
        (
            psi,
            multiscreen_parameter_count_from_psi(
                psi,
                vocab_size=vocab_size,
                d_key=d_key,
                d_value=d_value,
            ),
        )
        for psi in range(min_psi, max_psi + 1)
    ]
    recommended_psi, recommended_parameter_count = min(
        candidates,
        key=lambda item: (
            abs(item[1] - target_parameter_count),
            item[1] > target_parameter_count,
            item[0],
        ),
    )
    smaller = [(psi, count) for psi, count in candidates if count <= target_parameter_count]
    larger = [(psi, count) for psi, count in candidates if count >= target_parameter_count]
    smaller_psi, smaller_parameter_count = (smaller[-1] if smaller else (None, None))
    larger_psi, larger_parameter_count = (larger[0] if larger else (None, None))

    return MultiscreenSizeEstimate(
        token_count=token_count,
        tokens_per_parameter=float(tokens_per_parameter),
        target_parameter_count=target_parameter_count,
        recommended_psi=recommended_psi,
        recommended_parameter_count=recommended_parameter_count,
        recommended_tokens_per_parameter=float(token_count) / float(recommended_parameter_count),
        vocab_size=vocab_size,
        d_key=d_key,
        d_value=d_value,
        smaller_psi=smaller_psi,
        smaller_parameter_count=smaller_parameter_count,
        smaller_tokens_per_parameter=(
            None if smaller_parameter_count is None else float(token_count) / float(smaller_parameter_count)
        ),
        larger_psi=larger_psi,
        larger_parameter_count=larger_parameter_count,
        larger_tokens_per_parameter=(
            None if larger_parameter_count is None else float(token_count) / float(larger_parameter_count)
        ),
    )
