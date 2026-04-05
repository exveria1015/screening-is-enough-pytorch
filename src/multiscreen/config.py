"""Typed configuration for the Multiscreen architecture."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class MultiscreenConfig:
    """Model hyperparameters following the paper's default scaling rule."""

    vocab_size: int = 50_257
    d_model: int = 64
    n_layers: int = 8
    n_heads: int = 8
    d_key: int = 16
    d_value: int = 64
    mipe_threshold: float = 256.0
    max_seq_len: int = 4096
    max_train_seq_len: int = 4096
    eps: float = 1e-12

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.n_layers <= 0:
            raise ValueError("n_layers must be positive")
        if self.n_heads <= 0:
            raise ValueError("n_heads must be positive")
        if self.d_key < 2:
            raise ValueError("d_key must be at least 2 for MiPE rotation")
        if self.d_value <= 0:
            raise ValueError("d_value must be positive")
        if self.mipe_threshold <= 0.0:
            raise ValueError("mipe_threshold must be positive")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if self.max_train_seq_len <= 0:
            raise ValueError("max_train_seq_len must be positive")

    @classmethod
    def from_psi(
        cls,
        psi: int,
        *,
        vocab_size: int = 50_257,
        d_key: int = 16,
        d_value: int = 64,
        mipe_threshold: float = 256.0,
        max_seq_len: int = 4096,
        max_train_seq_len: int = 4096,
        eps: float = 1e-12,
    ) -> "MultiscreenConfig":
        """Build the default paper scaling where N_L = N_H = psi and d_E = psi^2."""

        if psi <= 0:
            raise ValueError("psi must be positive")
        return cls(
            vocab_size=vocab_size,
            d_model=psi * psi,
            n_layers=psi,
            n_heads=psi,
            d_key=d_key,
            d_value=d_value,
            mipe_threshold=mipe_threshold,
            max_seq_len=max_seq_len,
            max_train_seq_len=max_train_seq_len,
            eps=eps,
        )
