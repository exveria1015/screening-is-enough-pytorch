"""Minimal data helpers for causal language-model training."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class CausalLMBatch:
    """Input-label pair for next-token prediction."""

    input_ids: torch.Tensor
    labels: torch.Tensor


def causal_lm_batch_from_token_block(tokens: torch.Tensor) -> CausalLMBatch:
    """Convert a `[batch, seq + 1]` token block into next-token inputs and labels."""

    if tokens.ndim != 2:
        raise ValueError("tokens must have shape [batch, seq_plus_one]")
    if tokens.shape[1] < 2:
        raise ValueError("tokens must have at least two positions")
    tokens = tokens.to(dtype=torch.long)
    return CausalLMBatch(input_ids=tokens[:, :-1], labels=tokens[:, 1:])


def sample_token_blocks(
    token_ids: torch.Tensor,
    *,
    batch_size: int,
    block_size: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample contiguous token blocks from a 1D token stream."""

    if token_ids.ndim != 1:
        raise ValueError("token_ids must have shape [tokens]")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if block_size < 2:
        raise ValueError("block_size must be at least 2")
    if token_ids.numel() < block_size:
        raise ValueError("token stream is shorter than block_size")

    max_start = token_ids.numel() - block_size + 1
    starts = torch.randint(0, max_start, (batch_size,), generator=generator)
    blocks = [token_ids[start : start + block_size] for start in starts.tolist()]
    return torch.stack(blocks, dim=0)
