from __future__ import annotations

import torch

from multiscreen.generation import sample_next_token, truncate_prompt_tokens


def test_truncate_prompt_tokens_keeps_tail_within_context() -> None:
    prompt = torch.arange(10, dtype=torch.long)
    truncated = truncate_prompt_tokens(prompt, max_seq_len=4)
    assert torch.equal(truncated, torch.tensor([6, 7, 8, 9], dtype=torch.long))


def test_sample_next_token_greedy_returns_argmax() -> None:
    logits = torch.tensor([[0.0, 1.0, 3.0, 2.0]], dtype=torch.float32)
    sampled = sample_next_token(logits, greedy=True)
    assert torch.equal(sampled, torch.tensor([2], dtype=torch.long))


def test_sample_next_token_top_k_limits_candidates() -> None:
    logits = torch.tensor([[10.0, 9.0, -10.0, -11.0]], dtype=torch.float32)
    generator = torch.Generator().manual_seed(0)
    sampled = sample_next_token(logits, temperature=1.0, top_k=2, generator=generator)
    assert int(sampled.item()) in {0, 1}
