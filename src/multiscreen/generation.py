"""Autoregressive generation helpers for Multiscreen checkpoints."""

from __future__ import annotations

import torch

from multiscreen.model import MultiscreenLM


def truncate_prompt_tokens(token_ids: torch.Tensor, *, max_seq_len: int) -> torch.Tensor:
    """Keep the most recent prompt tokens that fit inside the model context."""

    if token_ids.ndim != 1:
        raise ValueError("token_ids must have shape [tokens]")
    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive")
    if token_ids.numel() <= max_seq_len:
        return token_ids
    return token_ids[-max_seq_len:]


def sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    greedy: bool = False,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample one token id from next-token logits."""

    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch, vocab]")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive when provided")

    if greedy:
        return torch.argmax(logits, dim=-1)

    scaled_logits = logits / temperature
    if top_k is not None and top_k < scaled_logits.shape[-1]:
        top_values, top_indices = torch.topk(scaled_logits, k=top_k, dim=-1)
        probs = torch.softmax(top_values, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)
        return top_indices.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)

    probs = torch.softmax(scaled_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)


@torch.no_grad()
def generate_tokens(
    model: MultiscreenLM,
    prompt_token_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    greedy: bool = False,
    eos_token_id: int | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Generate continuation tokens from a 1D prompt token tensor."""

    if prompt_token_ids.ndim != 1:
        raise ValueError("prompt_token_ids must have shape [tokens]")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")

    device = next(model.parameters()).device
    generated = truncate_prompt_tokens(prompt_token_ids.to(device=device, dtype=torch.long), max_seq_len=model.config.max_seq_len)

    model.eval()
    for _ in range(max_new_tokens):
        input_ids = generated.unsqueeze(0)
        logits, _ = model(input_ids, inference=True, return_relevances=False)
        next_token = sample_next_token(
            logits[:, -1, :],
            temperature=temperature,
            top_k=top_k,
            greedy=greedy,
            generator=generator,
        )
        generated = torch.cat([generated, next_token], dim=0)
        if eos_token_id is not None and int(next_token.item()) == eos_token_id:
            break
        if generated.numel() > model.config.max_seq_len:
            generated = generated[-model.config.max_seq_len :]
    return generated
