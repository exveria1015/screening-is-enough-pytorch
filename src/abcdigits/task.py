"""Training and evaluation helpers for ABCDigits."""

from __future__ import annotations

from dataclasses import dataclass
import inspect

import torch
from transformers import PreTrainedTokenizerBase

from abcdigits.generator import ABCDigitsConfig, build_abcdigits_example
from abcdigits.tokenization import (
    TokenizedABCDigitsExample,
    build_abcdigits_causal_lm_batch,
    tokenize_abcdigits_example,
)
from multiscreen.data import CausalLMBatch
from multiscreen.model import MultiscreenLM
from multiscreen.train import model_device


def _supports_return_relevances_kwarg(model: torch.nn.Module) -> bool:
    try:
        signature = inspect.signature(model.forward)
    except (TypeError, ValueError):
        return False
    return "return_relevances" in signature.parameters


def _forward_model_for_logits(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    inference: bool,
) -> tuple[torch.Tensor, object]:
    if _supports_return_relevances_kwarg(model):
        return model(input_ids, inference=inference, return_relevances=False)
    return model(input_ids, inference=inference)


@dataclass(slots=True)
class ABCDigitsEvalResult:
    """Exact-match evaluation summary for ABCDigits completion."""

    accuracy: float
    digit_accuracy: float
    count: int
    unique_prediction_count: int
    predictions: tuple[str, ...]
    targets: tuple[str, ...]


def sample_tokenized_abcdigits_examples(
    *,
    batch_size: int,
    config: ABCDigitsConfig,
    tokenizer: PreTrainedTokenizerBase,
    generator: torch.Generator | None = None,
    add_eos: bool = True,
) -> tuple[TokenizedABCDigitsExample, ...]:
    """Sample and tokenize a batch of ABCDigits examples."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return tuple(
        tokenize_abcdigits_example(
            build_abcdigits_example(config, generator=generator),
            tokenizer,
            add_eos=add_eos,
        )
        for _ in range(batch_size)
    )


def sample_abcdigits_causal_lm_batch(
    *,
    batch_size: int,
    config: ABCDigitsConfig,
    tokenizer: PreTrainedTokenizerBase,
    generator: torch.Generator | None = None,
    add_eos: bool = True,
    supervise_completion_only: bool = True,
    ignore_index: int = -100,
) -> tuple[CausalLMBatch, tuple[TokenizedABCDigitsExample, ...]]:
    """Sample tokenized ABCDigits examples and pad them into a training batch."""

    tokenized_examples = sample_tokenized_abcdigits_examples(
        batch_size=batch_size,
        config=config,
        tokenizer=tokenizer,
        generator=generator,
        add_eos=add_eos,
    )
    batch = build_abcdigits_causal_lm_batch(
        tokenized_examples,
        pad_token_id=int(tokenizer.pad_token_id),
        supervise_completion_only=supervise_completion_only,
        ignore_index=ignore_index,
    )
    return batch, tokenized_examples


@torch.no_grad()
def greedy_decode_completion(
    model: MultiscreenLM,
    prompt_ids: tuple[int, ...] | list[int],
    *,
    max_new_tokens: int,
) -> tuple[int, ...]:
    """Greedily decode a fixed number of tokens from a prompt."""

    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if len(prompt_ids) == 0:
        raise ValueError("prompt_ids must be non-empty")

    device = model_device(model)
    generated = torch.tensor([list(prompt_ids)], dtype=torch.long, device=device)
    predicted: list[int] = []
    model.eval()
    for _ in range(max_new_tokens):
        if generated.shape[1] > model.config.max_seq_len:
            raise ValueError("prompt plus generated tokens exceed model.config.max_seq_len")
        logits, _ = _forward_model_for_logits(model, generated, inference=True)
        next_token = int(torch.argmax(logits[0, -1]).item())
        predicted.append(next_token)
        next_token_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)
        generated = torch.cat((generated, next_token_tensor), dim=1)
    return tuple(predicted)


@torch.no_grad()
def greedy_decode_completion_text(
    model: MultiscreenLM,
    prompt_ids: tuple[int, ...] | list[int],
    tokenizer: PreTrainedTokenizerBase,
    *,
    target_text: str,
    max_new_tokens: int | None = None,
) -> str:
    """Greedily decode until the rendered completion reaches the target text length."""

    if not target_text:
        raise ValueError("target_text must be non-empty")

    device = model_device(model)
    generated = torch.tensor([list(prompt_ids)], dtype=torch.long, device=device)
    predicted: list[int] = []
    decoded = ""
    limit = max_new_tokens if max_new_tokens is not None else len(target_text)
    if limit <= 0:
        raise ValueError("max_new_tokens must be positive")

    model.eval()
    for _ in range(limit):
        if generated.shape[1] > model.config.max_seq_len:
            raise ValueError("prompt plus generated tokens exceed model.config.max_seq_len")
        logits, _ = _forward_model_for_logits(model, generated, inference=True)
        next_token = int(torch.argmax(logits[0, -1]).item())
        predicted.append(next_token)
        decoded = tokenizer.decode(predicted, clean_up_tokenization_spaces=False)
        next_token_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)
        generated = torch.cat((generated, next_token_tensor), dim=1)
        if len(decoded) >= len(target_text):
            break
    return decoded


@torch.no_grad()
def evaluate_abcdigits_exact_match(
    model: MultiscreenLM,
    tokenizer: PreTrainedTokenizerBase,
    tokenized_examples: list[TokenizedABCDigitsExample] | tuple[TokenizedABCDigitsExample, ...],
) -> ABCDigitsEvalResult:
    """Evaluate exact-match completion accuracy on tokenized ABCDigits examples."""

    if not tokenized_examples:
        raise ValueError("tokenized_examples must be non-empty")

    predictions: list[str] = []
    targets: list[str] = []
    correct = 0
    correct_digits = 0
    total_digits = 0
    for example in tokenized_examples:
        target = example.example.completion
        prediction = greedy_decode_completion_text(
            model,
            example.prompt_ids,
            tokenizer,
            target_text=target,
        )
        predictions.append(prediction)
        targets.append(target)
        correct += int(prediction == target)
        correct_digits += sum(
            int(predicted_digit == target_digit)
            for predicted_digit, target_digit in zip(prediction[: len(target)], target)
        )
        total_digits += len(target)
    return ABCDigitsEvalResult(
        accuracy=correct / float(len(tokenized_examples)),
        digit_accuracy=correct_digits / float(total_digits),
        count=len(tokenized_examples),
        unique_prediction_count=len(set(predictions)),
        predictions=tuple(predictions),
        targets=tuple(targets),
    )
