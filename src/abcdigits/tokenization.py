"""GPT-2 tokenization helpers for ABCDigits."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from abcdigits.generator import ABCDigitsExample
from multiscreen.data import CausalLMBatch, causal_lm_batch_from_token_block


@dataclass(slots=True)
class TokenizedABCDigitsExample:
    """Tokenized prompt/completion pair for ABCDigits evaluation or training."""

    example: ABCDigitsExample
    prompt_ids: tuple[int, ...]
    completion_ids: tuple[int, ...]
    full_ids: tuple[int, ...]

    @property
    def prompt_length(self) -> int:
        return len(self.prompt_ids)

    @property
    def completion_length(self) -> int:
        return len(self.completion_ids)


def build_gpt2_tokenizer(
    *,
    model_name: str = "gpt2",
    use_fast: bool = True,
    local_files_only: bool = False,
) -> PreTrainedTokenizerBase:
    """Load the GPT-2 tokenizer used by the paper."""

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast, local_files_only=local_files_only)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise ValueError("GPT-2 tokenizer must expose an eos token to derive a pad token")
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def tokenize_abcdigits_example(
    example: ABCDigitsExample,
    tokenizer: PreTrainedTokenizerBase,
    *,
    add_eos: bool = False,
) -> TokenizedABCDigitsExample:
    """Tokenize prompt and completion, preserving the prompt-token prefix."""

    prompt_ids = tokenizer.encode(example.prompt, add_special_tokens=False)
    full_text = example.prompt + example.completion
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("prompt tokenization is not a prefix of prompt+completion tokenization")
    completion_ids = full_ids[len(prompt_ids) :]
    if not completion_ids:
        raise ValueError("completion tokenization is empty")
    if add_eos:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer must provide eos_token_id when add_eos=True")
        full_ids = full_ids + [int(tokenizer.eos_token_id)]
    return TokenizedABCDigitsExample(
        example=example,
        prompt_ids=tuple(int(token_id) for token_id in prompt_ids),
        completion_ids=tuple(int(token_id) for token_id in completion_ids),
        full_ids=tuple(int(token_id) for token_id in full_ids),
    )


def build_abcdigits_token_block(
    tokenized_example: TokenizedABCDigitsExample,
) -> torch.Tensor:
    """Build a single token block from one tokenized ABCDigits example."""

    return torch.tensor(tokenized_example.full_ids, dtype=torch.long)


def build_abcdigits_causal_lm_batch(
    tokenized_examples: list[TokenizedABCDigitsExample] | tuple[TokenizedABCDigitsExample, ...],
    *,
    pad_token_id: int,
    supervise_completion_only: bool = False,
    ignore_index: int = -100,
) -> CausalLMBatch:
    """Pad tokenized ABCDigits examples into a causal-LM batch."""

    if not tokenized_examples:
        raise ValueError("tokenized_examples must be non-empty")

    max_length = max(len(example.full_ids) for example in tokenized_examples)
    padded = torch.full((len(tokenized_examples), max_length), pad_token_id, dtype=torch.long)
    for row, example in enumerate(tokenized_examples):
        padded[row, : len(example.full_ids)] = torch.tensor(example.full_ids, dtype=torch.long)

    batch = causal_lm_batch_from_token_block(padded)
    labels = batch.labels.clone()

    for row, example in enumerate(tokenized_examples):
        valid_label_length = len(example.full_ids) - 1
        if valid_label_length < labels.shape[1]:
            labels[row, valid_label_length:] = ignore_index
        if supervise_completion_only:
            prompt_only_label_count = max(example.prompt_length - 1, 0)
            labels[row, :prompt_only_label_count] = ignore_index

    return CausalLMBatch(input_ids=batch.input_ids, labels=labels)
