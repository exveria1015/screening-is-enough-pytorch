from __future__ import annotations

import torch

from abcdigits import (
    ABCDigitsConfig,
    ABCDigitsExample,
    TokenizedABCDigitsExample,
    build_gpt2_tokenizer,
    evaluate_abcdigits_exact_match,
    greedy_decode_completion,
    sample_abcdigits_causal_lm_batch,
    sample_tokenized_abcdigits_examples,
)
from multiscreen.config import MultiscreenConfig
from multiscreen.model import MultiscreenLM
from multiscreen.train import OptimizerConfig, build_optimizer, evaluate_loss, train_step


class StepwiseMockModel(torch.nn.Module):
    def __init__(self, vocab_size: int, scheduled_tokens: list[int], *, max_seq_len: int = 4096) -> None:
        super().__init__()
        self.config = type("Config", (), {"max_seq_len": max_seq_len})()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.vocab_size = vocab_size
        self.scheduled_tokens = scheduled_tokens

    def forward(self, input_ids: torch.Tensor, *, inference: bool = False):  # type: ignore[override]
        logits = torch.full(
            (input_ids.shape[0], input_ids.shape[1], self.vocab_size),
            fill_value=-1e9,
            dtype=torch.float32,
            device=input_ids.device,
        )
        step = input_ids.shape[1] - 1
        next_token = self.scheduled_tokens[step]
        logits[:, -1, next_token] = 1e9
        return logits, []


def test_sample_abcdigits_causal_lm_batch_returns_training_batch_and_examples() -> None:
    tokenizer = build_gpt2_tokenizer()
    batch, tokenized_examples = sample_abcdigits_causal_lm_batch(
        batch_size=3,
        config=ABCDigitsConfig(num_equations=30, depth=0.5),
        tokenizer=tokenizer,
        generator=torch.Generator().manual_seed(0),
    )
    assert batch.input_ids.shape[0] == 3
    assert batch.labels.shape == batch.input_ids.shape
    assert len(tokenized_examples) == 3


def test_greedy_decode_completion_follows_model_predictions() -> None:
    prompt_ids = (10, 11, 12)
    scheduled_tokens = [99, 99, 7, 8]
    model = StepwiseMockModel(vocab_size=128, scheduled_tokens=scheduled_tokens)
    predicted = greedy_decode_completion(model, prompt_ids, max_new_tokens=2)
    assert predicted == (7, 8)


def test_greedy_decode_completion_slides_context_when_prompt_hits_max_seq_len() -> None:
    class ContextAwareMockModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = type("Config", (), {"max_seq_len": 3})()
            self.weight = torch.nn.Parameter(torch.zeros(1))

        def forward(self, input_ids: torch.Tensor, *, inference: bool = False):  # type: ignore[override]
            logits = torch.full((1, input_ids.shape[1], 32), fill_value=-1e9, dtype=torch.float32, device=input_ids.device)
            last_token = int(input_ids[0, -1].item())
            next_token = 7 if last_token == 12 else 8
            logits[:, -1, next_token] = 1e9
            return logits, []

    model = ContextAwareMockModel()
    predicted = greedy_decode_completion(model, (10, 11, 12), max_new_tokens=2)
    assert predicted == (7, 8)


def test_evaluate_abcdigits_exact_match_counts_matches_correctly() -> None:
    tokenizer = build_gpt2_tokenizer()
    tokenized_examples = sample_tokenized_abcdigits_examples(
        batch_size=2,
        config=ABCDigitsConfig(num_equations=30, depth=0.5),
        tokenizer=tokenizer,
        generator=torch.Generator().manual_seed(1),
        add_eos=False,
    )
    first_completion = list(tokenized_examples[0].completion_ids)
    wrong_completion = [tokenized_examples[1].completion_ids[0]]
    if wrong_completion[0] == 0:
        wrong_completion[0] = 1
    else:
        wrong_completion[0] = 0
    scheduled_tokens = [0] * max(len(example.prompt_ids) + len(example.example.completion) for example in tokenized_examples)
    first_prompt_end = len(tokenized_examples[0].prompt_ids) - 1
    second_prompt_end = len(tokenized_examples[1].prompt_ids) - 1
    for offset, token in enumerate(first_completion):
        scheduled_tokens[first_prompt_end + offset] = token
    scheduled_tokens[second_prompt_end] = wrong_completion[0]
    model = StepwiseMockModel(
        vocab_size=tokenizer.vocab_size,
        scheduled_tokens=scheduled_tokens,
        max_seq_len=max(len(example.prompt_ids) for example in tokenized_examples)
        + max(len(example.example.completion) for example in tokenized_examples),
    )
    result = evaluate_abcdigits_exact_match(model, tokenizer, tokenized_examples)
    assert result.count == 2
    assert result.accuracy == 0.5
    assert 0.0 <= result.digit_accuracy < 1.0
    assert 1 <= result.unique_prediction_count <= 2


def test_evaluate_abcdigits_exact_match_uses_decoded_completion_string() -> None:
    tokenizer = build_gpt2_tokenizer()
    prompt = "Z="
    completion = "1234"
    prompt_ids = tuple(tokenizer.encode(prompt, add_special_tokens=False))
    full_ids = tuple(tokenizer.encode(prompt + completion, add_special_tokens=False))
    completion_ids = full_ids[len(prompt_ids) :]
    assert completion_ids != (10163, 19)

    example = ABCDigitsExample(
        config=ABCDigitsConfig(num_equations=26, depth=0.5, digits_per_value=4),
        prompt=prompt,
        completion=completion,
        target_letter="Z",
        target_value=completion,
        target_equation="Z=1234",
        target_equation_index=0,
        equations=("Z=1234",),
        letter_to_value={"Z": completion},
        non_target_weight_by_letter={},
    )
    tokenized_example = TokenizedABCDigitsExample(
        example=example,
        prompt_ids=prompt_ids,
        completion_ids=completion_ids,
        full_ids=full_ids,
    )
    scheduled_tokens = [0] * (len(prompt_ids) + 2)
    scheduled_tokens[len(prompt_ids) - 1] = 10163
    scheduled_tokens[len(prompt_ids)] = 19
    model = StepwiseMockModel(vocab_size=tokenizer.vocab_size, scheduled_tokens=scheduled_tokens, max_seq_len=8)

    result = evaluate_abcdigits_exact_match(model, tokenizer, (tokenized_example,))
    assert result.accuracy == 1.0
    assert result.digit_accuracy == 1.0
    assert result.unique_prediction_count == 1
    assert result.predictions == ("1234",)
    assert result.targets == ("1234",)


def test_abcdigits_training_smoke_reduces_loss_on_fixed_batch() -> None:
    tokenizer = build_gpt2_tokenizer()
    tokenized_examples = sample_tokenized_abcdigits_examples(
        batch_size=2,
        config=ABCDigitsConfig(num_equations=26, depth=0.5, digits_per_value=2),
        tokenizer=tokenizer,
        generator=torch.Generator().manual_seed(2),
        add_eos=True,
    )
    from abcdigits import build_abcdigits_causal_lm_batch

    batch = build_abcdigits_causal_lm_batch(
        tokenized_examples,
        pad_token_id=int(tokenizer.pad_token_id),
        supervise_completion_only=True,
    )
    max_length = batch.input_ids.shape[1]
    model = MultiscreenLM(
        MultiscreenConfig(
            vocab_size=tokenizer.vocab_size,
            d_model=16,
            n_layers=1,
            n_heads=1,
            d_key=4,
            d_value=8,
            max_seq_len=max_length,
            max_train_seq_len=max_length,
        )
    )
    optimizer = build_optimizer(model, OptimizerConfig(lr=1e-2))
    initial_loss = evaluate_loss(model, batch)
    for _ in range(10):
        train_step(model, optimizer, batch, grad_clip=1.0)
    final_loss = evaluate_loss(model, batch)
    assert final_loss < initial_loss
