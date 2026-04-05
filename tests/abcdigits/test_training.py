from __future__ import annotations

import torch

from abcdigits import (
    ABCDigitsConfig,
    ABCDigitsCurriculumConfig,
    ABCDigitsEvalCell,
    ABCDigitsExample,
    ABCDigitsTrainingPool,
    TokenizedABCDigitsExample,
    build_abcdigits_example,
    build_abcdigits_eval_suite,
    build_abcdigits_training_pool,
    build_gpt2_tokenizer,
    estimate_abcdigits_max_token_length,
    evaluate_abcdigits_grid,
    sample_abcdigits_curriculum_config,
    sample_abcdigits_training_batch,
    sample_abcdigits_training_batch_from_pool,
    tokenize_abcdigits_example,
)


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
        logits[:, -1, self.scheduled_tokens[step]] = 1e9
        return logits, []


def test_sample_abcdigits_curriculum_config_respects_requested_range() -> None:
    curriculum = ABCDigitsCurriculumConfig(
        min_num_equations=26,
        max_num_equations=40,
        depths=(0.1, 0.5, 0.9),
        digits_per_value=6,
    )
    generator = torch.Generator().manual_seed(0)
    for _ in range(20):
        config = sample_abcdigits_curriculum_config(curriculum, generator=generator)
        assert 26 <= config.num_equations <= 40
        assert config.depth in curriculum.depths
        assert config.digits_per_value == 6


def test_estimate_abcdigits_max_token_length_bounds_sampled_examples() -> None:
    tokenizer = build_gpt2_tokenizer()
    bound = estimate_abcdigits_max_token_length(num_equations=60, digits_per_value=6, add_eos=True)
    generator = torch.Generator().manual_seed(1)
    for _ in range(10):
        example = tokenize_abcdigits_example(
            build_abcdigits_example(
                ABCDigitsConfig(num_equations=60, depth=0.5, digits_per_value=6),
                generator=generator,
            ),
            tokenizer,
            add_eos=True,
        )
        assert len(example.full_ids) <= bound


def test_sample_abcdigits_training_batch_uses_sampled_config() -> None:
    tokenizer = build_gpt2_tokenizer()
    curriculum = ABCDigitsCurriculumConfig(
        min_num_equations=26,
        max_num_equations=30,
        depths=(0.3, 0.7),
        digits_per_value=6,
    )
    sampled = sample_abcdigits_training_batch(
        batch_size=3,
        curriculum=curriculum,
        tokenizer=tokenizer,
        generator=torch.Generator().manual_seed(2),
    )
    assert sampled.batch.input_ids.shape[0] == 3
    assert len(sampled.tokenized_examples) == 3
    assert sampled.config.num_equations in range(26, 31)
    assert sampled.config.depth in curriculum.depths


def test_build_abcdigits_training_pool_returns_requested_number_of_examples() -> None:
    tokenizer = build_gpt2_tokenizer()
    curriculum = ABCDigitsCurriculumConfig(
        min_num_equations=26,
        max_num_equations=26,
        depths=(0.5,),
        digits_per_value=6,
        add_eos=False,
    )
    pool = build_abcdigits_training_pool(
        pool_size=7,
        curriculum=curriculum,
        tokenizer=tokenizer,
        generator=torch.Generator().manual_seed(4),
    )
    assert isinstance(pool, ABCDigitsTrainingPool)
    assert len(pool.tokenized_examples) == 7
    assert all(example.example.config.num_equations == 26 for example in pool.tokenized_examples)


def test_sample_abcdigits_training_batch_from_pool_draws_examples_from_pool() -> None:
    tokenizer = build_gpt2_tokenizer()
    curriculum = ABCDigitsCurriculumConfig(
        min_num_equations=26,
        max_num_equations=26,
        depths=(0.5,),
        digits_per_value=6,
        add_eos=False,
    )
    pool = build_abcdigits_training_pool(
        pool_size=5,
        curriculum=curriculum,
        tokenizer=tokenizer,
        generator=torch.Generator().manual_seed(5),
    )
    sampled = sample_abcdigits_training_batch_from_pool(
        batch_size=3,
        training_pool=pool,
        tokenizer=tokenizer,
        generator=torch.Generator().manual_seed(6),
    )
    pool_full_ids = {example.full_ids for example in pool.tokenized_examples}
    assert sampled.batch.input_ids.shape[0] == 3
    assert sampled.config is not None
    assert all(example.full_ids in pool_full_ids for example in sampled.tokenized_examples)


def test_build_abcdigits_eval_suite_returns_requested_grid() -> None:
    tokenizer = build_gpt2_tokenizer()
    suite = build_abcdigits_eval_suite(
        num_equations_values=(26, 40),
        depths=(0.1, 0.9),
        examples_per_cell=3,
        tokenizer=tokenizer,
        digits_per_value=6,
        generator=torch.Generator().manual_seed(3),
    )
    assert len(suite) == 4
    assert {(cell.num_equations, cell.depth) for cell in suite} == {(26, 0.1), (26, 0.9), (40, 0.1), (40, 0.9)}
    assert all(len(cell.tokenized_examples) == 3 for cell in suite)


def test_evaluate_abcdigits_grid_aggregates_point_accuracies() -> None:
    tokenizer = build_gpt2_tokenizer()
    prompt = "Z="
    prompt_ids = tuple(tokenizer.encode(prompt, add_special_tokens=False))
    good_completion = "1234"
    bad_completion = "9999"

    def build_tokenized(completion: str) -> TokenizedABCDigitsExample:
        full_ids = tuple(tokenizer.encode(prompt + completion, add_special_tokens=False))
        completion_ids = full_ids[len(prompt_ids) :]
        example = ABCDigitsExample(
            config=ABCDigitsConfig(num_equations=26, depth=0.5, digits_per_value=4),
            prompt=prompt,
            completion=completion,
            target_letter="Z",
            target_value=completion,
            target_equation=f"Z={completion}",
            target_equation_index=0,
            equations=(f"Z={completion}",),
            letter_to_value={"Z": completion},
            non_target_weight_by_letter={},
        )
        return TokenizedABCDigitsExample(
            example=example,
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
            full_ids=full_ids,
        )

    suite = (
        ABCDigitsEvalCell(num_equations=26, depth=0.1, tokenized_examples=(build_tokenized(good_completion),)),
        ABCDigitsEvalCell(num_equations=26, depth=0.9, tokenized_examples=(build_tokenized(bad_completion),)),
    )
    scheduled_tokens = [0] * (len(prompt_ids) + 2)
    scheduled_tokens[len(prompt_ids) - 1] = 10163
    scheduled_tokens[len(prompt_ids)] = 19
    model = StepwiseMockModel(vocab_size=tokenizer.vocab_size, scheduled_tokens=scheduled_tokens, max_seq_len=8)

    result = evaluate_abcdigits_grid(model, tokenizer, suite)
    assert result.mean_accuracy == 0.5
    assert 0.0 <= result.mean_digit_accuracy <= 1.0
    assert 0.0 < result.mean_unique_prediction_ratio <= 1.0
    assert len(result.points) == 2
    assert result.points[0].accuracy == 1.0
    assert result.points[1].accuracy == 0.0
