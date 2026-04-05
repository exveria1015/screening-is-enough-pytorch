from __future__ import annotations

import torch

from abcdigits import (
    ABCDigitsConfig,
    build_abcdigits_causal_lm_batch,
    build_abcdigits_example,
    build_abcdigits_token_block,
    build_gpt2_tokenizer,
    tokenize_abcdigits_example,
)


def test_build_gpt2_tokenizer_loads_expected_vocabulary() -> None:
    tokenizer = build_gpt2_tokenizer()
    assert tokenizer.vocab_size == 50_257
    assert tokenizer.pad_token_id is not None


def test_tokenize_abcdigits_example_preserves_prompt_prefix() -> None:
    tokenizer = build_gpt2_tokenizer()
    example = build_abcdigits_example(ABCDigitsConfig(num_equations=30, depth=0.3), generator=torch.Generator().manual_seed(0))
    tokenized = tokenize_abcdigits_example(example, tokenizer)

    assert tokenized.prompt_length > 0
    assert tokenized.completion_length > 0
    assert tokenized.full_ids[: tokenized.prompt_length] == tokenized.prompt_ids
    assert tokenized.full_ids[tokenized.prompt_length :] == tokenized.completion_ids
    assert tokenizer.decode(list(tokenized.full_ids)) == example.prompt + example.completion


def test_build_abcdigits_token_block_matches_full_token_ids() -> None:
    tokenizer = build_gpt2_tokenizer()
    example = build_abcdigits_example(ABCDigitsConfig(num_equations=30, depth=0.5), generator=torch.Generator().manual_seed(1))
    tokenized = tokenize_abcdigits_example(example, tokenizer, add_eos=True)
    block = build_abcdigits_token_block(tokenized)
    assert block.dtype == torch.long
    assert tuple(block.tolist()) == tokenized.full_ids


def test_build_abcdigits_causal_lm_batch_pads_and_masks_prompt_positions() -> None:
    tokenizer = build_gpt2_tokenizer()
    examples = [
        build_abcdigits_example(ABCDigitsConfig(num_equations=30, depth=0.1), generator=torch.Generator().manual_seed(2)),
        build_abcdigits_example(ABCDigitsConfig(num_equations=45, depth=0.9), generator=torch.Generator().manual_seed(3)),
    ]
    tokenized_examples = [tokenize_abcdigits_example(example, tokenizer, add_eos=True) for example in examples]
    batch = build_abcdigits_causal_lm_batch(
        tokenized_examples,
        pad_token_id=int(tokenizer.pad_token_id),
        supervise_completion_only=True,
    )

    assert batch.input_ids.shape[0] == 2
    assert batch.labels.shape == batch.input_ids.shape

    first_prompt_only_count = tokenized_examples[0].prompt_length - 1
    assert torch.all(batch.labels[0, :first_prompt_only_count] == -100)

    first_valid_completion_index = first_prompt_only_count
    assert int(batch.labels[0, first_valid_completion_index].item()) == tokenized_examples[0].completion_ids[0]

    second_valid_length = len(tokenized_examples[1].full_ids) - 1
    assert torch.all(batch.labels[1, second_valid_length:] == -100)
