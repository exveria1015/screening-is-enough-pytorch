"""ABCDigits synthetic retrieval benchmark utilities."""

from abcdigits.generator import (
    ABCDigitsConfig,
    ABCDigitsExample,
    build_abcdigits_example,
    render_abcdigits_prompt,
    resolve_target_equation_index,
)
from abcdigits.task import (
    ABCDigitsEvalResult,
    evaluate_abcdigits_exact_match,
    greedy_decode_completion,
    sample_abcdigits_causal_lm_batch,
    sample_tokenized_abcdigits_examples,
)
from abcdigits.training import (
    ABCDigitsCurriculumConfig,
    ABCDigitsEvalCell,
    ABCDigitsEvalPoint,
    ABCDigitsGridEvalResult,
    ABCDigitsTrainingPool,
    SampledABCDigitsBatch,
    build_abcdigits_eval_suite,
    build_abcdigits_training_pool,
    estimate_abcdigits_max_token_length,
    evaluate_abcdigits_grid,
    sample_abcdigits_curriculum_config,
    sample_abcdigits_training_batch,
    sample_abcdigits_training_batch_from_pool,
)
from abcdigits.tokenization import (
    TokenizedABCDigitsExample,
    build_abcdigits_causal_lm_batch,
    build_abcdigits_token_block,
    build_gpt2_tokenizer,
    tokenize_abcdigits_example,
)

__all__ = [
    "ABCDigitsConfig",
    "ABCDigitsCurriculumConfig",
    "ABCDigitsEvalCell",
    "ABCDigitsEvalResult",
    "ABCDigitsEvalPoint",
    "ABCDigitsExample",
    "ABCDigitsGridEvalResult",
    "ABCDigitsTrainingPool",
    "SampledABCDigitsBatch",
    "TokenizedABCDigitsExample",
    "build_abcdigits_example",
    "build_abcdigits_causal_lm_batch",
    "build_abcdigits_eval_suite",
    "build_abcdigits_training_pool",
    "build_abcdigits_token_block",
    "build_gpt2_tokenizer",
    "estimate_abcdigits_max_token_length",
    "evaluate_abcdigits_exact_match",
    "evaluate_abcdigits_grid",
    "greedy_decode_completion",
    "render_abcdigits_prompt",
    "resolve_target_equation_index",
    "sample_abcdigits_causal_lm_batch",
    "sample_abcdigits_curriculum_config",
    "sample_abcdigits_training_batch",
    "sample_abcdigits_training_batch_from_pool",
    "sample_tokenized_abcdigits_examples",
    "tokenize_abcdigits_example",
]
