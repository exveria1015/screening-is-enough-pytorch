"""ABCDigits synthetic retrieval benchmark utilities."""

from __future__ import annotations

from importlib import import_module


_EXPORTS_BY_MODULE = {
    "abcdigits.generator": (
        "ABCDigitsConfig",
        "ABCDigitsExample",
        "build_abcdigits_example",
        "render_abcdigits_prompt",
        "resolve_target_equation_index",
    ),
    "abcdigits.task": (
        "ABCDigitsEvalResult",
        "evaluate_abcdigits_exact_match",
        "greedy_decode_completion",
        "sample_abcdigits_causal_lm_batch",
        "sample_tokenized_abcdigits_examples",
    ),
    "abcdigits.training": (
        "ABCDigitsCurriculumConfig",
        "ABCDigitsEvalCell",
        "ABCDigitsEvalPoint",
        "ABCDigitsGridEvalResult",
        "ABCDigitsTrainingPool",
        "SampledABCDigitsBatch",
        "build_abcdigits_eval_suite",
        "build_abcdigits_training_pool",
        "estimate_abcdigits_max_token_length",
        "evaluate_abcdigits_grid",
        "sample_abcdigits_curriculum_config",
        "sample_abcdigits_training_batch",
        "sample_abcdigits_training_batch_from_pool",
    ),
    "abcdigits.tokenization": (
        "TokenizedABCDigitsExample",
        "build_abcdigits_causal_lm_batch",
        "build_abcdigits_token_block",
        "build_gpt2_tokenizer",
        "tokenize_abcdigits_example",
    ),
}

_SYMBOL_TO_MODULE = {
    symbol: module_name
    for module_name, symbols in _EXPORTS_BY_MODULE.items()
    for symbol in symbols
}

__all__ = sorted(_SYMBOL_TO_MODULE)


def __getattr__(name: str):
    module_name = _SYMBOL_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
