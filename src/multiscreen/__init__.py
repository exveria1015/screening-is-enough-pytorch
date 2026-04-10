"""Paper-faithful Multiscreen implementation."""

from __future__ import annotations

from importlib import import_module


_EXPORTS_BY_MODULE = {
    "multiscreen.config": ("MultiscreenConfig",),
    "multiscreen.corpus_build": (
        "CorpusBuildReport",
        "CorpusBuildSpec",
        "CorpusBuildSplit",
        "CorpusDatasetSource",
        "CorpusSourceBuildReport",
        "CorpusSplitBuildReport",
        "allocate_source_document_counts",
        "build_corpus_from_spec",
        "iterate_source_texts",
        "load_corpus_build_spec",
        "normalize_corpus_text",
    ),
    "multiscreen.corpus": (
        "CorpusSplit",
        "TokenizedCorpusArtifact",
        "TokenizedCorpusMetadata",
        "build_fixed_causal_lm_batches",
        "build_token_stream_from_corpus",
        "expand_corpus_paths",
        "iter_corpus_documents",
        "load_corpus_documents",
        "load_tokenized_corpus_artifact",
        "save_tokenized_corpus_artifact",
        "split_token_stream",
        "tokenize_corpus_documents",
        "write_token_stream_from_corpus",
    ),
    "multiscreen.data": ("CausalLMBatch", "causal_lm_batch_from_token_block", "sample_token_blocks"),
    "multiscreen.generation": ("generate_tokens", "sample_next_token", "truncate_prompt_tokens"),
    "multiscreen.math": ("TanhNorm", "apply_mipe", "build_softmask", "normalize_unit", "trim_and_square"),
    "multiscreen.model": ("GatedScreeningTile", "MultiscreenLM", "ScreeningUnit"),
    "multiscreen.sizing": (
        "MultiscreenSizeEstimate",
        "estimate_multiscreen_size_from_token_count",
        "multiscreen_parameter_count",
        "multiscreen_parameter_count_from_dimensions",
        "multiscreen_parameter_count_from_psi",
    ),
    "multiscreen.train": (
        "OptimizerConfig",
        "TrainStepResult",
        "build_optimizer",
        "causal_lm_loss",
        "evaluate_loss",
        "model_device",
        "train_step",
    ),
    "multiscreen.triton_kernels": ("TritonScreeningSupport", "check_triton_screening_support", "triton_is_available"),
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
