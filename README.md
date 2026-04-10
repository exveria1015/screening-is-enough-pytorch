# screening-is-enough-pytorch

Unofficial PyTorch implementation of the Multiscreen architecture described in the paper *Screening Is Enough*.

> [!IMPORTANT]
> This repository is **not** the official implementation.
> It is an independent reimplementation and is **not affiliated with or endorsed by the paper author**.
> If this codebase differs from the paper, the paper is the authoritative reference.
> The reference track in this repository prioritizes correctness, readability, and architectural inspection over kernel-level optimization.
> Kernel and runtime experiments are isolated on dedicated branches so the paper-faithful path remains easy to inspect.

## Reference Paper

- Official paper: [Screening Is Enough](https://arxiv.org/abs/2604.01178)
- Author: Ken M. Nakanishi
- arXiv: `2604.01178`

## Project Scope

This repository is an implementation track for studying and reproducing the core ideas in Multiscreen using PyTorch.

The repository name is `screening-is-enough-pytorch`. The Python packages remain:

- `multiscreen`
- `abcdigits`

## Branch Policy

- `main`: paper-faithful PyTorch reference branch for inspecting the Multiscreen equations and model structure.
- `test-triton-kernel`: experimental branch split from the reference implementation for Triton/CUDA kernel work, backend profiling, and fusion experiments.

The goal of the experimental branch is to stay equation-equivalent to the paper while changing execution strategy. In practice that means preserving the same mathematical form, while allowing numerically close but not bitwise-identical floating-point results when reduction order changes on GPU.

## Current Status

Implemented in this repository today:

- Multiscreen configuration and parameter sizing utilities
- Screening Unit math, MiPE, softmask, and TanhNorm helpers
- Multiscreen language model layers with gated screening tiles
- Experimental opt-in Triton screening backend for exact-form aggregation without relevance materialization
- Equation-equivalent Triton kernel experiments for preprocessing, screening aggregation, and fused screening epilogues
- ABCDigits synthetic benchmark generation, tokenization, and evaluation
- Corpus building from YAML specifications via Hugging Face datasets
- One-time corpus tokenization and reusable token artifact generation
- Training loops for corpus pretraining and sampled ABCDigits training
- Unit tests for core math, model behavior, corpus tooling, and training paths

Not implemented or not yet reproduced:

- Transformer baseline from Appendix A
- Full paper-scale natural-language evaluation pipeline
- Long-context optimized inference kernels
- Official checkpoints or exact paper-result reproduction claims

## Known Differences From The Paper

This repository follows the paper as a reference for the Multiscreen architecture, but its experimental protocol is currently different from the paper in several important ways.

- Training data differs. The paper pretrains on SlimPajama, while this repository uses a generic Hugging Face dataset pipeline and the example config currently points to TinyStories.
- Training scale differs. The paper reports base pretraining with `2^38` tokens at sequence length `2^12`, followed by continual pretraining at sequence length `2^15` with an additional `2^27` tokens. The scripts in this repository are lightweight local training scripts intended for experimentation, not paper-scale reproduction.
- Batch sizing differs. The paper uses a global batch size of `2^22` tokens. The training scripts here default to much smaller local settings such as `--seq-len 512`, `--batch-size 8`, and limited step counts.
- Optimization differs. The paper uses AdamW with `(\beta_1, \beta_2) = (0.9, 0.95)`, `2^10` warmup steps during base pretraining, and a constant learning rate after warmup. This repository currently provides plain AdamW and optional ScheduleFree AdamW, and the plain AdamW path does not implement the paper's warmup-plus-constant schedule. The ScheduleFree option is also not part of the paper's reported setup.
- Continual pretraining protocol differs. The paper continues long-context training from pretrained checkpoints while inheriting optimizer state and applying no additional warmup. This repository does not currently provide the full paper-style continual pretraining workflow as a dedicated reproduction pipeline.
- Evaluation differs. The paper reports long-context perplexity on PG-19, ABCDigits retrieval across much larger context lengths, multi-seed averages, Transformer comparisons, RoPE scaling sweeps for the baseline, and 100K-context latency measurements. Those evaluations are not fully reproduced in this repository.
- ABCDigits training and evaluation differ in scale. The paper evaluates `2^12` to `2^17` context lengths with `1,000` examples per cell and reports multi-model averages, while the local ABCDigits scripts default to much smaller ranges intended for manageable experimentation and debugging.
- The optional Triton backend is an optimization path only. It preserves the same equations as the reference implementation, but floating-point reductions may differ slightly from the baseline PyTorch path, so the default execution path remains the reference PyTorch implementation.
- The `test-triton-kernel` branch is intentionally for runtime experiments. Treat `main` as the easiest branch to audit against the paper, and treat the Triton branch as the place where execution strategy is allowed to evolve without changing the target equations.

Because of these differences, results from this repository should be treated as exploratory or partial reproduction results, not as direct reproductions of the paper's reported numbers.

## Repository Layout

- `src/multiscreen/`: core model, math, data, corpus, sizing, and training utilities
- `src/abcdigits/`: ABCDigits generation, tokenization, curriculum sampling, and evaluation
- `scripts/build_corpus.py`: build JSONL corpora from YAML dataset specs
- `scripts/prepare_corpus.py`: tokenize corpora once and save reusable token artifacts
- `scripts/train_corpus.py`: train Multiscreen on raw or prepared corpora
- `scripts/train_abcdigits_smoke.py`: fixed-batch ABCDigits overfit smoke test
- `scripts/train_abcdigits.py`: sampled ABCDigits training with evaluation and checkpoints
- `tests/`: automated tests for the implemented components

## Installation

Python `3.12` to `3.13` is required.

```bash
python -m pip install -e .[dev]
pytest
```

The scripts in `scripts/` assume the package is installed in the active environment, for example via the editable install above.

## Quick Start

### 1. ABCDigits Smoke Test

```bash
python scripts/train_abcdigits_smoke.py --steps 40 --batch-size 4
```

### 2. Sampled ABCDigits Training

```bash
python scripts/train_abcdigits.py --steps 500 --batch-size 8
```

Example with ScheduleFree AdamW:

```bash
python scripts/train_abcdigits.py \
  --steps 500 \
  --batch-size 8 \
  --optimizer adamw_schedulefree \
  --schedulefree-warmup-steps 50
```

### 3. Corpus Workflow

Build a raw JSONL corpus from a YAML dataset spec:

```bash
python scripts/build_corpus.py \
  --config configs/corpus.example.yaml \
  --output-dir artifacts/corpus-raw
```

Prepare reusable token artifacts and a recommended size estimate:

```bash
python scripts/prepare_corpus.py \
  --train-path artifacts/corpus-raw/train.jsonl \
  --val-path artifacts/corpus-raw/val.jsonl \
  --output-dir artifacts/corpus-prepared
```

Train from the prepared corpus using the recommended Multiscreen configuration:

```bash
python scripts/train_corpus.py \
  --prepared-corpus artifacts/corpus-prepared \
  --use-recommended-config \
  --steps 100
```

Example with ScheduleFree AdamW:

```bash
python scripts/train_corpus.py \
  --prepared-corpus artifacts/corpus-prepared \
  --use-recommended-config \
  --optimizer adamw_schedulefree \
  --schedulefree-warmup-steps 200 \
  --steps 100
```

## Notes

- This repository is intended for reproduction, experimentation, and inspection of the Multiscreen architecture in PyTorch.
- It should not be cited or interpreted as the canonical implementation of the paper.
- If an official implementation is released separately, that implementation should be preferred for canonical comparisons.

## License

This repository is intended to be licensed under the Apache License 2.0. See `LICENSE`.
