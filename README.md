# screening-is-enough-pytorch

`screening-is-enough-pytorch/` is a fresh implementation track for reproducing the
Multiscreen architecture from the paper "Screening Is Enough".

The repository name is `screening-is-enough-pytorch`. The Python import
packages remain `multiscreen` and `abcdigits`.

This project intentionally does not inherit the old scaffold's architectural
choices. The current scope is:

- paper-shaped Multiscreen config scaling
- paper-shaped Screening Unit math
- parallel gated screening tiles per layer
- paper-specified parameter initialization
- paper-shaped ABCDigits generator
- GPT-2 tokenizer support for ABCDigits
- minimal causal-LM training utilities
- YAML-driven corpus building via Hugging Face datasets
- reusable tokenized corpus artifacts with model-size estimation
- GPT-2-tokenized corpus pretraining with saved checkpoints
- sampled ABCDigits training curriculum and evaluation grid
- checkpointed ABCDigits training CLI
- unit tests for the core formulas

Not implemented yet:

- Transformer baseline from Appendix A
- SlimPajama / PG-19 evaluation pipeline
- optimized kernels for long-context inference

## Layout

- `src/multiscreen/config.py`: typed model configuration
- `src/abcdigits/`: ABCDigits generator and rendering helpers
- `src/abcdigits/tokenization.py`: GPT-2 tokenizer integration for ABCDigits
- `src/abcdigits/task.py`: ABCDigits batch sampling and exact-match evaluation
- `src/abcdigits/training.py`: ABCDigits curriculum sampling and grid evaluation
- `src/multiscreen/data.py`: token-block sampling and batch shaping
- `src/multiscreen/corpus_build.py`: YAML corpus specs and JSONL builders
- `src/multiscreen/corpus.py`: corpus loading, tokenization, and saved artifacts
- `src/multiscreen/math.py`: screening math, MiPE, softmask, TanhNorm
- `src/multiscreen/model.py`: Multiscreen model, layers, tiles
- `src/multiscreen/train.py`: loss, optimizer, eval, train step
- `scripts/build_corpus.py`: build JSONL corpora from YAML dataset specs
- `scripts/prepare_corpus.py`: tokenize JSONL/text corpora and estimate Multiscreen size
- `scripts/train_abcdigits_smoke.py`: fixed-batch ABCDigits overfit smoke
- `scripts/train_abcdigits.py`: sampled ABCDigits training with metrics and checkpoints
- `scripts/train_corpus.py`: corpus pretraining from raw text or prepared token artifacts
- `tests/`: formula and shape tests

## Install

```bash
python -m pip install -e .[dev]
pytest
python scripts/build_corpus.py --config configs/corpus.example.yaml --output-dir artifacts/corpus-raw
python scripts/prepare_corpus.py --train-path artifacts/corpus-raw/train.jsonl --val-path artifacts/corpus-raw/val.jsonl --output-dir artifacts/corpus-prepared
python scripts/train_corpus.py --prepared-corpus artifacts/corpus-prepared --use-recommended-config --steps 100
python scripts/train_corpus.py --prepared-corpus artifacts/corpus-prepared --use-recommended-config --optimizer adamw_schedulefree --schedulefree-warmup-steps 200 --steps 100
python scripts/train_abcdigits_smoke.py --steps 40 --batch-size 4
python scripts/train_abcdigits.py --steps 500 --batch-size 8 --optimizer adamw_schedulefree --schedulefree-warmup-steps 50
```

## Notes

The current repository can train and evaluate on ABCDigits with GPT-2
tokenization, but it is not yet a full reproduction of the paper's
natural-language datasets, Transformer baseline, or long-context kernels.
