from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from abcdigits import build_gpt2_tokenizer  # noqa: E402
from multiscreen.corpus import (  # noqa: E402
    TokenizedCorpusArtifact,
    TokenizedCorpusMetadata,
    expand_corpus_paths,
    load_corpus_documents,
    load_tokenized_corpus_artifact,
    save_tokenized_corpus_artifact,
    split_token_stream,
    tokenize_corpus_documents,
    write_token_stream_from_corpus,
)
from multiscreen.sizing import estimate_multiscreen_size_from_token_count  # noqa: E402


def resolve_storage_dtype(name: str) -> torch.dtype:
    mapping = {
        "int32": torch.int32,
        "int64": torch.int64,
    }
    try:
        return mapping[name]
    except KeyError as error:
        raise ValueError(f"unsupported storage dtype: {name}") from error


def main() -> int:
    parser = argparse.ArgumentParser(description="Tokenize a text corpus once and save reusable artifacts.")
    parser.add_argument("--train-path", action="append", required=True, help="File, directory, or glob for training text/jsonl")
    parser.add_argument("--val-path", action="append", default=[], help="Optional file, directory, or glob for validation")
    parser.add_argument("--jsonl-text-key", type=str, default="text")
    parser.add_argument("--max-train-documents", type=int, default=None)
    parser.add_argument("--max-val-documents", type=int, default=None)
    parser.add_argument("--val-fraction", type=float, default=0.01)
    parser.add_argument("--tokenizer-name", type=str, default="gpt2")
    parser.add_argument("--append-eos", dest="append_eos", action="store_true")
    parser.add_argument("--no-append-eos", dest="append_eos", action="store_false")
    parser.add_argument("--tokens-per-parameter", type=float, default=20.0)
    parser.add_argument("--min-psi", type=int, default=1)
    parser.add_argument("--max-psi", type=int, default=64)
    parser.add_argument("--d-key", type=int, default=16)
    parser.add_argument("--d-value", type=int, default=64)
    parser.add_argument("--storage-dtype", choices=("int32", "int64"), default="int32")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--local-files-only", action="store_true")
    parser.set_defaults(append_eos=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    tokenizer = build_gpt2_tokenizer(model_name=args.tokenizer_name, local_files_only=args.local_files_only)
    if args.val_path:
        train_files, train_document_count, train_token_count = write_token_stream_from_corpus(
            args.output_dir / "train_tokens.bin",
            paths=args.train_path,
            tokenizer=tokenizer,
            jsonl_text_key=args.jsonl_text_key,
            max_documents=args.max_train_documents,
            add_eos=args.append_eos,
            storage_dtype=resolve_storage_dtype(args.storage_dtype),
        )
        val_files, val_document_count, val_token_count = write_token_stream_from_corpus(
            args.output_dir / "val_tokens.bin",
            paths=args.val_path,
            tokenizer=tokenizer,
            jsonl_text_key=args.jsonl_text_key,
            max_documents=args.max_val_documents,
            add_eos=args.append_eos,
            storage_dtype=resolve_storage_dtype(args.storage_dtype),
        )
        train_token_ids = None
        val_token_ids = None
    else:
        train_files = expand_corpus_paths(args.train_path)
        train_documents = load_corpus_documents(
            [str(path) for path in train_files],
            jsonl_text_key=args.jsonl_text_key,
            max_documents=args.max_train_documents,
        )
        val_files = ()
        full_token_ids = tokenize_corpus_documents(train_documents, tokenizer, add_eos=args.append_eos)
        split = split_token_stream(full_token_ids, val_fraction=args.val_fraction)
        train_token_ids = split.train_token_ids
        val_token_ids = split.val_token_ids
        train_document_count = len(train_documents)
        val_document_count = 0
        train_token_count = int(train_token_ids.numel())
        val_token_count = 0 if val_token_ids is None else int(val_token_ids.numel())

    size_estimate = estimate_multiscreen_size_from_token_count(
        train_token_count,
        vocab_size=tokenizer.vocab_size,
        tokens_per_parameter=args.tokens_per_parameter,
        d_key=args.d_key,
        d_value=args.d_value,
        min_psi=args.min_psi,
        max_psi=args.max_psi,
    )
    artifact = TokenizedCorpusArtifact(
        metadata=TokenizedCorpusMetadata(
            format_version=1,
            tokenizer_name=args.tokenizer_name,
            vocab_size=int(tokenizer.vocab_size),
            append_eos=bool(args.append_eos),
            jsonl_text_key=args.jsonl_text_key,
            train_files=tuple(str(path) for path in train_files),
            val_files=tuple(str(path) for path in val_files),
            train_documents=train_document_count,
            val_documents=val_document_count,
            train_tokens=train_token_count,
            val_tokens=val_token_count,
            total_tokens=train_token_count + val_token_count,
            storage_dtype=args.storage_dtype,
            size_estimate=size_estimate,
        ),
        train_token_ids=torch.empty(0, dtype=torch.long) if train_token_ids is None else train_token_ids,
        val_token_ids=val_token_ids,
    )
    if train_token_ids is not None:
        save_tokenized_corpus_artifact(
            args.output_dir,
            artifact=artifact,
            storage_dtype=resolve_storage_dtype(args.storage_dtype),
        )
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "metadata.json").write_text(
            json.dumps(artifact.metadata.to_dict(), sort_keys=True, indent=2),
            encoding="utf-8",
        )

    reloaded = load_tokenized_corpus_artifact(args.output_dir)
    print(
        "prepared corpus:",
        f"output_dir={args.output_dir}",
        f"tokenizer={reloaded.metadata.tokenizer_name}",
        f"train_docs={reloaded.metadata.train_documents}",
        f"val_docs={reloaded.metadata.val_documents}",
        f"train_tokens={reloaded.metadata.train_tokens}",
        f"val_tokens={reloaded.metadata.val_tokens}",
        f"storage_dtype={reloaded.metadata.storage_dtype}",
    )
    print(
        "size estimate:",
        f"target_params={size_estimate.target_parameter_count}",
        f"recommended_psi={size_estimate.recommended_psi}",
        f"recommended_params={size_estimate.recommended_parameter_count}",
        f"tokens_per_parameter={size_estimate.recommended_tokens_per_parameter:.2f}",
    )
    if size_estimate.smaller_psi is not None:
        print(
            "smaller candidate:",
            f"psi={size_estimate.smaller_psi}",
            f"params={size_estimate.smaller_parameter_count}",
        )
    if size_estimate.larger_psi is not None:
        print(
            "larger candidate:",
            f"psi={size_estimate.larger_psi}",
            f"params={size_estimate.larger_parameter_count}",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
