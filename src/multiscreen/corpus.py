"""Text-corpus loading and token-stream helpers for language-model pretraining."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import glob
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from transformers import PreTrainedTokenizerBase

from multiscreen.data import CausalLMBatch, causal_lm_batch_from_token_block, sample_token_blocks
from multiscreen.sizing import MultiscreenSizeEstimate


_SUPPORTED_TEXT_SUFFIXES = {".jsonl", ".md", ".text", ".txt"}
_ARTIFACT_METADATA_FILENAME = "metadata.json"
_ARTIFACT_TRAIN_TOKENS_BINARY_FILENAME = "train_tokens.bin"
_ARTIFACT_VAL_TOKENS_BINARY_FILENAME = "val_tokens.bin"
_ARTIFACT_TRAIN_TOKENS_FILENAME = "train_tokens.pt"
_ARTIFACT_VAL_TOKENS_FILENAME = "val_tokens.pt"


@dataclass(slots=True)
class CorpusSplit:
    """Train/validation token streams."""

    train_token_ids: torch.Tensor
    val_token_ids: torch.Tensor | None


@dataclass(slots=True)
class TokenizedCorpusMetadata:
    """Metadata persisted next to a tokenized corpus artifact."""

    format_version: int
    tokenizer_name: str
    vocab_size: int
    append_eos: bool
    jsonl_text_key: str
    train_files: tuple[str, ...]
    val_files: tuple[str, ...]
    train_documents: int
    val_documents: int
    train_tokens: int
    val_tokens: int
    total_tokens: int
    storage_dtype: str
    size_estimate: MultiscreenSizeEstimate | None = None

    def to_dict(self) -> dict:
        payload = asdict(self)
        if self.size_estimate is not None:
            payload["size_estimate"] = asdict(self.size_estimate)
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> "TokenizedCorpusMetadata":
        size_estimate_payload = payload.get("size_estimate")
        size_estimate = None
        if isinstance(size_estimate_payload, dict):
            size_estimate = MultiscreenSizeEstimate(**size_estimate_payload)
        return cls(
            format_version=int(payload["format_version"]),
            tokenizer_name=str(payload["tokenizer_name"]),
            vocab_size=int(payload["vocab_size"]),
            append_eos=bool(payload["append_eos"]),
            jsonl_text_key=str(payload["jsonl_text_key"]),
            train_files=tuple(str(path) for path in payload.get("train_files", [])),
            val_files=tuple(str(path) for path in payload.get("val_files", [])),
            train_documents=int(payload["train_documents"]),
            val_documents=int(payload["val_documents"]),
            train_tokens=int(payload["train_tokens"]),
            val_tokens=int(payload["val_tokens"]),
            total_tokens=int(payload["total_tokens"]),
            storage_dtype=str(payload["storage_dtype"]),
            size_estimate=size_estimate,
        )


@dataclass(slots=True)
class TokenizedCorpusArtifact:
    """Reusable tokenized train/validation streams plus metadata."""

    metadata: TokenizedCorpusMetadata
    train_token_ids: torch.Tensor
    val_token_ids: torch.Tensor | None


def _is_glob_pattern(path: str) -> bool:
    return any(character in path for character in "*?[]")


def expand_corpus_paths(paths: list[str] | tuple[str, ...]) -> tuple[Path, ...]:
    """Resolve files, directories, and glob patterns into concrete corpus files."""

    if not paths:
        raise ValueError("paths must be non-empty")

    resolved: list[Path] = []
    seen: set[Path] = set()
    for raw_path in paths:
        if _is_glob_pattern(raw_path):
            candidates = [Path(match) for match in glob.glob(raw_path, recursive=True)]
        else:
            path = Path(raw_path)
            if path.is_dir():
                candidates = [candidate for candidate in sorted(path.rglob("*")) if candidate.is_file()]
            else:
                candidates = [path]

        for candidate in sorted(candidates):
            if not candidate.exists():
                continue
            if candidate.suffix.lower() not in _SUPPORTED_TEXT_SUFFIXES:
                continue
            canonical = candidate.resolve()
            if canonical in seen:
                continue
            seen.add(canonical)
            resolved.append(canonical)

    if not resolved:
        raise ValueError("no supported corpus files were found")
    return tuple(resolved)


def load_corpus_documents(
    paths: list[str] | tuple[str, ...],
    *,
    jsonl_text_key: str = "text",
    max_documents: int | None = None,
) -> tuple[str, ...]:
    """Load documents from plain-text or JSONL corpus files."""

    if not jsonl_text_key:
        raise ValueError("jsonl_text_key must be non-empty")
    if max_documents is not None and max_documents <= 0:
        raise ValueError("max_documents must be positive when provided")

    documents: list[str] = []
    for path in expand_corpus_paths(paths):
        if path.suffix.lower() == ".jsonl":
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    payload = line.strip()
                    if not payload:
                        continue
                    record = json.loads(payload)
                    text = record.get(jsonl_text_key)
                    if not isinstance(text, str):
                        raise ValueError(f"missing string field {jsonl_text_key!r} in {path}:{line_number}")
                    documents.append(text)
                    if max_documents is not None and len(documents) >= max_documents:
                        return tuple(documents)
        else:
            documents.append(path.read_text(encoding="utf-8"))
            if max_documents is not None and len(documents) >= max_documents:
                return tuple(documents)

    if not documents:
        raise ValueError("no corpus documents were loaded")
    return tuple(documents)


def iter_corpus_documents(
    paths: list[str] | tuple[str, ...],
    *,
    jsonl_text_key: str = "text",
    max_documents: int | None = None,
) -> tuple[tuple[Path, ...], Iterable[str]]:
    """Stream documents from plain-text or JSONL corpus files."""

    if not jsonl_text_key:
        raise ValueError("jsonl_text_key must be non-empty")
    if max_documents is not None and max_documents <= 0:
        raise ValueError("max_documents must be positive when provided")

    resolved_paths = expand_corpus_paths(paths)
    def _generator() -> Iterable[str]:
        count = 0
        yielded_any = False
        for path in resolved_paths:
            if path.suffix.lower() == ".jsonl":
                with path.open("r", encoding="utf-8") as handle:
                    for line_number, line in enumerate(handle, start=1):
                        payload = line.strip()
                        if not payload:
                            continue
                        record = json.loads(payload)
                        text = record.get(jsonl_text_key)
                        if not isinstance(text, str):
                            raise ValueError(f"missing string field {jsonl_text_key!r} in {path}:{line_number}")
                        yielded_any = True
                        yield text
                        count += 1
                        if max_documents is not None and count >= max_documents:
                            return
            else:
                yielded_any = True
                yield path.read_text(encoding="utf-8")
                count += 1
                if max_documents is not None and count >= max_documents:
                    return
        if not yielded_any:
            raise ValueError("no corpus documents were loaded")

    return resolved_paths, _generator()


def tokenize_corpus_documents(
    documents: list[str] | tuple[str, ...],
    tokenizer: PreTrainedTokenizerBase,
    *,
    add_eos: bool = True,
) -> torch.Tensor:
    """Tokenize documents into one contiguous token stream."""

    if not documents:
        raise ValueError("documents must be non-empty")

    token_ids: list[int] = []
    for document in documents:
        encoded = tokenizer.encode(document, add_special_tokens=False)
        token_ids.extend(int(token_id) for token_id in encoded)
        if add_eos:
            if tokenizer.eos_token_id is None:
                raise ValueError("tokenizer must define eos_token_id when add_eos=True")
            token_ids.append(int(tokenizer.eos_token_id))

    if len(token_ids) < 2:
        raise ValueError("tokenized corpus must contain at least two tokens")
    return torch.tensor(token_ids, dtype=torch.long)


def build_token_stream_from_corpus(
    paths: list[str] | tuple[str, ...],
    tokenizer: PreTrainedTokenizerBase,
    *,
    jsonl_text_key: str = "text",
    max_documents: int | None = None,
    add_eos: bool = True,
) -> torch.Tensor:
    """Load and tokenize a text corpus into a single 1D token stream."""

    documents = load_corpus_documents(paths, jsonl_text_key=jsonl_text_key, max_documents=max_documents)
    return tokenize_corpus_documents(documents, tokenizer, add_eos=add_eos)


def _resolve_storage_dtype(storage_dtype: torch.dtype) -> tuple[str, np.dtype]:
    if storage_dtype == torch.int32:
        return "int32", np.int32
    if storage_dtype == torch.int64:
        return "int64", np.int64
    raise ValueError("storage_dtype must be torch.int32 or torch.int64")


def _resolve_numpy_dtype(storage_dtype: str) -> np.dtype:
    if storage_dtype == "int32":
        return np.int32
    if storage_dtype == "int64":
        return np.int64
    raise ValueError(f"unsupported storage dtype: {storage_dtype}")


def write_token_stream_from_corpus(
    output_path: str | Path,
    *,
    paths: list[str] | tuple[str, ...],
    tokenizer: PreTrainedTokenizerBase,
    jsonl_text_key: str = "text",
    max_documents: int | None = None,
    add_eos: bool = True,
    storage_dtype: torch.dtype = torch.int32,
) -> tuple[tuple[Path, ...], int, int]:
    """Stream-tokenize one corpus split directly into a binary token file."""

    dtype_name, numpy_dtype = _resolve_storage_dtype(storage_dtype)
    del dtype_name
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    resolved_paths, documents = iter_corpus_documents(
        paths,
        jsonl_text_key=jsonl_text_key,
        max_documents=max_documents,
    )
    document_count = 0
    token_count = 0
    with output_path.open("wb") as handle:
        for document in documents:
            encoded = tokenizer.encode(document, add_special_tokens=False)
            if add_eos:
                if tokenizer.eos_token_id is None:
                    raise ValueError("tokenizer must define eos_token_id when add_eos=True")
                encoded.append(int(tokenizer.eos_token_id))
            if encoded:
                np.asarray(encoded, dtype=numpy_dtype).tofile(handle)
                token_count += len(encoded)
            document_count += 1

    if token_count < 2:
        raise ValueError("tokenized corpus must contain at least two tokens")
    return resolved_paths, document_count, token_count


def split_token_stream(token_ids: torch.Tensor, *, val_fraction: float) -> CorpusSplit:
    """Split one token stream into contiguous train and validation segments."""

    if token_ids.ndim != 1:
        raise ValueError("token_ids must have shape [tokens]")
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1)")
    if val_fraction == 0.0:
        return CorpusSplit(train_token_ids=token_ids, val_token_ids=None)

    val_tokens = int(token_ids.numel() * val_fraction)
    if val_tokens < 2 or token_ids.numel() - val_tokens < 2:
        raise ValueError("val_fraction leaves too few tokens for a train/validation split")
    return CorpusSplit(
        train_token_ids=token_ids[:-val_tokens],
        val_token_ids=token_ids[-val_tokens:],
    )


def build_fixed_causal_lm_batches(
    token_ids: torch.Tensor,
    *,
    batch_size: int,
    seq_len: int,
    num_batches: int,
    generator: torch.Generator | None = None,
) -> tuple[CausalLMBatch, ...]:
    """Pre-sample fixed causal-LM batches from one token stream."""

    if num_batches <= 0:
        raise ValueError("num_batches must be positive")

    block_size = seq_len + 1
    batches: list[CausalLMBatch] = []
    for _ in range(num_batches):
        blocks = sample_token_blocks(
            token_ids,
            batch_size=batch_size,
            block_size=block_size,
            generator=generator,
        )
        batches.append(causal_lm_batch_from_token_block(blocks))
    return tuple(batches)


def save_tokenized_corpus_artifact(
    output_dir: str | Path,
    *,
    artifact: TokenizedCorpusArtifact,
    storage_dtype: torch.dtype = torch.int32,
) -> Path:
    """Persist tokenized train/validation streams and metadata for later training."""

    if storage_dtype not in {torch.int32, torch.int64}:
        raise ValueError("storage_dtype must be torch.int32 or torch.int64")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    storage_dtype_name, numpy_dtype = _resolve_storage_dtype(storage_dtype)
    train_token_ids = artifact.train_token_ids.detach().cpu().to(dtype=storage_dtype)
    if train_token_ids.ndim != 1:
        raise ValueError("artifact.train_token_ids must have shape [tokens]")
    np.asarray(train_token_ids.numpy(), dtype=numpy_dtype).tofile(output_path / _ARTIFACT_TRAIN_TOKENS_BINARY_FILENAME)

    val_token_ids = artifact.val_token_ids
    if val_token_ids is not None:
        val_to_save = val_token_ids.detach().cpu().to(dtype=storage_dtype)
        if val_to_save.ndim != 1:
            raise ValueError("artifact.val_token_ids must have shape [tokens]")
        np.asarray(val_to_save.numpy(), dtype=numpy_dtype).tofile(output_path / _ARTIFACT_VAL_TOKENS_BINARY_FILENAME)

    metadata = TokenizedCorpusMetadata(
        format_version=artifact.metadata.format_version,
        tokenizer_name=artifact.metadata.tokenizer_name,
        vocab_size=artifact.metadata.vocab_size,
        append_eos=artifact.metadata.append_eos,
        jsonl_text_key=artifact.metadata.jsonl_text_key,
        train_files=artifact.metadata.train_files,
        val_files=artifact.metadata.val_files,
        train_documents=artifact.metadata.train_documents,
        val_documents=artifact.metadata.val_documents,
        train_tokens=int(train_token_ids.numel()),
        val_tokens=0 if val_token_ids is None else int(val_token_ids.numel()),
        total_tokens=int(train_token_ids.numel()) + (0 if val_token_ids is None else int(val_token_ids.numel())),
        storage_dtype=storage_dtype_name,
        size_estimate=artifact.metadata.size_estimate,
    )
    metadata_path = output_path / _ARTIFACT_METADATA_FILENAME
    metadata_path.write_text(json.dumps(metadata.to_dict(), sort_keys=True, indent=2), encoding="utf-8")
    return output_path


def load_tokenized_corpus_artifact(path: str | Path) -> TokenizedCorpusArtifact:
    """Load tokenized train/validation streams and metadata from disk."""

    artifact_path = Path(path)
    metadata_path = artifact_path / _ARTIFACT_METADATA_FILENAME
    if not metadata_path.exists():
        raise ValueError(f"missing corpus metadata file: {metadata_path}")

    metadata = TokenizedCorpusMetadata.from_dict(json.loads(metadata_path.read_text(encoding="utf-8")))
    train_binary_path = artifact_path / _ARTIFACT_TRAIN_TOKENS_BINARY_FILENAME
    if train_binary_path.exists():
        numpy_dtype = _resolve_numpy_dtype(metadata.storage_dtype)
        train_array = np.memmap(train_binary_path, dtype=numpy_dtype, mode="r+", shape=(metadata.train_tokens,))
        train_token_ids = torch.from_numpy(train_array)
    else:
        train_token_ids = torch.load(artifact_path / _ARTIFACT_TRAIN_TOKENS_FILENAME, weights_only=False).to(dtype=torch.long)
    if train_token_ids.ndim != 1:
        raise ValueError("saved train_token_ids must have shape [tokens]")

    val_path = artifact_path / _ARTIFACT_VAL_TOKENS_BINARY_FILENAME
    val_token_ids = None
    if val_path.exists():
        numpy_dtype = _resolve_numpy_dtype(metadata.storage_dtype)
        val_array = np.memmap(val_path, dtype=numpy_dtype, mode="r+", shape=(metadata.val_tokens,))
        val_token_ids = torch.from_numpy(val_array)
        if val_token_ids.ndim != 1:
            raise ValueError("saved val_token_ids must have shape [tokens]")
    else:
        legacy_val_path = artifact_path / _ARTIFACT_VAL_TOKENS_FILENAME
        if legacy_val_path.exists():
            val_token_ids = torch.load(legacy_val_path, weights_only=False).to(dtype=torch.long)
            if val_token_ids.ndim != 1:
                raise ValueError("saved val_token_ids must have shape [tokens]")

    return TokenizedCorpusArtifact(
        metadata=metadata,
        train_token_ids=train_token_ids,
        val_token_ids=val_token_ids,
    )
