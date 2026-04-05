from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from abcdigits import build_gpt2_tokenizer
from multiscreen.corpus import (
    TokenizedCorpusArtifact,
    TokenizedCorpusMetadata,
    build_fixed_causal_lm_batches,
    build_token_stream_from_corpus,
    expand_corpus_paths,
    iter_corpus_documents,
    load_corpus_documents,
    load_tokenized_corpus_artifact,
    save_tokenized_corpus_artifact,
    split_token_stream,
    tokenize_corpus_documents,
    write_token_stream_from_corpus,
)
from multiscreen.sizing import estimate_multiscreen_size_from_token_count


def test_expand_corpus_paths_supports_files_directories_and_globs(tmp_path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    text_path = docs / "a.txt"
    jsonl_path = docs / "b.jsonl"
    ignored_path = docs / "c.bin"
    text_path.write_text("hello", encoding="utf-8")
    jsonl_path.write_text(json.dumps({"text": "world"}) + "\n", encoding="utf-8")
    ignored_path.write_bytes(b"\x00")

    resolved = expand_corpus_paths([str(text_path), str(docs), str(docs / "*.jsonl")])
    assert text_path.resolve() in resolved
    assert jsonl_path.resolve() in resolved
    assert ignored_path.resolve() not in resolved


def test_load_corpus_documents_reads_text_and_jsonl(tmp_path) -> None:
    text_path = tmp_path / "train.txt"
    jsonl_path = tmp_path / "train.jsonl"
    text_path.write_text("alpha", encoding="utf-8")
    jsonl_path.write_text(
        json.dumps({"text": "beta"}) + "\n" + json.dumps({"text": "gamma"}) + "\n",
        encoding="utf-8",
    )

    documents = load_corpus_documents([str(text_path), str(jsonl_path)])
    assert documents == ("alpha", "beta", "gamma")


def test_tokenize_corpus_documents_appends_eos_between_documents() -> None:
    tokenizer = build_gpt2_tokenizer()
    token_ids = tokenize_corpus_documents(("hello", "world"), tokenizer, add_eos=True)
    assert token_ids.dtype == torch.long
    assert int(token_ids[-1].item()) == int(tokenizer.eos_token_id)
    assert (token_ids == int(tokenizer.eos_token_id)).sum().item() == 2


def test_build_token_stream_from_corpus_tokenizes_loaded_documents(tmp_path) -> None:
    text_path = tmp_path / "train.txt"
    text_path.write_text("hello world", encoding="utf-8")
    tokenizer = build_gpt2_tokenizer()

    token_ids = build_token_stream_from_corpus([str(text_path)], tokenizer, add_eos=False)
    assert token_ids.ndim == 1
    assert token_ids.numel() >= 2


def test_split_token_stream_returns_contiguous_train_and_validation_segments() -> None:
    token_ids = torch.arange(20, dtype=torch.long)
    split = split_token_stream(token_ids, val_fraction=0.2)
    assert torch.equal(split.train_token_ids, torch.arange(16, dtype=torch.long))
    assert torch.equal(split.val_token_ids, torch.arange(16, 20, dtype=torch.long))


def test_build_fixed_causal_lm_batches_returns_requested_batch_count() -> None:
    token_ids = torch.arange(100, dtype=torch.long)
    batches = build_fixed_causal_lm_batches(
        token_ids,
        batch_size=2,
        seq_len=8,
        num_batches=3,
        generator=torch.Generator().manual_seed(0),
    )
    assert len(batches) == 3
    for batch in batches:
        assert batch.input_ids.shape == (2, 8)
        assert batch.labels.shape == (2, 8)


def test_load_corpus_documents_rejects_missing_jsonl_text_field(tmp_path) -> None:
    jsonl_path = tmp_path / "broken.jsonl"
    jsonl_path.write_text(json.dumps({"body": "missing"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing string field"):
        load_corpus_documents([str(jsonl_path)], jsonl_text_key="text")


def test_iter_corpus_documents_streams_text_and_jsonl(tmp_path) -> None:
    text_path = tmp_path / "a.txt"
    jsonl_path = tmp_path / "b.jsonl"
    text_path.write_text("alpha", encoding="utf-8")
    jsonl_path.write_text(json.dumps({"text": "beta"}) + "\n", encoding="utf-8")
    resolved, documents = iter_corpus_documents([str(text_path), str(jsonl_path)])
    assert tuple(path.name for path in resolved) == ("a.txt", "b.jsonl")
    assert tuple(documents) == ("alpha", "beta")


def test_write_token_stream_from_corpus_creates_binary_stream(tmp_path) -> None:
    text_path = tmp_path / "train.txt"
    text_path.write_text("hello world\nagain", encoding="utf-8")
    tokenizer = build_gpt2_tokenizer()
    resolved, document_count, token_count = write_token_stream_from_corpus(
        tmp_path / "train_tokens.bin",
        paths=[str(text_path)],
        tokenizer=tokenizer,
        add_eos=True,
        storage_dtype=torch.int32,
    )
    assert resolved == (text_path.resolve(),)
    assert document_count == 1
    assert token_count >= 2
    reloaded = load_tokenized_corpus_artifact(
        save_tokenized_corpus_artifact(
            tmp_path / "artifact",
            artifact=TokenizedCorpusArtifact(
                metadata=TokenizedCorpusMetadata(
                    format_version=1,
                    tokenizer_name="gpt2",
                    vocab_size=tokenizer.vocab_size,
                    append_eos=True,
                    jsonl_text_key="text",
                    train_files=(str(text_path.resolve()),),
                    val_files=(),
                    train_documents=1,
                    val_documents=0,
                    train_tokens=token_count,
                    val_tokens=0,
                    total_tokens=token_count,
                    storage_dtype="int32",
                    size_estimate=None,
                ),
                train_token_ids=torch.from_numpy(np.fromfile(tmp_path / "train_tokens.bin", dtype="int32")).to(dtype=torch.long),
                val_token_ids=None,
            ),
            storage_dtype=torch.int32,
        )
    )
    assert reloaded.train_token_ids.numel() == token_count


def test_tokenized_corpus_artifact_roundtrips_tokens_and_metadata(tmp_path) -> None:
    estimate = estimate_multiscreen_size_from_token_count(1_000_000, vocab_size=50_257, min_psi=1, max_psi=16)
    artifact = TokenizedCorpusArtifact(
        metadata=TokenizedCorpusMetadata(
            format_version=1,
            tokenizer_name="gpt2",
            vocab_size=50_257,
            append_eos=True,
            jsonl_text_key="text",
            train_files=("train.txt",),
            val_files=("val.txt",),
            train_documents=10,
            val_documents=2,
            train_tokens=6,
            val_tokens=2,
            total_tokens=8,
            storage_dtype="int32",
            size_estimate=estimate,
        ),
        train_token_ids=torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.long),
        val_token_ids=torch.tensor([7, 8], dtype=torch.long),
    )
    save_tokenized_corpus_artifact(tmp_path / "prepared", artifact=artifact, storage_dtype=torch.int32)

    loaded = load_tokenized_corpus_artifact(tmp_path / "prepared")
    assert torch.equal(loaded.train_token_ids, artifact.train_token_ids)
    assert torch.equal(loaded.val_token_ids, artifact.val_token_ids)
    assert loaded.metadata.tokenizer_name == "gpt2"
    assert loaded.metadata.size_estimate is not None
    assert loaded.metadata.size_estimate.recommended_psi == estimate.recommended_psi
