from __future__ import annotations

import json
from pathlib import Path

from multiscreen.corpus_build import (
    CorpusBuildSpec,
    CorpusBuildSplit,
    CorpusDatasetSource,
    allocate_source_document_counts,
    build_corpus_from_spec,
    load_corpus_build_spec,
    normalize_corpus_text,
)


def test_allocate_source_document_counts_matches_total_and_ratios() -> None:
    counts = allocate_source_document_counts(total_documents=10, ratios=[3.0, 2.0, 1.0])
    assert sum(counts) == 10
    assert counts == (5, 3, 2)


def test_normalize_corpus_text_strips_and_normalizes_newlines() -> None:
    assert normalize_corpus_text("  a\r\nb\r  ") == "a\nb"


def test_load_corpus_build_spec_reads_yaml_like_mapping(tmp_path) -> None:
    config_path = tmp_path / "corpus.yaml"
    config_path.write_text(
        json.dumps(
            {
                "version": 1,
                "splits": {
                    "train": {
                        "output": "train.jsonl",
                        "total_documents": 4,
                        "seed": 7,
                        "min_chars": 5,
                        "datasets": [
                            {
                                "path": "dataset-a",
                                "split": "train",
                                "ratio": 0.75,
                                "text_key": "body",
                                "streaming": False,
                            }
                        ],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    spec = load_corpus_build_spec(config_path)
    assert spec.version == 1
    assert spec.splits["train"].output == "train.jsonl"
    assert spec.splits["train"].min_chars == 5
    assert spec.splits["train"].datasets[0].text_key == "body"
    assert spec.splits["train"].datasets[0].streaming is False


def test_build_corpus_from_spec_writes_requested_ratio_counts(tmp_path, monkeypatch) -> None:
    spec = CorpusBuildSpec(
        version=1,
        splits={
            "train": CorpusBuildSplit(
                output="train.jsonl",
                total_documents=5,
                seed=0,
                min_chars=1,
                datasets=(
                    CorpusDatasetSource(path="dataset-a", split="train", ratio=3.0),
                    CorpusDatasetSource(path="dataset-b", split="train", ratio=2.0),
                ),
            )
        },
    )

    source_texts = {
        ("dataset-a", "train"): [f"a{i}" for i in range(10)],
        ("dataset-b", "train"): [f"b{i}" for i in range(10)],
    }

    def fake_iterate_source_texts(source: CorpusDatasetSource, *, seed: int, min_chars: int):
        del seed
        for text in source_texts[(source.path, source.split)]:
            if len(text) >= min_chars:
                yield text

    monkeypatch.setattr("multiscreen.corpus_build.iterate_source_texts", fake_iterate_source_texts)
    report = build_corpus_from_spec(spec, output_dir=tmp_path / "out")

    assert report.splits[0].written_documents == 5
    assert tuple(source.written_documents for source in report.splits[0].sources) == (3, 2)

    lines = (tmp_path / "out" / "train.jsonl").read_text(encoding="utf-8").strip().splitlines()
    payloads = [json.loads(line) for line in lines]
    assert [payload["text"] for payload in payloads] == ["a0", "a1", "a2", "b0", "b1"]
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["splits"][0]["written_documents"] == 5


def test_example_yaml_loads() -> None:
    config_path = Path(__file__).resolve().parents[2] / "configs" / "corpus.example.yaml"
    spec = load_corpus_build_spec(config_path)
    assert "train" in spec.splits
    assert "validation" in spec.splits
    assert spec.splits["train"].datasets[0].path == "roneneldan/TinyStories"
    assert spec.splits["validation"].datasets[0].split == "validation"
