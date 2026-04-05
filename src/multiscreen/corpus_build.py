"""YAML-driven corpus construction on top of Hugging Face datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable


@dataclass(slots=True)
class CorpusDatasetSource:
    """One dataset source inside a split build spec."""

    path: str
    split: str
    ratio: float
    text_key: str = "text"
    name: str | None = None
    revision: str | None = None
    data_files: str | list[str] | dict[str, str] | dict[str, list[str]] | None = None
    streaming: bool = True
    shuffle: bool = True
    shuffle_buffer: int = 10_000


@dataclass(slots=True)
class CorpusBuildSplit:
    """Build instructions for one output split."""

    output: str
    total_documents: int
    seed: int = 0
    min_chars: int = 1
    datasets: tuple[CorpusDatasetSource, ...] = ()


@dataclass(slots=True)
class CorpusBuildSpec:
    """Top-level corpus build configuration."""

    version: int
    splits: dict[str, CorpusBuildSplit]


@dataclass(slots=True)
class CorpusSourceBuildReport:
    """Observed output count for one source."""

    path: str
    split: str
    ratio: float
    requested_documents: int
    written_documents: int


@dataclass(slots=True)
class CorpusSplitBuildReport:
    """Observed output stats for one built split."""

    name: str
    output_path: str
    total_documents: int
    written_documents: int
    total_characters: int
    sources: tuple[CorpusSourceBuildReport, ...]


@dataclass(slots=True)
class CorpusBuildReport:
    """Manifest written after corpus construction finishes."""

    version: int
    splits: tuple[CorpusSplitBuildReport, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_yaml_or_json(path: Path) -> dict[str, Any]:
    payload = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(payload)
    except ModuleNotFoundError:
        data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("corpus build config must be a mapping")
    return data


def _require_mapping(value: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a mapping")
    return value


def _require_sequence(value: Any, *, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a list")
    return value


def load_corpus_build_spec(path: str | Path) -> CorpusBuildSpec:
    """Load a corpus build spec from YAML."""

    payload = _load_yaml_or_json(Path(path))
    version = int(payload.get("version", 1))
    splits_payload = _require_mapping(payload.get("splits"), context="splits")
    splits: dict[str, CorpusBuildSplit] = {}
    for split_name, split_payload_raw in splits_payload.items():
        split_payload = _require_mapping(split_payload_raw, context=f"splits.{split_name}")
        dataset_payloads = _require_sequence(split_payload.get("datasets"), context=f"splits.{split_name}.datasets")
        datasets: list[CorpusDatasetSource] = []
        for index, source_payload_raw in enumerate(dataset_payloads):
            source_payload = _require_mapping(source_payload_raw, context=f"splits.{split_name}.datasets[{index}]")
            datasets.append(
                CorpusDatasetSource(
                    path=str(source_payload["path"]),
                    split=str(source_payload["split"]),
                    ratio=float(source_payload.get("ratio", 1.0)),
                    text_key=str(source_payload.get("text_key", "text")),
                    name=None if source_payload.get("name") is None else str(source_payload["name"]),
                    revision=None if source_payload.get("revision") is None else str(source_payload["revision"]),
                    data_files=source_payload.get("data_files"),
                    streaming=bool(source_payload.get("streaming", True)),
                    shuffle=bool(source_payload.get("shuffle", True)),
                    shuffle_buffer=int(source_payload.get("shuffle_buffer", 10_000)),
                )
            )
        splits[str(split_name)] = CorpusBuildSplit(
            output=str(split_payload["output"]),
            total_documents=int(split_payload["total_documents"]),
            seed=int(split_payload.get("seed", 0)),
            min_chars=int(split_payload.get("min_chars", 1)),
            datasets=tuple(datasets),
        )
    return CorpusBuildSpec(version=version, splits=splits)


def allocate_source_document_counts(*, total_documents: int, ratios: Iterable[float]) -> tuple[int, ...]:
    """Allocate an exact document budget according to source ratios."""

    ratio_list = [float(ratio) for ratio in ratios]
    if total_documents <= 0:
        raise ValueError("total_documents must be positive")
    if not ratio_list:
        raise ValueError("ratios must be non-empty")
    if any(ratio <= 0.0 for ratio in ratio_list):
        raise ValueError("all ratios must be positive")

    total_ratio = sum(ratio_list)
    raw_counts = [float(total_documents) * ratio / total_ratio for ratio in ratio_list]
    counts = [int(raw_count) for raw_count in raw_counts]
    remainder = total_documents - sum(counts)
    order = sorted(
        range(len(raw_counts)),
        key=lambda index: (raw_counts[index] - counts[index], -index),
        reverse=True,
    )
    for index in order[:remainder]:
        counts[index] += 1
    return tuple(counts)


def normalize_corpus_text(text: str) -> str:
    """Apply light normalization before writing JSONL records."""

    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def _load_huggingface_dataset(source: CorpusDatasetSource) -> Any:
    try:
        from datasets import load_dataset  # type: ignore
    except ModuleNotFoundError as error:
        raise RuntimeError("datasets is required for build_corpus.py; install the project dependencies first") from error

    return load_dataset(
        path=source.path,
        name=source.name,
        split=source.split,
        revision=source.revision,
        data_files=source.data_files,
        streaming=source.streaming,
    )


def iterate_source_texts(
    source: CorpusDatasetSource,
    *,
    seed: int,
    min_chars: int,
) -> Iterable[str]:
    """Yield normalized text examples from one dataset source."""

    dataset = _load_huggingface_dataset(source)
    if source.shuffle:
        shuffle_kwargs: dict[str, Any] = {"seed": seed}
        if source.streaming:
            shuffle_kwargs["buffer_size"] = source.shuffle_buffer
        dataset = dataset.shuffle(**shuffle_kwargs)

    for example in dataset:
        if not isinstance(example, dict):
            raise ValueError(f"dataset example must be a mapping for source {source.path}:{source.split}")
        value = example.get(source.text_key)
        if not isinstance(value, str):
            raise ValueError(f"missing string field {source.text_key!r} in source {source.path}:{source.split}")
        text = normalize_corpus_text(value)
        if len(text) < min_chars:
            continue
        yield text


def build_corpus_from_spec(spec: CorpusBuildSpec, *, output_dir: str | Path) -> CorpusBuildReport:
    """Build JSONL corpus files from a YAML spec."""

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    split_reports: list[CorpusSplitBuildReport] = []
    for split_name, split_spec in spec.splits.items():
        if split_spec.total_documents <= 0:
            raise ValueError(f"{split_name}.total_documents must be positive")
        if split_spec.min_chars <= 0:
            raise ValueError(f"{split_name}.min_chars must be positive")
        if not split_spec.datasets:
            raise ValueError(f"{split_name}.datasets must be non-empty")

        output_path = output_root / split_spec.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        requested_counts = allocate_source_document_counts(
            total_documents=split_spec.total_documents,
            ratios=[source.ratio for source in split_spec.datasets],
        )

        total_written = 0
        total_characters = 0
        source_reports: list[CorpusSourceBuildReport] = []
        with output_path.open("w", encoding="utf-8") as handle:
            for source_index, (source, requested_documents) in enumerate(zip(split_spec.datasets, requested_counts, strict=True)):
                written_documents = 0
                iterator = iterate_source_texts(
                    source,
                    seed=split_spec.seed + source_index,
                    min_chars=split_spec.min_chars,
                )
                for text in iterator:
                    payload = {
                        "text": text,
                        "source_path": source.path,
                        "source_name": source.name,
                        "source_split": source.split,
                    }
                    handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
                    written_documents += 1
                    total_written += 1
                    total_characters += len(text)
                    if written_documents >= requested_documents:
                        break
                if written_documents != requested_documents:
                    raise ValueError(
                        f"source {source.path}:{source.split} ended after {written_documents} documents, "
                        f"expected {requested_documents}"
                    )
                source_reports.append(
                    CorpusSourceBuildReport(
                        path=source.path,
                        split=source.split,
                        ratio=source.ratio,
                        requested_documents=requested_documents,
                        written_documents=written_documents,
                    )
                )

        split_reports.append(
            CorpusSplitBuildReport(
                name=split_name,
                output_path=str(output_path),
                total_documents=split_spec.total_documents,
                written_documents=total_written,
                total_characters=total_characters,
                sources=tuple(source_reports),
            )
        )

    report = CorpusBuildReport(version=spec.version, splits=tuple(split_reports))
    (output_root / "manifest.json").write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report
