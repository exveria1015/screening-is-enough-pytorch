from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

from multiscreen.corpus_build import build_corpus_from_spec, load_corpus_build_spec


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a JSONL corpus from a YAML spec using Hugging Face datasets.")
    parser.add_argument("--config", type=Path, required=True, help="YAML corpus build specification")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for built JSONL files")
    args = parser.parse_args()

    spec = load_corpus_build_spec(args.config)
    report = build_corpus_from_spec(spec, output_dir=args.output_dir)
    for split in report.splits:
        print(
            "built split:",
            f"name={split.name}",
            f"output={split.output_path}",
            f"documents={split.written_documents}",
            f"characters={split.total_characters}",
        )
    print(f"manifest: {args.output_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    exit_code = main()
    # `datasets` streaming can abort during interpreter teardown on some local installs.
    # At this point every file has been flushed and closed, so a hard exit is safer.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
