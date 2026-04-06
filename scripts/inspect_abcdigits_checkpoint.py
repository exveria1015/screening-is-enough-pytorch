from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import torch

from abcdigits import (
    build_abcdigits_eval_suite,
    build_gpt2_tokenizer,
    evaluate_abcdigits_exact_match,
)
from multiscreen.config import MultiscreenConfig
from multiscreen.model import MultiscreenLM


def parse_float_csv(value: str) -> tuple[float, ...]:
    parsed = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("expected at least one float")
    return parsed


def parse_int_csv(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return parsed


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def tail_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return "..." + text[-max_chars:]


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect ABCDigits predictions from a saved checkpoint.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--tokenizer-name", type=str, default=None)
    parser.add_argument("--num-equations", type=parse_int_csv, default=None)
    parser.add_argument("--depths", type=parse_float_csv, default=None)
    parser.add_argument("--examples-per-cell", type=int, default=5)
    parser.add_argument("--digits-per-value", type=int, default=None)
    parser.add_argument("--separator", type=str, default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--prompt-tail-chars", type=int, default=80)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_args = checkpoint.get("args", {})
    model = MultiscreenLM(MultiscreenConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"])

    device = resolve_device(args.device)
    model.to(device)

    tokenizer_name = args.tokenizer_name or checkpoint_args.get("tokenizer_name", "gpt2")
    digits_per_value = (
        args.digits_per_value if args.digits_per_value is not None else int(checkpoint_args.get("digits_per_value", 6))
    )
    separator = args.separator if args.separator is not None else str(checkpoint_args.get("separator", " "))
    num_equations = (
        args.num_equations
        if args.num_equations is not None
        else tuple(int(value) for value in checkpoint_args.get("eval_num_equations", [26]))
    )
    depths = (
        args.depths
        if args.depths is not None
        else tuple(float(value) for value in checkpoint_args.get("eval_depths", [0.5]))
    )

    tokenizer = build_gpt2_tokenizer(model_name=tokenizer_name, local_files_only=args.local_files_only)
    eval_suite = build_abcdigits_eval_suite(
        num_equations_values=num_equations,
        depths=depths,
        examples_per_cell=args.examples_per_cell,
        tokenizer=tokenizer,
        digits_per_value=digits_per_value,
        separator=separator,
        generator=torch.Generator().manual_seed(args.seed),
        add_eos=False,
    )

    print(f"checkpoint: {args.checkpoint}")
    print(f"saved_step: {checkpoint.get('step')}")
    print(f"saved_best_accuracy: {checkpoint.get('best_accuracy')}")
    print(f"device: {device}")
    print(f"tokenizer: {tokenizer_name}")
    print(
        "eval setup:",
        f"num_equations={num_equations}",
        f"depths={depths}",
        f"examples_per_cell={args.examples_per_cell}",
        f"digits_per_value={digits_per_value}",
        f"seed={args.seed}",
    )

    for cell in eval_suite:
        result = evaluate_abcdigits_exact_match(model, tokenizer, cell.tokenized_examples)
        prediction_counts = Counter(result.predictions)
        print(
            f"\ncell N={cell.num_equations} depth={cell.depth:.1f} "
            f"accuracy={result.accuracy:.3f} unique_predictions={len(prediction_counts)}"
        )
        top_predictions = ", ".join(
            f"{repr(prediction)} x{count}" for prediction, count in prediction_counts.most_common(3)
        )
        if top_predictions:
            print(f"top_predictions: {top_predictions}")
        for index, (example, prediction, target) in enumerate(
            zip(cell.tokenized_examples, result.predictions, result.targets, strict=True)
        ):
            match = prediction == target
            prompt_tail = tail_text(example.example.prompt, args.prompt_tail_chars)
            print(
                f"[{index}] query={example.example.query!r} pred={prediction!r} "
                f"target={target!r} match={match} prompt_tail={prompt_tail!r}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
