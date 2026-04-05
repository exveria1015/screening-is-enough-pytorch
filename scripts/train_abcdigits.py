from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from abcdigits import (  # noqa: E402
    ABCDigitsCurriculumConfig,
    build_abcdigits_eval_suite,
    build_abcdigits_training_pool,
    build_gpt2_tokenizer,
    estimate_abcdigits_max_token_length,
    evaluate_abcdigits_grid,
    sample_abcdigits_training_batch,
    sample_abcdigits_training_batch_from_pool,
)
from multiscreen.config import MultiscreenConfig  # noqa: E402
from multiscreen.model import MultiscreenLM  # noqa: E402
from multiscreen.train import (  # noqa: E402
    OptimizerConfig,
    build_optimizer,
    set_optimizer_eval_mode,
    train_step,
)


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


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def save_checkpoint(
    path: Path,
    *,
    model: MultiscreenLM,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    step: int,
    best_accuracy: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "best_accuracy": best_accuracy,
            "args": vars(args),
            "model_config": asdict(model.config),
        },
        path,
    )


def format_grid_summary(points: tuple, *, max_points: int = 6) -> str:
    preview = []
    for point in points[:max_points]:
        preview.append(
            f"N={point.num_equations},d={point.depth:.1f}:"
            f"exact={point.accuracy:.3f},digit={point.digit_accuracy:.3f},uniq={point.unique_prediction_ratio:.3f}"
        )
    return " ".join(preview)


def summarize_training_batch(tokenized_examples: tuple) -> tuple[str, str]:
    num_equations_values = sorted({example.example.config.num_equations for example in tokenized_examples})
    depth_values = sorted({example.example.config.depth for example in tokenized_examples})
    equation_summary = (
        str(num_equations_values[0])
        if len(num_equations_values) == 1
        else f"{num_equations_values[0]}-{num_equations_values[-1]}"
    )
    depth_summary = (
        f"{depth_values[0]:.1f}"
        if len(depth_values) == 1
        else ",".join(f"{depth:.1f}" for depth in depth_values)
    )
    return equation_summary, depth_summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Multiscreen on sampled ABCDigits batches.")
    parser.add_argument("--tokenizer-name", type=str, default="gpt2")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--train-min-num-equations", type=int, default=26)
    parser.add_argument("--train-max-num-equations", type=int, default=52)
    parser.add_argument("--train-depths", type=parse_float_csv, default=(0.1, 0.3, 0.5, 0.7, 0.9))
    parser.add_argument("--train-pool-size", type=int, default=0)
    parser.add_argument("--digits-per-value", type=int, default=6)
    parser.add_argument("--separator", type=str, default=" ")
    parser.add_argument("--train-add-eos", dest="train_add_eos", action="store_true")
    parser.add_argument("--train-no-add-eos", dest="train_add_eos", action="store_false")
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=2)
    parser.add_argument("--d-key", type=int, default=16)
    parser.add_argument("--d-value", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--optimizer", choices=("adamw", "adamw_schedulefree"), default="adamw")
    parser.add_argument("--schedulefree-warmup-steps", type=int, default=0)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-examples-per-cell", type=int, default=20)
    parser.add_argument("--eval-num-equations", type=parse_int_csv, default=(26, 52, 104))
    parser.add_argument("--eval-depths", type=parse_float_csv, default=(0.1, 0.3, 0.5, 0.7, 0.9))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/abcdigits"))
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.set_defaults(train_add_eos=False)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    train_generator = torch.Generator().manual_seed(args.seed + 1)
    eval_generator = torch.Generator().manual_seed(args.seed + 2)
    pool_generator = torch.Generator().manual_seed(args.seed + 3)
    pool_sample_generator = torch.Generator().manual_seed(args.seed + 4)
    tokenizer = build_gpt2_tokenizer(model_name=args.tokenizer_name, local_files_only=args.local_files_only)

    curriculum = ABCDigitsCurriculumConfig(
        min_num_equations=args.train_min_num_equations,
        max_num_equations=args.train_max_num_equations,
        depths=args.train_depths,
        digits_per_value=args.digits_per_value,
        separator=args.separator,
        add_eos=args.train_add_eos,
    )
    eval_suite = build_abcdigits_eval_suite(
        num_equations_values=args.eval_num_equations,
        depths=args.eval_depths,
        examples_per_cell=args.eval_examples_per_cell,
        tokenizer=tokenizer,
        digits_per_value=args.digits_per_value,
        separator=args.separator,
        generator=eval_generator,
        add_eos=False,
    )
    training_pool = None
    if args.train_pool_size > 0:
        training_pool = build_abcdigits_training_pool(
            pool_size=args.train_pool_size,
            curriculum=curriculum,
            tokenizer=tokenizer,
            generator=pool_generator,
        )

    max_num_equations = max(args.train_max_num_equations, max(args.eval_num_equations))
    max_seq_len = estimate_abcdigits_max_token_length(
        num_equations=max_num_equations,
        digits_per_value=args.digits_per_value,
        separator=args.separator,
        add_eos=args.train_add_eos,
    )
    model = MultiscreenLM(
        MultiscreenConfig(
            vocab_size=tokenizer.vocab_size,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            d_key=args.d_key,
            d_value=args.d_value,
            max_seq_len=max_seq_len,
            max_train_seq_len=max_seq_len,
        )
    )
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    model.to(device)

    optimizer = build_optimizer(
        model,
        OptimizerConfig(
            lr=args.lr,
            optimizer_name=args.optimizer,
            beta1=args.beta1,
            beta2=args.beta2,
            weight_decay=args.weight_decay,
            warmup_steps=args.schedulefree_warmup_steps,
        ),
    )

    metrics_path = args.output_dir / "metrics.jsonl"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "run_config.json").write_text(json.dumps(vars(args), default=str, indent=2), encoding="utf-8")

    best_accuracy = -1.0
    print(
        "train setup:",
        f"device={device}",
        f"steps={args.steps}",
        f"batch_size={args.batch_size}",
        f"optimizer={args.optimizer}",
        f"schedulefree_warmup_steps={args.schedulefree_warmup_steps}",
        f"train_equations=[{args.train_min_num_equations},{args.train_max_num_equations}]",
        f"train_add_eos={args.train_add_eos}",
        f"train_pool_size={args.train_pool_size}",
        f"max_seq_len={max_seq_len}",
    )
    for step in range(1, args.steps + 1):
        if training_pool is None:
            sampled = sample_abcdigits_training_batch(
                batch_size=args.batch_size,
                curriculum=curriculum,
                tokenizer=tokenizer,
                generator=train_generator,
                supervise_completion_only=True,
            )
        else:
            sampled = sample_abcdigits_training_batch_from_pool(
                batch_size=args.batch_size,
                training_pool=training_pool,
                tokenizer=tokenizer,
                generator=pool_sample_generator,
                supervise_completion_only=True,
            )
        result = train_step(model, optimizer, sampled.batch, grad_clip=args.grad_clip)
        num_equations_summary, depth_summary = summarize_training_batch(sampled.tokenized_examples)
        train_payload = {
            "kind": "train",
            "step": step,
            "loss": result.loss,
            "grad_norm": result.grad_norm,
            "num_equations": num_equations_summary,
            "depth": depth_summary,
            "train_pool_size": args.train_pool_size,
        }
        append_jsonl(metrics_path, train_payload)
        if step == 1 or step % args.log_interval == 0 or step == args.steps:
            print(
                f"step {step}: loss={result.loss:.6f} grad_norm={result.grad_norm:.6f} "
                f"num_equations={num_equations_summary} depth={depth_summary}"
            )

        should_eval = args.eval_interval > 0 and (step % args.eval_interval == 0 or step == args.steps)
        if should_eval:
            set_optimizer_eval_mode(optimizer)
            grid = evaluate_abcdigits_grid(model, tokenizer, eval_suite)
            eval_payload = {
                "kind": "eval",
                "step": step,
                "mean_accuracy": grid.mean_accuracy,
                "mean_digit_accuracy": grid.mean_digit_accuracy,
                "mean_unique_prediction_ratio": grid.mean_unique_prediction_ratio,
                "points": [
                    {
                        "num_equations": point.num_equations,
                        "depth": point.depth,
                        "accuracy": point.accuracy,
                        "digit_accuracy": point.digit_accuracy,
                        "count": point.count,
                        "unique_prediction_count": point.unique_prediction_count,
                        "unique_prediction_ratio": point.unique_prediction_ratio,
                    }
                    for point in grid.points
                ],
            }
            append_jsonl(metrics_path, eval_payload)
            print(
                f"eval {step}: mean_accuracy={grid.mean_accuracy:.6f} "
                f"mean_digit_accuracy={grid.mean_digit_accuracy:.6f} "
                f"mean_unique_prediction_ratio={grid.mean_unique_prediction_ratio:.6f} "
                f"{format_grid_summary(grid.points)}"
            )
            save_checkpoint(
                args.output_dir / "checkpoint_last.pt",
                model=model,
                optimizer=optimizer,
                args=args,
                step=step,
                best_accuracy=max(best_accuracy, grid.mean_accuracy),
            )
            if grid.mean_accuracy > best_accuracy:
                best_accuracy = grid.mean_accuracy
                save_checkpoint(
                    args.output_dir / "checkpoint_best.pt",
                    model=model,
                    optimizer=optimizer,
                    args=args,
                    step=step,
                    best_accuracy=best_accuracy,
                )

        if args.save_every > 0 and step % args.save_every == 0:
            set_optimizer_eval_mode(optimizer)
            save_checkpoint(
                args.output_dir / f"checkpoint_step_{step}.pt",
                model=model,
                optimizer=optimizer,
                args=args,
                step=step,
                best_accuracy=best_accuracy,
            )

    print(f"finished: best_mean_accuracy={best_accuracy:.6f} output_dir={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
