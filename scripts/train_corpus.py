from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from abcdigits import build_gpt2_tokenizer
from multiscreen.config import MultiscreenConfig
from multiscreen.corpus import (
    TokenizedCorpusArtifact,
    build_fixed_causal_lm_batches,
    build_token_stream_from_corpus,
    expand_corpus_paths,
    load_tokenized_corpus_artifact,
    split_token_stream,
)
from multiscreen.data import causal_lm_batch_from_token_block, sample_token_blocks
from multiscreen.model import MultiscreenLM
from multiscreen.sizing import multiscreen_parameter_count
from multiscreen.train import (
    OptimizerConfig,
    build_optimizer,
    evaluate_loss,
    set_optimizer_eval_mode,
    should_log_step,
    train_step,
)


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
    best_val_loss: float | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "best_val_loss": best_val_loss,
            "args": vars(args),
            "model_config": asdict(model.config),
        },
        path,
    )


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def build_model_config(
    args: argparse.Namespace,
    *,
    vocab_size: int,
    prepared_artifact: TokenizedCorpusArtifact | None = None,
) -> MultiscreenConfig:
    if args.use_recommended_config:
        if prepared_artifact is None:
            raise ValueError("--use-recommended-config requires --prepared-corpus")
        size_estimate = prepared_artifact.metadata.size_estimate
        if size_estimate is None:
            raise ValueError("prepared corpus does not contain a size estimate")
        return size_estimate.build_config(
            max_seq_len=args.seq_len,
            max_train_seq_len=args.seq_len,
        )
    if args.psi is not None:
        return MultiscreenConfig.from_psi(
            psi=args.psi,
            vocab_size=vocab_size,
            d_key=args.d_key,
            d_value=args.d_value,
            max_seq_len=args.seq_len,
            max_train_seq_len=args.seq_len,
        )
    return MultiscreenConfig(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_key=args.d_key,
        d_value=args.d_value,
        max_seq_len=args.seq_len,
        max_train_seq_len=args.seq_len,
    )


def load_training_token_streams(
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor | None, int, str, TokenizedCorpusArtifact | None]:
    if args.prepared_corpus is not None:
        artifact = load_tokenized_corpus_artifact(args.prepared_corpus)
        return (
            artifact.train_token_ids,
            artifact.val_token_ids,
            artifact.metadata.vocab_size,
            f"prepared:{args.prepared_corpus}",
            artifact,
        )

    tokenizer = build_gpt2_tokenizer(model_name=args.tokenizer_name, local_files_only=args.local_files_only)
    train_files = expand_corpus_paths(args.train_path)
    if args.val_path:
        val_files = expand_corpus_paths(args.val_path)
        train_token_ids = build_token_stream_from_corpus(
            [str(path) for path in train_files],
            tokenizer,
            jsonl_text_key=args.jsonl_text_key,
            max_documents=args.max_train_documents,
            add_eos=args.append_eos,
        )
        val_token_ids = build_token_stream_from_corpus(
            [str(path) for path in val_files],
            tokenizer,
            jsonl_text_key=args.jsonl_text_key,
            max_documents=args.max_val_documents,
            add_eos=args.append_eos,
        )
    else:
        full_token_ids = build_token_stream_from_corpus(
            [str(path) for path in train_files],
            tokenizer,
            jsonl_text_key=args.jsonl_text_key,
            max_documents=args.max_train_documents,
            add_eos=args.append_eos,
        )
        split = split_token_stream(full_token_ids, val_fraction=args.val_fraction)
        train_token_ids = split.train_token_ids
        val_token_ids = split.val_token_ids
    return train_token_ids, val_token_ids, tokenizer.vocab_size, "raw", None


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Multiscreen on a text corpus with GPT-2 tokenization.")
    parser.add_argument("--train-path", action="append", default=[], help="File, directory, or glob for training text/jsonl")
    parser.add_argument("--val-path", action="append", default=[], help="Optional file, directory, or glob for validation")
    parser.add_argument("--prepared-corpus", type=Path, default=None, help="Directory produced by prepare_corpus.py")
    parser.add_argument("--use-recommended-config", action="store_true")
    parser.add_argument("--jsonl-text-key", type=str, default="text")
    parser.add_argument("--max-train-documents", type=int, default=None)
    parser.add_argument("--max-val-documents", type=int, default=None)
    parser.add_argument("--val-fraction", type=float, default=0.01)
    parser.add_argument("--tokenizer-name", type=str, default="gpt2")
    parser.add_argument("--append-eos", dest="append_eos", action="store_true")
    parser.add_argument("--no-append-eos", dest="append_eos", action="store_false")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--psi", type=int, default=None)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=16)
    parser.add_argument("--n-heads", type=int, default=16)
    parser.add_argument("--d-key", type=int, default=16)
    parser.add_argument("--d-value", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2 ** -4)
    parser.add_argument("--optimizer", choices=("adamw", "adamw_schedulefree"), default="adamw")
    parser.add_argument("--schedulefree-warmup-steps", type=int, default=0)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/corpus-train"))
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.set_defaults(append_eos=True)
    args = parser.parse_args()

    if args.seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if args.steps <= 0:
        raise ValueError("steps must be positive")
    if args.log_interval < 0:
        raise ValueError("log_interval must be non-negative")
    if args.eval_interval < 0:
        raise ValueError("eval_interval must be non-negative")
    if args.save_every < 0:
        raise ValueError("save_every must be non-negative")
    if args.prepared_corpus is None and not args.train_path:
        raise ValueError("either --prepared-corpus or --train-path must be provided")
    if args.prepared_corpus is not None and args.train_path:
        raise ValueError("--prepared-corpus and --train-path are mutually exclusive")
    if args.prepared_corpus is not None and args.val_path:
        raise ValueError("--prepared-corpus and --val-path are mutually exclusive")

    torch.manual_seed(args.seed)
    train_generator = torch.Generator().manual_seed(args.seed + 1)
    eval_generator = torch.Generator().manual_seed(args.seed + 2)
    train_token_ids, val_token_ids, vocab_size, data_source, prepared_artifact = load_training_token_streams(args)

    block_size = args.seq_len + 1
    if train_token_ids.numel() < block_size:
        raise ValueError("training token stream is shorter than seq_len + 1")
    fixed_eval_batches = None
    if val_token_ids is not None:
        if val_token_ids.numel() < block_size:
            raise ValueError("validation token stream is shorter than seq_len + 1")
        fixed_eval_batches = build_fixed_causal_lm_batches(
            val_token_ids,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            num_batches=args.eval_batches,
            generator=eval_generator,
        )

    model = MultiscreenLM(build_model_config(args, vocab_size=vocab_size, prepared_artifact=prepared_artifact))
    device = resolve_device(args.device)
    model.to(device)
    parameter_count = multiscreen_parameter_count(model.config)
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

    best_val_loss = None
    append_eos = args.append_eos if prepared_artifact is None else prepared_artifact.metadata.append_eos
    print(
        "train setup:",
        f"device={device}",
        f"data_source={data_source}",
        f"steps={args.steps}",
        f"seq_len={args.seq_len}",
        f"batch_size={args.batch_size}",
        f"parameters={parameter_count}",
        f"optimizer={args.optimizer}",
        f"schedulefree_warmup_steps={args.schedulefree_warmup_steps}",
        f"train_tokens={train_token_ids.numel()}",
        f"val_tokens={0 if val_token_ids is None else val_token_ids.numel()}",
        f"append_eos={append_eos}",
    )
    for step in range(1, args.steps + 1):
        blocks = sample_token_blocks(
            train_token_ids,
            batch_size=args.batch_size,
            block_size=block_size,
            generator=train_generator,
        )
        batch = causal_lm_batch_from_token_block(blocks)
        result = train_step(model, optimizer, batch, grad_clip=args.grad_clip)
        train_payload = {
            "kind": "train",
            "step": step,
            "loss": result.loss,
            "grad_norm": result.grad_norm,
        }
        append_jsonl(metrics_path, train_payload)
        if should_log_step(step, total_steps=args.steps, log_interval=args.log_interval):
            print(f"step {step}: loss={result.loss:.6f} grad_norm={result.grad_norm:.6f}")

        should_eval = fixed_eval_batches is not None and args.eval_interval > 0 and (
            step % args.eval_interval == 0 or step == args.steps
        )
        if should_eval:
            val_losses = [evaluate_loss(model, eval_batch, optimizer=optimizer) for eval_batch in fixed_eval_batches]
            mean_val_loss = sum(val_losses) / float(len(val_losses))
            eval_payload = {
                "kind": "eval",
                "step": step,
                "val_loss": mean_val_loss,
                "eval_batches": len(val_losses),
            }
            append_jsonl(metrics_path, eval_payload)
            print(f"eval {step}: val_loss={mean_val_loss:.6f}")

            save_checkpoint(
                args.output_dir / "checkpoint_last.pt",
                model=model,
                optimizer=optimizer,
                args=args,
                step=step,
                best_val_loss=mean_val_loss if best_val_loss is None else min(best_val_loss, mean_val_loss),
            )
            if best_val_loss is None or mean_val_loss < best_val_loss:
                best_val_loss = mean_val_loss
                save_checkpoint(
                    args.output_dir / "checkpoint_best.pt",
                    model=model,
                    optimizer=optimizer,
                    args=args,
                    step=step,
                    best_val_loss=best_val_loss,
                )

        if args.save_every > 0 and step % args.save_every == 0:
            set_optimizer_eval_mode(optimizer)
            save_checkpoint(
                args.output_dir / f"checkpoint_step_{step}.pt",
                model=model,
                optimizer=optimizer,
                args=args,
                step=step,
                best_val_loss=best_val_loss,
            )

    print(f"finished: best_val_loss={best_val_loss} output_dir={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
