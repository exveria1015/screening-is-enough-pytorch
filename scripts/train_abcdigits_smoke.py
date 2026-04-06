from __future__ import annotations

import argparse

import torch

from abcdigits import (
    ABCDigitsConfig,
    build_abcdigits_causal_lm_batch,
    build_gpt2_tokenizer,
    evaluate_abcdigits_exact_match,
    sample_tokenized_abcdigits_examples,
)
from multiscreen.config import MultiscreenConfig
from multiscreen.model import MultiscreenLM
from multiscreen.train import OptimizerConfig, build_optimizer, evaluate_loss, train_step


def main() -> int:
    parser = argparse.ArgumentParser(description="Overfit a Multiscreen model on a fixed ABCDigits batch.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--num-equations", type=int, default=26)
    parser.add_argument("--depth", type=float, default=0.5)
    parser.add_argument("--digits-per-value", type=int, default=6)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--n-heads", type=int, default=1)
    parser.add_argument("--d-key", type=int, default=8)
    parser.add_argument("--d-value", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    tokenizer = build_gpt2_tokenizer()
    generator = torch.Generator().manual_seed(args.seed)
    tokenized_examples = sample_tokenized_abcdigits_examples(
        batch_size=args.batch_size,
        config=ABCDigitsConfig(
            num_equations=args.num_equations,
            depth=args.depth,
            digits_per_value=args.digits_per_value,
        ),
        tokenizer=tokenizer,
        generator=generator,
        add_eos=True,
    )
    batch = build_abcdigits_causal_lm_batch(
        tokenized_examples,
        pad_token_id=int(tokenizer.pad_token_id),
        supervise_completion_only=True,
    )
    max_train_seq_len = batch.input_ids.shape[1]
    max_eval_seq_len = max(
        len(example.prompt_ids) + len(example.example.completion) - 1 for example in tokenized_examples
    )
    model = MultiscreenLM(
        MultiscreenConfig(
            vocab_size=tokenizer.vocab_size,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            d_key=args.d_key,
            d_value=args.d_value,
            max_seq_len=max(max_train_seq_len, max_eval_seq_len),
            max_train_seq_len=max_train_seq_len,
        )
    )
    optimizer = build_optimizer(model, OptimizerConfig(lr=args.lr))

    print(f"initial_loss: {evaluate_loss(model, batch):.6f}")
    print(f"initial_accuracy: {evaluate_abcdigits_exact_match(model, tokenizer, tokenized_examples).accuracy:.6f}")
    for step in range(args.steps):
        result = train_step(model, optimizer, batch, grad_clip=args.grad_clip)
        if step in {0, args.steps // 2, args.steps - 1}:
            print(f"step {step + 1}: loss={result.loss:.6f} grad_norm={result.grad_norm:.6f}")
    print(f"final_loss: {evaluate_loss(model, batch):.6f}")
    print(f"final_accuracy: {evaluate_abcdigits_exact_match(model, tokenizer, tokenized_examples).accuracy:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
