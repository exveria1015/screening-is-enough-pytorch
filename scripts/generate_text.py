from __future__ import annotations

import argparse
from pathlib import Path

import torch

from abcdigits import build_gpt2_tokenizer
from multiscreen.config import MultiscreenConfig
from multiscreen.generation import generate_tokens, truncate_prompt_tokens
from multiscreen.model import MultiscreenLM


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate text from a trained Multiscreen checkpoint.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--tokenizer-name", type=str, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--hide-prompt", action="store_true")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_args = checkpoint.get("args", {})
    tokenizer_name = args.tokenizer_name or checkpoint_args.get("tokenizer_name", "gpt2")
    tokenizer = build_gpt2_tokenizer(model_name=tokenizer_name, local_files_only=args.local_files_only)

    model = MultiscreenLM(MultiscreenConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    device = resolve_device(args.device)
    model.to(device)

    prompt_token_ids = torch.tensor(tokenizer.encode(args.prompt, add_special_tokens=False), dtype=torch.long)
    truncated_prompt = truncate_prompt_tokens(prompt_token_ids, max_seq_len=model.config.max_seq_len)
    if truncated_prompt.numel() == 0:
        raise ValueError("prompt must tokenize to at least one token")

    generator = torch.Generator(device=device.type).manual_seed(args.seed)
    generated = generate_tokens(
        model,
        truncated_prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=None if args.greedy else args.top_k,
        greedy=args.greedy,
        eos_token_id=tokenizer.eos_token_id,
        generator=generator,
    ).cpu()

    prompt_text = tokenizer.decode(truncated_prompt.tolist(), clean_up_tokenization_spaces=False)
    generated_text = tokenizer.decode(generated.tolist(), clean_up_tokenization_spaces=False)
    completion_text = generated_text[len(prompt_text) :] if generated_text.startswith(prompt_text) else generated_text

    print(f"checkpoint: {args.checkpoint}")
    print(f"device: {device}")
    print(f"step: {checkpoint.get('step')}")
    print(f"max_seq_len: {model.config.max_seq_len}")
    print(f"prompt_tokens: {int(prompt_token_ids.numel())} -> used_prompt_tokens: {int(truncated_prompt.numel())}")
    if not args.hide_prompt:
        print("\n[PROMPT]")
        print(prompt_text)
    print("\n[COMPLETION]")
    print(completion_text)
    print("\n[GENERATED]")
    print(generated_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
