"""Training utilities for Multiscreen language-model experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F

from multiscreen.data import CausalLMBatch
from multiscreen.model import MultiscreenLM


@dataclass(slots=True)
class OptimizerConfig:
    """Optimizer hyperparameters used for training."""

    lr: float
    optimizer_name: str = "adamw"
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.0
    warmup_steps: int = 0

    def __post_init__(self) -> None:
        if self.lr <= 0.0:
            raise ValueError("lr must be positive")
        if self.optimizer_name not in {"adamw", "adamw_schedulefree"}:
            raise ValueError("optimizer_name must be 'adamw' or 'adamw_schedulefree'")
        if not 0.0 <= self.beta1 < 1.0:
            raise ValueError("beta1 must be in [0, 1)")
        if not 0.0 <= self.beta2 < 1.0:
            raise ValueError("beta2 must be in [0, 1)")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")


@dataclass(slots=True)
class TrainStepResult:
    """Outputs from a single optimization step."""

    loss: float
    grad_norm: float


def model_device(model: torch.nn.Module) -> torch.device:
    """Return the device of the first model parameter."""

    return next(model.parameters()).device


def causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor, *, ignore_index: int = -100) -> torch.Tensor:
    """Cross-entropy loss for next-token prediction."""

    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, seq, vocab]")
    if labels.ndim != 2:
        raise ValueError("labels must have shape [batch, seq]")
    if logits.shape[:2] != labels.shape:
        raise ValueError("logits and labels must agree on batch and seq dimensions")
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=ignore_index)


def build_optimizer(model: MultiscreenLM, config: OptimizerConfig) -> torch.optim.Optimizer:
    """Construct the requested optimizer for Multiscreen training."""

    if config.optimizer_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay,
        )

    try:
        from schedulefree import AdamWScheduleFree
    except ImportError as exc:
        raise RuntimeError(
            "adamw_schedulefree requires the 'schedulefree' package to be installed"
        ) from exc

    return AdamWScheduleFree(
        model.parameters(),
        lr=config.lr,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
    )


def set_optimizer_train_mode(optimizer: torch.optim.Optimizer) -> None:
    """Switch optimizers with explicit train/eval state into train mode."""

    train_method = getattr(optimizer, "train", None)
    if callable(train_method):
        train_method()


def set_optimizer_eval_mode(optimizer: torch.optim.Optimizer) -> None:
    """Switch optimizers with explicit train/eval state into eval mode."""

    eval_method = getattr(optimizer, "eval", None)
    if callable(eval_method):
        eval_method()


def compute_grad_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    """Return the global L2 norm of all finite gradients."""

    grads = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not grads:
        return 0.0
    stacked = torch.stack([torch.linalg.vector_norm(grad.detach()) for grad in grads])
    return float(torch.linalg.vector_norm(stacked))


def should_log_step(step: int, *, total_steps: int, log_interval: int) -> bool:
    """Return whether a training loop should emit a progress log on this step."""

    if step <= 0:
        raise ValueError("step must be positive")
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if log_interval < 0:
        raise ValueError("log_interval must be non-negative")
    if step > total_steps:
        raise ValueError("step must not exceed total_steps")
    return step == 1 or step == total_steps or (log_interval > 0 and step % log_interval == 0)


def train_step(
    model: MultiscreenLM,
    optimizer: torch.optim.Optimizer,
    batch: CausalLMBatch,
    *,
    grad_clip: float | None = None,
) -> TrainStepResult:
    """Run one training step and update model parameters."""

    model.train()
    set_optimizer_train_mode(optimizer)
    optimizer.zero_grad(set_to_none=True)
    device = model_device(model)
    input_ids = batch.input_ids.to(device)
    labels = batch.labels.to(device)
    logits, _ = model(input_ids, return_relevances=False)
    loss = causal_lm_loss(logits, labels)
    loss.backward()
    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
    grad_norm = compute_grad_norm(model.parameters())
    optimizer.step()
    return TrainStepResult(loss=float(loss.detach()), grad_norm=grad_norm)


@torch.no_grad()
def evaluate_loss(
    model: MultiscreenLM,
    batch: CausalLMBatch,
    *,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    """Evaluate next-token loss without updating the model."""

    model.eval()
    if optimizer is not None:
        set_optimizer_eval_mode(optimizer)
    device = model_device(model)
    input_ids = batch.input_ids.to(device)
    labels = batch.labels.to(device)
    logits, _ = model(input_ids, return_relevances=False)
    return float(causal_lm_loss(logits, labels).detach())
