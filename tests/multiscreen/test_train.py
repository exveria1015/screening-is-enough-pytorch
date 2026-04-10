from __future__ import annotations

import torch

from multiscreen.config import MultiscreenConfig
from multiscreen.data import causal_lm_batch_from_token_block, sample_token_blocks
from multiscreen.model import MultiscreenLM
from multiscreen.train import (
    OptimizerConfig,
    build_optimizer,
    causal_lm_loss,
    evaluate_loss,
    set_optimizer_eval_mode,
    set_optimizer_train_mode,
    should_log_step,
    train_step,
)


def build_model() -> MultiscreenLM:
    torch.manual_seed(0)
    config = MultiscreenConfig(
        vocab_size=32,
        d_model=16,
        n_layers=1,
        n_heads=1,
        d_key=4,
        d_value=8,
        max_seq_len=8,
        max_train_seq_len=8,
    )
    return MultiscreenLM(config)


def test_causal_lm_batch_from_token_block_splits_inputs_and_labels() -> None:
    tokens = torch.tensor([[3, 4, 5, 6], [7, 8, 9, 10]], dtype=torch.long)
    batch = causal_lm_batch_from_token_block(tokens)
    assert torch.equal(batch.input_ids, torch.tensor([[3, 4, 5], [7, 8, 9]], dtype=torch.long))
    assert torch.equal(batch.labels, torch.tensor([[4, 5, 6], [8, 9, 10]], dtype=torch.long))


def test_sample_token_blocks_returns_contiguous_subsequences() -> None:
    generator = torch.Generator().manual_seed(123)
    token_ids = torch.arange(12, dtype=torch.long)
    blocks = sample_token_blocks(token_ids, batch_size=3, block_size=4, generator=generator)
    assert blocks.shape == (3, 4)
    for row in blocks:
        assert torch.equal(row[1:] - row[:-1], torch.ones(3, dtype=torch.long))


def test_causal_lm_loss_matches_cross_entropy_reference() -> None:
    logits = torch.tensor([[[2.0, 0.0], [0.0, 2.0]]], dtype=torch.float32)
    labels = torch.tensor([[0, 1]], dtype=torch.long)
    actual = causal_lm_loss(logits, labels)
    expected = torch.nn.functional.cross_entropy(logits.reshape(-1, 2), labels.reshape(-1))
    assert torch.allclose(actual, expected, atol=1e-6)


def test_build_optimizer_uses_requested_hyperparameters() -> None:
    model = build_model()
    optimizer = build_optimizer(model, OptimizerConfig(lr=1e-3, beta1=0.8, beta2=0.9, weight_decay=0.1))
    group = optimizer.param_groups[0]
    assert group["lr"] == 1e-3
    assert group["betas"] == (0.8, 0.9)
    assert group["weight_decay"] == 0.1


def test_build_schedulefree_optimizer_uses_requested_hyperparameters() -> None:
    model = build_model()
    optimizer = build_optimizer(
        model,
        OptimizerConfig(
            lr=2e-3,
            optimizer_name="adamw_schedulefree",
            beta1=0.8,
            beta2=0.9,
            weight_decay=0.1,
            warmup_steps=7,
        ),
    )
    group = optimizer.param_groups[0]
    assert optimizer.__class__.__name__ == "AdamWScheduleFree"
    assert group["lr"] == 2e-3
    assert group["betas"] == (0.8, 0.9)
    assert group["weight_decay"] == 0.1
    assert group["warmup_steps"] == 7


def test_schedulefree_optimizer_train_and_eval_modes_toggle_group_state() -> None:
    model = build_model()
    optimizer = build_optimizer(model, OptimizerConfig(lr=1e-3, optimizer_name="adamw_schedulefree"))
    group = optimizer.param_groups[0]
    assert group["train_mode"] is False
    set_optimizer_train_mode(optimizer)
    assert group["train_mode"] is True
    set_optimizer_eval_mode(optimizer)
    assert group["train_mode"] is False


def test_train_step_updates_parameters_and_returns_finite_stats() -> None:
    model = build_model()
    optimizer = build_optimizer(model, OptimizerConfig(lr=5e-3))
    tokens = torch.tensor(
        [
            [1, 2, 3, 1, 2, 3, 1, 2],
            [1, 2, 3, 1, 2, 3, 1, 2],
        ],
        dtype=torch.long,
    )
    batch = causal_lm_batch_from_token_block(tokens)
    before = model.embedding.weight.detach().clone()
    result = train_step(model, optimizer, batch, grad_clip=1.0)
    after = model.embedding.weight.detach()
    assert result.loss > 0.0
    assert result.grad_norm > 0.0
    assert torch.isfinite(torch.tensor(result.loss))
    assert torch.isfinite(torch.tensor(result.grad_norm))
    assert not torch.allclose(before, after)


def test_train_step_and_evaluate_loss_work_with_schedulefree_optimizer() -> None:
    model = build_model()
    optimizer = build_optimizer(
        model,
        OptimizerConfig(
            lr=5e-3,
            optimizer_name="adamw_schedulefree",
            warmup_steps=3,
        ),
    )
    tokens = torch.tensor(
        [
            [1, 2, 3, 1, 2, 3, 1, 2],
            [1, 2, 3, 1, 2, 3, 1, 2],
        ],
        dtype=torch.long,
    )
    batch = causal_lm_batch_from_token_block(tokens)
    initial_loss = evaluate_loss(model, batch, optimizer=optimizer)
    result = train_step(model, optimizer, batch, grad_clip=1.0)
    final_loss = evaluate_loss(model, batch, optimizer=optimizer)
    assert result.loss > 0.0
    assert result.grad_norm > 0.0
    assert initial_loss != final_loss
    assert optimizer.param_groups[0]["train_mode"] is False


def test_repeated_train_steps_reduce_loss_on_simple_pattern() -> None:
    model = build_model()
    optimizer = build_optimizer(model, OptimizerConfig(lr=1e-2))
    tokens = torch.tensor(
        [
            [1, 2, 3, 1, 2, 3, 1, 2],
            [1, 2, 3, 1, 2, 3, 1, 2],
            [1, 2, 3, 1, 2, 3, 1, 2],
            [1, 2, 3, 1, 2, 3, 1, 2],
        ],
        dtype=torch.long,
    )
    batch = causal_lm_batch_from_token_block(tokens)
    initial_loss = evaluate_loss(model, batch)
    for _ in range(25):
        train_step(model, optimizer, batch, grad_clip=1.0)
    final_loss = evaluate_loss(model, batch)
    assert final_loss < initial_loss


def test_should_log_step_allows_zero_interval_without_modulo_errors() -> None:
    assert should_log_step(1, total_steps=5, log_interval=0) is True
    assert should_log_step(3, total_steps=5, log_interval=0) is False
    assert should_log_step(5, total_steps=5, log_interval=0) is True


def test_should_log_step_rejects_negative_interval() -> None:
    try:
        should_log_step(1, total_steps=5, log_interval=-1)
    except ValueError as exc:
        assert "log_interval" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for negative log_interval")
