from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from multiscreen.config import MultiscreenConfig
from multiscreen.math import normalize_unit
from multiscreen.model import GatedScreeningTile, MultiscreenLM, MultiscreenLayer, ScreeningUnit


def test_from_psi_uses_default_scaling_rule() -> None:
    config = MultiscreenConfig.from_psi(psi=4)
    assert config.n_layers == 4
    assert config.n_heads == 4
    assert config.d_model == 16
    assert config.d_key == 16
    assert config.d_value == 64


def test_multiscreen_forward_shapes() -> None:
    config = MultiscreenConfig(
        vocab_size=101,
        d_model=16,
        n_layers=2,
        n_heads=2,
        d_key=4,
        d_value=8,
        max_seq_len=8,
        max_train_seq_len=8,
    )
    model = MultiscreenLM(config)
    input_ids = torch.randint(0, config.vocab_size, (3, 5), dtype=torch.long)
    logits, relevances = model(input_ids, return_relevances=True)
    assert logits.shape == (3, 5, config.vocab_size)
    assert len(relevances) == config.n_layers
    assert len(relevances[0]) == config.n_heads
    assert relevances[0][0].shape == (3, 5, 5)


def test_multiscreen_layer_contains_only_parallel_tiles_and_no_block_ffn() -> None:
    config = MultiscreenConfig(d_model=16, n_layers=2, n_heads=2, d_key=4, d_value=8)
    layer = MultiscreenLayer(config)
    children = dict(layer.named_children())
    assert list(children.keys()) == ["tiles"]
    assert isinstance(layer.tiles, torch.nn.ModuleList)
    assert len(layer.tiles) == config.n_heads
    assert all(isinstance(tile, GatedScreeningTile) for tile in layer.tiles)
    assert not hasattr(layer, "mlp")
    assert not hasattr(layer, "ffn")


def test_multiscreen_layer_matches_explicit_tile_execution() -> None:
    torch.manual_seed(0)
    config = MultiscreenConfig(
        d_model=16,
        n_layers=2,
        n_heads=3,
        d_key=4,
        d_value=8,
        max_seq_len=8,
        max_train_seq_len=4,
    )
    layer = MultiscreenLayer(config)
    x = torch.randn(2, 5, config.d_model, dtype=torch.float32)

    expected_updates: list[torch.Tensor] = []
    expected_relevances: list[torch.Tensor] = []
    for tile in layer.tiles:
        update, relevance = tile(x, inference=True)
        expected_updates.append(update)
        expected_relevances.append(relevance)
    expected_hidden = x + torch.stack(expected_updates, dim=0).sum(dim=0)

    actual_hidden, actual_relevances = layer(x, inference=True, return_relevances=True)

    assert torch.allclose(actual_hidden, expected_hidden, atol=1e-6, rtol=1e-6)
    assert len(actual_relevances) == len(expected_relevances)
    for actual, expected in zip(actual_relevances, expected_relevances, strict=True):
        assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_multiscreen_layer_packed_qkvg_projection_matches_explicit_tiles() -> None:
    torch.manual_seed(0)
    config = MultiscreenConfig(
        d_model=16,
        n_layers=2,
        n_heads=3,
        d_key=4,
        d_value=8,
        max_seq_len=8,
        max_train_seq_len=4,
    )
    layer = MultiscreenLayer(config)
    x = torch.randn(2, 5, config.d_model, dtype=torch.float32)

    expected_q = torch.stack([tile.q_proj(x) for tile in layer.tiles], dim=1)
    expected_k = torch.stack([tile.k_proj(x) for tile in layer.tiles], dim=1)
    expected_v = torch.stack([tile.v_proj(x) for tile in layer.tiles], dim=1)
    expected_g = torch.stack([tile.g_proj(x) for tile in layer.tiles], dim=1)

    actual_q, actual_k, actual_v, actual_g = layer._project_qkvg(x)

    assert torch.allclose(actual_q, expected_q, atol=1e-6, rtol=1e-6)
    assert torch.allclose(actual_k, expected_k, atol=1e-6, rtol=1e-6)
    assert torch.allclose(actual_v, expected_v, atol=1e-6, rtol=1e-6)
    assert torch.allclose(actual_g, expected_g, atol=1e-6, rtol=1e-6)


def test_multiscreen_layer_can_skip_relevance_materialization_without_changing_output() -> None:
    torch.manual_seed(0)
    config = MultiscreenConfig(
        d_model=16,
        n_layers=2,
        n_heads=3,
        d_key=4,
        d_value=8,
        max_seq_len=8,
        max_train_seq_len=4,
    )
    layer = MultiscreenLayer(config)
    x = torch.randn(2, 5, config.d_model, dtype=torch.float32)

    expected_hidden, expected_relevances = layer(x, inference=True, return_relevances=True)
    actual_hidden, actual_relevances = layer(x, inference=True, return_relevances=False, query_chunk_size=2)

    assert actual_relevances is None
    assert expected_relevances is not None
    assert torch.allclose(actual_hidden, expected_hidden, atol=1e-6, rtol=1e-6)


def test_multiscreen_backward_produces_finite_gradients() -> None:
    torch.manual_seed(0)
    config = MultiscreenConfig(
        vocab_size=101,
        d_model=16,
        n_layers=1,
        n_heads=1,
        d_key=4,
        d_value=8,
        max_seq_len=8,
        max_train_seq_len=8,
    )
    model = MultiscreenLM(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 5), dtype=torch.long)
    labels = torch.randint(0, config.vocab_size, (2, 5), dtype=torch.long)
    logits, _ = model(input_ids, return_relevances=False)
    loss = F.cross_entropy(logits.reshape(-1, config.vocab_size), labels.reshape(-1))
    loss.backward()

    params = [
        model.embedding.weight,
        model.s_e,
        model.s_f,
        model.layers[0].tiles[0].q_proj.weight,
        model.layers[0].tiles[0].screening.s_w,
        model.layers[0].tiles[0].screening.s_r,
        model.layers[0].tiles[0].s_o,
    ]
    for parameter in params:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()

    assert float(torch.linalg.vector_norm(model.embedding.weight.grad)) > 0.0
    assert float(torch.linalg.vector_norm(model.layers[0].tiles[0].q_proj.weight.grad)) > 0.0


def test_layer_initializes_head_windows_linearly_and_output_scale_correctly() -> None:
    config = MultiscreenConfig(
        d_model=16,
        n_layers=4,
        n_heads=4,
        d_key=4,
        d_value=8,
    )
    layer = MultiscreenLayer(config)
    expected_s_ws = torch.linspace(0.0, torch.log(torch.tensor(config.mipe_threshold)), steps=config.n_heads)
    actual_s_ws = torch.stack([tile.screening.s_w.detach() for tile in layer.tiles])
    assert torch.allclose(actual_s_ws, expected_s_ws, atol=1e-6)

    expected_s_o = -0.5 * torch.log(torch.tensor(float(config.n_heads * config.n_layers)))
    actual_s_os = torch.stack([tile.s_o.detach() for tile in layer.tiles])
    assert torch.allclose(actual_s_os, expected_s_o.expand_as(actual_s_os), atol=1e-6)


def test_projection_and_embedding_initialization_match_paper_scales() -> None:
    torch.manual_seed(0)
    config = MultiscreenConfig(
        vocab_size=4096,
        d_model=256,
        n_layers=4,
        n_heads=4,
        d_key=16,
        d_value=64,
    )
    model = MultiscreenLM(config)
    tile = model.layers[0].tiles[0]

    expected_stds = {
        "embedding": 0.1 / math.sqrt(config.d_model),
        "q_proj": 0.1 / math.sqrt(config.d_key),
        "k_proj": 0.1 / math.sqrt(config.d_key),
        "v_proj": 0.1 / math.sqrt(config.d_value),
        "g_proj": 0.1,
        "o_proj": 0.1 / math.sqrt(config.d_model),
    }
    actual_tensors = {
        "embedding": model.embedding.weight.detach(),
        "q_proj": tile.q_proj.weight.detach(),
        "k_proj": tile.k_proj.weight.detach(),
        "v_proj": tile.v_proj.weight.detach(),
        "g_proj": tile.g_proj.weight.detach(),
        "o_proj": tile.o_proj.weight.detach(),
    }

    for name, expected_std in expected_stds.items():
        actual = actual_tensors[name]
        assert float(actual.std()) == pytest.approx(expected_std, rel=0.2)
        assert abs(float(actual.mean())) < expected_std * 0.2

    assert float(model.s_e.detach()) == pytest.approx(0.0, abs=1e-8)
    assert float(model.s_f.detach()) == pytest.approx(0.5 * math.log(config.d_model), abs=1e-8)
    assert float(tile.screening.s_r.detach()) == pytest.approx(0.0, abs=1e-8)


def test_screening_unit_normalizes_values_before_aggregation() -> None:
    config = MultiscreenConfig(d_model=8, n_layers=1, n_heads=1, d_key=4, d_value=4, max_seq_len=4, max_train_seq_len=4)
    unit = ScreeningUnit(config, initial_s_w=10.0)
    q = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], dtype=torch.float32)
    k = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], dtype=torch.float32)
    v_small = torch.tensor([[[1.0, 2.0, 0.0, 0.0]]], dtype=torch.float32)
    v_large = 100.0 * v_small

    out_small, _ = unit(q, k, v_small)
    out_large, _ = unit(q, k, v_large)
    assert torch.allclose(out_small, out_large, atol=1e-6)


def test_screening_unit_chunked_path_matches_dense_path_exactly() -> None:
    torch.manual_seed(0)
    config = MultiscreenConfig(d_model=8, n_layers=1, n_heads=1, d_key=4, d_value=4, max_seq_len=8, max_train_seq_len=8)
    unit = ScreeningUnit(config, initial_s_w=0.0)
    q = torch.randn(2, 5, 4, dtype=torch.float32)
    k = torch.randn(2, 5, 4, dtype=torch.float32)
    v = torch.randn(2, 5, 4, dtype=torch.float32)

    dense_output, dense_relevance = unit(q, k, v, return_relevance=True)
    chunked_output, chunked_relevance = unit(q, k, v, return_relevance=False, query_chunk_size=2)

    assert dense_relevance is not None
    assert chunked_relevance is None
    assert torch.allclose(chunked_output, dense_output, atol=1e-6, rtol=1e-6)


def test_screening_relevance_is_not_constrained_to_sum_to_one() -> None:
    config = MultiscreenConfig(d_model=8, n_layers=1, n_heads=1, d_key=4, d_value=4, max_seq_len=3, max_train_seq_len=3)
    unit = ScreeningUnit(config, initial_s_w=10.0)
    q = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )
    k = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )
    v = torch.ones(1, 3, 4, dtype=torch.float32)

    _, relevance = unit(q, k, v)
    assert float(relevance[0, 2, 0].detach()) == pytest.approx(1.0, abs=1e-6)
    assert float(relevance[0, 2, 1].detach()) == pytest.approx(1.0, abs=1e-6)
    assert float(relevance[0, 2].sum().detach()) > 1.0


def test_screening_relevance_can_be_exactly_zero_for_all_keys() -> None:
    config = MultiscreenConfig(d_model=8, n_layers=1, n_heads=1, d_key=4, d_value=4, max_seq_len=3, max_train_seq_len=3)
    unit = ScreeningUnit(config, initial_s_w=10.0)
    q = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )
    k = torch.tensor(
        [[[0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )
    v = torch.ones(1, 3, 4, dtype=torch.float32)

    _, relevance = unit(q, k, v)
    assert torch.allclose(relevance[0, 2].detach(), torch.zeros(3, dtype=torch.float32), atol=1e-6)


def test_screening_unit_backward_reaches_window_parameter() -> None:
    config = MultiscreenConfig(d_model=8, n_layers=1, n_heads=1, d_key=4, d_value=4, max_seq_len=4, max_train_seq_len=4)
    unit = ScreeningUnit(config, initial_s_w=0.0)
    q = torch.ones(1, 4, 4, dtype=torch.float32, requires_grad=False)
    k = torch.ones(1, 4, 4, dtype=torch.float32, requires_grad=False)
    v = torch.ones(1, 4, 4, dtype=torch.float32, requires_grad=False)

    output, _ = unit(q, k, v)
    output.sum().backward()

    assert unit.s_w.grad is not None
    assert torch.isfinite(unit.s_w.grad).all()
    assert abs(float(unit.s_w.grad)) > 0.0


def test_screening_unit_backward_reaches_relevance_parameter_on_partial_similarity() -> None:
    config = MultiscreenConfig(d_model=8, n_layers=1, n_heads=1, d_key=4, d_value=4, max_seq_len=4, max_train_seq_len=4)
    unit = ScreeningUnit(config, initial_s_w=0.0)
    q = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], dtype=torch.float32)
    k = torch.tensor([[[0.75, (1.0 - 0.75**2) ** 0.5, 0.0, 0.0]]], dtype=torch.float32)
    v = torch.tensor([[[1.0, 2.0, 0.0, 0.0]]], dtype=torch.float32)

    output, _ = unit(q, k, v)
    output.sum().backward()

    assert unit.s_r.grad is not None
    assert torch.isfinite(unit.s_r.grad).all()
    assert abs(float(unit.s_r.grad)) > 0.0


def test_inference_promotes_large_window_to_full_causal_mask() -> None:
    config = MultiscreenConfig(d_model=8, n_layers=1, n_heads=1, d_key=4, d_value=4, max_seq_len=5, max_train_seq_len=2)
    unit = ScreeningUnit(config, initial_s_w=float(torch.log(torch.tensor(2.0))))
    q = torch.ones(1, 5, 4, dtype=torch.float32)
    k = torch.ones(1, 5, 4, dtype=torch.float32)
    v = torch.eye(5, 4, dtype=torch.float32).unsqueeze(0)

    _, relevance_train = unit(q, k, v, inference=False)
    _, relevance_infer = unit(q, k, v, inference=True)

    assert float(relevance_train[0, 4, 0].detach()) == 0.0
    assert float(relevance_infer[0, 4, 0].detach()) > 0.0


def test_inference_full_causal_mask_matches_expected_matrix() -> None:
    config = MultiscreenConfig(d_model=8, n_layers=1, n_heads=1, d_key=4, d_value=4, max_seq_len=5, max_train_seq_len=2)
    unit = ScreeningUnit(config, initial_s_w=float(torch.log(torch.tensor(2.0))))
    q = torch.ones(1, 5, 4, dtype=torch.float32)
    k = torch.ones(1, 5, 4, dtype=torch.float32)
    v = torch.ones(1, 5, 4, dtype=torch.float32)

    _, relevance = unit(q, k, v, inference=True)
    expected = torch.tril(torch.ones(5, 5, dtype=torch.float32))
    assert torch.allclose(relevance[0].detach(), expected, atol=1e-6)


def test_input_and_output_share_same_normalized_embedding_matrix() -> None:
    config = MultiscreenConfig(vocab_size=4, d_model=3, n_layers=1, n_heads=1, d_key=2, d_value=2, max_seq_len=4)
    model = MultiscreenLM(config)
    with torch.no_grad():
        model.embedding.weight.copy_(
            torch.tensor(
                [
                    [3.0, 0.0, 0.0],
                    [0.0, 4.0, 0.0],
                    [0.0, 0.0, 5.0],
                    [1.0, 2.0, 2.0],
                ],
                dtype=torch.float32,
            )
        )
        model.s_e.fill_(math.log(2.0))
        model.s_f.fill_(math.log(3.0))

    normalized = normalize_unit(model.embedding.weight.detach(), eps=config.eps)
    input_ids = torch.tensor([[0, 1, 3]], dtype=torch.long)
    hidden = torch.tensor([[[1.0, 2.0, 3.0]]], dtype=torch.float32)

    expected_embed = torch.exp(model.s_e.detach()) * normalized[input_ids]
    expected_logits = hidden @ (torch.exp(model.s_f.detach()) * normalized).transpose(0, 1)

    assert torch.allclose(model.embed(input_ids), expected_embed, atol=1e-6)
    assert torch.allclose(model.logits(hidden), expected_logits, atol=1e-6)


def test_model_logits_match_with_and_without_returning_relevances() -> None:
    torch.manual_seed(0)
    config = MultiscreenConfig(vocab_size=32, d_model=16, n_layers=2, n_heads=2, d_key=4, d_value=8, max_seq_len=8)
    model = MultiscreenLM(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 5), dtype=torch.long)

    logits_with_relevance, relevances = model(input_ids, return_relevances=True)
    logits_without_relevance, skipped = model(input_ids, return_relevances=False, query_chunk_size=2)

    assert relevances is not None
    assert skipped is None
    assert torch.allclose(logits_with_relevance, logits_without_relevance, atol=1e-6, rtol=1e-6)


def test_model_auto_backend_falls_back_to_torch_on_cpu() -> None:
    torch.manual_seed(0)
    config = MultiscreenConfig(vocab_size=32, d_model=16, n_layers=2, n_heads=2, d_key=4, d_value=8, max_seq_len=8)
    model = MultiscreenLM(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 5), dtype=torch.long)

    logits_torch, _ = model(input_ids, return_relevances=False, screening_backend="torch")
    logits_auto, _ = model(input_ids, return_relevances=False, screening_backend="auto")

    assert torch.allclose(logits_auto, logits_torch, atol=1e-6, rtol=1e-6)


def test_model_rejects_unknown_screening_backend() -> None:
    config = MultiscreenConfig(vocab_size=32, d_model=16, n_layers=1, n_heads=1, d_key=4, d_value=8, max_seq_len=8)
    model = MultiscreenLM(config)
    input_ids = torch.randint(0, config.vocab_size, (1, 5), dtype=torch.long)

    with pytest.raises(ValueError, match="screening_backend"):
        model(input_ids, screening_backend="bogus")


def test_model_explicit_triton_backend_raises_when_unavailable() -> None:
    config = MultiscreenConfig(vocab_size=32, d_model=16, n_layers=1, n_heads=1, d_key=4, d_value=8, max_seq_len=8)
    model = MultiscreenLM(config)
    input_ids = torch.randint(0, config.vocab_size, (1, 5), dtype=torch.long)

    with pytest.raises(RuntimeError, match="Triton screening backend is unavailable"):
        model(input_ids, return_relevances=False, screening_backend="triton")


def test_forward_rejects_sequences_longer_than_config() -> None:
    config = MultiscreenConfig(vocab_size=32, d_model=16, n_layers=1, n_heads=1, d_key=4, d_value=8, max_seq_len=4)
    model = MultiscreenLM(config)
    input_ids = torch.randint(0, config.vocab_size, (1, 5), dtype=torch.long)
    with pytest.raises(ValueError, match="sequence length exceeds config.max_seq_len"):
        model(input_ids)
