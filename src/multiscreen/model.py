"""Paper-faithful Multiscreen model components."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from multiscreen.config import MultiscreenConfig
from multiscreen.math import TanhNorm, apply_mipe, build_softmask, normalize_unit, trim_and_square, window_from_parameter


def _normal_init(parameter: torch.Tensor, std: float) -> None:
    nn.init.normal_(parameter, mean=0.0, std=std)


class NormalizedEmbedding(nn.Module):
    """Embedding matrix with row normalization used at input and output."""

    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, d_model))
        _normal_init(self.weight, 0.1 / math.sqrt(d_model))

    def normalized_weight(self, eps: float) -> torch.Tensor:
        return normalize_unit(self.weight, eps=eps)

    def forward(self, token_ids: torch.Tensor, *, eps: float) -> torch.Tensor:
        return F.embedding(token_ids, self.normalized_weight(eps))


class ScreeningUnit(nn.Module):
    """Single-head screening unit defined in Section 3.2."""

    def __init__(self, config: MultiscreenConfig, initial_s_w: float) -> None:
        super().__init__()
        self.config = config
        self.s_w = nn.Parameter(torch.tensor(float(initial_s_w)))
        self.s_r = nn.Parameter(torch.tensor(0.0))
        self.tanh_norm = TanhNorm(eps=config.eps)

    def effective_window(self, *, inference: bool) -> torch.Tensor:
        window = window_from_parameter(self.s_w)
        if inference and window.item() > self.config.max_train_seq_len:
            return torch.tensor(float("inf"), dtype=window.dtype, device=window.device)
        return window

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        inference: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
            raise ValueError("q, k, v must have shape [batch, seq, dim]")
        if q.shape[:2] != k.shape[:2] or q.shape[:2] != v.shape[:2]:
            raise ValueError("q, k, v must agree on batch and seq dimensions")

        q_unit = normalize_unit(q, eps=self.config.eps)
        k_unit = normalize_unit(k, eps=self.config.eps)
        v_unit = normalize_unit(v, eps=self.config.eps)

        seq_len = q.shape[1]
        positions = torch.arange(seq_len, device=q.device)
        window = self.effective_window(inference=inference).to(dtype=q.dtype, device=q.device)

        q_mipe = apply_mipe(q_unit, positions, window, self.config.mipe_threshold)
        k_mipe = apply_mipe(k_unit, positions, window, self.config.mipe_threshold)

        similarity = torch.einsum("bqd,bkd->bqk", q_mipe, k_mipe).clamp(min=-1.0, max=1.0)
        relevance = trim_and_square(similarity, self.s_r.to(dtype=q.dtype, device=q.device))
        softmask = build_softmask(positions, positions, window, dtype=q.dtype).to(device=q.device)
        distance_aware_relevance = relevance * softmask.unsqueeze(0)
        aggregated = torch.einsum("bqk,bkd->bqd", distance_aware_relevance, v_unit)
        return self.tanh_norm(aggregated), distance_aware_relevance


class GatedScreeningTile(nn.Module):
    """Head-level tile defined in Section 3.3."""

    def __init__(self, config: MultiscreenConfig, *, initial_s_w: float, initial_s_o: float) -> None:
        super().__init__()
        self.config = config
        self.q_proj = nn.Linear(config.d_model, config.d_key, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_key, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_value, bias=False)
        self.g_proj = nn.Linear(config.d_model, config.d_value, bias=False)
        self.o_proj = nn.Linear(config.d_value, config.d_model, bias=False)
        self.screening = ScreeningUnit(config, initial_s_w=initial_s_w)
        self.s_o = nn.Parameter(torch.tensor(float(initial_s_o)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        _normal_init(self.q_proj.weight, 0.1 / math.sqrt(self.config.d_key))
        _normal_init(self.k_proj.weight, 0.1 / math.sqrt(self.config.d_key))
        _normal_init(self.v_proj.weight, 0.1 / math.sqrt(self.config.d_value))
        _normal_init(self.g_proj.weight, 0.1)
        _normal_init(self.o_proj.weight, 0.1 / math.sqrt(self.config.d_model))

    def forward(self, x: torch.Tensor, *, inference: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        g = self.g_proj(x)
        screened, relevance = self.screening(q, k, v, inference=inference)
        gate = torch.tanh(F.silu(g))
        update = self.o_proj(screened * gate) * torch.exp(self.s_o)
        return update, relevance


class MultiscreenLayer(nn.Module):
    """Residual layer with N_H parallel gated screening tiles."""

    def __init__(self, config: MultiscreenConfig) -> None:
        super().__init__()
        if config.n_heads == 1:
            initial_s_ws = [0.0]
        else:
            max_value = math.log(config.mipe_threshold)
            initial_s_ws = [head * max_value / (config.n_heads - 1) for head in range(config.n_heads)]
        initial_s_o = math.log(1.0 / math.sqrt(config.n_heads * config.n_layers))
        self.tiles = nn.ModuleList(
            GatedScreeningTile(config, initial_s_w=initial_s_w, initial_s_o=initial_s_o)
            for initial_s_w in initial_s_ws
        )

    def forward(self, x: torch.Tensor, *, inference: bool = False) -> tuple[torch.Tensor, list[torch.Tensor]]:
        updates: list[torch.Tensor] = []
        relevances: list[torch.Tensor] = []
        for tile in self.tiles:
            update, relevance = tile(x, inference=inference)
            updates.append(update)
            relevances.append(relevance)
        stacked = torch.stack(updates, dim=0).sum(dim=0)
        return x + stacked, relevances


class MultiscreenLM(nn.Module):
    """Multiscreen language model defined in Section 3."""

    def __init__(self, config: MultiscreenConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = NormalizedEmbedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList(MultiscreenLayer(config) for _ in range(config.n_layers))
        self.s_e = nn.Parameter(torch.tensor(0.0))
        self.s_f = nn.Parameter(torch.tensor(0.5 * math.log(config.d_model)))

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.s_e) * self.embedding(input_ids, eps=self.config.eps)

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        output_matrix = torch.exp(self.s_f) * self.embedding.normalized_weight(self.config.eps)
        return torch.matmul(hidden, output_matrix.transpose(0, 1))

    def forward(self, input_ids: torch.Tensor, *, inference: bool = False) -> tuple[torch.Tensor, list[list[torch.Tensor]]]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, seq]")
        if input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError("sequence length exceeds config.max_seq_len")

        hidden = self.embed(input_ids)
        all_relevances: list[list[torch.Tensor]] = []
        for layer in self.layers:
            hidden, layer_relevances = layer(hidden, inference=inference)
            all_relevances.append(layer_relevances)
        return self.logits(hidden), all_relevances
