"""Paper-faithful Multiscreen model components."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from multiscreen.config import MultiscreenConfig
from multiscreen.math import (
    build_mipe_rotation,
    build_softmask,
    normalize_and_apply_mipe,
    normalize_unit,
    relevance_width_from_parameter,
    tanh_norm,
    trim_and_square,
    window_from_parameter,
)
from multiscreen.triton_kernels import (
    check_triton_screening_support,
    triton_normalize_and_apply_mipe,
    triton_normalize_unit,
    triton_screening_aggregate_q_fused,
    triton_screening_aggregate,
    triton_screening_aggregate_fused,
)

_DEFAULT_STREAMING_QUERY_CHUNK_SIZE = 128
_STREAMING_MIN_SEQ_LEN = 512
_VALID_SCREENING_BACKENDS = {"torch", "triton", "auto"}


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


def _effective_window(s_w: torch.Tensor, *, config: MultiscreenConfig, inference: bool) -> torch.Tensor:
    window = window_from_parameter(s_w)
    if not inference:
        return window
    max_train_seq_len = torch.as_tensor(config.max_train_seq_len, dtype=window.dtype, device=window.device)
    return torch.where(window > max_train_seq_len, torch.full_like(window, float("inf")), window)


def _resolve_query_chunk_size(
    seq_len: int,
    *,
    return_relevance: bool,
    query_chunk_size: int | None,
) -> int | None:
    if return_relevance:
        return None
    if query_chunk_size is not None:
        if query_chunk_size <= 0:
            raise ValueError("query_chunk_size must be positive when provided")
        return min(query_chunk_size, seq_len)
    if seq_len >= _STREAMING_MIN_SEQ_LEN:
        return min(_DEFAULT_STREAMING_QUERY_CHUNK_SIZE, seq_len)
    return None


def _normalize_screening_backend(screening_backend: str) -> str:
    if screening_backend not in _VALID_SCREENING_BACKENDS:
        raise ValueError(f"screening_backend must be one of {_VALID_SCREENING_BACKENDS}")
    return screening_backend


def _select_screening_backend(
    screening_backend: str,
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    return_relevance: bool,
) -> str:
    backend = _normalize_screening_backend(screening_backend)
    if q.ndim != 4:
        if backend == "triton":
            raise RuntimeError("Triton screening backend is unavailable: only head-packed 4D tensors are supported")
        return "torch"
    if return_relevance:
        if backend == "triton":
            raise RuntimeError("Triton screening backend is unavailable: return_relevance=True is not supported")
        return "torch"
    support = check_triton_screening_support(q, k, v)
    if backend == "triton":
        if not support.supported:
            raise RuntimeError(f"Triton screening backend is unavailable: {support.reason}")
        return "triton"
    if backend == "auto" and support.supported:
        return "triton"
    return "torch"


def _dense_screening_aggregation(
    q_mipe: torch.Tensor,
    k_mipe: torch.Tensor,
    v_unit: torch.Tensor,
    *,
    positions: torch.Tensor,
    window: torch.Tensor,
    s_r: torch.Tensor,
    eps: float,
    return_relevance: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    similarity = torch.einsum("bhqd,bhkd->bhqk", q_mipe, k_mipe).clamp(min=-1.0, max=1.0)
    relevance = trim_and_square(similarity, s_r.to(dtype=q_mipe.dtype, device=q_mipe.device))
    softmask = build_softmask(positions, positions, window, dtype=q_mipe.dtype).to(device=q_mipe.device)
    distance_aware_relevance = relevance * softmask.unsqueeze(0)
    aggregated = torch.einsum("bhqk,bhkd->bhqd", distance_aware_relevance, v_unit)
    return tanh_norm(aggregated, eps=eps), distance_aware_relevance if return_relevance else None


def _triton_screening_aggregation(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    window: torch.Tensor,
    s_r: torch.Tensor,
    eps: float,
    mipe_threshold: float,
    fuse_preprocessing: bool,
) -> torch.Tensor:
    r = relevance_width_from_parameter(s_r).to(dtype=q.dtype, device=q.device)
    if fuse_preprocessing:
        return triton_screening_aggregate_fused(
            q,
            k,
            v,
            r=r,
            window=window,
            full_causal=torch.isinf(window),
            eps=eps,
            mipe_threshold=mipe_threshold,
            apply_tanh_norm=True,
        )
    k_mipe = triton_normalize_and_apply_mipe(
        k,
        window=window,
        eps=eps,
        mipe_threshold=mipe_threshold,
    )
    v_unit = triton_normalize_unit(v, eps=eps)
    return triton_screening_aggregate_q_fused(
        q,
        k_mipe,
        v_unit,
        r=r,
        window=window,
        full_causal=torch.isinf(window),
        eps=eps,
        mipe_threshold=mipe_threshold,
        apply_tanh_norm=True,
    )


def _streaming_screening_aggregation(
    q_mipe: torch.Tensor,
    k_mipe: torch.Tensor,
    v_unit: torch.Tensor,
    *,
    positions: torch.Tensor,
    window: torch.Tensor,
    s_r: torch.Tensor,
    eps: float,
    query_chunk_size: int,
) -> torch.Tensor:
    aggregated_chunks: list[torch.Tensor] = []
    seq_len = q_mipe.shape[-2]
    for start in range(0, seq_len, query_chunk_size):
        stop = min(start + query_chunk_size, seq_len)
        chunk_positions = positions[start:stop]
        q_chunk = q_mipe[:, :, start:stop, :]
        similarity_chunk = torch.einsum("bhcd,bhkd->bhck", q_chunk, k_mipe).clamp(min=-1.0, max=1.0)
        relevance_chunk = trim_and_square(similarity_chunk, s_r.to(dtype=q_mipe.dtype, device=q_mipe.device))
        softmask_chunk = build_softmask(chunk_positions, positions, window, dtype=q_mipe.dtype).to(device=q_mipe.device)
        distance_aware_chunk = relevance_chunk * softmask_chunk.unsqueeze(0)
        aggregated_chunks.append(torch.einsum("bhck,bhkd->bhcd", distance_aware_chunk, v_unit))
    return tanh_norm(torch.cat(aggregated_chunks, dim=2), eps=eps)


def _screening_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    s_w: torch.Tensor,
    s_r: torch.Tensor,
    config: MultiscreenConfig,
    inference: bool = False,
    return_relevance: bool = True,
    query_chunk_size: int | None = None,
    screening_backend: str = "torch",
    triton_fuse_preprocessing: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if q.ndim not in {3, 4} or k.ndim != q.ndim or v.ndim != q.ndim:
        raise ValueError("q, k, v must all have shape [batch, seq, dim] or [batch, head, seq, dim]")
    if q.shape[:-1] != k.shape[:-1] or q.shape[:-1] != v.shape[:-1]:
        raise ValueError("q, k, v must agree on their non-feature dimensions")

    squeeze_head = q.ndim == 3
    if squeeze_head:
        q = q.unsqueeze(1)
        k = k.unsqueeze(1)
        v = v.unsqueeze(1)
        if s_w.ndim == 0:
            s_w = s_w.unsqueeze(0)
        if s_r.ndim == 0:
            s_r = s_r.unsqueeze(0)

    seq_len = q.shape[-2]
    positions = torch.arange(seq_len, device=q.device)
    window = _effective_window(s_w, config=config, inference=inference).to(dtype=q.dtype, device=q.device)
    backend = _select_screening_backend(
        screening_backend,
        q=q,
        k=k,
        v=v,
        return_relevance=return_relevance,
    )
    if backend == "triton":
        aggregated = _triton_screening_aggregation(
            q,
            k,
            v,
            window=window,
            s_r=s_r,
            eps=config.eps,
            mipe_threshold=config.mipe_threshold,
            fuse_preprocessing=triton_fuse_preprocessing,
        )
        if squeeze_head:
            aggregated = aggregated[:, 0]
        return aggregated, None
    else:
        mipe_rotation = build_mipe_rotation(
            positions,
            window,
            config.mipe_threshold,
            dtype=q.dtype,
            device=q.device,
        )
        q_mipe = normalize_and_apply_mipe(
            q,
            positions,
            window,
            config.mipe_threshold,
            eps=config.eps,
            rotation=mipe_rotation,
        )
        k_mipe = normalize_and_apply_mipe(
            k,
            positions,
            window,
            config.mipe_threshold,
            eps=config.eps,
            rotation=mipe_rotation,
        )
        v_unit = normalize_unit(v, eps=config.eps)

    chunk_size = _resolve_query_chunk_size(
        seq_len,
        return_relevance=return_relevance,
        query_chunk_size=query_chunk_size,
    )
    if chunk_size is None or chunk_size >= seq_len:
        aggregated, distance_aware_relevance = _dense_screening_aggregation(
            q_mipe,
            k_mipe,
            v_unit,
            positions=positions,
            window=window,
            s_r=s_r,
            eps=config.eps,
            return_relevance=return_relevance,
        )
    else:
        aggregated = _streaming_screening_aggregation(
            q_mipe,
            k_mipe,
            v_unit,
            positions=positions,
            window=window,
            s_r=s_r,
            eps=config.eps,
            query_chunk_size=chunk_size,
        )
        distance_aware_relevance = None

    if squeeze_head:
        aggregated = aggregated[:, 0]
        if distance_aware_relevance is not None:
            distance_aware_relevance = distance_aware_relevance[:, 0]
    return aggregated, distance_aware_relevance


class ScreeningUnit(nn.Module):
    """Single-head screening unit defined in Section 3.2."""

    def __init__(self, config: MultiscreenConfig, initial_s_w: float) -> None:
        super().__init__()
        self.config = config
        self.s_w = nn.Parameter(torch.tensor(float(initial_s_w)))
        self.s_r = nn.Parameter(torch.tensor(0.0))

    def effective_window(self, *, inference: bool) -> torch.Tensor:
        return _effective_window(self.s_w, config=self.config, inference=inference)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        inference: bool = False,
        return_relevance: bool = True,
        query_chunk_size: int | None = None,
        screening_backend: str = "torch",
        triton_fuse_preprocessing: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return _screening_forward(
            q,
            k,
            v,
            s_w=self.s_w,
            s_r=self.s_r,
            config=self.config,
            inference=inference,
            return_relevance=return_relevance,
            query_chunk_size=query_chunk_size,
            screening_backend=screening_backend,
            triton_fuse_preprocessing=triton_fuse_preprocessing,
        )


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

    def forward(
        self,
        x: torch.Tensor,
        *,
        inference: bool = False,
        return_relevance: bool = True,
        query_chunk_size: int | None = None,
        screening_backend: str = "torch",
        triton_fuse_preprocessing: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        g = self.g_proj(x)
        screened, relevance = self.screening(
            q,
            k,
            v,
            inference=inference,
            return_relevance=return_relevance,
            query_chunk_size=query_chunk_size,
            screening_backend=screening_backend,
            triton_fuse_preprocessing=triton_fuse_preprocessing,
        )
        gate = torch.tanh(F.silu(g))
        update = self.o_proj(screened * gate) * torch.exp(self.s_o)
        return update, relevance


class MultiscreenLayer(nn.Module):
    """Residual layer with N_H parallel gated screening tiles."""

    def __init__(self, config: MultiscreenConfig) -> None:
        super().__init__()
        self.config = config
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

    def _project_qkvg(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        projection_width = 2 * self.config.d_key + 2 * self.config.d_value
        packed_weight = torch.cat(
            [
                torch.cat(
                    (tile.q_proj.weight, tile.k_proj.weight, tile.v_proj.weight, tile.g_proj.weight),
                    dim=0,
                )
                for tile in self.tiles
            ],
            dim=0,
        )
        packed = F.linear(x, packed_weight)
        packed = packed.reshape(x.shape[0], x.shape[1], len(self.tiles), projection_width).movedim(2, 1)
        return tuple(
            projection.contiguous()
            for projection in torch.split(
                packed,
                (self.config.d_key, self.config.d_key, self.config.d_value, self.config.d_value),
                dim=-1,
            )
        )

    def _stack_head_parameters(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.stack([tile.screening.s_w for tile in self.tiles]),
            torch.stack([tile.screening.s_r for tile in self.tiles]),
            torch.stack([tile.s_o for tile in self.tiles]),
            torch.stack([tile.o_proj.weight for tile in self.tiles]),
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        inference: bool = False,
        return_relevances: bool = True,
        query_chunk_size: int | None = None,
        screening_backend: str = "torch",
        triton_fuse_preprocessing: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        q, k, v, g = self._project_qkvg(x)
        s_w, s_r, s_o, o_weights = self._stack_head_parameters()
        screened, relevance = _screening_forward(
            q,
            k,
            v,
            s_w=s_w,
            s_r=s_r,
            config=self.tiles[0].screening.config,
            inference=inference,
            return_relevance=return_relevances,
            query_chunk_size=query_chunk_size,
            screening_backend=screening_backend,
            triton_fuse_preprocessing=triton_fuse_preprocessing,
        )
        gate = torch.tanh(F.silu(g))
        update = torch.einsum("bhtv,hev->bhte", screened * gate, o_weights)
        update = update * torch.exp(s_o).view(1, -1, 1, 1)
        if relevance is None:
            return x + update.sum(dim=1), None
        return x + update.sum(dim=1), [relevance[:, head] for head in range(relevance.shape[1])]


class MultiscreenLM(nn.Module):
    """Multiscreen language model defined in Section 3."""

    def __init__(self, config: MultiscreenConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = NormalizedEmbedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList(MultiscreenLayer(config) for _ in range(config.n_layers))
        self.screening_backend = "torch"
        self.triton_fuse_preprocessing = False
        self.s_e = nn.Parameter(torch.tensor(0.0))
        self.s_f = nn.Parameter(torch.tensor(0.5 * math.log(config.d_model)))

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.s_e) * self.embedding(input_ids, eps=self.config.eps)

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        output_matrix = torch.exp(self.s_f) * self.embedding.normalized_weight(self.config.eps)
        return torch.matmul(hidden, output_matrix.transpose(0, 1))

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        inference: bool = False,
        return_relevances: bool = False,
        query_chunk_size: int | None = None,
        screening_backend: str | None = None,
        triton_fuse_preprocessing: bool | None = None,
    ) -> tuple[torch.Tensor, list[list[torch.Tensor]] | None]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, seq]")
        if input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError("sequence length exceeds config.max_seq_len")

        hidden = self.embed(input_ids)
        selected_backend = _normalize_screening_backend(self.screening_backend if screening_backend is None else screening_backend)
        selected_triton_fusion = self.triton_fuse_preprocessing if triton_fuse_preprocessing is None else triton_fuse_preprocessing
        all_relevances: list[list[torch.Tensor]] | None = [] if return_relevances else None
        for layer in self.layers:
            hidden, layer_relevances = layer(
                hidden,
                inference=inference,
                return_relevances=return_relevances,
                query_chunk_size=query_chunk_size,
                screening_backend=selected_backend,
                triton_fuse_preprocessing=selected_triton_fusion,
            )
            if all_relevances is not None:
                assert layer_relevances is not None
                all_relevances.append(layer_relevances)
        return self.logits(hidden), all_relevances
