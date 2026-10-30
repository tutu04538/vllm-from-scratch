"""Gemma4 MTP assistant 模型（80E.E1；对应冻结 `model_executor/models/gemma4_mtp.py`）。

**它不是"又一个 Qwen MTP"**。Gemma4 assistant 与 target 的宽度、层结构、注意力和权重命名都
不同，E1 要先把模型层做对，E2 才谈真实 target 接入：

    Gemma4MTP
      model: Gemma4MultiTokenPredictor
        embed_tokens                 target 共享的词嵌入（backbone 宽）
        pre_projection               2 * backbone -> assistant
        layers[]                     Gemma4MTPDecoderLayer（Q-only attention + Gemma4MLP）
        norm                         assistant 宽
        post_projection              assistant -> backbone（下一步反馈）
      lm_head                        assistant 宽 -> 词表
      masked_embedding               可选：centroid 选簇后只算部分词表 logits

固定例子：backbone=16、assistant=8。输入 `[embedding(16) ‖ hidden(16)]` 先拼成 32，再
`pre_projection 32→8`；decoder 里全是 8 宽；`draft_hidden 8` 给出 logits，
`post_projection 8→16` 才是下一步要喂回的 feedback hidden。**两条宽度不能混**。

本文件只做 E1 的模型层与权重路由；不接 V2/V1 提议者，也不加载真实 target。真实接入需要
Gemma4 target、81 的多 group/SWA/full/backend 前置，属 80E.E2。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import torch
from torch import nn

from ..attention import Attention
from ..layers import (ColumnParallelLinear, MergedColumnParallelLinear, ParallelLMHead,
                      RMSNorm, RotaryEmbedding, RowParallelLinear,
                      VocabParallelEmbedding, get_act_and_mul_fn, get_rope)
from ..layers.logits_processor import LogitsProcessor
from ..model_loader.auto_weights_loader import AutoWeightsLoader, WeightsMapper
from .qwen3 import maybe_prefix
from .registry import register_model

DEFAULT_LAYER_TYPE = "full_attention"


def _cfg(config: Any, key: str, default: Any = None) -> Any:
    """同时读 dict 与配置对象（上游 hf_config 是对象，本仓库常传 dict）。"""
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def _text_config(config: Any) -> Any:
    if isinstance(config, Mapping):
        nested = config.get("text_config")
        if nested:
            return nested
    getter = getattr(config, "get_text_config", None)
    if callable(getter):
        return getter()
    return config


def _int(config: Any, key: str, default: int | None = None) -> int:
    value = _cfg(config, key, default)
    if value is None:
        raise ValueError(f"Gemma4 配置缺少必需的整数字段 {key!r}")
    return int(value)


def _layer_types(text_config: Any, num_layers: int) -> list[str]:
    types = _cfg(text_config, "layer_types", None)
    if not types:
        return [DEFAULT_LAYER_TYPE] * num_layers
    types = [str(item) for item in types]
    if len(types) < num_layers:
        types = types + [DEFAULT_LAYER_TYPE] * (num_layers - len(types))
    return types[:num_layers]


def _layer_geometry(text_config: Any, layer_idx: int) -> tuple[str, int, int]:
    """返回 `(layer_type, head_dim, num_kv_heads)`，对齐 `gemma4_layer_config()`。

    Gemma4 的 full 层可以用更大的 head_dim（以及 `attention_k_eq_v` 时更多的 KV heads）。
    这两项必须按**层**解析，不能拿 text config 的 flat 值一把梭——那会让 full 层的
    Q 形状与 target 的 K/V 对不上。
    """
    num_layers = _int(text_config, "num_hidden_layers")
    types = _layer_types(text_config, num_layers)
    layer_type = types[layer_idx]
    head_dim = _int(text_config, "head_dim")
    num_kv_heads = _int(text_config, "num_key_value_heads")

    per_layer = _cfg(text_config, "per_layer_config", None)
    if bool(_cfg(text_config, "is_heterogeneous", False)) and per_layer:
        if layer_idx >= len(per_layer):
            raise ValueError(
                f"per_layer_config 只有 {len(per_layer)} 项，取不到第 {layer_idx} 层")
        layer_cfg = per_layer[layer_idx]
        return (str(_cfg(layer_cfg, "layer_type", layer_type)),
                _int(layer_cfg, "head_dim"),
                _int(layer_cfg, "num_key_value_heads"))

    if layer_type == "full_attention":
        global_head_dim = _cfg(text_config, "global_head_dim", None)
        if global_head_dim:
            head_dim = int(global_head_dim)
        if bool(_cfg(text_config, "attention_k_eq_v", False)):
            global_kv_heads = _cfg(text_config, "num_global_key_value_heads", None)
            if global_kv_heads:
                num_kv_heads = int(global_kv_heads)
    return layer_type, head_dim, num_kv_heads


def _rope_parameters(text_config: Any, layer_type: str, is_sliding: bool) -> dict[str, Any]:
    all_parameters = _cfg(text_config, "rope_parameters", None) or {}
    if isinstance(all_parameters, Mapping) and isinstance(
            all_parameters.get(layer_type), Mapping):
        parameters = dict(all_parameters[layer_type])
    else:
        parameters = dict(all_parameters) if isinstance(all_parameters, Mapping) else {}

    if "rope_theta" not in parameters:
        if is_sliding:
            parameters["rope_theta"] = _cfg(
                text_config, "rope_local_base_freq", 10000.0)
        else:
            parameters["rope_theta"] = _cfg(text_config, "rope_theta", 10000.0)
    return parameters


def _make_rotary_embedding(text_config: Any, layer_idx: int, head_dim: int,
                           layer_type: str, is_sliding: bool) -> RotaryEmbedding:
    """配置转发到统一 `get_rope`；**不在这里自己建表**。

    Gemma4 target/assistant 的 35+4 层最终只会落到两个 key（sliding/default 与
    full/proportional）。层号只用于报错和日志，不进 cache key；否则每层都会
    重新建一份相同的表，共享失效。
    """
    del layer_idx  # 层号不影响 RoPE 数学；保留参数是为了调用点/报错上下文清晰
    parameters = _rope_parameters(text_config, layer_type, is_sliding)
    return get_rope(
        head_size=head_dim,
        max_position=_int(text_config, "max_position_embeddings"),
        is_neox_style=True,
        rope_parameters=parameters,
    )


class Gemma4MLP(nn.Module):
    """Gemma4 的 gated MLP（对应冻结 `gemma4.py::Gemma4MLP`）。"""

    def __init__(self, text_config: Any, prefix: str = "",
                 intermediate_size: int | None = None) -> None:
        super().__init__()
        hidden_size = _int(text_config, "hidden_size")
        if intermediate_size is None:
            intermediate_size = _int(text_config, "intermediate_size")
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size, intermediate_size],
            bias=False,
            prefix=maybe_prefix(prefix, "gate_up_proj"),
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            prefix=maybe_prefix(prefix, "down_proj"),
        )
        self.act_fn = get_act_and_mul_fn(
            str(_cfg(text_config, "hidden_activation", "gelu_pytorch_tanh")))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class Gemma4MTPMaskedEmbedder(nn.Module):
    """centroid 选簇 + 稀疏 logits（对应冻结同名类）。

    完整词表 logits 是 `[num_tokens, vocab]`；这里先算 `[num_tokens, num_centroids]`，
    每个 token 只取 top-k 个簇，簇内 `vocab_size / num_centroids` 个 token 做点积。
    输出仍然是完整 `[num_tokens, vocab]`（未选中的列填 `-inf`），但**不物化**完整词表权重。
    """

    def __init__(self, hidden_size: int, vocab_size: int, num_centroids: int,
                 centroid_intermediate_top_k: int) -> None:
        super().__init__()
        if num_centroids <= 0 or vocab_size % num_centroids != 0:
            raise ValueError(
                f"vocab_size={vocab_size} 必须能被 num_centroids={num_centroids} 整除")
        if not 0 < centroid_intermediate_top_k <= num_centroids:
            raise ValueError(
                f"centroid_intermediate_top_k={centroid_intermediate_top_k} 必须在 "
                f"[1, {num_centroids}]")
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_centroids = num_centroids
        self.centroid_intermediate_top_k = centroid_intermediate_top_k
        self.vocab_size_per_centroid = vocab_size // num_centroids
        self.num_selected = (
            centroid_intermediate_top_k * self.vocab_size_per_centroid)

        self.centroids = nn.Linear(hidden_size, num_centroids, bias=False)
        self.register_buffer(
            "token_ordering", torch.empty(vocab_size, dtype=torch.long))

    def _select_and_score(self, hidden_states: torch.Tensor,
                          lm_head_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        num_tokens = hidden_states.shape[0]
        _, top_k_indices = torch.topk(
            self.centroids(hidden_states),
            k=self.centroid_intermediate_top_k,
            dim=-1,
        )
        clusters = self.token_ordering.view(
            self.num_centroids, self.vocab_size_per_centroid)
        selected = clusters[top_k_indices]
        embeddings = lm_head_weight[selected.reshape(-1)].view(
            num_tokens, self.num_selected, self.hidden_size)
        logits = torch.einsum("td,tsd->ts", hidden_states, embeddings)
        return logits, selected.view(num_tokens, -1)

    def forward(self, hidden_states: torch.Tensor,
                lm_head_weight: torch.Tensor) -> torch.Tensor:
        logits, indices = self._select_and_score(hidden_states, lm_head_weight)
        output = torch.full(
            (hidden_states.shape[0], self.vocab_size),
            fill_value=torch.finfo(hidden_states.dtype).min,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        return output.scatter_(-1, indices, logits)

    def get_top_tokens(self, hidden_states: torch.Tensor,
                       lm_head_weight: torch.Tensor) -> torch.Tensor:
        logits, indices = self._select_and_score(hidden_states, lm_head_weight)
        return indices.gather(-1, logits.argmax(-1, keepdim=True)).squeeze(-1)


class Gemma4MTPAttention(nn.Module):
    """Q-only attention（对应冻结 `Gemma4MTPAttention`）。

    **没有 K/V 投影**：只算 Q，K/V 由 target 写入共享 cache，本层只读。`scaling=1.0`
    是 Gemma4 的固定值（不是常见的 `1/sqrt(head_dim)`）；`q_norm` 作用在每个 head 的
    `head_dim` 上；RoPE 按层类型选 sliding/full，full 层用 proportional。
    """

    def __init__(self, text_config: Any, layer_idx: int, prefix: str = "") -> None:
        super().__init__()
        hidden_size = _int(text_config, "hidden_size")
        num_heads = _int(text_config, "num_attention_heads")
        layer_type, head_dim, num_kv_heads = _layer_geometry(text_config, layer_idx)
        self.layer_idx = layer_idx
        self.layer_type = layer_type
        self.is_sliding = layer_type == "sliding_attention"
        self.per_layer_sliding_window = (
            _cfg(text_config, "sliding_window", None) if self.is_sliding else None)
        attn_soft_cap = _cfg(text_config, "attn_logit_softcapping", None)
        if attn_soft_cap is not None:
            raise NotImplementedError(
                f"Gemma4 第 {layer_idx} 层声明了 attn_logit_softcapping="
                f"{attn_soft_cap}；本仓库的 Torch 注意力后端尚未实现该 softcap，"
                f"不能静默忽略（会改变注意力分布）")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.q_size = num_heads * head_dim
        self.scaling = 1.0

        self.q_proj = ColumnParallelLinear(
            hidden_size,
            self.q_size,
            bias=bool(_cfg(text_config, "attention_bias", False)),
            prefix=maybe_prefix(prefix, "q_proj"),
        )
        self.o_proj = RowParallelLinear(
            self.q_size,
            hidden_size,
            bias=bool(_cfg(text_config, "attention_bias", False)),
            prefix=maybe_prefix(prefix, "o_proj"),
        )
        self.q_norm = RMSNorm(head_dim, eps=float(_cfg(text_config, "rms_norm_eps", 1e-6)))
        self.rotary_emb = _make_rotary_embedding(
            text_config, layer_idx, head_dim, layer_type, self.is_sliding)

        self.attn = Attention(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            scale=self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_name=maybe_prefix(prefix, "attn"),
        )
        # Gemma4 assistant 的每一层都是 Q-only；80D 的映射会再写一遍
        # `kv_sharing_target_layer_name`，这里先把"不能写 KV"的行为定死。
        self.attn.q_only = True
        self.attn.kv_sharing_target_layer_name = None
        self.attn.per_layer_sliding_window = self.per_layer_sliding_window
        self.is_kv_shared_layer = True

    def attention_pre(self, positions: torch.Tensor,
                      hidden_states: torch.Tensor) -> torch.Tensor:
        """只算 Q：投影 → per-head RMSNorm → RoPE。"""
        q, _ = self.q_proj(hidden_states)
        q = q.unflatten(-1, (self.num_heads, self.head_dim))
        q = self.q_norm(q)
        q = q.flatten(-2, -1)
        q, _ = self.rotary_emb(positions, q, None)
        return q

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor,
                **kwargs: Any) -> torch.Tensor:
        q = self.attention_pre(positions, hidden_states)
        num_tokens = q.shape[0]
        # 与上游相同的占位 K/V：Attention 的 API 需要三个输入，但 Q-only 路径不会读它们。
        kv_dummy = torch.empty(
            num_tokens,
            self.num_kv_heads * self.head_dim,
            dtype=q.dtype,
            device=q.device,
        )
        attn_output = self.attn(q, kv_dummy, kv_dummy)
        output, _ = self.o_proj(attn_output)
        return output


class Gemma4MTPDecoderLayer(nn.Module):
    """Gemma4 assistant 的一层（对应冻结 `Gemma4MTPDecoderLayer`）。

    残差顺序与 Qwen3 不同：这里是**显式**的 `residual = hidden_states` 加回，不是
    `RMSNorm(x, residual)` 的融合签名；最后还要乘 `layer_scalar`。照抄冻结结构，别拿
    Qwen3 的 layer 替换。
    """

    def __init__(self, text_config: Any, layer_idx: int, prefix: str = "") -> None:
        super().__init__()
        hidden_size = _int(text_config, "hidden_size")
        eps = float(_cfg(text_config, "rms_norm_eps", 1e-6))
        self.self_attn = Gemma4MTPAttention(
            text_config, layer_idx, prefix=maybe_prefix(prefix, "self_attn"))
        if bool(_cfg(text_config, "enable_moe_block", False)) or bool(
                _cfg(text_config, "use_double_wide_mlp", False)):
            raise NotImplementedError(
                "Gemma4 MTP 的 MoE / double-wide MLP 结构尚未适配；本实现只覆盖"
                "稠密 Gemma4MLP")
        self.mlp = Gemma4MLP(text_config, prefix=maybe_prefix(prefix, "mlp"))
        self.input_layernorm = RMSNorm(hidden_size, eps=eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=eps)
        self.pre_feedforward_layernorm = RMSNorm(hidden_size, eps=eps)
        self.post_feedforward_layernorm = RMSNorm(hidden_size, eps=eps)
        self.register_buffer("layer_scalar", torch.ones(1))

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor,
                residual: torch.Tensor | None, **kwargs: Any):
        residual = hidden_states
        hidden_states = self.input_layernorm(residual)

        hidden_states = self.self_attn(
            positions=positions, hidden_states=hidden_states, **kwargs)

        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = hidden_states + residual
        residual = hidden_states

        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = hidden_states + residual

        hidden_states = hidden_states * self.layer_scalar
        return hidden_states, None


class Gemma4MultiTokenPredictor(nn.Module):
    """assistant 主干：embedding + 两个跨宽度投影 + 全部 decoder 层 + final norm。"""

    def __init__(self, config: Any, prefix: str = "") -> None:
        super().__init__()
        text_config = _text_config(config)
        self.config = text_config
        self.hidden_size = _int(text_config, "hidden_size")
        self.backbone_hidden_size = int(_cfg(
            config, "backbone_hidden_size", self.hidden_size))
        self.vocab_size = _int(text_config, "vocab_size")
        self.num_mtp_layers = _int(text_config, "num_hidden_layers")
        if int(_cfg(text_config, "hidden_size_per_layer_input", 0) or 0) != 0 or int(
                _cfg(text_config, "vocab_size_per_layer_input", 0) or 0) != 0:
            raise NotImplementedError(
                "Gemma4 的 per-layer input embedding 尚未适配；"
                "assistant 检查点要求这两个字段为 0")

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size, self.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"))
        self.pre_projection = ColumnParallelLinear(
            2 * self.backbone_hidden_size,
            self.hidden_size,
            bias=False,
            return_bias=False,
            prefix=maybe_prefix(prefix, "pre_projection"),
        )
        self.layers = nn.ModuleList([
            Gemma4MTPDecoderLayer(
                text_config, idx, prefix=maybe_prefix(prefix, f"layers.{idx}"))
            for idx in range(self.num_mtp_layers)
        ])
        self.norm = RMSNorm(
            self.hidden_size, eps=float(_cfg(text_config, "rms_norm_eps", 1e-6)))
        self.post_projection = RowParallelLinear(
            self.hidden_size,
            self.backbone_hidden_size,
            bias=False,
            prefix=maybe_prefix(prefix, "post_projection"),
        )
        # embedding 与 target 共享后是 backbone 宽；上游按 sqrt(backbone) 缩放。
        self.register_buffer(
            "normalizer",
            torch.tensor(self.backbone_hidden_size ** 0.5),
            persistent=False,
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids) * self.normalizer

    def forward(self, input_ids: torch.Tensor | None, positions: torch.Tensor,
                hidden_states: torch.Tensor,
                intermediate_tensors: Any = None,
                inputs_embeds: torch.Tensor | None = None,
                spec_step_idx: int = 0):
        del intermediate_tensors, spec_step_idx
        if hidden_states.shape[-1] != self.backbone_hidden_size:
            raise ValueError(
                f"Gemma4 assistant 的 hidden_states 必须是 backbone 宽 "
                f"{self.backbone_hidden_size}，收到 {hidden_states.shape[-1]}；"
                f"不要把 assistant 宽的 draft hidden 当反馈输入")
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Gemma4 MTP 需要 input_ids 或 inputs_embeds 之一")
            inputs_embeds = self.embed_input_ids(input_ids)
        if inputs_embeds.shape[-1] != self.backbone_hidden_size:
            raise ValueError(
                f"Gemma4 assistant 的 inputs_embeds 必须是 backbone 宽 "
                f"{self.backbone_hidden_size}，收到 {inputs_embeds.shape[-1]}")

        combined = torch.cat([inputs_embeds, hidden_states], dim=-1)
        hidden_states = self.pre_projection(combined)

        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )

        draft_hidden_states = self.norm(hidden_states)
        backbone_hidden_states, _ = self.post_projection(draft_hidden_states)
        return draft_hidden_states, backbone_hidden_states


@register_model("Gemma4MTPModel")
class Gemma4MTP(nn.Module):
    """顶层 assistant（对应冻结 `Gemma4MTP`）：主干 + draft lm_head + 可选 masked embedder。"""

    def __init__(self, config: Any, prefix: str = "") -> None:
        super().__init__()
        text_config = _text_config(config)
        self.config = config
        self.text_config = text_config
        self.model = Gemma4MultiTokenPredictor(
            config, prefix=maybe_prefix(prefix, "model"))

        self.lm_head = ParallelLMHead(
            _int(text_config, "vocab_size"),
            _int(text_config, "hidden_size"),
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.tie_word_embeddings = bool(_cfg(
            config, "tie_word_embeddings",
            _cfg(text_config, "tie_word_embeddings", True)))
        if self.tie_word_embeddings:
            self.lm_head.tie_weights(self.model.embed_tokens)

        self.logits_processor = LogitsProcessor(
            _int(text_config, "vocab_size"),
            soft_cap=_cfg(text_config, "final_logit_softcapping", None),
        )

        self.use_ordered_embeddings = bool(
            _cfg(config, "use_ordered_embeddings", False))
        if self.use_ordered_embeddings:
            self.masked_embedding: Gemma4MTPMaskedEmbedder | None = (
                Gemma4MTPMaskedEmbedder(
                    hidden_size=_int(text_config, "hidden_size"),
                    vocab_size=_int(text_config, "vocab_size"),
                    num_centroids=int(_cfg(config, "num_centroids", 2048)),
                    centroid_intermediate_top_k=int(
                        _cfg(config, "centroid_intermediate_top_k", 32)),
                ))
        else:
            self.masked_embedding = None

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(self, input_ids: torch.Tensor | None, positions: torch.Tensor,
                hidden_states: torch.Tensor,
                intermediate_tensors: Any = None,
                inputs_embeds: torch.Tensor | None = None,
                spec_step_idx: int = 0, **kwargs: Any):
        return self.model(
            input_ids=input_ids,
            positions=positions,
            hidden_states=hidden_states,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            spec_step_idx=spec_step_idx,
        )

    def _get_full_lm_head_weight(self) -> torch.Tensor:
        # TP=1：没有 all-gather；只按 masked embedder 的词表宽度截断。
        assert self.masked_embedding is not None
        return self.lm_head.weight[: self.masked_embedding.vocab_size]

    def compute_logits(self, hidden_states: torch.Tensor,
                       spec_step_idx: int = 0) -> torch.Tensor:
        del spec_step_idx
        if self.masked_embedding is not None:
            return self.masked_embedding(
                hidden_states, self._get_full_lm_head_weight())
        return self.logits_processor(self.lm_head, hidden_states)

    def get_top_tokens(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.masked_embedding is None:
            raise NotImplementedError(
                "get_top_tokens 只用于 use_ordered_embeddings=True 的 centroid 路径")
        return self.masked_embedding.get_top_tokens(
            hidden_states, self._get_full_lm_head_weight())

    # -------- 权重路由（对应冻结 `hf_to_vllm_mapper` + `load_weights`） --------

    def _route_checkpoint_name(self, name: str) -> str:
        # 检查点把两个跨宽度投影放在顶层；模型里它们在 `model.` 下。
        if name.startswith(("pre_projection.", "post_projection.")):
            return "model." + name
        return name

    def _load_buffer(self, name: str, tensor: torch.Tensor,
                     buffers: dict[str, torch.Tensor]) -> bool:
        target = buffers.get(name)
        if target is None:
            return False
        if tuple(target.shape) != tuple(tensor.shape):
            raise ValueError(
                f"Gemma4 buffer {name!r} 形状不匹配：目标 {tuple(target.shape)}，"
                f"收到 {tuple(tensor.shape)}")
        with torch.no_grad():
            target.copy_(tensor)
        return True

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        buffers = dict(self.named_buffers())
        loaded_buffers: set[str] = set()

        def routed():
            for name, tensor in weights:
                routed_name = self._route_checkpoint_name(name)
                if self._load_buffer(routed_name, tensor, buffers):
                    loaded_buffers.add(routed_name)
                    continue
                yield routed_name, tensor

        mapper = WeightsMapper(
            orig_to_new_stacked={
                ".gate_proj": (".gate_up_proj", 0),
                ".up_proj": (".gate_up_proj", 1),
            },
        )
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["lm_head."] if self.tie_word_embeddings else None,
        )
        loaded = loader.load_weights(routed(), mapper=mapper)
        if (self.masked_embedding is not None
                and "masked_embedding.token_ordering" not in loaded_buffers):
            raise ValueError(
                "use_ordered_embeddings=True 时检查点必须提供 "
                "`masked_embedding.token_ordering`；否则 centroid 选簇会在 forward 时越界")
        return loaded | loaded_buffers
