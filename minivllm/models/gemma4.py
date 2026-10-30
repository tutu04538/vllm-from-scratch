"""Gemma4 文本 target（80E.E2；对应冻结 `model_executor/models/gemma4.py` 的文本子集）。

这一关第一次把 Gemma4 的 **target 自身**接进本仓库。它和普通 Qwen 解码器有四处硬差异，
缺一项都会"能跑但数值不对"：

1. **YOCO 跨层 KV 共享**：最后 `num_kv_shared_layers` 层只算 Q，读前面同类型
   （sliding/full）最后一个非共享层的 paged KV。结构上它们仍走统一的 `qkv_proj`
   （checkpoint 里的 K/V 权重也会装进来），只是 forward 跳过 K/V 的 norm/RoPE。
   目标模型自己也会产生 Q-only 层，不只是 assistant 才 Q-only。
2. **混合注意力**：sliding 层 `head_dim=256` + default RoPE + window；full 层
   `head_dim=512` + proportional RoPE。两层型是两个 KV group，块表与 metadata 必须分开。
3. **Per-Layer Embedding（PLE）**：每层除了主干 hidden 还有一份
   `[vocab, num_layers * ple_dim]` 的 token 侧嵌入与 `[hidden -> num_layers*ple_dim]`
   的上下文投影；逐层 gate + projection + norm 后加到 residual 上。
4. **double-wide MLP**：共享层（YOCO 后半段）的 intermediate_size 翻倍，检查点里的
   `gate_proj/up_proj` 宽度也随之翻倍；不能拿前半段的形状去加载。

本文件只实现**文本**路径；视觉/音频塔不加载也不假装支持（真正的多模态输入会在
Processor 层被拒绝，不存在"图像 token 静默当文本"的路径）。权重命名同时接受
HF 文本模型（`model.*`）和条件生成包装（`model.language_model.*`）。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import torch
from torch import nn

from ..attention import Attention
from ..layers import (ColumnParallelLinear, MergedColumnParallelLinear,
                      ParallelLMHead, QKVParallelLinear, RMSNorm,
                      RowParallelLinear, VocabParallelEmbedding)
from ..layers.logits_processor import LogitsProcessor
from ..model_loader.auto_weights_loader import AutoWeightsLoader, WeightsMapper
from .gemma4_mtp import (Gemma4MLP, _cfg, _int, _layer_geometry, _layer_types,
                         _make_rotary_embedding, _text_config)
from .qwen3 import maybe_prefix
from .registry import register_model


def _is_shared_layer(text_config: Any, layer_idx: int) -> bool:
    num_layers = _int(text_config, "num_hidden_layers")
    num_shared = int(_cfg(text_config, "num_kv_shared_layers", 0) or 0)
    if num_shared < 0 or num_shared > num_layers:
        raise ValueError(
            f"num_kv_shared_layers={num_shared} 必须在 [0, {num_layers}]")
    return num_shared > 0 and layer_idx >= num_layers - num_shared


def _kv_source_layer(text_config: Any, layer_idx: int) -> int | None:
    """共享层读哪一层的 cache：前面同类型**最后一个**非共享层。"""
    if not _is_shared_layer(text_config, layer_idx):
        return None
    num_layers = _int(text_config, "num_hidden_layers")
    num_shared = int(_cfg(text_config, "num_kv_shared_layers", 0) or 0)
    first_shared = num_layers - num_shared
    types = _layer_types(text_config, num_layers)
    current = types[layer_idx]
    for idx in range(first_shared - 1, -1, -1):
        if types[idx] == current:
            return idx
    raise ValueError(
        f"Gemma4 第 {layer_idx} 层是 {current} 共享层，但前面没有同类型的非共享源层")


class Gemma4Attention(nn.Module):
    """target 的一层注意力。

    上游 target 的所有层都走**一个** `qkv_proj`；共享层也保留 K/V 权重和 k_norm/v_norm，
    只是 forward 里跳过它们，直接读源层 cache。不要按 mini 早期“共享层只有 q_proj”的
    结构理解上游。
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
            _int(text_config, "sliding_window") if self.is_sliding else None)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.qkv_proj = QKVParallelLinear(
            hidden_size, head_dim, num_heads, num_kv_heads,
            bias=bool(_cfg(text_config, "attention_bias", False)),
            prefix=maybe_prefix(prefix, "qkv_proj"))
        self.q_size = self.qkv_proj.q_size
        self.kv_size = self.qkv_proj.kv_size
        self.scaling = 1.0
        self.attention_bias = bool(_cfg(text_config, "attention_bias", False))
        self.attention_k_eq_v = bool(_cfg(text_config, "attention_k_eq_v", False))
        self.is_kv_shared_layer = _is_shared_layer(text_config, layer_idx)
        source_idx = _kv_source_layer(text_config, layer_idx)

        self.o_proj = RowParallelLinear(
            self.q_size, hidden_size, bias=self.attention_bias,
            prefix=maybe_prefix(prefix, "o_proj"))
        self.q_norm = RMSNorm(head_dim, eps=float(_cfg(
            text_config, "rms_norm_eps", 1e-6)))
        if self.is_kv_shared_layer:
            # 共享层不消费 K/V，不建这两个 norm；真实 checkpoint 里即使有对应
            # 权重也在加载器里显式跳过（无参数可装，不能静默丢）。
            self.k_norm = None
            self.v_norm = None
        else:
            self.k_norm = RMSNorm(head_dim, eps=float(_cfg(
                text_config, "rms_norm_eps", 1e-6)))
            self.v_norm = RMSNorm(head_dim, eps=float(_cfg(
                text_config, "rms_norm_eps", 1e-6)), has_weight=False)
        self.rotary_emb = _make_rotary_embedding(
            text_config, layer_idx, head_dim, layer_type, self.is_sliding)

        self.attn = Attention(
            num_heads=num_heads,
            head_size=head_dim,
            scale=self.scaling,
            num_kv_heads=num_kv_heads,
            layer_name=maybe_prefix(prefix, "attn"),
        )
        self.attn.q_only = self.is_kv_shared_layer
        self.attn.kv_sharing_target_layer_name = (
            None if source_idx is None
            else f"{_attn_prefix(prefix)}.{source_idx}.self_attn.attn")
        self.attn.per_layer_sliding_window = self.per_layer_sliding_window

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor,
                **kwargs: Any) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split(
            [self.q_size, self.kv_size, self.kv_size], dim=-1)

        q = q.unflatten(-1, (self.num_heads, self.head_dim))
        q = self.q_norm(q)
        q = q.flatten(-2, -1)

        if self.is_kv_shared_layer:
            # 共享层只处理 Q；K/V 已经按检查点加载，但这里不读取/不归一化。
            q, _ = self.rotary_emb(positions, q, None)
            attn_output = self.attn(q, k, v)
        else:
            k = k.unflatten(-1, (self.num_kv_heads, self.head_dim))
            k = self.k_norm(k)
            k = k.flatten(-2, -1)
            q, k = self.rotary_emb(positions, q, k)
            v = v.unflatten(-1, (self.num_kv_heads, self.head_dim))
            v = self.v_norm(v)
            v = v.flatten(-2, -1)
            attn_output = self.attn(q, k, v)

        output, _ = self.o_proj(attn_output)
        return output


def _attn_prefix(prefix: str) -> str:
    """`model.layers.3.self_attn` → `model.layers`（源层名只换层号）。"""
    marker = ".layers."
    if marker not in f".{prefix}.":
        raise ValueError(
            f"Gemma4 Attention 的 prefix 里必须含 '.layers.'，收到 {prefix!r}")
    before, _ = prefix.split(marker, 1)
    return before + ".layers"


class Gemma4DecoderLayer(nn.Module):
    """target 解码层：显式 residual + 四个 RMSNorm + 可选 PLE + layer_scalar。"""

    def __init__(self, text_config: Any, layer_idx: int, prefix: str = "") -> None:
        super().__init__()
        hidden_size = _int(text_config, "hidden_size")
        eps = float(_cfg(text_config, "rms_norm_eps", 1e-6))
        self.layer_idx = layer_idx
        self.hidden_size_per_layer_input = int(_cfg(
            text_config, "hidden_size_per_layer_input", 0) or 0)
        self.self_attn = Gemma4Attention(
            text_config, layer_idx, prefix=maybe_prefix(prefix, "self_attn"))

        first_shared = _int(text_config, "num_hidden_layers") - int(
            _cfg(text_config, "num_kv_shared_layers", 0) or 0)
        double_wide = (bool(_cfg(text_config, "use_double_wide_mlp", False))
                       and layer_idx >= first_shared and first_shared > 0)
        intermediate = _int(text_config, "intermediate_size") * (2 if double_wide else 1)
        self.mlp = Gemma4MLP(
            text_config, prefix=maybe_prefix(prefix, "mlp"),
            intermediate_size=intermediate)

        self.input_layernorm = RMSNorm(hidden_size, eps=eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=eps)
        self.pre_feedforward_layernorm = RMSNorm(hidden_size, eps=eps)
        self.post_feedforward_layernorm = RMSNorm(hidden_size, eps=eps)
        self.register_buffer("layer_scalar", torch.ones(1))

        if self.hidden_size_per_layer_input > 0:
            self.per_layer_input_gate = ColumnParallelLinear(
                hidden_size, self.hidden_size_per_layer_input, bias=False,
                return_bias=False, prefix=maybe_prefix(prefix, "per_layer_input_gate"))
            self.per_layer_projection = RowParallelLinear(
                self.hidden_size_per_layer_input, hidden_size, bias=False,
                prefix=maybe_prefix(prefix, "per_layer_projection"))
            self.post_per_layer_input_norm = RMSNorm(hidden_size, eps=eps)
        else:
            self.per_layer_input_gate = None
            self.per_layer_projection = None
            self.post_per_layer_input_norm = None

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor,
                residual: torch.Tensor | None = None,
                per_layer_input: torch.Tensor | None = None,
                **kwargs: Any):
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

        if per_layer_input is not None and self.per_layer_input_gate is not None:
            gate = self.per_layer_input_gate(hidden_states)
            gate = torch.nn.functional.gelu(gate, approximate="tanh")
            contribution, _ = self.per_layer_projection(gate * per_layer_input)
            contribution = self.post_per_layer_input_norm(contribution)
            hidden_states = hidden_states + contribution

        hidden_states = hidden_states * self.layer_scalar
        return hidden_states, None


class Gemma4TextModel(nn.Module):
    """文本主干：embedding + PLE + 全部 decoder 层 + final norm。"""

    def __init__(self, config: Any, prefix: str = "model") -> None:
        super().__init__()
        self.config = config
        self.text_config = _text_config(config)
        text_config = self.text_config
        self.hidden_size = _int(text_config, "hidden_size")
        self.num_hidden_layers = _int(text_config, "num_hidden_layers")
        self.vocab_size = _int(text_config, "vocab_size")
        self.hidden_size_per_layer_input = int(_cfg(
            text_config, "hidden_size_per_layer_input", 0) or 0)
        self.vocab_size_per_layer_input = int(_cfg(
            text_config, "vocab_size_per_layer_input", self.vocab_size))

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size, self.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"))
        self.register_buffer(
            "normalizer", torch.tensor(self.hidden_size ** 0.5), persistent=False)

        if self.hidden_size_per_layer_input > 0:
            total_ple_dim = self.hidden_size_per_layer_input * self.num_hidden_layers
            self.embed_tokens_per_layer = VocabParallelEmbedding(
                self.vocab_size_per_layer_input, total_ple_dim,
                prefix=maybe_prefix(prefix, "embed_tokens_per_layer"))
            self.register_buffer(
                "embed_scale_per_layer",
                torch.tensor(self.hidden_size_per_layer_input ** 0.5),
                persistent=False)
            self.per_layer_model_projection = ColumnParallelLinear(
                self.hidden_size, total_ple_dim, bias=False, return_bias=False,
                prefix=maybe_prefix(prefix, "per_layer_model_projection"))
            self.per_layer_projection_norm = RMSNorm(
                self.hidden_size_per_layer_input,
                eps=float(_cfg(text_config, "rms_norm_eps", 1e-6)))
            self.register_buffer(
                "per_layer_input_scale", torch.tensor(2.0) ** -0.5,
                persistent=False)
            self.register_buffer(
                "per_layer_projection_scale",
                torch.tensor(self.hidden_size) ** -0.5, persistent=False)
        else:
            self.embed_tokens_per_layer = None
            self.per_layer_model_projection = None
            self.per_layer_projection_norm = None

        self.layers = nn.ModuleList([
            Gemma4DecoderLayer(
                text_config, idx, prefix=maybe_prefix(prefix, f"layers.{idx}"))
            for idx in range(self.num_hidden_layers)
        ])
        self.norm = RMSNorm(
            self.hidden_size, eps=float(_cfg(text_config, "rms_norm_eps", 1e-6)))

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids) * self.normalizer

    def get_per_layer_inputs(self, input_ids: torch.Tensor) -> torch.Tensor | None:
        if self.embed_tokens_per_layer is None:
            return None
        valid = torch.logical_and(
            input_ids >= 0, input_ids < self.vocab_size_per_layer_input)
        safe_ids = torch.where(valid, input_ids, torch.zeros_like(input_ids))
        embeds = self.embed_tokens_per_layer(safe_ids) * self.embed_scale_per_layer
        return embeds.reshape(
            *input_ids.shape, self.num_hidden_layers,
            self.hidden_size_per_layer_input)

    def project_per_layer_inputs(
        self, inputs_embeds: torch.Tensor,
        per_layer_inputs: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if self.per_layer_model_projection is None:
            return None
        projection = self.per_layer_model_projection(inputs_embeds)
        projection = projection * self.per_layer_projection_scale
        projection = projection.reshape(
            *inputs_embeds.shape[:-1], self.num_hidden_layers,
            self.hidden_size_per_layer_input)
        projection = self.per_layer_projection_norm(projection)
        if per_layer_inputs is None:
            return projection
        return (projection + per_layer_inputs) * self.per_layer_input_scale

    def forward(self, input_ids: torch.Tensor | None, positions: torch.Tensor,
                inputs_embeds: torch.Tensor | None = None,
                per_layer_inputs: torch.Tensor | None = None,
                **kwargs: Any) -> torch.Tensor:
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Gemma4 需要 input_ids 或 inputs_embeds 之一")
            hidden_states = self.embed_input_ids(input_ids)
            raw_ple = self.get_per_layer_inputs(input_ids)
        else:
            hidden_states = inputs_embeds
            raw_ple = per_layer_inputs
            if self.hidden_size_per_layer_input > 0 and raw_ple is None:
                raise NotImplementedError(
                    "Gemma4 的 inputs_embeds 路径需要显式 per_layer_inputs："
                    "从 embedding 反推 token id 只对精确匹配有效，不能用于生产输入；"
                    "文本路径请直接传 input_ids")
        per_layer_inputs = self.project_per_layer_inputs(hidden_states, raw_ple)

        residual = None
        for idx, layer in enumerate(self.layers):
            layer_input = (
                per_layer_inputs[:, idx, :]
                if per_layer_inputs is not None else None)
            hidden_states, residual = layer(
                positions, hidden_states, residual,
                per_layer_input=layer_input, **kwargs)
        if residual is not None:
            hidden_states, _ = self.norm(hidden_states, residual)
        else:
            hidden_states = self.norm(hidden_states)
        return hidden_states


@register_model("Gemma4ForCausalLM", "Gemma4ForConditionalGeneration")
class Gemma4ForCausalLM(nn.Module):
    """文本 LM 头。条件生成包装也路由到这里，但只加载文本栈。"""

    def __init__(self, config: Any, prefix: str = "") -> None:
        super().__init__()
        self.config = config
        self.text_config = _text_config(config)
        self.model = Gemma4TextModel(
            config, prefix=maybe_prefix(prefix, "model"))
        self.lm_head = ParallelLMHead(
            _int(self.text_config, "vocab_size"),
            _int(self.text_config, "hidden_size"),
            prefix=maybe_prefix(prefix, "lm_head"))
        self.tie_word_embeddings = bool(_cfg(
            self.text_config, "tie_word_embeddings",
            _cfg(config, "tie_word_embeddings", True)))
        if self.tie_word_embeddings:
            self.lm_head.tie_weights(self.model.embed_tokens)
        self.logits_processor = LogitsProcessor(
            _int(self.text_config, "vocab_size"),
            soft_cap=_cfg(self.text_config, "final_logit_softcapping", None))

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(self, input_ids: torch.Tensor | None, positions: torch.Tensor,
                inputs_embeds: torch.Tensor | None = None,
                per_layer_inputs: torch.Tensor | None = None,
                **kwargs: Any) -> torch.Tensor:
        return self.model(
            input_ids=input_ids, positions=positions,
            inputs_embeds=inputs_embeds,
            per_layer_inputs=per_layer_inputs, **kwargs)

    def compute_logits(self, hidden_states: torch.Tensor,
                       spec_step_idx: int = 0) -> torch.Tensor:
        del spec_step_idx
        return self.logits_processor(self.lm_head, hidden_states)

    # -------- 权重路由 --------

    def _route_name(self, name: str) -> str:
        # 条件生成包装：model.language_model.* → model.*。文本模型本身不改。
        return name.replace("language_model.", "")

    def _full_attention_layer_indices(self) -> set[int]:
        return {
            idx for idx in range(self.model.num_hidden_layers)
            if self.model.layers[idx].self_attn.layer_type == "full_attention"
        }

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
        attns = [layer.self_attn for layer in self.model.layers]
        for attn in attns:
            attn.qkv_proj.reset_load_state()
        k_eq_v = bool(_cfg(self.text_config, "attention_k_eq_v", False))
        full_layers = self._full_attention_layer_indices()
        shared_layers = {
            idx for idx, attn in enumerate(attns)
            if attn.is_kv_shared_layer
        }

        def routed():
            for name, tensor in weights:
                name = self._route_name(name)
                if name.startswith(("vision_tower.", "audio_tower.",
                                    "embed_vision.", "embed_audio.")):
                    continue
                if name.endswith(".layer_scalar"):
                    if self._load_buffer(name, tensor, buffers):
                        loaded_buffers.add(name)
                        continue
                    raise ValueError(f"检查点里的 {name!r} 没有对应 buffer")
                if name in buffers and self._load_buffer(name, tensor, buffers):
                    loaded_buffers.add(name)
                    continue

                # 共享层没有 k_norm/v_norm 参数（forward 不消费 K/V）；真实
                # checkpoint 里可能仍有这两份权重，显式跳过，不能当成未知名字。
                if ("self_attn." in name
                        and name.rsplit(".", 2)[-2] in ("k_norm", "v_norm")
                        and ".layers." in name):
                    try:
                        layer_idx = int(
                            name.split(".layers.", 1)[1].split(".", 1)[0])
                    except (IndexError, ValueError):
                        layer_idx = -1
                    if layer_idx in shared_layers:
                        continue

                # 上游 `_weight_iterator`：k_eq_v 的 full 层 checkpoint 只有 k_proj，
                # 把同一份 K 权重复制到 V 槽。复制后 V 仍会走自己的 v_norm（无权重），
                # 不是“把已经 k_norm/RoPE 的 k 直接当 v”。
                if (k_eq_v and "self_attn.k_proj" in name
                        and ".layers." in name):
                    try:
                        layer_idx = int(
                            name.split(".layers.", 1)[1].split(".", 1)[0])
                    except (IndexError, ValueError):
                        layer_idx = -1
                    if layer_idx in full_layers:
                        yield name, tensor
                        yield name.replace("k_proj", "v_proj"), tensor.clone()
                        continue
                yield name, tensor

        mapper = WeightsMapper(
            orig_to_new_stacked={
                ".q_proj": (".qkv_proj", "q"),
                ".k_proj": (".qkv_proj", "k"),
                ".v_proj": (".qkv_proj", "v"),
                ".gate_proj": (".gate_up_proj", 0),
                ".up_proj": (".gate_up_proj", 1),
            },
        )
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.tie_word_embeddings else None),
            skip_substrs=["vision_tower.", "audio_tower.", "embed_vision.",
                          "embed_audio."],
        )
        loaded = loader.load_weights(routed(), mapper=mapper)
        # 非共享层必须拿到 q/k/v 三个 shard；共享层 forward 只消费 Q，允许
        # checkpoint 只提供 q_proj（上游 tiny/HF 文本栈就是这样），但缺 Q 仍报错。
        for idx, attn in enumerate(attns):
            required = {"q"} if attn.is_kv_shared_layer else {"q", "k", "v"}
            try:
                attn.qkv_proj.validate_loaded_shards(required)
            except ValueError as exc:
                raise ValueError(f"Gemma4 第 {idx} 层：{exc}") from exc
        return loaded | loaded_buffers
