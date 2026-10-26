"""Step 3.5（`step3p5`）文本模型与 MTP 的 PyTorch 子集。

对应冻结上游：

    vllm/model_executor/models/step3p5.py      Step3p5Attention / DecoderLayer / ForCausalLM
    vllm/model_executor/models/step3p5_mtp.py  Step3p5MTP / Step3p5AMultiTokenPredictor

本文件只覆盖 80F 需要的文本路径，并明确保留三条能力边界：

1. **MTP 层是 dense MLP**（真实配置 `moe_layers_enum` 不含 45–47）；目标模型的 MoE
   只在 tiny 配置里用来验证路由/共享专家，真实 196B target 不加载、不声称对齐。
2. **full-attention 层的 llama3 rope scaling 未实现**；MTP 的三层都是 sliding，
   `yarn_only_types=["full_attention"]` 会把 scaling 丢掉，所以真实 MTP 路径不受影响。
   加载真实 target 的 full 层会在初始化时明确报错，而不是静默用错 RoPE。
3. **真实 target 的 fused MoE 权重名**没有映射；真实 196B 权重不在本机，也不适合 25GB GPU。
"""

from __future__ import annotations

import math
import re

import torch
import torch.nn.functional as F
from torch import nn

from ..attention import Attention
from ..layers import (ColumnParallelLinear, GemmaRMSNorm, MergedColumnParallelLinear,
                      ParallelLMHead, QKVParallelLinear, RMSNorm, RotaryEmbedding,
                      RowParallelLinear, SiluAndMul, SwigluStepAndMul,
                      VocabParallelEmbedding)
from ..model_loader.auto_weights_loader import AutoWeightsLoader, WeightsMapper
from .qwen3 import maybe_prefix
from .registry import register_model
from .utils import get_spec_layer_idx_from_weight_name, skip_spec_layer_weights


def _cfg(config: dict, key: str, default=None):
    value = config.get(key, default)
    return default if value is None else value


def _layer_index(prefix: str) -> int:
    match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", prefix)
    if match is None:
        raise ValueError(f"Step3p5 层前缀里找不到 layer index：{prefix!r}")
    return int(match.group(1))


def _layer_type(config: dict, layer_idx: int) -> str:
    types = _cfg(config, "layer_types", []) or []
    if layer_idx < len(types):
        return types[layer_idx]
    return "sliding_attention" if layer_idx % 2 == 0 else "full_attention"


def _swiglu_limit(values, layer_idx: int) -> float | None:
    if not values or layer_idx >= len(values):
        return None
    value = values[layer_idx]
    return None if value in (None, 0, 0.0) else float(value)


class Step3p5MLP(nn.Module):
    """dense SwiGLU（对应上游 `Step3p5MLP`）。"""

    def __init__(self, config: dict, *, prefix: str = "", limit: float | None = None,
                 intermediate_size: int | None = None) -> None:
        super().__init__()
        hidden_size = int(config["hidden_size"])
        intermediate = int(intermediate_size if intermediate_size is not None
                           else config["intermediate_size"])
        self.hidden_size = hidden_size
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate, intermediate], bias=False,
            prefix=maybe_prefix(prefix, "gate_up_proj"))
        self.down_proj = RowParallelLinear(
            intermediate, hidden_size, bias=False,
            prefix=maybe_prefix(prefix, "down_proj"))
        self.limit = limit
        self.act_fn = (SwigluStepAndMul(limit) if limit not in (None, 0, 0.0)
                       else SiluAndMul())

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(hidden_states)
        act = self.act_fn(gate_up)
        output, _ = self.down_proj(act)
        return output


class Step3p5MoE(nn.Module):
    """Step3p5 的稀疏 MoE 教学实现（router + shared expert + top-k experts）。

    这**不是**上游 `FusedMoE` 的 kernel 级等价：这里用 Python 逐个 expert 算，只用于 tiny
    配置验证路由/加权/共享专家语义。真实 196B target 的 fused 权重名不在本文件的加载范围内。
    """

    def __init__(self, config: dict, *, prefix: str = "", layer_idx: int = 0) -> None:
        super().__init__()
        hidden = int(config["hidden_size"])
        num_experts = int(config["moe_num_experts"])
        self.num_experts = num_experts
        self.top_k = int(config.get("moe_top_k", 1))
        self.scaling = float(config.get("moe_router_scaling_factor", 1.0))
        self.normalize = bool(config.get("norm_expert_weight", True))
        self.activation = str(config.get("moe_router_activation", "sigmoid"))
        if self.activation not in ("sigmoid", "softmax"):
            raise NotImplementedError(f"不支持的 MoE router 激活 {self.activation!r}")
        if self.top_k > num_experts:
            raise ValueError(f"moe_top_k={self.top_k} 大于专家数 {num_experts}")

        self.gate = ColumnParallelLinear(
            hidden, num_experts, bias=False, return_bias=False,
            prefix=maybe_prefix(prefix, "gate"))
        # 上游 router bias 是 FP32 buffer；本仓库用 Parameter 以便加载覆盖检查。
        self.router_bias = nn.Parameter(torch.zeros(num_experts, dtype=torch.float32),
                                        requires_grad=False)
        limit = _swiglu_limit(config.get("swiglu_limits"), layer_idx)
        self.share_expert = Step3p5MLP(
            config, prefix=maybe_prefix(prefix, "share_expert"), limit=limit,
            intermediate_size=int(config.get("share_expert_dim") or
                                  config["moe_intermediate_size"]))
        expert_size = int(config.get("moe_intermediate_size", config["intermediate_size"]))
        self.experts = nn.ModuleList([
            Step3p5MLP(config, prefix=maybe_prefix(prefix, f"experts.{index}"),
                       limit=limit, intermediate_size=expert_size)
            for index in range(num_experts)
        ])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # gate 用 FP32（上游 need_fp32_gate=True）。
        # gate 权重在 bf16 checkpoint 里也可能是 bf16；先转 FP32 再做 router。
        router = F.linear(hidden_states.float(), self.gate.weight.float(),
                          None) + self.router_bias.float()
        if self.activation == "sigmoid":
            scores = torch.sigmoid(router)
        else:
            scores = torch.softmax(router, dim=-1)
        top_values, top_indices = torch.topk(scores, self.top_k, dim=-1)
        if self.normalize:
            top_values = top_values / top_values.sum(dim=-1, keepdim=True).clamp_min(1e-20)
        out = torch.zeros_like(hidden_states)
        # tiny 路径：逐 token、逐专家；真实大模型应换 fused kernel，不能拿这个跑 196B。
        for token in range(hidden_states.shape[0]):
            row = hidden_states[token:token + 1]
            token_out = torch.zeros_like(row)
            for slot in range(self.top_k):
                expert_idx = int(top_indices[token, slot])
                weight = top_values[token, slot].to(row.dtype)
                token_out = token_out + weight * self.experts[expert_idx](row)
            out[token] = token_out
        out = out * self.scaling
        out = out + self.share_expert(hidden_states)
        return out


class Step3p5Attention(nn.Module):
    """GQA + qk-norm + partial RoPE + 可选 head-wise gate（上游 `Step3p5Attention`）。"""

    def __init__(self, config: dict, *, prefix: str = "") -> None:
        super().__init__()
        hidden = int(config["hidden_size"])
        layer_idx = _layer_index(prefix)
        layer_type = _layer_type(config, layer_idx)
        num_heads = int(config["num_attention_heads"])
        num_kv_heads = int(config.get("num_attention_groups", num_heads))
        head_dim = int(config.get("head_dim") or hidden // num_heads)
        other = config.get("attention_other_setting") or {}
        if other and layer_type == other.get("attention_type"):
            num_heads = int(other.get("num_attention_heads", num_heads))
            num_kv_heads = int(other.get("num_attention_groups", num_kv_heads))
            head_dim = int(other.get("head_dim", head_dim))

        sliding_window = config.get("sliding_window")
        if layer_type != "sliding_attention" or sliding_window is None:
            sliding_window = None
        self.sliding_window = None if sliding_window is None else int(sliding_window)

        rope_theta = config.get("rope_theta", 10000.0)
        if isinstance(rope_theta, list):
            rope_theta = rope_theta[layer_idx]
        rope_scaling = config.get("rope_scaling")
        yarn_only = config.get("yarn_only_types") or []
        if yarn_only and layer_type not in yarn_only:
            rope_scaling = None
        if rope_scaling is not None:
            rope_type = (rope_scaling or {}).get("rope_type", "default")
            if rope_type not in ("default", None):
                raise NotImplementedError(
                    f"Step3p5 的 {rope_type!r} RoPE scaling 尚未实现（层 {layer_idx}，"
                    f"type={layer_type}）；MTP 的 sliding 层不会走到这里，"
                    f"但真实 target 的 full 层需要它")

        partial = _cfg(config, "partial_rotary_factors", []) or []
        partial_factor = float(partial[layer_idx]) if layer_idx < len(partial) else 1.0
        if partial_factor not in (0.5, 1.0):
            raise ValueError(f"Step3p5 partial_rotary_factor 只支持 0.5/1.0，收到 {partial_factor}")
        use_rope_layers = config.get("use_rope_layers") or []
        self.use_rope = (bool(use_rope_layers[layer_idx])
                         if layer_idx < len(use_rope_layers) else True)

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.q_size = num_heads * head_dim
        self.kv_size = num_kv_heads * head_dim
        self.qkv_proj = QKVParallelLinear(
            hidden, head_dim, num_heads, num_kv_heads,
            bias=bool(config.get("attention_bias", False)),
            prefix=maybe_prefix(prefix, "qkv_proj"))
        self.o_proj = RowParallelLinear(
            num_heads * head_dim, hidden, bias=False,
            prefix=maybe_prefix(prefix, "o_proj"))
        self.q_norm = GemmaRMSNorm(head_dim, eps=float(_cfg(config, "rms_norm_eps", 1e-6)))
        self.k_norm = GemmaRMSNorm(head_dim, eps=float(_cfg(config, "rms_norm_eps", 1e-6)))
        self.rotary_emb = None
        if self.use_rope:
            self.rotary_emb = RotaryEmbedding(
                head_size=head_dim,
                rotary_dim=int(head_dim * partial_factor),
                max_position_embeddings=int(config["max_position_embeddings"]),
                base=float(rope_theta))

        self.use_head_wise_attn_gate = bool(config.get("use_head_wise_attn_gate", False))
        if self.use_head_wise_attn_gate:
            self.g_proj = ColumnParallelLinear(
                hidden, num_heads, bias=False, return_bias=False,
                prefix=maybe_prefix(prefix, "g_proj"))
        else:
            self.g_proj = None

        self.attn = Attention(
            num_heads=num_heads, head_size=head_dim, scale=head_dim ** -0.5,
            num_kv_heads=num_kv_heads, layer_name=maybe_prefix(prefix, "attn"))
        self.attn.per_layer_sliding_window = self.sliding_window

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = self.q_norm(q.view(-1, self.num_heads, self.head_dim)).view(q.shape)
        k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim)).view(k.shape)
        if self.rotary_emb is not None:
            q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        if self.g_proj is not None:
            gate = torch.sigmoid(self.g_proj(hidden_states))
            attn_output = (attn_output.view(-1, self.num_heads, self.head_dim)
                           * gate.unsqueeze(-1)).view(attn_output.shape)
        output, _ = self.o_proj(attn_output)
        return output


class Step3p5DecoderLayer(nn.Module):
    """一层 Step3p5 decoder；按 `moe_layers_enum` 选 MLP 或 MoE。"""

    def __init__(self, config: dict, *, prefix: str = "") -> None:
        super().__init__()
        layer_idx = _layer_index(prefix)
        self.layer_idx = layer_idx
        self.self_attn = Step3p5Attention(
            config, prefix=maybe_prefix(prefix, "self_attn"))
        moe_layers = config.get("moe_layers_enum")
        if moe_layers is None:
            moe_indices = set(range(1, int(config["num_hidden_layers"])))
        else:
            moe_indices = {int(x) for x in str(moe_layers).replace(" ", "").split(",")
                           if x != ""}
        self.use_moe = bool(config.get("use_moe", False)) and layer_idx in moe_indices
        if self.use_moe:
            self.mlp = Step3p5MoE(config, prefix=maybe_prefix(prefix, "moe"),
                                  layer_idx=layer_idx)
            self.moe = self.mlp
        else:
            self.mlp = Step3p5MLP(
                config, prefix=maybe_prefix(prefix, "mlp"),
                limit=_swiglu_limit(config.get("swiglu_limits_shared"), layer_idx))
            self.moe = None
        eps = float(_cfg(config, "rms_norm_eps", 1e-6))
        self.input_layernorm = GemmaRMSNorm(int(config["hidden_size"]), eps=eps)
        self.post_attention_layernorm = GemmaRMSNorm(int(config["hidden_size"]), eps=eps)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states) + residual
        return hidden_states


class Step3p5Model(nn.Module):
    hf_to_vllm_mapper = WeightsMapper(orig_to_new_stacked={
        ".q_proj": (".qkv_proj", "q"),
        ".k_proj": (".qkv_proj", "k"),
        ".v_proj": (".qkv_proj", "v"),
        ".gate_proj": (".gate_up_proj", 0),
        ".up_proj": (".gate_up_proj", 1),
    })

    def __init__(self, config: dict, prefix: str = "") -> None:
        super().__init__()
        self.config = config
        self.vocab_size = int(config["vocab_size"])
        self.hidden_size = int(config["hidden_size"])
        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size, self.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"))
        self.layers = nn.ModuleList([
            Step3p5DecoderLayer(config, prefix=maybe_prefix(prefix, f"layers.{index}"))
            for index in range(int(config["num_hidden_layers"]))
        ])
        self.norm = GemmaRMSNorm(
            self.hidden_size, eps=float(_cfg(config, "rms_norm_eps", 1e-6)))

    def forward(self, input_ids, positions):
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(positions, hidden_states)
        return self.norm(hidden_states)

    def load_weights(self, weights):
        return AutoWeightsLoader(self).load_weights(
            skip_spec_layer_weights(self.config, weights),
            mapper=self.hf_to_vllm_mapper)


@register_model("Step3p5ForCausalLM")
class Step3p5ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, config: dict, prefix: str = "") -> None:
        super().__init__()
        self.config = config
        self.model = Step3p5Model(config, prefix=maybe_prefix(prefix, "model"))
        self.lm_head = ParallelLMHead(
            int(config["vocab_size"]), int(config["hidden_size"]),
            prefix=maybe_prefix(prefix, "lm_head"))

    def embed_input_ids(self, input_ids):
        return self.model.embed_tokens(input_ids)

    def forward(self, input_ids, positions):
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states):
        return self.lm_head(hidden_states)

    def load_weights(self, weights):
        return AutoWeightsLoader(self).load_weights(
            skip_spec_layer_weights(self.config, weights),
            mapper=Step3p5Model.hf_to_vllm_mapper)


class Step3p5SharedHead(nn.Module):
    def __init__(self, config: dict, prefix: str = "") -> None:
        super().__init__()
        self.norm = GemmaRMSNorm(
            int(config["hidden_size"]), eps=float(_cfg(config, "rms_norm_eps", 1e-6)))
        self.head = ParallelLMHead(
            int(config["vocab_size"]), int(config["hidden_size"]),
            prefix=maybe_prefix(prefix, "head"))

    def forward(self, hidden_states):
        return self.head(self.norm(hidden_states))


class Step3p5MTPLayer(nn.Module):
    def __init__(self, config: dict, prefix: str = "") -> None:
        super().__init__()
        hidden = int(config["hidden_size"])
        self.enorm = GemmaRMSNorm(hidden, eps=float(_cfg(config, "rms_norm_eps", 1e-6)))
        self.hnorm = GemmaRMSNorm(hidden, eps=float(_cfg(config, "rms_norm_eps", 1e-6)))
        self.eh_proj = ColumnParallelLinear(
            hidden * 2, hidden, bias=False, return_bias=False,
            prefix=maybe_prefix(prefix, "eh_proj"))
        self.shared_head = Step3p5SharedHead(
            config, prefix=maybe_prefix(prefix, "shared_head"))
        self.mtp_block = Step3p5DecoderLayer(
            config, prefix=maybe_prefix(prefix, "mtp_block"))

    def forward(self, input_ids, positions, previous_hidden_states,
                inputs_embeds=None, spec_step_index: int = 0):
        if inputs_embeds is None:
            raise ValueError("Step3p5 MTP 需要 inputs_embeds（由 predictor 的 embed_tokens 提供）")
        inputs_embeds = self.enorm(inputs_embeds)
        previous_hidden_states = self.hnorm(previous_hidden_states)
        hidden_states = self.eh_proj(
            torch.cat([inputs_embeds, previous_hidden_states], dim=-1))
        return self.mtp_block(positions, hidden_states)


class Step3p5MultiTokenPredictor(nn.Module):
    def __init__(self, config: dict, prefix: str = "") -> None:
        super().__init__()
        self.config = config
        self.vocab_size = int(config["vocab_size"])
        self.hidden_size = int(config["hidden_size"])
        self.mtp_start_layer_idx = int(config["num_hidden_layers"])
        self.num_mtp_layers = int(config.get("num_nextn_predict_layers") or 1)
        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size, self.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"))
        self.layers = nn.ModuleDict({
            str(index): Step3p5MTPLayer(
                config, prefix=maybe_prefix(prefix, f"layers.{index}"))
            for index in range(self.mtp_start_layer_idx,
                               self.mtp_start_layer_idx + self.num_mtp_layers)
        })

    def embed_input_ids(self, input_ids):
        return self.embed_tokens(input_ids)

    def forward(self, input_ids, positions, previous_hidden_states,
                inputs_embeds=None, spec_step_idx: int = 0):
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        current = spec_step_idx % self.num_mtp_layers
        layer = self.layers[str(self.mtp_start_layer_idx + current)]
        return layer(input_ids, positions, previous_hidden_states,
                     inputs_embeds, current)

    def compute_logits(self, hidden_states, spec_step_idx: int = 0):
        current = spec_step_idx % self.num_mtp_layers
        layer = self.layers[str(self.mtp_start_layer_idx + current)]
        return layer.shared_head(hidden_states)


@register_model("Step3p5MTP")
class Step3p5MTP(nn.Module):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, config: dict, prefix: str = "") -> None:
        super().__init__()
        self.config = config
        self.model = Step3p5MultiTokenPredictor(
            config, prefix=maybe_prefix(prefix, "model"))

    def embed_input_ids(self, input_ids):
        return self.model.embed_input_ids(input_ids)

    def forward(self, input_ids, positions, hidden_states, spec_step_idx: int = 0):
        return self.model(input_ids, positions, hidden_states,
                          spec_step_idx=spec_step_idx)

    def compute_logits(self, hidden_states, spec_step_idx: int = 0):
        return self.model.compute_logits(hidden_states, spec_step_idx)

    def _spec_weights(self, weights):
        """只保留 MTP 层 + 自己的 embed_tokens；把真实 checkpoint 的 name 改成模块路径。"""
        start = self.model.mtp_start_layer_idx
        end = start + self.model.num_mtp_layers
        for name, weight in weights:
            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
            if spec_layer is None:
                if "embed_tokens" in name and "layers." not in name:
                    yield name, weight
                # target 的 45 层与 lm_head 都不属于 MTP
                continue
            if not (start <= spec_layer < end):
                continue
            # 上游 Step3p5MTP.load_weights：shared_head.output -> shared_head.head，
            # .transformer. 去掉；本仓库的 block 在 mtp_block 下，再补这一层前缀。
            rewritten = name.replace(".transformer.", ".")
            rewritten = rewritten.replace("shared_head.output", "shared_head.head")
            prefix = f"model.layers.{spec_layer}."
            if not rewritten.startswith(prefix):
                raise ValueError(f"MTP 权重名 {name!r} 不以层前缀 {prefix!r} 开头")
            rest = rewritten[len(prefix):]
            if rest.startswith(("self_attn.", "mlp.", "input_layernorm.",
                                "post_attention_layernorm.")):
                rewritten = prefix + "mtp_block." + rest
            yield rewritten, weight

    def load_weights(self, weights):
        return AutoWeightsLoader(self).load_weights(
            self._spec_weights(weights), mapper=Step3p5Model.hf_to_vllm_mapper)


__all__ = ["Step3p5Attention", "Step3p5DecoderLayer", "Step3p5ForCausalLM",
           "Step3p5MLP", "Step3p5MoE", "Step3p5Model", "Step3p5MTP",
           "Step3p5MTPLayer", "Step3p5MultiTokenPredictor", "Step3p5SharedHead"]
