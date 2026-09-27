"""外部 Qwen3 目录的适配：把别人的配置字段和参数名翻译成内部那一套。

不改外部文件，也不重新初始化参数；只做名字、结构和精度的对应。
"""

import torch

from ..model import normalize_eos_ids

MODEL_TYPE = "qwen3"
# 文件可以是 FP32 或 BF16；进来之后由 read_raw_weights() 统一转成 FP32
WEIGHT_DTYPES = (torch.float32, torch.bfloat16)

# 顶层参数：外部名 -> 内部名。这些线性层都是 PyTorch 的 [out, in] 布局，不需要转置
_TOP_LEVEL = {
    "model.embed_tokens.weight": "token_embedding.weight",
    "model.norm.weight": "norm.weight",
    "lm_head.weight": "lm_head.weight",
}

# 每层参数：外部名的后缀 -> 内部名的后缀（前面补 layers.<i>.）
_PER_LAYER = {
    "input_layernorm.weight": "norm1.weight",
    "post_attention_layernorm.weight": "norm2.weight",
    "self_attn.q_proj.weight": "q_proj.weight",
    "self_attn.k_proj.weight": "k_proj.weight",
    "self_attn.v_proj.weight": "v_proj.weight",
    "self_attn.o_proj.weight": "o_proj.weight",
    "self_attn.q_norm.weight": "q_norm.weight",
    "self_attn.k_norm.weight": "k_norm.weight",
    "mlp.gate_proj.weight": "gate_proj.weight",
    "mlp.up_proj.weight": "up_proj.weight",
    "mlp.down_proj.weight": "down_proj.weight",
}

# 外部配置字段名 -> 内部配置字段名。head_dim 不在表里：它必须取目录里的值，
# 不能像内部自定义格式那样在缺省时按 d_model // num_q_heads 推导。
_FIELD_NAMES = {
    "hidden_size": "d_model",
    "num_hidden_layers": "num_layers",
    "num_attention_heads": "num_q_heads",
    "num_key_value_heads": "num_kv_heads",
    "max_position_embeddings": "max_seq_len",
    "intermediate_size": "intermediate_size",
    "rms_norm_eps": "rms_norm_eps",
    "vocab_size": "vocab_size",
    "head_dim": "head_dim",
}
_INT_FIELDS = ("vocab_size", "d_model", "max_seq_len", "num_q_heads", "num_kv_heads",
               "num_layers", "intermediate_size", "head_dim")

EMBEDDING_WEIGHT = "token_embedding.weight"
LM_HEAD_WEIGHT = "lm_head.weight"


def _reject_unsupported(raw):
    # 本关只支持一种外部配置：dense qwen3、无 bias、完整因果 attention、silu、普通 RoPE。
    # 超出范围的一律明确拒绝，不假装支持。
    for name in ("num_experts", "num_local_experts"):
        if raw.get(name):
            raise ValueError(f"不支持 MoE 配置（{name}={raw[name]}），本实现只支持 dense qwen3")

    # 精度声明可能在 dtype 或 torch_dtype；两个名字都认，但仍然只接受 FP32 / BF16
    declared = raw.get("dtype", raw.get("torch_dtype"))
    if declared not in (None, "float32", "bfloat16"):
        raise ValueError(f"不支持的 dtype={declared!r}，本实现只支持 float32 或 bfloat16")
    if raw.get("attention_bias"):
        raise ValueError("不支持 attention_bias=True，本实现的 Q/K/V/O 都不带 bias")
    if raw.get("use_sliding_window") or raw.get("sliding_window") is not None:
        raise ValueError(f"不支持 sliding window（sliding_window={raw.get('sliding_window')!r}、"
                         f"use_sliding_window={raw.get('use_sliding_window')!r}），"
                         f"本实现只有完整因果 attention")
    # layer_types 声明了逐层类型；不能只看滑动开关就放过
    layer_types = raw.get("layer_types")
    if layer_types is not None:
        unusual = sorted({t for t in layer_types if t != "full_attention"})
        if unusual:
            raise ValueError(f"不支持这些层类型 {unusual}，本实现只支持 full_attention")
    if raw.get("hidden_act") not in (None, "silu"):
        raise ValueError(f"不支持的 hidden_act={raw['hidden_act']!r}，本实现的 MLP 固定为 silu")


def _rope_theta(raw):
    # 两种普通 RoPE 写法都认：rope_parameters 里的 theta，或者顶层 rope_theta。
    # 任何 scaling 都会改变旋转方式，一律拒绝。
    if raw.get("rope_scaling") is not None:
        raise ValueError(f"不支持的 rope_scaling={raw['rope_scaling']!r}，本实现只有普通 RoPE")

    rope = raw.get("rope_parameters")
    if isinstance(rope, dict):
        if rope.get("rope_type") != "default":
            raise ValueError(f"不支持的 rope_type={rope.get('rope_type')!r}，本实现只有普通 RoPE")
        theta, where = rope.get("rope_theta"), "rope_parameters.rope_theta"
    else:
        theta, where = raw.get("rope_theta"), "rope_theta"

    if not isinstance(theta, (int, float)) or isinstance(theta, bool) or theta <= 0:
        raise ValueError(f"外部配置的 {where} 必须是正数，收到 {theta!r}")
    return float(theta)


def _eos_ids(raw, generation, vocab_size):
    # 停止规则优先级：generation_config.json 声明 > config.json 声明。
    # 两个文件都没写就不猜——真实模型的普通 token 4 不是停止符。
    if isinstance(generation, dict) and generation.get("eos_token_id") is not None:
        value, where = generation["eos_token_id"], "generation_config.json"
    elif raw.get("eos_token_id") is not None:
        value, where = raw["eos_token_id"], "config.json"
    else:
        raise ValueError("外部目录的 generation_config.json 与 config.json 都没有声明 "
                         "eos_token_id，本实现不猜停止规则")
    return list(normalize_eos_ids(value, vocab_size, where))


def to_internal_config(raw, generation=None):
    _reject_unsupported(raw)

    config = {}
    for external, internal in _FIELD_NAMES.items():
        if external not in raw:
            raise ValueError(f"外部配置缺少字段 {external}")
        config[internal] = raw[external]
    for name in _INT_FIELDS:
        value = config[name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"外部配置字段 {name} 必须是整数，收到 {value!r}")

    config["rope_theta"] = _rope_theta(raw)
    if not isinstance(config["rms_norm_eps"], (int, float)) or isinstance(config["rms_norm_eps"], bool):
        raise ValueError(f"外部配置字段 rms_norm_eps 必须是数值，收到 {config['rms_norm_eps']!r}")

    # Qwen3 没有这个字段：它的结构里 Q/K 投影之后一定有归一化。
    # 不能因为配置里没写就当关闭，那会静默换成另一套公式。
    config["use_qk_norm"] = True
    config["eos_token_ids"] = _eos_ids(raw, generation, config["vocab_size"])
    return config


def _to_internal_name(name):
    internal = _TOP_LEVEL.get(name)
    if internal is not None:
        return internal

    prefix = "model.layers."
    if not name.startswith(prefix):
        return None
    parts = name[len(prefix):].split(".", 1)        # ["0", "self_attn.q_proj.weight"]
    if len(parts) != 2 or not parts[0].isdigit():
        return None
    suffix = _PER_LAYER.get(parts[1])
    return None if suffix is None else f"layers.{parts[0]}.{suffix}"


def _reconcile_tied_weights(weights, raw):
    # tie_word_embeddings=true：embedding 与 lm_head 是同一套数值。
    # 目录里可能只存一个名字，也可能两个都存（本地这份 1.5 GB 权重就是两个都有且相等）。
    if not raw.get("tie_word_embeddings"):
        return weights        # 两个独立参数，缺哪个由 strict=True 报错，不擅自复制

    has_embedding = EMBEDDING_WEIGHT in weights
    has_head = LM_HEAD_WEIGHT in weights
    if has_embedding and has_head:
        if not torch.equal(weights[EMBEDDING_WEIGHT], weights[LM_HEAD_WEIGHT]):
            raise ValueError("tie_word_embeddings=true，但目录里 embedding 与 lm_head 的数值不一致；"
                             "不擅自覆盖其中一个，请确认导出是否正确")
    elif has_embedding:
        weights[LM_HEAD_WEIGHT] = weights[EMBEDDING_WEIGHT]
    elif has_head:
        weights[EMBEDDING_WEIGHT] = weights[LM_HEAD_WEIGHT]
    else:
        raise ValueError("tie_word_embeddings=true，但权重里 embedding 与 lm_head 一个都没有")

    # 本关只要求推理等价：两份参数数值相同即可，先不做物理存储去重
    return weights


def to_internal_weights(weights, raw_config=None):
    # 逐个改名；不认识的参数名不丢弃，直接报错。
    # 少装、shape 不符会由 load_state_dict(strict=True) 拦下。
    renamed = {}
    unknown = []
    for name, tensor in weights.items():
        internal = _to_internal_name(name)
        if internal is None:
            unknown.append(name)
        else:
            renamed[internal] = tensor

    if unknown:
        raise ValueError(f"权重里有本实现不认识的参数名: {sorted(unknown)}")
    return _reconcile_tied_weights(renamed, raw_config or {})
