"""外部 Qwen3 目录的适配：把别人的配置字段和参数名翻译成内部那一套。

不改外部文件，也不重新初始化参数；只做名字和结构的对应。
"""

MODEL_TYPE = "qwen3"

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


def _reject_unsupported(raw):
    # 本关只支持一种外部配置：dense qwen3、无 bias、完整因果 attention、silu、default RoPE。
    # 超出范围的一律明确拒绝，不假装支持。
    for name in ("num_experts", "num_local_experts"):
        if raw.get(name):
            raise ValueError(f"不支持 MoE 配置（{name}={raw[name]}），本实现只支持 dense qwen3")

    if raw.get("dtype") not in (None, "float32"):
        raise ValueError(f"不支持的 dtype={raw['dtype']!r}，本实现只支持 float32")
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
    if raw.get("tie_word_embeddings"):
        raise ValueError("不支持 tie_word_embeddings=True，本实现的 embedding 与 lm_head 是两份权重")
    # EOS 是 Engine 的固定约定，配置里声明别的值就必须报错，不能默默按 4 处理
    if raw.get("eos_token_id") != 4:
        raise ValueError(f"不支持 eos_token_id={raw.get('eos_token_id')!r}，"
                         f"本实现沿用 Engine 约定 eos_token_id=4")


def _rope_theta(raw):
    rope = raw.get("rope_parameters")
    if not isinstance(rope, dict):
        raise ValueError("外部配置缺少 rope_parameters，本实现只支持该字段形式给出的 RoPE")
    if rope.get("rope_type") != "default":
        raise ValueError(f"不支持的 rope_type={rope.get('rope_type')!r}，本实现只有普通 RoPE")
    theta = rope.get("rope_theta")
    if not isinstance(theta, (int, float)) or isinstance(theta, bool) or theta <= 0:
        raise ValueError(f"rope_parameters.rope_theta 必须是正数，收到 {theta!r}")
    return float(theta)


def to_internal_config(raw):
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


def to_internal_weights(weights):
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
    return renamed
