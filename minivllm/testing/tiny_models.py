"""**只给测试用**的 tiny Qwen3 模型目录生成器（不再往仓库里放 safetensors）。

原来的 `fixtures/step30_qwen3/{tiny_gqa,tiny_mqa}` 是提交进仓库的随机初始化权重（约 200 KB），
由外部工具用 transformers `save_pretrained` 导出。现在改为**现场生成**：

- 配置与形状与原来**完全一致**（对照脚本依赖 `vocab_size=11/13`、GQA 4/2 与 MQA 4/1、
  `head_dim`、`rms_norm_eps` 这些规格，不能变）；
- 权重按固定 `seed` 生成 → 同一 seed 每次一致，但**与原来那两份权重无关**；
  所以任何"记住某个 token id / 某条 logits"的断言都不能依赖它，只能做自洽对照
  （与 HF / 真实 vLLM 读同一份目录比数值）；
- 生成到临时目录，进程内缓存：多个用例共用一份，不重复写盘。

    from minivllm.testing.tiny_models import tiny_qwen3_dir
    model_dir = tiny_qwen3_dir("tiny_gqa")

要按**旧仓库布局**重建（给按 `fixtures/step30_qwen3/...` 路径写的老脚本/验收探针用）：

    python -m minivllm.testing.tiny_models --out fixtures/step30_qwen3

生成器不属于生产路径：`minivllm` 的引擎代码不 import 它，只依赖 torch + safetensors。
"""

import json
import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file

# 与旧 fixtures 的 config.json 逐字段一致（含 transformers 写下的默认字段）
_TINY_GQA = {
    "architectures": ["Qwen3ForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": None,
    "dtype": "float32",
    "eos_token_id": 4,
    "head_dim": 16,
    "hidden_act": "silu",
    "hidden_size": 32,
    "initializer_range": 0.02,
    "intermediate_size": 48,
    "layer_types": ["full_attention"] * 2,
    "max_position_embeddings": 64,
    "max_window_layers": 28,
    "model_type": "qwen3",
    "num_attention_heads": 4,
    "num_hidden_layers": 2,
    "num_key_value_heads": 2,
    "pad_token_id": None,
    "rms_norm_eps": 1e-06,
    "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
    "sliding_window": None,
    "tie_word_embeddings": False,
    "transformers_version": "5.14.1",
    "use_cache": True,
    "use_sliding_window": False,
    "vocab_size": 11,
}

_TINY_MQA = {
    "architectures": ["Qwen3ForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": None,
    "dtype": "float32",
    "eos_token_id": 4,
    "head_dim": 6,
    "hidden_act": "silu",
    "hidden_size": 30,
    "initializer_range": 0.02,
    "intermediate_size": 41,
    "layer_types": ["full_attention"] * 3,
    "max_position_embeddings": 64,
    "max_window_layers": 28,
    "model_type": "qwen3",
    "num_attention_heads": 4,
    "num_hidden_layers": 3,
    "num_key_value_heads": 1,
    "pad_token_id": None,
    "rms_norm_eps": 0.003,
    "rope_parameters": {"rope_theta": 500000.0, "rope_type": "default"},
    "sliding_window": None,
    "tie_word_embeddings": False,
    "transformers_version": "5.14.1",
    "use_cache": True,
    "use_sliding_window": False,
    "vocab_size": 13,
}

_CONFIGS = {"tiny_gqa": _TINY_GQA, "tiny_mqa": _TINY_MQA}
_CACHE: dict[tuple[str, int], str] = {}


def tiny_qwen3_config(name: str) -> dict:
    """取模型配置（深拷贝：调用方改它不会污染模板）。"""
    if name not in _CONFIGS:
        raise ValueError(f"没有名为 {name!r} 的 tiny 模型；可选：{sorted(_CONFIGS)}")
    return json.loads(json.dumps(_CONFIGS[name]))


def write_tiny_qwen3(name: str, out_dir, seed: int = 0) -> Path:
    """把 tiny 模型写到 `out_dir`（config.json + generation_config.json + model.safetensors）。

    权重是**随机但确定性**的：同一 `seed` 每次字节一致，与旧 fixtures 无关。
    目录里**没有 tokenizer**——本项目的引擎只读 config 与 safetensors。
    """
    config = tiny_qwen3_config(name)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (out / "generation_config.json").write_text(json.dumps(
        {"bos_token_id": config["bos_token_id"], "eos_token_id": config["eos_token_id"],
         "pad_token_id": config["pad_token_id"]}, indent=2) + "\n")
    save_file(_weights(config, seed), str(out / "model.safetensors"))
    return out


def tiny_qwen3_dir(name: str, seed: int = 0) -> str:
    """生成到临时目录并返回路径（**进程内缓存**：同名同 seed 只写一次）。

    返回 `str` 而不是 `Path`：调用方多数在拼 `f"{dir}/config.json"` 这类字符串。
    """
    key = (name, seed)
    cached = _CACHE.get(key)
    if cached is None:
        cached = tempfile.mkdtemp(prefix=f"minivllm_{name}_")
        write_tiny_qwen3(name, cached, seed)
        _CACHE[key] = cached
    return cached


def tiny_eagle3_dir(target_name: str = "tiny_gqa", *, num_aux_layers: int = 2,
                    aux_layers=(0, 1), seed: int = 0) -> str:
    """生成一份**与 tiny target 规格匹配**的 EAGLE3 draft（随机但确定；只给测试）。

    与 `tiny_qwen3_dir` 的区别（这些就是 63 关要适配的东西）：
      - 只有 **1 层**（`midlayer.*`），forward 语义是 EAGLE3 那套（见 `models/qwen3_eagle3.py`）；
      - `fc.weight` 把 `num_aux_layers` 个 target 辅助层特征投到 hidden（输入宽度 = hidden × 层数）；
      - 检查点里**没有 embed_tokens**（与 target 共享）、`lm_head` 是 draft 自己那份（本 tiny 用同词表）；
      - `eagle_aux_hidden_state_layer_ids` 显式给出（tiny target 只有 2 层，上游默认值
        `(2, n//2, n-3)` 对 2 层不成立，所以这里必须显式写）。
    """
    target_config = tiny_qwen3_config(target_name)
    hidden = target_config["hidden_size"]
    head_dim = target_config["head_dim"]
    num_heads = target_config["num_attention_heads"]
    num_kv_heads = target_config["num_key_value_heads"]
    vocab = target_config["vocab_size"]
    config = {
        "architectures": ["Eagle3Qwen3ForCausalLM"],
        "model_type": "qwen3",                      # Qwen3 风格 → 带 q/k norm
        "hidden_size": hidden,
        "num_hidden_layers": 1,
        "num_attention_heads": num_heads,
        "num_key_value_heads": num_kv_heads,
        "head_dim": head_dim,
        "intermediate_size": target_config["intermediate_size"],
        "vocab_size": vocab,                        # 同词表：异构词表的采样空间语义属 67 关
        "max_position_embeddings": target_config["max_position_embeddings"],
        "rms_norm_eps": target_config["rms_norm_eps"],
        "rope_theta": target_config.get("rope_theta", 10000.0),
        "target_hidden_size": hidden,
        "num_aux_layers": num_aux_layers,
        "eagle_aux_hidden_state_layer_ids": list(aux_layers),
        "tie_word_embeddings": False,               # draft 的 lm_head 是自己的（同词表）
        "torch_dtype": target_config.get("torch_dtype", "float32"),
    }
    generator = torch.Generator().manual_seed(seed)
    scale = 0.02
    def rand(*shape):
        return (torch.randn(*shape, generator=generator) * scale).to(torch.float32)
    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim
    weights = {
        "fc.weight": rand(hidden, hidden * num_aux_layers),
        "norm.weight": torch.ones(hidden, dtype=torch.float32),
        "lm_head.weight": rand(vocab, hidden),
        "midlayer.hidden_norm.weight": torch.ones(hidden, dtype=torch.float32),
        "midlayer.input_layernorm.weight": torch.ones(hidden, dtype=torch.float32),
        "midlayer.post_attention_layernorm.weight": torch.ones(hidden, dtype=torch.float32),
        # 第一层的 qkv 输入是 2*hidden（embeds + 特征拼接）
        "midlayer.self_attn.q_proj.weight": rand(q_size, 2 * hidden),
        "midlayer.self_attn.k_proj.weight": rand(kv_size, 2 * hidden),
        "midlayer.self_attn.v_proj.weight": rand(kv_size, 2 * hidden),
        "midlayer.self_attn.o_proj.weight": rand(hidden, q_size),
        "midlayer.mlp.gate_proj.weight": rand(config["intermediate_size"], hidden),
        "midlayer.mlp.up_proj.weight": rand(config["intermediate_size"], hidden),
        "midlayer.mlp.down_proj.weight": rand(hidden, config["intermediate_size"]),
        "midlayer.self_attn.q_norm.weight": torch.ones(head_dim, dtype=torch.float32),
        "midlayer.self_attn.k_norm.weight": torch.ones(head_dim, dtype=torch.float32),
    }
    out = Path(tempfile.mkdtemp(prefix=f"minivllm_eagle3_{target_name}_"))
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    save_file(weights, str(out / "model.safetensors"))
    return str(out)


def _weights(config: dict, seed: int) -> dict:
    """按 HF 的初始化方式生成权重：线性/嵌入 ~ N(0, initializer_range)，RMSNorm 全 1。

    `initializer_range` 与形状全部从 config 推导——改配置就不会出现形状对不上。
    """
    hidden = config["hidden_size"]
    inter = config["intermediate_size"]
    heads = config["num_attention_heads"]
    kv_heads = config["num_key_value_heads"]
    head_dim = config["head_dim"]
    vocab = config["vocab_size"]
    num_layers = config["num_hidden_layers"]
    generator = torch.Generator().manual_seed(seed)

    def normal(*shape):
        return torch.empty(*shape, dtype=torch.float32).normal_(
            0.0, config["initializer_range"], generator=generator)

    def ones(*shape):
        return torch.ones(*shape, dtype=torch.float32)

    weights = {
        "model.embed_tokens.weight": normal(vocab, hidden),
        "lm_head.weight": normal(vocab, hidden),
        "model.norm.weight": ones(hidden),
    }
    for layer in range(num_layers):
        prefix = f"model.layers.{layer}."
        weights[prefix + "input_layernorm.weight"] = ones(hidden)
        weights[prefix + "post_attention_layernorm.weight"] = ones(hidden)
        weights[prefix + "self_attn.q_proj.weight"] = normal(heads * head_dim, hidden)
        weights[prefix + "self_attn.k_proj.weight"] = normal(kv_heads * head_dim, hidden)
        weights[prefix + "self_attn.v_proj.weight"] = normal(kv_heads * head_dim, hidden)
        weights[prefix + "self_attn.o_proj.weight"] = normal(hidden, heads * head_dim)
        weights[prefix + "self_attn.q_norm.weight"] = ones(head_dim)
        weights[prefix + "self_attn.k_norm.weight"] = ones(head_dim)
        weights[prefix + "mlp.gate_proj.weight"] = normal(inter, hidden)
        weights[prefix + "mlp.up_proj.weight"] = normal(inter, hidden)
        weights[prefix + "mlp.down_proj.weight"] = normal(hidden, inter)
    return weights


def _main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="生成 tiny Qwen3 测试模型目录")
    parser.add_argument("--out", required=True,
                        help="输出根目录；会在其下写 tiny_gqa/ 与 tiny_mqa/")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true",
                        help="目录非空时也覆盖（默认跳过，避免盖掉手头的东西）")
    args = parser.parse_args(argv)

    root = Path(args.out)
    for name in _CONFIGS:
        target = root / name
        if target.exists() and any(target.iterdir()) and not args.force:
            print(f"跳过（已存在，非空）：{target}")
            continue
        write_tiny_qwen3(name, target, seed=args.seed)
        print(f"已生成：{target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
