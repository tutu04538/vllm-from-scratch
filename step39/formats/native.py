"""自定义模型目录格式（`save_model()` 写出来的那种）。

字段名、参数名与内部模型一致，所以这里只需要「校验 + 原样透传」。
"""

import json
import pathlib

import torch

from safetensors.torch import save_file as _save_safetensors

from ..model import DEFAULT_EOS_TOKEN_IDS, normalize_eos_ids

MODEL_TYPE = "tiny_rope_decoder"
MODEL_DTYPE = "float32"
# v1/v2：没有 eos_token_ids，那时的停止规则就是写死的 4
# v3：写明 eos_token_ids，停止规则随模型一起保存
# v4：QKV 与 gate/up 合并成 qkv_proj / gate_up_proj，参数名不再与 HF 一一对应
#     旧版本目录里的 q_proj/k_proj/v_proj 与 gate_proj/up_proj 仍能装进来，
#     合成发生在 DecoderLayer._load_from_state_dict（这里只负责认版本号）
FORMAT_VERSION = 4
COMPATIBLE_FORMAT_VERSIONS = (1, 2, 3, 4)
# 自己写出来的目录永远是 FP32
WEIGHT_DTYPES = (torch.float32,)

# JSON 里必须出现的结构字段；运行选项（device、后端、并发数……）不属于模型结构
_STRUCT_FIELDS = ("vocab_size", "d_model", "max_seq_len", "num_q_heads", "num_kv_heads",
                  "num_layers", "intermediate_size", "rms_norm_eps", "rope_theta")
_INT_FIELDS = ("vocab_size", "d_model", "max_seq_len", "num_q_heads", "num_kv_heads",
               "num_layers", "intermediate_size")
_FLOAT_FIELDS = ("rms_norm_eps", "rope_theta")


def to_internal_config(raw, generation=None):
    # 读取并检查配置；任何一项不对就报错，不返回半份配置
    version = raw.get("format_version")
    if version not in COMPATIBLE_FORMAT_VERSIONS:
        raise ValueError(f"不支持的 format_version={version!r}，"
                         f"本实现支持 {list(COMPATIBLE_FORMAT_VERSIONS)}")
    config = dict(raw)
    if version <= 1:
        # 旧目录没有这两项，按旧公式处理；不静默忽略「已写明」的新配置
        config.setdefault("head_dim", None)
        config.setdefault("use_qk_norm", False)
    else:
        missing = [name for name in ("head_dim", "use_qk_norm") if name not in config]
        if missing:
            raise ValueError(f"format_version={version} 的配置缺少字段: {missing}")

    if config["head_dim"] is not None:
        if isinstance(config["head_dim"], bool) or not isinstance(config["head_dim"], int):
            raise ValueError(f"head_dim 必须是整数或 null，收到 {config['head_dim']!r}")
    if not isinstance(config["use_qk_norm"], bool):
        raise ValueError(f"use_qk_norm 必须是布尔值，收到 {config['use_qk_norm']!r}")
    if config.get("model_type") != MODEL_TYPE:
        raise ValueError(f"不支持的 model_type={config.get('model_type')!r}，本实现只支持 {MODEL_TYPE!r}")
    if config.get("dtype") != MODEL_DTYPE:
        raise ValueError(f"不支持的 dtype={config.get('dtype')!r}，本实现只支持 {MODEL_DTYPE!r}")

    missing = [name for name in _STRUCT_FIELDS if name not in config]
    if missing:
        raise ValueError(f"配置缺少必需字段: {missing}")
    for name in _INT_FIELDS:
        if isinstance(config[name], bool) or not isinstance(config[name], int):
            raise ValueError(f"配置字段 {name} 必须是整数，收到 {config[name]!r}")
    for name in _FLOAT_FIELDS:
        if isinstance(config[name], bool) or not isinstance(config[name], (int, float)):
            raise ValueError(f"配置字段 {name} 必须是数值，收到 {config[name]!r}")

    # 停止规则：v3 起必须写明；v1/v2 是更早写出来的目录，那时的行为就是固定的 {4}
    if version >= 3:
        if "eos_token_ids" not in config:
            raise ValueError(f"format_version={version} 的配置缺少字段: ['eos_token_ids']")
        config["eos_token_ids"] = normalize_eos_ids(config["eos_token_ids"], config["vocab_size"], "配置")
    else:
        config["eos_token_ids"] = list(DEFAULT_EOS_TOKEN_IDS)
    return config


def to_internal_weights(weights, raw_config=None):
    # 文件里的参数名就是内部参数名，不需要改名；
    # 多出来的、少掉的、shape 不符的都由 load_state_dict(strict=True) 拦下
    return weights


def save_model(model, model_dir):
    # 把模型配置和全部参数写进目录；不改动原模型（权重、device、已捕获的图都还能用）
    model_dir = pathlib.Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    config = {"format_version": FORMAT_VERSION, "model_type": MODEL_TYPE, "dtype": MODEL_DTYPE}
    config.update(model.model_config())          # 取实际模型结构，不写死数值
    # 停止规则不是模型结构，和 format_version/dtype 一样属于「重建这个部署」要带上的信息
    config["eos_token_ids"] = list(model.eos_token_ids)
    (model_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    # safetensors 要求稠密连续张量；保存到 CPU float32，与模型当前所在设备无关
    # tied 权重共享底层存储时每个名字各存一份，保存不做去重
    weights = {name: tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
               for name, tensor in model.state_dict().items()}
    _save_safetensors(weights, model_dir / "model.safetensors")
    return model_dir
