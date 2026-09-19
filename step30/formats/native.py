"""自定义模型目录格式（`save_model()` 写出来的那种）。

字段名、参数名与内部模型一致，所以这里只需要「校验 + 原样透传」。
"""

import json
import pathlib

import torch

from safetensors.torch import save_file as _save_safetensors

MODEL_TYPE = "tiny_rope_decoder"
MODEL_DTYPE = "float32"
FORMAT_VERSION = 2
# v1：没有 head_dim / use_qk_norm，按旧公式（推导 head_dim、不做 Q/K norm）
COMPATIBLE_FORMAT_VERSIONS = (1, 2)

# JSON 里必须出现的结构字段；运行选项（device、后端、并发数……）不属于模型结构
_STRUCT_FIELDS = ("vocab_size", "d_model", "max_seq_len", "num_q_heads", "num_kv_heads",
                  "num_layers", "intermediate_size", "rms_norm_eps", "rope_theta")
_INT_FIELDS = ("vocab_size", "d_model", "max_seq_len", "num_q_heads", "num_kv_heads",
               "num_layers", "intermediate_size")
_FLOAT_FIELDS = ("rms_norm_eps", "rope_theta")


def to_internal_config(raw):
    # 读取并检查配置；任何一项不对就报错，不返回半份配置
    version = raw.get("format_version")
    if version not in COMPATIBLE_FORMAT_VERSIONS:
        raise ValueError(f"不支持的 format_version={version!r}，"
                         f"本实现支持 {list(COMPATIBLE_FORMAT_VERSIONS)}")
    config = dict(raw)
    if version == 1:
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
    return config


def to_internal_weights(weights):
    # 文件里的参数名就是内部参数名，不需要改名；
    # 多出来的、少掉的、shape 不符的都由 load_state_dict(strict=True) 拦下
    return weights


def save_model(model, model_dir):
    # 把模型配置和全部参数写进目录；不改动原模型（权重、device、已捕获的图都还能用）
    model_dir = pathlib.Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    config = {"format_version": FORMAT_VERSION, "model_type": MODEL_TYPE, "dtype": MODEL_DTYPE}
    config.update(model.model_config())          # 取实际模型配置，不写死数值
    (model_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    # safetensors 要求稠密连续张量；保存到 CPU float32，与模型当前所在设备无关
    weights = {name: tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
               for name, tensor in model.state_dict().items()}
    _save_safetensors(weights, model_dir / "model.safetensors")
    return model_dir
