"""模型目录适配层：把不同来源的目录翻译成同一种内部表示。

外部目录之间差的只是「字段怎么叫、参数名怎么排」。翻译完之后交给上层的东西
完全一样：一份内部结构配置 + 一份「内部参数名 -> Tensor」。

适配只在加载时发生一次，不在每次 step 里做；上层（Engine）不需要知道目录是谁导出的。
"""

import json
import pathlib

import torch

from safetensors.torch import load_file as _load_safetensors

from . import native, qwen3

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "model.safetensors"
INDEX_NAME = "model.safetensors.index.json"

# model_type -> 适配器。就一个分支，不是注册中心：加格式时加一行
_ADAPTERS = {
    native.MODEL_TYPE: native,
    qwen3.MODEL_TYPE: qwen3,
}


def read_raw_config(model_dir):
    """读外部 config.json 并选出适配器；两者一起返回，调用方不必重复读文件"""
    model_dir = pathlib.Path(model_dir)
    config_path = model_dir / CONFIG_NAME
    if not config_path.is_file():
        raise FileNotFoundError(f"缺少配置文件 {config_path}")

    try:
        raw = json.loads(config_path.read_text())
    except json.JSONDecodeError as ex:
        raise ValueError(f"{config_path} 不是合法 JSON: {ex}") from ex
    if not isinstance(raw, dict):
        raise ValueError(f"{config_path} 的顶层必须是对象")

    model_type = raw.get("model_type")
    adapter = _ADAPTERS.get(model_type)
    if adapter is None:
        raise ValueError(f"不支持的 model_type={model_type!r}；本实现支持 {sorted(_ADAPTERS)}，"
                         f"自定义格式的版本号见 {native.MODEL_TYPE}")
    return adapter, raw


def read_raw_weights(model_dir):
    """读外部权重文件；两种格式的文件同名，dtype 在这一层统一把关"""
    model_dir = pathlib.Path(model_dir)
    if (model_dir / INDEX_NAME).is_file():
        raise ValueError(f"{model_dir} 是分片权重目录（有 {INDEX_NAME}），"
                         f"本实现只支持单个 {WEIGHTS_NAME}")

    weights_path = model_dir / WEIGHTS_NAME
    if not weights_path.is_file():
        raise FileNotFoundError(f"缺少权重文件 {weights_path}")

    weights = _load_safetensors(weights_path)
    # load_state_dict 会静默把 dtype 转成目标参数的类型，所以在改名之前自己确认
    for name, tensor in weights.items():
        if tensor.dtype != torch.float32:
            raise ValueError(f"权重 {name} 的 dtype 是 {tensor.dtype}，本实现只支持 float32")
    return weights


__all__ = ["native", "qwen3", "read_raw_config", "read_raw_weights",
           "CONFIG_NAME", "WEIGHTS_NAME", "INDEX_NAME"]
