"""模型目录适配层：把不同来源的目录翻译成同一种内部表示。

外部目录之间差的只是「字段怎么叫、参数名怎么排」。翻译完之后交给上层的东西
完全一样：一份内部结构配置 + 一份「内部参数名 -> Tensor」+ 停止规则。

适配只在加载时发生一次，不在每次 step 里做；上层（Engine）不需要知道目录是谁导出的。
"""

import json
import pathlib

import torch

from safetensors.torch import load_file as _load_safetensors

from . import native, qwen3

CONFIG_NAME = "config.json"
GENERATION_CONFIG_NAME = "generation_config.json"
WEIGHTS_NAME = "model.safetensors"
INDEX_NAME = "model.safetensors.index.json"

# model_type -> 适配器。就一个分支，不是注册中心：加格式时加一行
_ADAPTERS = {
    native.MODEL_TYPE: native,
    qwen3.MODEL_TYPE: qwen3,
}


def _read_json(path, required):
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"缺少配置文件 {path}")
        return None
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as ex:
        raise ValueError(f"{path} 不是合法 JSON: {ex}") from ex
    if not isinstance(data, dict):
        raise ValueError(f"{path} 的顶层必须是对象")
    return data


def read_raw_config(model_dir):
    """读外部目录的配置，选出适配器；三者一起返回，调用方不必重复读文件。

    config.json 必须有；generation_config.json 可选（真实模型用它声明停止 token）。
    """
    model_dir = pathlib.Path(model_dir)
    raw = _read_json(model_dir / CONFIG_NAME, required=True)
    generation = _read_json(model_dir / GENERATION_CONFIG_NAME, required=False)

    model_type = raw.get("model_type")
    adapter = _ADAPTERS.get(model_type)
    if adapter is None:
        raise ValueError(f"不支持的 model_type={model_type!r}；本实现支持 {sorted(_ADAPTERS)}，"
                         f"自定义格式的版本号见 {native.MODEL_TYPE}")
    return adapter, raw, generation


def read_raw_weights(model_dir, allowed_dtypes):
    """读外部权重文件并检查实际 dtype；**原样返回**，不做精度转换。

    装进模型时用哪种精度由调用方决定（`t.to(model.dtype)`）：文件是 FP32 而模型跑
    BF16 就在那里舍入一次，文件是 BF16 就直接装入。两种格式的文件同名，读法只有一份；
    允许哪些 dtype 由适配器声明。
    """
    model_dir = pathlib.Path(model_dir)
    if (model_dir / INDEX_NAME).is_file():
        raise ValueError(f"{model_dir} 是分片权重目录（有 {INDEX_NAME}），"
                         f"本实现只支持单个 {WEIGHTS_NAME}")

    weights_path = model_dir / WEIGHTS_NAME
    if not weights_path.is_file():
        raise FileNotFoundError(f"缺少权重文件 {weights_path}")

    weights = _load_safetensors(weights_path)
    # 不看配置里写的字符串，直接读每个 Tensor 的实际 dtype
    for name, tensor in weights.items():
        if tensor.dtype not in allowed_dtypes:
            allowed = " 或 ".join(str(d).replace("torch.", "") for d in allowed_dtypes)
            raise ValueError(f"权重 {name} 的 dtype 是 {tensor.dtype}，本实现只支持 {allowed}")

    return weights


__all__ = ["native", "qwen3", "read_raw_config", "read_raw_weights",
           "CONFIG_NAME", "GENERATION_CONFIG_NAME", "WEIGHTS_NAME", "INDEX_NAME"]
