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


def _check_shard_name(shard, index_path):
    """分片名必须是**目录内的文件名**：不许绝对路径、不许带子目录、不许跳出目录。

    索引文件来自外部目录，路径拼接前必须挡住 `../../etc/passwd` 这类写法。
    """
    if not isinstance(shard, str) or not shard.endswith(".safetensors"):
        raise ValueError(f"{index_path} 里的分片名必须是 .safetensors 文件名，收到 {shard!r}")
    if shard != pathlib.PurePosixPath(shard).name or shard in (".", ".."):
        raise ValueError(f"{index_path} 里的分片名必须是目录内的文件名（不许绝对路径、"
                         f"不许带子目录），收到 {shard!r}")


def _read_sharded_weights(model_dir, index_path):
    """按 `model.safetensors.index.json` 的 `weight_map` 读分片，合并成一份权重字典。

    第五十五关补上：1.7B 这类真实模型是按 `weight_map`（参数名 -> 分片文件名）分片的。
    检查四件事，缺一不可：

    - 分片文件存在（缺文件明确报错，不去猜别的名字）；
    - 每个分片**含有**索引为它声明的全部参数（缺参数报错）；
    - 分片里的参数确实声明属于这个分片（多出来、放错分片、重复存放都报错）；
    - 分片路径合法（见 `_check_shard_name`）。

    每个唯一分片只 `load_file` 一次；本关先合并成一份字典（峰值内存 = 全部权重），
    不做流式低峰值加载。
    """
    index = _read_json(index_path, required=True)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"{index_path} 里缺少非空的 weight_map")

    shards = {}
    for name, shard in weight_map.items():
        _check_shard_name(shard, index_path)
        shards.setdefault(shard, []).append(name)

    weights = {}
    for shard in sorted(shards):
        path = model_dir / shard
        if not path.is_file():
            raise FileNotFoundError(f"索引 {index_path} 声明的分片不存在：{path}")
        part = _load_safetensors(path)
        missing = [name for name in shards[shard] if name not in part]
        if missing:
            raise ValueError(f"分片 {shard} 缺少索引为它声明的参数：{sorted(missing)}")
        for name in part:
            if name not in weight_map:
                raise ValueError(f"分片 {shard} 里的参数 {name} 没有出现在索引的 weight_map 里")
            if weight_map[name] != shard:
                raise ValueError(f"参数 {name} 在索引里声明属于 {weight_map[name]}，"
                                 f"却出现在分片 {shard} 里（重复或放错分片）")
        weights.update(part)
    return weights


def read_raw_weights(model_dir, allowed_dtypes):
    """读外部权重文件并检查实际 dtype；**原样返回**，不做精度转换。

    装进模型时用哪种精度由调用方决定（`t.to(model.dtype)`）：文件是 FP32 而模型跑
    BF16 就在那里舍入一次，文件是 BF16 就直接装入。两种格式的文件同名，读法只有一份；
    允许哪些 dtype 由适配器声明。

    单个 `model.safetensors` 与分片目录（有 `model.safetensors.index.json`）都支持。
    """
    model_dir = pathlib.Path(model_dir)
    index_path = model_dir / INDEX_NAME
    if index_path.is_file():
        weights = _read_sharded_weights(model_dir, index_path)
    else:
        weights_path = model_dir / WEIGHTS_NAME
        if not weights_path.is_file():
            raise FileNotFoundError(f"缺少权重文件 {weights_path}"
                                    f"（也没有分片索引 {INDEX_NAME}）")
        weights = _load_safetensors(weights_path)

    # 不看配置里写的字符串，直接读每个 Tensor 的实际 dtype
    for name, tensor in weights.items():
        if tensor.dtype not in allowed_dtypes:
            allowed = " 或 ".join(str(d).replace("torch.", "") for d in allowed_dtypes)
            raise ValueError(f"权重 {name} 的 dtype 是 {tensor.dtype}，本实现只支持 {allowed}")

    return weights


__all__ = ["native", "qwen3", "read_raw_config", "read_raw_weights",
           "CONFIG_NAME", "GENERATION_CONFIG_NAME", "WEIGHTS_NAME", "INDEX_NAME"]
