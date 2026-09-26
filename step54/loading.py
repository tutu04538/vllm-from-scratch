"""模型装配与目录加载：外部目录 -> 内部配置与权重 -> 模型。

这一层只认内部那套字段名，格式差异全在 `formats/` 里消化掉。它不认识 Engine，
也不碰运行时（KV 池、调度器）——那几样由 `engine.Engine._init_runtime()` 装。

**engine.py 里按裸名字重导出这些函数**（`from .loading import ...`），
`Engine.from_model_dir()` 调用时写的也是裸名字：这样
`step54.engine.load_model_config = 假的` 这种**打桩**才对调用点生效
（Python 每次调用都去 engine 模块的全局里查一次名字）。
"""

import torch

from .formats import native, read_raw_config, read_raw_weights
from .model import TinyCausalLM


def build_model_from_config(config, device, attention_backend, max_num_batched_tokens, use_cuda_graph,
                            dtype=torch.float32, norm_backend="torch", rope_backend="torch"):
    # 按内部配置构造模型；维度合法性由 TinyCausalLM 的校验负责（缺字段、非法维度都会明确报错）
    return TinyCausalLM(
        vocab_size=config["vocab_size"], d_model=config["d_model"], max_seq_len=config["max_seq_len"],
        num_q_heads=config["num_q_heads"], num_kv_heads=config["num_kv_heads"],
        num_layers=config["num_layers"], intermediate_size=config["intermediate_size"],
        rms_norm_eps=config["rms_norm_eps"], rope_theta=config["rope_theta"],
        head_dim=config["head_dim"], use_qk_norm=config["use_qk_norm"],
        eos_token_ids=config["eos_token_ids"], dtype=dtype, norm_backend=norm_backend,
        rope_backend=rope_backend,
        device=device, attention_backend=attention_backend,
        max_num_query_tokens=max_num_batched_tokens, use_cuda_graph=use_cuda_graph)


def load_model_config(model_dir):
    # 读目录里的外部配置，翻译成内部字段；支持哪些来源由 formats/ 按 model_type 分派
    adapter, raw, generation = read_raw_config(model_dir)
    return adapter.to_internal_config(raw, generation)


def _load_weights_into(adapter, model_dir, raw, model):
    # 适配器把外部参数名翻成内部参数名，再严格装入已经建在目标设备上的模型。
    # 装入前显式转成模型的运行精度：FP32 文件进 BF16 模型就在这里舍入一次，
    # BF16 文件进 FP32 模型是精确扩宽（不恢复文件里本来就没有的信息）。
    # strict=True：缺参数、多参数、shape 不符都会抛，不会留下混着随机参数的模型
    weights = read_raw_weights(model_dir, adapter.WEIGHT_DTYPES)
    mapped = adapter.to_internal_weights(weights, raw)
    model.load_state_dict({name: t.to(model.dtype) for name, t in mapped.items()}, strict=True)
    return model


def load_model_weights(model_dir, model):
    adapter, raw, _ = read_raw_config(model_dir)
    return _load_weights_into(adapter, model_dir, raw, model)


# 自定义格式的公开常量，保持与之前一致（engine 也会重导出这几个）
FORMAT_VERSION = native.FORMAT_VERSION
COMPATIBLE_FORMAT_VERSIONS = native.COMPATIBLE_FORMAT_VERSIONS
MODEL_TYPE = native.MODEL_TYPE
MODEL_DTYPE = native.MODEL_DTYPE
