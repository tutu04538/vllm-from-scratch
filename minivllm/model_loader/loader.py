"""加载器入口：`get_model(model_config, device)`（对应 vLLM `model_executor/model_loader/loader.py`）。

Worker/Engine 只调这一个函数，不关心权重是 safetensors 还是别的格式——**分工**是这一层存在的
理由：换格式（bin / GGUF / 量化 / HF hub）改的是这里选出来的 loader，不是 Runner。

选法按目录内容决定（vLLM 按 `load_format` 配置决定，本关先用文件本身判断，少一个配置项）：

    model.safetensors / model.safetensors.index.json  → DefaultModelLoader
    *.bin / *.pt / *.gguf                             → NotImplementedError（明确报错）
    都没有                                             → FileNotFoundError

"明确报错"这条很重要：静默退回空模型会让引擎跑起来但输出胡话，比直接失败难查得多。
"""

import os

from .default_loader import DefaultModelLoader


def get_model_loader(model_config):
    """按模型目录的布局选出加载器。"""
    model_dir = model_config.model
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(
            f"模型目录不存在：{model_dir!r}。本关只支持本地目录（不做 HF hub 下载）。")
    if (os.path.isfile(os.path.join(model_dir, "model.safetensors"))
            or os.path.isfile(os.path.join(model_dir, "model.safetensors.index.json"))):
        return DefaultModelLoader(model_dir)

    unsupported = sorted(name for name in os.listdir(model_dir)
                         if name.endswith((".bin", ".pt", ".pth", ".gguf")))
    if unsupported:
        raise NotImplementedError(
            f"{model_dir} 里只有本关未实现的权重格式 {unsupported}；"
            f"本关只读 safetensors（单文件或带 index 的分片）。")
    raise FileNotFoundError(
        f"{model_dir} 里找不到 safetensors 权重：既没有 model.safetensors，"
        f"也没有 model.safetensors.index.json（目录内容：{sorted(os.listdir(model_dir))}）")


def get_model(model_config, device):
    """构造并加载模型。**这是 Runner.load_model 唯一该调的函数。**"""
    return get_model_loader(model_config).load_model(model_config, device)
