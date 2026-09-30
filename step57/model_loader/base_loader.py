"""`BaseModelLoader`：**选模型类 → 构造 → 灌权重**这三步的通用部分（对应 vLLM
`model_executor/model_loader/base_loader.py`）。

198 §8 特意点了一句："初始化模型的通用逻辑在 model_loader 包/基类附近，不要误说成全部都在
default_loader.py"。这里是那句话的落点：

    BaseModelLoader.load_model()      通用流程，与权重从哪来无关
      ├─ _get_model_class()           查 registry："architectures" → 类（vLLM 是 ModelRegistry）
      └─ load_weights()               调模型自己的 load_weights（名字路由在模型那边）
    DefaultModelLoader                **只**负责"权重从本地 safetensors 目录来"

换成"从 HF hub 下""从 torch.save 的 bin 读""从量化检查点读"，改的都是 `get_all_weights()`，
流程不变——这就是分层的价值。
"""

import torch
from torch import nn

# 配置里的 dtype 字符串 → torch dtype。放在这里而不是模型里：**精度是加载决策**，
# 模型只按收到的 dtype 建参数（所以"同一份权重要跑 FP32 还是 BF16"不改模型代码）。
_DTYPES = {"float32": torch.float32, "fp32": torch.float32,
           "float16": torch.float16, "half": torch.float16,
           "bfloat16": torch.bfloat16, "bf16": torch.bfloat16}


def _dtype_of(model_config):
    name = str(model_config.dtype)
    if name not in _DTYPES:
        raise ValueError(f"不支持的 dtype {name!r}；本关支持 {sorted(set(_DTYPES))}")
    return _DTYPES[name]


class BaseModelLoader:
    """模型加载器的公共骨架。子类只需要回答"权重从哪来"。"""

    def get_all_weights(self, model: nn.Module):
        """产出 `(名字, 张量)`。子类实现。"""
        raise NotImplementedError

    def _get_model_class(self, model_config):
        """`config.json` 的 `architectures` → 模型类。查表，不写 if/elif。"""
        from ..models.registry import get_model_class

        architectures = (model_config.hf_config or {}).get("architectures") or []
        if len(architectures) != 1:
            raise ValueError(
                f"本关只支持 config 里写明唯一 architectures 的模型，收到 {architectures!r}；"
                f"多结构（如投机草稿模型共存）属于 57E 的加载路径")
        return get_model_class(architectures[0])

    def _initialize_model(self, model_config, device) -> nn.Module:
        """构造空模型（权重还是随机/未初始化的 `torch.empty`）。"""
        model_class = self._get_model_class(model_config)
        model = model_class(model_config.hf_config)
        return model.to(device=device, dtype=_dtype_of(model_config))

    def load_model(self, model_config, device) -> nn.Module:
        model = self._initialize_model(model_config, device)
        loaded = model.load_weights(self.get_all_weights(model))
        if loaded is None:
            raise ValueError(f"{type(model).__name__}.load_weights() 必须返回已加载参数名集合"
                             f"（本关靠它做覆盖检查，不能静默返回 None）")
        return model
