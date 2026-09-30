"""模型加载（对应 vLLM `vllm/model_executor/model_loader/` 的子集）。

四层，一层一个问题（198 §8）：

    loader.py              选哪种加载器（按目录布局）——"权重从哪来"
    base_loader.py         选模型类、构造、灌权重——与来源无关的通用流程
    default_loader.py      本地 safetensors 目录
    weight_utils.py        文件 → (名字, 张量) 的流式读取与检查点校验
    auto_weights_loader.py 名字路由（打包映射）+ 递归委派 + 覆盖检查

调用方（Runner）只用 `get_model(model_config, device)`。
"""

from .auto_weights_loader import AutoWeightsLoader, WeightsMapper
from .base_loader import BaseModelLoader
from .default_loader import DefaultModelLoader
from .loader import get_model, get_model_loader
from .weight_utils import iter_weights

__all__ = ["get_model", "get_model_loader", "BaseModelLoader", "DefaultModelLoader",
           "AutoWeightsLoader", "WeightsMapper", "iter_weights"]
