"""`DefaultModelLoader`：从**本地目录**读 safetensors 的加载器（对应 vLLM
`model_executor/model_loader/default_loader.py`）。

它只回答一件事：**权重从哪来**。模型类怎么选、名字怎么路由、参数怎么填，都在基类和模型那边
（见 `base_loader.py` 的说明）。对应 vLLM 里就是 `get_all_weights()` 这一个方法：

    DefaultModelLoader.get_all_weights(model)
      → 读 config 里的模型目录
      → weight_utils.iter_weights()  逐 shard 流式产出 (名字, 张量)
      → 按需 rename（本关没有：Qwen3 的检查点名与模型名只差打包那一层，
        路由交给 WeightsMapper，不在这里改）

**与 vLLM 的差异**：vLLM 还会处理 `model_config.model_impl`、draft 模型、量化、`download_dir`、
mmproj 等；本关只有"本地目录 + safetensors"，并把"不支持"明确报出来。
"""

from .base_loader import BaseModelLoader
from .weight_utils import iter_weights


class DefaultModelLoader(BaseModelLoader):
    def __init__(self, model_dir: str) -> None:
        self.model_dir = model_dir

    def get_all_weights(self, model):
        # 迭代器原样交出去：模型自己的 load_weights 会边取边写，不需要（也不该）在这里
        # 攒成一个 dict。`model` 参数是给"权重来源依赖模型结构"的加载器（如量化）留的接口。
        return iter_weights(self.model_dir)
