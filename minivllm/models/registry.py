"""模型族注册表（对应 vLLM `model_executor/models/registry.py` 的极小化版本）。

存在的理由：加载器要能"按 config 里的 architectures 选类"，而不该长出一串 if/elif。
真实 vLLM 用 `_ModelRegistry` + `@register_model` 装饰器 + 惰性 import（避免一次导入所有模型）；
本关只有一个模型族，用普通字典 + 显式注册，**不假装已有惰性加载机制**。
"""

_REGISTRY: dict[str, object] = {}


def register_model(*architectures: str):
    def decorator(cls):
        for architecture in architectures:
            _REGISTRY[architecture] = cls
        return cls
    return decorator


def get_model_class(architecture: str):
    if architecture not in _REGISTRY:
        raise ValueError(f"没有注册的模型结构 {architecture!r}；已注册：{sorted(_REGISTRY)}")
    return _REGISTRY[architecture]


from .medusa import Medusa  # noqa: E402  （注册发生在 import 时）
from .qwen3 import Qwen3ForCausalLM  # noqa: E402
from .qwen3_dflash import DFlashForCausalLM  # noqa: E402
from .qwen3_dflash2 import DFlash2Qwen3ForCausalLM  # noqa: E402
from .qwen3_dspark import Qwen3DSparkForCausalLM  # noqa: E402

register_model("Qwen3ForCausalLM")(Qwen3ForCausalLM)
# 66 关：Medusa head 的注册名（上游 `model_executor/models/registry.py:678`
# `"MedusaModel": ("medusa", "Medusa")` 逐字相同）。上游还注册了
# `"MLPSpeculatorPreTrainedModel"`，但那行是**注释掉的**（`# Temporarily disabled.`）——
# 本仓库同样不注册，见 `tests/step66/test_mlp_support_boundary.py` 的版本缺口说明。
register_model("MedusaModel")(Medusa)
# 76 关：DFlash 草稿（上游注册名 `DFlashQwen3ForCausalLM`；`DFlashLagunaForCausalLM`
# 是另一种基座，本仓库没有权重也没验证过 → 不注册，加载到它就是"没有注册的模型结构"）
register_model("DFlashQwen3ForCausalLM")(DFlashForCausalLM)
# 77 关：DSpark 草稿。注册名用 checkpoint 的 `architectures` 字符串（上游
# `registry.py:638` 就是 `"Qwen3DSparkModel": ("qwen3_dspark", "Qwen3DSparkForCausalLM")`）；
# 上游另有 `K3DSparkModel` / `Gemma4DSparkModel` / `DSparkDraftModel` 三族，本仓库没有权重也
# 没验证过 → 不注册（加载到它们是"没有注册的模型结构"；配置期另有更早的明确报错）
register_model("Qwen3DSparkModel")(Qwen3DSparkForCausalLM)
# 78 关：DFlash2 草稿。注册名逐字照上游 `registry.py:628`
# `"DFlash2DraftModel": ("qwen3_dflash2", "DFlash2Qwen3ForCausalLM")`；**不是**类名。
# 这个字符串同时是 `init_speculator()` 选 `DFlash2Speculator` 的唯一依据（上游同一条判据）：
# 方法仍是 `dflash`，没有 `method="dflash2"`。
register_model("DFlash2DraftModel")(DFlash2Qwen3ForCausalLM)
