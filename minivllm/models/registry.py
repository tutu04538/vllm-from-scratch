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

register_model("Qwen3ForCausalLM")(Qwen3ForCausalLM)
# 66 关：Medusa head 的注册名（上游 `model_executor/models/registry.py:678`
# `"MedusaModel": ("medusa", "Medusa")` 逐字相同）。上游还注册了
# `"MLPSpeculatorPreTrainedModel"`，但那行是**注释掉的**（`# Temporarily disabled.`）——
# 本仓库同样不注册，见 `tests/step66/test_mlp_support_boundary.py` 的版本缺口说明。
register_model("MedusaModel")(Medusa)
