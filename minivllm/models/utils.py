"""模型侧的小工具（对应 vLLM `model_executor/models/utils.py` 的子集）。

本仓库只放进**真的被生产路径调用**的东西。`maybe_prefix()` 还在 `models/qwen3.py`
（历史位置，63/64/65 的模型都从那里 import），这里放 65 关的 MTP 权重名识别：

    get_spec_layer_idx_from_weight_name()   检查点里的名字属于**哪一个 spec 层**（MTP 层）
    skip_spec_layer_weights()               加载 **target** 时把这些名字整份丢掉
"""


def get_spec_layer_idx_from_weight_name(config: dict, weight_name: str) -> int | None:
    """这个权重名属于哪个 MTP（spec）层？不是 spec 层就返回 None。

    对应上游 `model_executor/models/utils.py:496-515`，规则逐字照抄：

        if not (n := getattr(config, "num_nextn_predict_layers", 0)):
            return None
        base = config.num_hidden_layers
        for i in range(n):
            if weight_name.startswith((f"model.layers.{base + i}.", f"layers.{base + i}.")):
                return base + i
        return None

    **为什么要识别绝对层号**：DeepSeek 一族的 MTP 权重就排在 target 的最后一层后面
    （`num_hidden_layers=61` → spec 层是 `model.layers.61.*`），加载时要"认出它 → 改写成
    draft 模型内部的相对层号"。判定必须用 `startswith("model.layers.{base+i}.")` 这种**带点号**的
    前缀：不带点会同时命中 `model.layers.6` 与 `model.layers.61`（10 倍层号那种错）。

    本仓库的 MTP（Qwen3 稠密家族）用的是 `mtp.*` 命名（上游 `qwen3_next_mtp.py` 那一派），
    这个函数服务的是"绝对层号命名"那一派；两派都走同一个 loader 入口，见
    `models/qwen3_mtp.py::Qwen3MTP.load_weights`。
    """
    num_nextn_predict_layers = int(config.get("num_nextn_predict_layers") or 0)
    if num_nextn_predict_layers == 0:
        return None
    base = int(config["num_hidden_layers"])
    for index in range(num_nextn_predict_layers):
        if weight_name.startswith((f"model.layers.{base + index}.",
                                   f"layers.{base + index}.")):
            return base + index
    return None


def skip_spec_layer_weights(config: dict, weights):
    """丢掉"绝对层号"命名的 spec（MTP）层权重——**加载 target 时用**。

    对应上游 `deepseek_v2.py:1575-1577` 的两行：

        spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
        if spec_layer is not None:
            continue  # skip spec decode layers for main model

    **为什么 target 需要它**：MTP 的权重就排在 target 最后一层后面（与 target 同一个文件），
    而 target 模型只建了 `num_hidden_layers` 层——不丢掉这些名字，加载器会报"没有这个参数"；
    丢掉之后 target 照常加载，MTP 那边再由它自己的 `load_weights()` 认领。
    `mtp.` 前缀那一派由加载器的 `skip_prefixes=["mtp."]` 处理（上游 qwen3_next.py:846）。
    """
    for name, weight in weights:
        if get_spec_layer_idx_from_weight_name(config, name) is not None:
            continue
        yield name, weight
