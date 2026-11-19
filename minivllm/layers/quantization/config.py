"""冻结 vLLM 的量化配置解析子集。

这里描述 checkpoint 的格式，不实现参数加载或矩阵计算。具体 kernel 的能力
必须在实现后独立检查，不能仅凭最低 SM 判断层的 shape 是否可运行。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class QuantizationConfig(ABC):
    source: str

    @classmethod
    @abstractmethod
    def from_config(cls, config: dict[str, Any], source: str):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_name(cls) -> str:
        raise NotImplementedError

    @classmethod
    def get_supported_act_dtypes(cls):
        return [torch.float16, torch.bfloat16]

    @classmethod
    @abstractmethod
    def get_min_capability(cls) -> int:
        raise NotImplementedError


def _required(config, *keys):
    for key in keys:
        if key in config:
            return config[key]
    raise ValueError(f"量化配置缺少必填字段: {' / '.join(keys)}")


def _integer(value, name):
    if type(value) is not int:
        raise ValueError(f"量化字段 {name} 必须为整数")
    return value


def _boolean(value, name):
    if type(value) is not bool:
        raise ValueError(f"量化字段 {name} 必须为布尔值")
    return value


def _names(value, name):
    if value is None:
        return ()
    if not isinstance(value, list) or any(not isinstance(x, str) or not x for x in value):
        raise ValueError(f"量化字段 {name} 必须为非空字符串列表")
    return tuple(value)


@dataclass(frozen=True)
class AutoAWQConfig(QuantizationConfig):
    weight_bits: int
    group_size: int
    zero_point: bool
    lm_head_quantized: bool
    modules_to_not_convert: tuple[str, ...]

    @classmethod
    def get_name(cls):
        return "auto_awq"

    @classmethod
    def get_min_capability(cls):
        return 75

    @classmethod
    def from_config(cls, config, source):
        return cls(source, _integer(_required(config, "w_bit", "bits"), "bits"),
                   _integer(_required(config, "q_group_size", "group_size"), "group_size"),
                   _boolean(_required(config, "zero_point"), "zero_point"),
                   _boolean(config.get("lm_head", False), "lm_head"),
                   _names(config.get("modules_to_not_convert"), "modules_to_not_convert"))


@dataclass(frozen=True)
class AutoGPTQConfig(QuantizationConfig):
    weight_bits: int
    group_size: int
    desc_act: bool
    is_sym: bool
    lm_head_quantized: bool
    dynamic: dict
    modules_in_block_to_quantize: tuple[str, ...]

    @classmethod
    def get_name(cls):
        return "auto_gptq"

    @classmethod
    def get_min_capability(cls):
        return 60

    @classmethod
    def from_config(cls, config, source):
        dynamic = config.get("dynamic") or {}
        if not isinstance(dynamic, dict):
            raise ValueError("量化字段 dynamic 必须为字典")
        return cls(source, _integer(_required(config, "bits"), "bits"),
                   _integer(_required(config, "group_size"), "group_size"),
                   _boolean(_required(config, "desc_act"), "desc_act"),
                   _boolean(_required(config, "sym"), "sym"),
                   _boolean(config.get("lm_head", False), "lm_head"), dynamic.copy(),
                   _names(config.get("modules_in_block_to_quantize"), "modules_in_block_to_quantize"))


@dataclass(frozen=True)
class Fp8Config(QuantizationConfig):
    is_checkpoint_fp8_serialized: bool
    activation_scheme: str
    ignored_layers: tuple[str, ...]
    weight_block_size: tuple[int, ...] | None
    store_dtype: str | None

    @classmethod
    def get_name(cls):
        return "fp8"

    @classmethod
    def get_min_capability(cls):
        return 75

    @classmethod
    def get_supported_act_dtypes(cls):
        return [torch.bfloat16, torch.float16]

    @classmethod
    def from_config(cls, config, source):
        method = _required(config, "quant_method")
        if not isinstance(method, str):
            raise ValueError("quant_method 必须为字符串")
        scheme = _required(config, "activation_scheme")
        if scheme not in ("dynamic", "static"):
            raise ValueError(f"不支持的 activation_scheme: {scheme!r}")
        block = config.get("weight_block_size")
        if block is not None and (not isinstance(block, list) or len(block) != 2 or
                                  any(type(v) is not int or v <= 0 for v in block)):
            raise ValueError("weight_block_size 必须为两个正整数")
        ignored = config.get("ignored_layers") or config.get("modules_to_not_convert")
        return cls(source, "fp8" in method.lower(), scheme, _names(ignored, "ignored_layers"),
                   tuple(block) if block is not None else None, config.get("store_dtype"))


_METHODS = {"awq": AutoAWQConfig, "awq_marlin": AutoAWQConfig,
            "auto_awq": AutoAWQConfig, "gptq": AutoGPTQConfig,
            "gptq_marlin": AutoGPTQConfig, "auto_gptq": AutoGPTQConfig,
            "fp8": Fp8Config}


def get_quantization_config(method: str):
    if not isinstance(method, str):
        raise ValueError("量化方法必须为字符串")
    cls = _METHODS.get(method.lower())
    if cls is None:
        raise NotImplementedError(f"尚不支持量化方法 {method!r}")
    return cls


def resolve_quantization_config(user_method: str | None, checkpoint_config: dict | None,
                                external_config: dict | None = None):
    """按 checkpoint 身份解析配置，冲突直接拒绝，绝不更改 checkpoint 方法。"""
    if checkpoint_config is not None and not isinstance(checkpoint_config, dict):
        raise ValueError("checkpoint quantization_config 必须为字典")
    if external_config is not None and not isinstance(external_config, dict):
        raise ValueError("外置量化配置必须为字典")
    if checkpoint_config is not None and external_config is not None and checkpoint_config != external_config:
        raise ValueError("checkpoint 与外置量化配置冲突")
    config = checkpoint_config if checkpoint_config is not None else external_config
    if config is None:
        if user_method is not None:
            raise ValueError("指定量化方法，但 checkpoint 没有量化配置")
        return None
    source = "checkpoint" if checkpoint_config is not None else "external"
    # 上游外置 AWQ/GPTQ 配置可没有 quant_method；此时格式由显式用户方法确定。
    # checkpoint 内声明的方法不允许由默认值猜测或覆盖。
    if source == "external" and "quant_method" not in config and user_method is not None:
        cls = get_quantization_config(user_method)
        if cls is Fp8Config:
            raise ValueError("外置 FP8 配置必须声明 quant_method")
        config = {**config, "quant_method": "awq" if cls is AutoAWQConfig else "gptq"}
    method = _required(config, "quant_method")
    if not isinstance(method, str):
        raise ValueError("quant_method 必须为字符串")
    method = method.lower()
    cls = get_quantization_config(method)
    if user_method is not None:
        if not isinstance(user_method, str) or _METHODS.get(user_method.lower()) is not cls:
            raise ValueError(f"用户量化方法 {user_method!r} 与 {source} 的 {method!r} 冲突")
    parsed = cls.from_config(config, source)
    # 明确区分可解析的上游格式和本专题计划支持的子集。
    if isinstance(parsed, AutoAWQConfig):
        if (parsed.weight_bits, parsed.group_size, parsed.zero_point) != (4, 128, True):
            raise NotImplementedError("本项目 AWQ 仅计划支持 W4、group=128、zero_point=true")
        if parsed.lm_head_quantized:
            raise NotImplementedError("本项目暂不支持量化 lm_head")
    elif isinstance(parsed, AutoGPTQConfig):
        if (parsed.weight_bits, parsed.group_size, parsed.is_sym, parsed.desc_act) != (4, 128, True, False):
            raise NotImplementedError("本项目 GPTQ 仅计划支持 W4、group=128、sym=true、desc_act=false")
        if parsed.lm_head_quantized or parsed.dynamic or parsed.modules_in_block_to_quantize:
            raise NotImplementedError("本项目暂不支持 GPTQ 的 lm_head 或逐层配置")
        checkpoint_format = config.get("checkpoint_format", config.get("format", "gptq"))
        if checkpoint_format != "gptq":
            raise NotImplementedError(f"本项目暂不支持 GPTQ checkpoint 格式 {checkpoint_format!r}")
    else:
        if not parsed.is_checkpoint_fp8_serialized or parsed.activation_scheme != "dynamic":
            raise NotImplementedError("本项目 FP8 仅计划支持预量化权重和 dynamic activation")
        if parsed.weight_block_size is not None or parsed.store_dtype not in (None, "fp8"):
            raise NotImplementedError("本项目暂不支持 block FP8 或其他存储格式")
    return parsed
