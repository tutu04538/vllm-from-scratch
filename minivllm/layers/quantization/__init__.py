"""量化配置入口；当前只解析配置，量化执行尚未开放。"""

from .config import get_quantization_config, resolve_quantization_config

__all__ = ["get_quantization_config", "resolve_quantization_config"]
