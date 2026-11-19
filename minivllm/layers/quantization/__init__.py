"""量化配置入口；当前只解析配置，量化执行尚未开放。"""

from .config import resolve_quantization_config

__all__ = ["resolve_quantization_config"]
