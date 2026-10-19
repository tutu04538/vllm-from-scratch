"""DSpark 的 V2 提议者（对应 vLLM `v1/worker/gpu/spec_decode/dspark/`，需求 077）。"""

from .speculator import DSparkSpeculator
from .utils import load_dspark_model

__all__ = ["DSparkSpeculator", "load_dspark_model"]
