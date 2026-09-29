"""engine 层：`LLMEngine`（用户侧）→ `EngineCoreClient`/`InprocClient`（传输）→ `EngineCore`。

对应 vLLM `vllm/v1/engine/`。vLLM 把这个包的 `__init__.py` 当协议入口用
（`from vllm.v1.engine import EngineCoreRequest, EngineCoreOutput, FinishReason`），
这里照做：协议数据统一从 `step57.outputs` 定义、从这个包重导出，调用方的 import 路径就与
vLLM 同形。
"""

from ..outputs import EngineCoreOutput, EngineCoreOutputs, EngineCoreRequest, FinishReason
from .core import EngineCore
from .core_client import EngineCoreClient, InprocClient
from .llm_engine import LLMEngine
from .output_processor import OutputProcessor, OutputProcessorResult

__all__ = [
    "EngineCore", "EngineCoreClient", "InprocClient", "LLMEngine",
    "OutputProcessor", "OutputProcessorResult",
    "EngineCoreRequest", "EngineCoreOutput", "EngineCoreOutputs", "FinishReason",
]
