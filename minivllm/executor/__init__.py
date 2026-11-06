"""executor 层：把"这一轮的调度结果"交给某一个执行部署。

对应 vLLM `vllm/v1/executor/`。本关只有单进程一种（`uniproc_executor.py`）。
"""

from .uniproc_executor import UniProcExecutor
from .factory import create_executor, get_executor_class

__all__ = ["UniProcExecutor", "create_executor", "get_executor_class"]
