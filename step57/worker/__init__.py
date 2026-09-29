"""worker 层：一个执行端的运行环境（设备准备、模型装载、转发执行/采样）。

对应 vLLM `vllm/v1/worker/`。真实模型、KV 分配、CUDA Graph 属于 57B/57D。
"""

from .worker import Worker

__all__ = ["Worker"]
