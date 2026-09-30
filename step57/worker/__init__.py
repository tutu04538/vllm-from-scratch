"""worker 层：一个执行端的运行环境（设备准备、模型装载、转发执行/采样）。

对应 vLLM `vllm/v1/worker/`。四个文件各回答一个问题：

    worker.py           运行环境：设备、模型装载、把执行/采样转发下去
    gpu_model_runner.py 一轮怎么跑：更新镜像 → 准备输入 → 模型 → 采样 → 记账
    gpu_input_batch.py  批状态缓冲：请求 ↔ batch 行的账本（定长缓冲，原地写）
    block_table.py      块表镜像：逻辑块号 → 物理块号，以及 slot_mapping
"""

from .block_table import BlockTable
from .gpu_input_batch import InputBatch
from .gpu_model_runner import CachedRequestState, ExecuteModelState, GPUModelRunner, PreparedInputs
from .worker import Worker

__all__ = ["Worker", "GPUModelRunner", "CachedRequestState", "ExecuteModelState",
           "PreparedInputs", "InputBatch", "BlockTable"]
