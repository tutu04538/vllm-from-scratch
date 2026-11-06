"""执行器选择：未实现的多进程后端必须在创建 worker 前失败。"""

from .uniproc_executor import UniProcExecutor


def get_executor_class(vllm_config):
    vllm_config._validate_parallel_config()
    if vllm_config.device_config.device.startswith("cuda"):
        import torch
        count = torch.cuda.device_count()
        vllm_config.parallel_config.validate_devices(count)
        device = torch.device(vllm_config.device_config.device)
        if device.index is not None and device.index >= count:
            raise ValueError(f"CUDA 设备编号 {device.index} 超出可见 GPU 数 {count}")
        ids = vllm_config.parallel_config.device_ids
        if ids is not None and device.index is None and ids != (torch.cuda.current_device(),):
            raise ValueError("device_ids 与当前 CUDA 设备不一致")
    backend = vllm_config.parallel_config.resolved_backend
    if backend == "uni":
        return UniProcExecutor
    raise NotImplementedError(f"executor 后端 {backend!r} 尚未开放")


def create_executor(vllm_config):
    return get_executor_class(vllm_config)(vllm_config)
