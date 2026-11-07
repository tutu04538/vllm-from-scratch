"""按配置选择单进程或多进程执行器。"""

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
        if vllm_config.parallel_config.world_size == 1 and ids is not None and device.index is None and ids != (torch.cuda.current_device(),):
            raise ValueError("device_ids 与当前 CUDA 设备不一致")
    backend = vllm_config.parallel_config.resolved_backend
    if backend == "uni":
        return UniProcExecutor
    if backend == "mp":
        from .multiproc_executor import MultiprocExecutor
        return MultiprocExecutor
    raise NotImplementedError(f"executor 后端 {backend!r} 尚未开放")


def create_executor(vllm_config):
    return get_executor_class(vllm_config)(vllm_config)
