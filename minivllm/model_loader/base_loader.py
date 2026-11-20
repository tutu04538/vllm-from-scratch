"""`BaseModelLoader`：**选模型类 → 构造 → 灌权重**这三步的通用部分（对应 vLLM
`model_executor/model_loader/base_loader.py`）。

198 §8 特意点了一句："初始化模型的通用逻辑在 model_loader 包/基类附近，不要误说成全部都在
default_loader.py"。这里是那句话的落点：

    BaseModelLoader.load_model()      通用流程，与权重从哪来无关
      ├─ _get_model_class()           查 registry："architectures" → 类（vLLM 是 ModelRegistry）
      └─ load_weights()               调模型自己的 load_weights（名字路由在模型那边）
    DefaultModelLoader                **只**负责"权重从本地 safetensors 目录来"

换成"从 HF hub 下""从 torch.save 的 bin 读""从量化检查点读"，改的都是 `get_all_weights()`，
流程不变——这就是分层的价值。
"""

import torch
from torch import nn

# 配置里的 dtype 字符串 → torch dtype。放在这里而不是模型里：**精度是加载决策**，
# 模型只按收到的 dtype 建参数（所以"同一份权重要跑 FP32 还是 BF16"不改模型代码）。
_DTYPES = {"float32": torch.float32, "fp32": torch.float32,
           "float16": torch.float16, "half": torch.float16,
           "bfloat16": torch.bfloat16, "bf16": torch.bfloat16}


def _dtype_of(model_config):
    name = str(model_config.dtype)
    if name not in _DTYPES:
        raise ValueError(f"不支持的 dtype {name!r}；本关支持 {sorted(set(_DTYPES))}")
    return _DTYPES[name]


class BaseModelLoader:
    """模型加载器的公共骨架。子类只需要回答"权重从哪来"。"""

    def get_all_weights(self, model: nn.Module):
        """产出 `(名字, 张量)`。子类实现。"""
        raise NotImplementedError

    def _get_model_class(self, model_config):
        """`config.json` 的 `architectures` → 模型类。查表，不写 if/elif。"""
        from ..models.registry import get_model_class

        architectures = (model_config.hf_config or {}).get("architectures") or []
        if len(architectures) != 1:
            raise ValueError(
                f"本关只支持 config 里写明唯一 architectures 的模型，收到 {architectures!r}；"
                f"多结构（如投机草稿模型共存）属于 57E 的加载路径")
        return get_model_class(architectures[0])

    def _initialize_model(self, model_config, device, *, representation_only=False) -> nn.Module:
        """构造空模型（权重还是随机/未初始化的 `torch.empty`）。

        大模型（Gemma4 E2B 约 5.1B 参数）不能先在 CPU 建 float32 再 `.to()`：那会临时
        占约 20 GB 主机内存，本机 31 GB 总内存会进入 swap 甚至 OOM。非 CPU 设备上直接
        在目标 device/默认 dtype 下构造，再 `.to()` 只是形状/设备校验。默认 dtype 是
        进程级状态，必须用 try/finally 恢复。
        """
        # 在构造模型之前识别 checkpoint 的量化身份。后续关卡提供真实执行方法；
        # 当前不允许把量化权重误当作 BF16 权重加载。
        from ..layers.quantization import get_quantization_config, resolve_quantization_config
        import json
        from pathlib import Path

        hf_config = model_config.hf_config or {}
        checkpoint_config = hf_config.get("quantization_config")
        if checkpoint_config is None and isinstance(hf_config.get("text_config"), dict):
            checkpoint_config = hf_config["text_config"].get("quantization_config")
        if checkpoint_config is None:
            checkpoint_config = hf_config.get("compression_config")
        user_method = model_config.quantization
        external_config = None
        if checkpoint_config is None and user_method is not None:
            config_class = get_quantization_config(user_method)
            filenames = config_class.get_config_filenames()
            paths = [Path(model_config.model) / name for name in filenames
                     if (Path(model_config.model) / name).is_file()]
            if len(paths) > 1:
                raise ValueError(f"存在多个外置量化配置文件: {paths}")
            if paths:
                with paths[0].open(encoding="utf-8") as stream:
                    external_config = json.load(stream)
        quant_config = resolve_quantization_config(user_method, checkpoint_config, external_config)
        if quant_config is not None:
            architectures = hf_config.get("architectures") or []
            if architectures != ["Qwen3ForCausalLM"]:
                raise NotImplementedError(
                    f"量化执行尚未开放: {quant_config.get_name()}（{quant_config.source}），"
                    f"模型结构 {architectures!r}")
            dtype = _dtype_of(model_config)
            if dtype not in quant_config.get_supported_act_dtypes():
                raise ValueError(f"量化方法 {quant_config.get_name()} 不支持激活 dtype {dtype}")
            target_device = torch.device(device)
            if target_device.type == "cuda":
                capability = torch.cuda.get_device_capability(target_device)
                numeric_capability = capability[0] * 10 + capability[1]
                if numeric_capability < quant_config.get_min_capability():
                    raise NotImplementedError(
                        f"量化方法 {quant_config.get_name()} 需要 SM{quant_config.get_min_capability()}，"
                        f"当前 SM{numeric_capability}")
            from ..layers.quantization.config import AutoAWQConfig
            if not isinstance(quant_config, AutoAWQConfig):
                raise NotImplementedError(f"量化执行尚未开放: {quant_config.get_name()}")
            if not representation_only and (target_device.type != 'cuda' or dtype != torch.bfloat16):
                raise NotImplementedError(
                    f"量化执行尚未开放: {quant_config.get_name()}；当前仅支持 CUDA BF16")
            model_class = self._get_model_class(model_config)
            old_dtype = torch.get_default_dtype()
            torch.set_default_dtype(dtype)
            try:
                with torch.device(target_device):
                    model = model_class(hf_config, quant_config=quant_config)
                if not representation_only:
                    from ..layers.quantization.awq import AutoAWQMarlinLinearMethod
                    for module in model.modules():
                        method = getattr(module, 'quant_method', None)
                        if isinstance(method, AutoAWQMarlinLinearMethod):
                            method.supports_execution = True
                return model
            finally:
                torch.set_default_dtype(old_dtype)
        if representation_only:
            raise ValueError("表示审计入口要求 AWQ checkpoint 配置")
        model_class = self._get_model_class(model_config)
        dtype = _dtype_of(model_config)
        target_device = torch.device(device)
        if target_device.type == "cpu":
            model = model_class(model_config.hf_config)
            return model.to(device=target_device, dtype=dtype)
        old_dtype = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        try:
            with torch.device(target_device):
                model = model_class(model_config.hf_config)
        finally:
            torch.set_default_dtype(old_dtype)
        return model.to(device=target_device, dtype=dtype)

    def load_model(self, model_config, device) -> nn.Module:
        model = self._initialize_model(model_config, device)
        return self._load_initialized_model(model, model_config, device)

    def load_model_for_representation(self, model_config, device) -> nn.Module:
        """只供 AWQ 参数加载与布局审计；返回的模型不支持量化 forward。"""
        model = self._initialize_model(model_config, device, representation_only=True)
        model = self._load_initialized_model(model, model_config, device)
        if model._quant_postload_state != 'representation_ready':
            raise ValueError("AWQ 表示审计没有量化线性层")
        return model

    def _load_initialized_model(self, model, model_config, device) -> nn.Module:
        # 加载期间不允许执行；全部方法后处理成功后才一起开放 forward。
        linear_layers = [module for module in model.modules()
                         if getattr(module, '_method_weights_created', False)]
        for layer in linear_layers:
            layer._method_weights_ready = False
        model._quant_postload_state = 'loading'
        try:
            loaded = model.load_weights(self.get_all_weights(model))
        except BaseException:
            model._quant_postload_state = 'failed'
            raise
        if loaded is None:
            model._quant_postload_state = 'failed'
            raise ValueError(f"{type(model).__name__}.load_weights() 必须返回已加载参数名集合"
                             f"（本关靠它做覆盖检查，不能静默返回 None）")
        from .post_load import process_weights_after_loading
        process_weights_after_loading(model, model_config, device)
        for layer in linear_layers:
            layer._method_weights_ready = True
        # 加载完就切推理模式（对应 vLLM 的 `model.eval()`；204 §6 要求的收尾项）。
        # 与 Runner 的 `torch.inference_mode()` 分工不同：eval 改的是模块自身的**行为**
        # （dropout/BN 等训练态分支），inference_mode 管的是**梯度记录**。两者都不能省。
        model.eval()
        return model
