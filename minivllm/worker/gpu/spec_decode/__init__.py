"""V2 提议者的工厂（对应 vLLM `v1/worker/gpu/spec_decode/__init__.py::init_speculator`）。

上游这里是一条**逐方法分派**的长链（dflash / dspark / gemma4 / multi-module MTP / mtp /
EAGLE 系），最后 `else: raise NotImplementedError(f"{method} is not supported yet.")`。

本仓库 73 关只实现 EAGLE 系（`use_eagle()`：eagle / eagle3 / mtp）里的 **EAGLE3 与 EAGLE-1**
分支，其余分支保持"明确拒绝"——**不把普通 draft/ngram 等自动转接到别的算法上**
（那会让配置说一套、跑的是另一套）。每个未实现分支都写明归属关卡。
"""

import torch

from ....config import VllmConfig


def init_speculator(vllm_config: VllmConfig, device: torch.device):
    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    if speculative_config.method == "dflash":
        raise NotImplementedError(
            "V2 的 DFlash speculator（上下文 KV + 并行 query）属 76 关；"
            "本关 V2 只支持 EAGLE/EAGLE3（可用 VLLM_USE_V2_MODEL_RUNNER=0 走 V1）"
        )
    elif speculative_config.method == "dspark":
        raise NotImplementedError(
            "V2 的 DSpark speculator（条件采样 + 实际 proposal 分布）属 77 关"
        )
    elif speculative_config.use_multi_module_mtp():
        raise NotImplementedError(
            "V2 的多模块 MTP speculator 属 80 关"
        )
    elif speculative_config.method == "mtp":
        raise NotImplementedError(
            "V2 的原生 MTP speculator（MTPSpeculator，权重在 target checkpoint 里）"
            "本关未接：本机没有带 MTP 层的 checkpoint 可验收，属 80 关"
            "（65 关的 MTP 走 V1 路径，仍然可用）"
        )
    elif speculative_config.use_eagle():
        from .eagle.speculator import EagleSpeculator

        return EagleSpeculator(vllm_config, device)
    else:
        raise NotImplementedError(
            f"{speculative_config.method} is not supported yet."
        )
