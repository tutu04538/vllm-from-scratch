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
        if speculative_config.draft_model_config is None:
            raise NotImplementedError(
                "V2 的 mtp 需要**显式**给出派生好的 draft 配置：MTP 的权重在 target 的 "
                "checkpoint 里，draft 配置要用 "
                "`SpeculativeConfig.derive_mtp_draft_config(target_model_config)` 从 target 派生"
                "（上游在配置期做，本仓库的 SpeculativeConfig 拿不到 target —— V1 由 Runner "
                "派生，V2 侧的接线属 80 关）。用法：\n"
                "    draft_hf = {**target_hf, \"n_predict\": n, "
                "\"architectures\": [\"Qwen3MTPModel\"]}\n"
                "    spec = SpeculativeConfig(method=\"mtp\", num_speculative_tokens=K, "
                "draft_model_config=ModelConfig(model=<target 目录>, hf_config=draft_hf))\n"
                "（或用 VLLM_USE_V2_MODEL_RUNNER=0 走 V1）"
            )
        from .mtp.speculator import MTPSpeculator

        return MTPSpeculator(vllm_config, device)
    elif speculative_config.use_eagle():
        from .eagle.speculator import EagleSpeculator

        return EagleSpeculator(vllm_config, device)
    else:
        raise NotImplementedError(
            f"{speculative_config.method} is not supported yet."
        )
