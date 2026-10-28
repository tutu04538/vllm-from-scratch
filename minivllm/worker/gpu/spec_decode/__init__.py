"""V2 提议者的工厂（对应 vLLM `v1/worker/gpu/spec_decode/__init__.py::init_speculator`）。

上游这里是一条**逐方法分派**的长链（dflash / dspark / gemma4 / multi-module MTP / mtp /
EAGLE 系），最后 `else: raise NotImplementedError(f"{method} is not supported yet.")`。

本仓库 73 关实现 EAGLE 系（`use_eagle()`：eagle / eagle3 / mtp）里的 **EAGLE3 与 EAGLE-1**、
76 关加 **DFlash**、77 关加 **DSpark**、80B 加 **多模块 MTP**（`use_multi_module_mtp()`）、
80D 加 **Gemma4 Q-only 共享映射**（映射本身在 `gemma4/kv_sharing.py`；这里只明确拒绝，
模型结构/真实接入已由 80E 完成）；其余分支保持"明确拒绝"——**不把普通 draft/ngram 等自动
转接到别的算法上**（那会让配置说一套、跑的是另一套）。每个未实现分支都写明归属关卡。
"""

import torch

from ....config import VllmConfig


def init_speculator(vllm_config: VllmConfig, device: torch.device):
    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    if speculative_config.method == "dflash":
        # 78 关：方法仍是 dflash，靠 **checkpoint 的 architectures** 分出 DFlash2
        # （上游同一条判据：`"DFlash2DraftModel" in draft_model_config.architectures`）。
        # 没有 `method="dflash2"`——目录名/方法名都不是分派依据。
        if speculative_config.uses_dflash2():
            from .dflash2.speculator import DFlash2Speculator

            return DFlash2Speculator(vllm_config, device)
        # 76 关：上下文 KV + 并行 query 的独立提议者（不继承自回归那条路）
        from .dflash.speculator import DFlashSpeculator

        return DFlashSpeculator(vllm_config, device)
    elif speculative_config.method == "dspark":
        # 77 关：DSpark = DFlash 的并行主干 + 串行 Markov 采样（q 必须是"实际条件分布"）
        from .dspark.speculator import DSparkSpeculator

        return DSparkSpeculator(vllm_config, device)
    elif speculative_config.use_gemma4_mtp():
        # 80E.E2：assistant（Q-only、跨宽度投影）+ 常量位置 + 多 KV group 元数据。
        from .gemma4.speculator import Gemma4Speculator

        return Gemma4Speculator(vllm_config, device)
    elif speculative_config.use_multi_module_mtp():
        # 80B/80C：单卡、纯文本、支持整段 K 步循环 CUDA Graph 的
        # MultiModuleMTPSpeculator。Gemma4 已在上面明确拒绝；Step3p5、稀疏索引
        # 分别属 80F/80G，不在这个分支里偷偷替换。
        from .multi_module_mtp.speculator import MultiModuleMTPSpeculator

        return MultiModuleMTPSpeculator(vllm_config, device)
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
