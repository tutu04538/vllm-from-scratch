"""提议者的公共基类（对应 vLLM `v1/worker/gpu/spec_decode/speculator.py`）。

V2 的提议者（speculator）与 V1 的 proposer 最大的区别**不是算法**，而是接口形态：

    V1 `propose(rows, all_token_ids, input_batch, ...)`   CPU 侧行对象 + 逐请求 Python 数据
    V2 `propose(input_batch, attn_metadata, slot_mappings, hidden_states, ...)`
                                                          批次张量 + 常驻 slot 寻址

V2 的提议者**全程只说"张量 + slot"**：`input_batch.idx_mapping` 是 batch 行→slot，
`last_sampled`/`temperature`/`seeds`/`draft_tokens` 都是 `[max_num_reqs, ...]` 且按 slot 索引，
`num_sampled`/`num_rejected` 由 GPU 采样器算好（无需 CPU 知道被拒了几枚）。这正是
"不等结果回 CPU 就能准备下一步输入"的前提（70 关留下的异步调度缺口）。

本文件只保留本仓库 V2 通路要用的部分；裁剪项（LoRA/多模态/EPLB/DP/local-argmax/
multi-module MTP）逐条记在 `docs/step73_alignment.md` 差异账本。
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from ....config import VllmConfig
from ....utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE
from ..input_batch import InputBuffers


class BaseSpeculator(ABC):
    @abstractmethod
    def load_model(self, target_model: nn.Module) -> None:
        pass

    @abstractmethod
    def init_cudagraph_manager(self, cudagraph_mode) -> None:
        pass

    @abstractmethod
    def capture(self) -> None:
        pass

    @abstractmethod
    def propose(
        self,
        input_batch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        # [num_tokens, hidden_size]
        last_hidden_states: torch.Tensor,
        # num_layers x [num_tokens, hidden_size]
        aux_hidden_states: list[torch.Tensor] | None,
        # [num_reqs]
        num_sampled: torch.Tensor,
        # [num_reqs]
        num_rejected: torch.Tensor,
        # [max_num_reqs]
        last_sampled: torch.Tensor,
        # [num_prefill_lookahead, max_num_reqs]
        next_prefill_tokens: torch.Tensor,
        # [max_num_reqs]
        temperature: torch.Tensor,
        # [max_num_reqs]
        seeds: torch.Tensor,
        dummy_run: bool = False,
        is_profile: bool = False,
    ) -> torch.Tensor:
        pass


class DraftModelSpeculator(BaseSpeculator):
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        self.vllm_config = vllm_config
        self.device = device

        assert vllm_config.speculative_config is not None
        self.speculative_config = vllm_config.speculative_config
        self.method = self.speculative_config.method
        self.num_speculative_steps = self.speculative_config.num_speculative_tokens

        self.scheduler_config = vllm_config.scheduler_config
        self.max_num_reqs = self.scheduler_config.max_num_seqs
        self.max_num_tokens = self.scheduler_config.max_num_batched_tokens
        self.max_model_len = vllm_config.model_config.max_model_len
        self.draft_max_seq_len = self.max_model_len
        # We need to get the hidden size from the draft model config because
        # the draft model's hidden size can be different from the target model's
        # hidden size (e.g., Llama 3.3 70B).
        self.hidden_size = self.speculative_config.draft_model_config.get_hidden_size()
        self.vocab_size = self.speculative_config.draft_model_config.get_vocab_size()
        # 上游这里是 torch.dtype；本仓库的配置里是字符串，过一遍映射表（同上游同名表）。
        self.dtype = STR_DTYPE_TO_TORCH_DTYPE.get(
            str(vllm_config.model_config.dtype), torch.float32
        )
        # 上游读 `model_config.use_fp64_gumbel`（H100/Ada/Blackwell 上 fp64 更慢）；
        # 本仓库没有这个配置项 → 恒 fp32（差异账本第 "gumbel 精度" 条）。
        self.use_fp64_gumbel = False

        self.input_buffers = InputBuffers(
            max_num_reqs=self.max_num_reqs,
            max_num_tokens=self.max_num_tokens,
            device=device,
        )
        self.idx_mapping = torch.zeros(
            self.max_num_reqs, dtype=torch.int32, device=device
        )
        self.temperature = torch.zeros(
            self.max_num_reqs, dtype=torch.float32, device=device
        )
        self.seeds = torch.zeros(self.max_num_reqs, dtype=torch.int64, device=device)
        self.draft_tokens = torch.zeros(
            self.max_num_reqs,
            self.num_speculative_steps,
            dtype=torch.int64,
            device=device,
        )
        self.arange = torch.arange(
            self.max_num_reqs + 1, dtype=torch.int32, device="cpu"
        )

        self.draft_logits: torch.Tensor | None = None
        if self.speculative_config.draft_sample_method == "probabilistic":
            # Pre-temperature logits, cached from the previous decode step.
            dtype, fill = self.draft_logits_spec(vllm_config)
            self.draft_logits = torch.full(
                (
                    self.max_num_reqs,
                    self.num_speculative_steps,
                    self.vocab_size,
                ),
                fill,
                dtype=dtype,
                device=device,
            )

    @abstractmethod
    def load_draft_model(self, target_model: nn.Module) -> nn.Module:
        pass

    def load_model(self, target_model: nn.Module) -> None:
        self.model = self.load_draft_model(target_model)

    def draft_logits_spec(self, vllm_config: VllmConfig) -> tuple[torch.dtype, float]:
        """Dtype and fill for the cached proposal distribution.

        Speculators that write only a subset of columns each step override this.
        """
        return vllm_config.model_config.dtype, 0.0

    @property
    def attn_vllm_config(self) -> VllmConfig:
        """Config for the draft's attention metadata builders."""
        return self.vllm_config

    def set_attn(self, block_tables, target_input_buffers: InputBuffers) -> None:
        """把 draft 的注意力上下文接到 target 的块表与输入缓冲上。

        上游还有 `model_state` / `kv_cache_config` / `attn_groups`（多后端 + 图捕获用）；
        本仓库 V2 只有一个注意力后端且本关不做图，所以只留**真正被用到**的两样：
        target 的块表（草稿与 target 用同一套物理块号）与 target 的输入缓冲
        （draft prefill 直接复用 target 的 attention metadata）。
        """
        self.block_tables = block_tables
        self.target_input_buffers = target_input_buffers

    def _copy_request_inputs(
        self,
        num_reqs: int,
        # [num_reqs]
        idx_mapping: torch.Tensor,
        # [max_num_reqs]
        temperature: torch.Tensor,
        # [max_num_reqs]
        seeds: torch.Tensor,
    ) -> None:
        # Copy temperature, seeds, and idx mapping to the pre-allocated buffers.
        # NOTE(woosuk): For draft sampling, we only consider the temperature
        # and ignore the other sampling parameters such as top_k and top_p,
        # for simplicity and performance.
        # While this may slightly degrade the acceptance rate, it does not
        # affect the output distribution after rejection sampling.
        self.temperature.copy_(temperature)
        self.seeds.copy_(seeds)
        self.idx_mapping[:num_reqs].copy_(idx_mapping)
        # idx_mapping for CG padded requests points to -1, which is ignored
        # during sampling to prevent writing stale values to draft logits.
        self.idx_mapping[num_reqs:].fill_(-1)

    def _greedy_sample_draft(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.model.compute_logits(hidden_states)
        return logits.argmax(dim=-1)

    def sample_draft(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        idx_mapping: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        draft_step: torch.Tensor,
        draft_logits: torch.Tensor | None,
    ) -> torch.Tensor:
        if draft_logits is not None:
            from ..sample.gumbel import gumbel_sample

            logits = self.model.compute_logits(hidden_states)
            # NOTE(woosuk): We must add 1 to the positions to match the Gumbel noise
            # used for draft and target sampling.
            return gumbel_sample(
                logits,
                idx_mapping,
                temperature,
                seeds,
                positions + 1,
                apply_temperature=True,
                logits_cache=draft_logits,
                logits_cache_col=draft_step,
                use_fp64=self.use_fp64_gumbel,
            )
        return self._greedy_sample_draft(hidden_states)
