"""Engine：把模型与运行时装起来，并按一轮一轮地推进。

前向编排只有三步（见 `step()`）：调度器排出本轮计划 → 模型一次 forward → 采样。
采样那一层的实现在 `sample_runtime.py`（`SampleRuntime`），配置校验在 `validation.py`，
模型装配与目录加载在 `loading.py`——这里只做装配与编排。

`load_model_config` 这几个加载入口在这里**重导出**，老 import 路径继续可用
（`from step55.engine import load_model_config`）。
"""

import math

import torch

from .attention import AttentionMetadata
from .cache import KVCachePool
from .draft import DraftModelProposer
from .formats import CONFIG_NAME as MODEL_CONFIG_NAME
from .formats import WEIGHTS_NAME as MODEL_WEIGHTS_NAME
from .loading import (COMPATIBLE_FORMAT_VERSIONS, FORMAT_VERSION, MODEL_DTYPE, MODEL_TYPE,
                      build_model_from_config, load_model_config, load_model_from_dir,
                      load_model_weights)
from .model import TinyCausalLM
from .sample_runtime import SampleRuntime
from .sampling import TorchSampler
from .scheduler import Scheduler
from .validation import (SCHEDULING_POLICIES, SPECULATIVE_MODES, check_draft_model,
                         check_runtime, check_scheduling_policy, check_speculative,
                         resolve_device)


class Engine:

    def __init__(self, max_num_seqs=1, max_num_batched_tokens=4,block_size=4, num_kv_blocks=8, vocab_size=5, d_model=8, max_seq_len=32, on_finished=None, enable_prefix_caching=True, device=None, attention_backend="torch", use_cuda_graph=False, num_q_heads=1, num_kv_heads=1,
                 num_layers=2, intermediate_size=64, rms_norm_eps=1e-6, rope_theta=10000.0,
                 head_dim=None, use_qk_norm=False, eos_token_ids=None, dtype=torch.float32,
                 norm_backend="torch", rope_backend="torch", model=None, *, on_token=None,
                 scheduling_policy="fcfs",
                 speculative_mode=None, num_speculative_tokens=2, prompt_lookup_n=2,
                 draft_model=None, draft_num_kv_blocks=None,
                 draft_max_num_batched_tokens=None):
        # on_token 是 keyword-only：它放在**所有旧参数之后**，旧的位置参数一个都没挪位。
        # 插在中间会让 Engine(..., None, False) 里的 False 从 enable_prefix_caching
        # 变成 on_token —— Python 按位置配对，不会知道调用者的原意。
        # model 给定时用它，不再按上面的维度参数随机初始化（加载路径走这里）
        # 后端与 Graph 开关也以模型上的为准，避免两边不一致
        check_scheduling_policy(scheduling_policy)
        if model is None:
            device = resolve_device(device)
            check_runtime(device, attention_backend, use_cuda_graph, dtype, norm_backend,
                           rope_backend)
            model = TinyCausalLM(vocab_size=vocab_size, d_model=d_model, max_seq_len=max_seq_len,
                                 device=device, attention_backend=attention_backend,
                                 max_num_query_tokens=max_num_batched_tokens,
                                 use_cuda_graph=use_cuda_graph,
                                 num_q_heads=num_q_heads, num_kv_heads=num_kv_heads,
                                 num_layers=num_layers, intermediate_size=intermediate_size,
                                 rms_norm_eps=rms_norm_eps, rope_theta=rope_theta,
                                 head_dim=head_dim, use_qk_norm=use_qk_norm,
                                 eos_token_ids=eos_token_ids, dtype=dtype, norm_backend=norm_backend,
                                 rope_backend=rope_backend)

        self._init_runtime(model, max_num_seqs, max_num_batched_tokens, block_size, num_kv_blocks,
                           on_finished, enable_prefix_caching, on_token,
                           scheduling_policy, speculative_mode, num_speculative_tokens,
                           prompt_lookup_n, draft_model, draft_num_kv_blocks,
                           draft_max_num_batched_tokens)

    def _init_runtime(self, model, max_num_seqs, max_num_batched_tokens, block_size, num_kv_blocks,
                      on_finished, enable_prefix_caching, on_token=None,
                      scheduling_policy="fcfs", speculative_mode=None,
                      num_speculative_tokens=2, prompt_lookup_n=2, draft_model=None,
                      draft_num_kv_blocks=None, draft_max_num_batched_tokens=None):
        # 模型已经就位（随机初始化或从目录加载），这里只装运行时：元数据、KV 池、调度器
        device = model.device
        check_runtime(device, model.attention_backend, model.use_cuda_graph, model.dtype,
                      model.norm_backend, model.rope_backend)
        # 组合校验放在这里：模型已经就位，attention 后端与 Graph 开关都以它为准
        check_speculative(speculative_mode, num_speculative_tokens, prompt_lookup_n,
                          scheduling_policy, enable_prefix_caching,
                          model.attention_backend, model.use_cuda_graph)
        # draft 模型那一侧的校验：两个模型必须能读同一段 token id
        if draft_max_num_batched_tokens is None:
            # 默认与 target 同一个预算；显式给值就是**分开计数**的两个预算
            draft_max_num_batched_tokens = max_num_batched_tokens
        if speculative_mode == "draft_model":
            check_draft_model(model, draft_model, draft_num_kv_blocks,
                              draft_max_num_batched_tokens)
        elif draft_model is not None:
            raise ValueError("给了 draft_model 却没有开 speculative_mode='draft_model'；"
                             "两个模型只有在这条模式下才会一起跑")

        self.model = model
        self.model.eval()
        # 采样只保留 Torch 一条路径；Triton 采样 kernel 与 beam 已移出主线
        self.sampler = TorchSampler()
        self.device = device
        self.attention_backend = model.attention_backend
        self.enable_prefix_caching = enable_prefix_caching
        self.scheduling_policy = scheduling_policy
        self.speculative_mode = speculative_mode

        # 容量按引擎配置一次分配，与首次出现的 batch 大小无关
        if model.attention_backend == "triton":
            model.attention_metadata = AttentionMetadata(
                max_num_seqs=max_num_seqs,
                max_num_query_tokens=max_num_batched_tokens,
                max_blocks_per_request=math.ceil(model.max_seq_len / block_size),
                device=device,
            )

        self.kv_cache_pool = KVCachePool(block_size, num_kv_blocks, model.num_kv_heads,
                                         model.head_dim, device, self.enable_prefix_caching,
                                         num_layers=model.num_layers, dtype=model.dtype)
        # draft model 那一层（第五十五关）：第二个池子按**它自己**的层数/头数/精度建，
        # 与 target 的池子没有任何共享。draft 池不做前缀缓存——两套缓存交互的复杂度
        # 这一关照需求控制住（恢复时从真实历史补算，见 draft.DraftModelProposer）。
        self.draft_model = draft_model
        self.draft_proposer = None
        if speculative_mode == "draft_model":
            self.draft_kv_pool = KVCachePool(block_size, draft_num_kv_blocks,
                                             draft_model.num_kv_heads, draft_model.head_dim,
                                             draft_model.device, False,
                                             num_layers=draft_model.num_layers,
                                             dtype=draft_model.dtype)
            self.draft_proposer = DraftModelProposer(
                draft_model, self.draft_kv_pool, self.sampler, draft_model.eos_token_ids,
                draft_max_num_batched_tokens)

        # 采样执行层：组合一个 SampleRuntime，把采样后端、KV 池、停止 token 交给它
        # （draft 那一层也交给它：验证之后两套 KV 的回滚/对齐要在同一时刻做）
        self.sample_runtime = SampleRuntime(self.sampler, self.kv_cache_pool,
                                            model.eos_token_ids, self.draft_proposer)
        # 增量输出回调。不传就是 None —— 那时采样层连事件字典都不建
        self.on_token = on_token
        self.scheduler = Scheduler(max_num_seqs=max_num_seqs, max_num_batched_tokens=max_num_batched_tokens, block_size=block_size, enable_prefix_caching=enable_prefix_caching, on_finished=on_finished, kv_cache_pool=self.kv_cache_pool, eos_token_ids=model.eos_token_ids,
                          scheduling_policy=scheduling_policy,
                          speculative_mode=speculative_mode,
                          num_speculative_tokens=num_speculative_tokens,
                          prompt_lookup_n=prompt_lookup_n,
                          max_seq_len=model.max_seq_len,
                          vocab_size=model.vocab_size,
                          draft_proposer=self.draft_proposer)

    @classmethod
    def from_model_dir(cls, model_dir, device=None, attention_backend="torch", use_cuda_graph=False,
                       max_num_seqs=1, max_num_batched_tokens=4, block_size=4, num_kv_blocks=8,
                       on_finished=None, enable_prefix_caching=True, dtype=torch.float32,
                       norm_backend="torch", rope_backend="torch", *, on_token=None,
                       scheduling_policy="fcfs",
                       speculative_mode=None, num_speculative_tokens=2, prompt_lookup_n=2,
                       draft_model_dir=None, draft_num_kv_blocks=None,
                       draft_max_num_batched_tokens=None):
        # 只给目录和运行选项，模型结构全部来自目录；失败时不会交出半个 Engine。
        check_scheduling_policy(scheduling_policy)
        device = resolve_device(device)
        check_runtime(device, attention_backend, use_cuda_graph, dtype, norm_backend, rope_backend)
        # 「读配置 -> 建模型 -> 装权重」整条流程在 loading.py
        model = load_model_from_dir(model_dir, device, attention_backend,
                                    max_num_batched_tokens, use_cuda_graph,
                                    dtype, norm_backend, rope_backend)
        draft_model = None
        if draft_model_dir is not None:
            # draft 按**它自己的目录**读结构与权重：层数、hidden、头数都可以不同，
            # 也不能套用 target 的维度。它的固定输入缓冲按 draft 自己的预算开。
            if draft_max_num_batched_tokens is None:
                draft_max_num_batched_tokens = max_num_batched_tokens
            draft_model = load_model_from_dir(draft_model_dir, device, attention_backend,
                                              draft_max_num_batched_tokens, use_cuda_graph,
                                              dtype, norm_backend, rope_backend)
        return cls(model=model, draft_model=draft_model,
                   draft_num_kv_blocks=draft_num_kv_blocks,
                   draft_max_num_batched_tokens=draft_max_num_batched_tokens,
                   max_num_seqs=max_num_seqs,
                   max_num_batched_tokens=max_num_batched_tokens, block_size=block_size,
                   num_kv_blocks=num_kv_blocks, on_finished=on_finished, on_token=on_token,
                   enable_prefix_caching=enable_prefix_caching,
                   scheduling_policy=scheduling_policy,
                   speculative_mode=speculative_mode,
                   num_speculative_tokens=num_speculative_tokens,
                   prompt_lookup_n=prompt_lookup_n)

    def add_request(self, request):
        self.scheduler.add_request(request)

    def has_unfinished_requests(self):
        return self.scheduler.has_unfinished_requests()

    def step(self):

        with torch.inference_mode():

            self.scheduler.schedule()

            if not self.scheduler.has_unfinished_requests():
                return self.scheduler.step_done

            scheduled_items = self.scheduler.scheduled_items

            if scheduled_items:
                # draft 阶段（第五十五关）：补算 draft 的历史 + 逐位置批量提议，把草稿
                # 写回本轮计划。**必须在组装输入之前**——它改的正是本轮的输入行与
                # token 计数；也必须在调度器之外，因为这是跑**第二个模型**。
                if self.draft_proposer is not None:
                    self.draft_proposer.run_round(scheduled_items)

                # 本轮所有真实 token 拼成一维，prefill 与 decode 共用一次模型调用
                # 先在 CPU 组装，再由图外的准备步骤一次写进固定 GPU 缓冲
                input_ids = torch.tensor([token for item in scheduled_items for token in item["input_ids"]], dtype=torch.long)
                num_scheduled_tokens = [item["num_scheduled_tokens"] for item in scheduled_items]
                past_kv = [item["request"].cache for item in scheduled_items]

                # 采样计划在 forward 之前就定好：模型只拿行号，采样层只拿请求
                sample_rows, picked = self.sample_runtime.plan_sample_rows(scheduled_items)
                logits = self.model._forward_append(input_ids, num_scheduled_tokens, past_kv,
                                                    self.kv_cache_pool, sample_rows=sample_rows)
                self.sample_runtime.run(logits, picked, self.on_token)

            self.scheduler.post_step()

        return self.scheduler.step_done
