"""`ExtractHiddenStatesProposer`（对应 vLLM `v1/spec_decode/extract_hidden_states.py`）。

**它不是一个"提议者"，而是一段借投机框架跑的缓存写入**（需求 064 §1）。上游注释写得很直白：
这个方法不真的投机——它把 target 自己采出的 token 当"草稿"返回。于是每轮：

    target 前向（本轮 query 行，K=1 时每请求 2 行：b + 1 枚"草稿"）
      → 采样/验证
      → 本提议者：把本轮**每个 query 行**的辅助层特征 [T, L, H] 按**同一份 slot_mapping**
        散射进 cache-only 层的分页缓存（借 KV 的寻址、块生命周期、prefix 复用）
      → 返回 `sampled_token_ids[:, :1]`：target 本轮采出来的第一列，当作下一轮的草稿

为什么这样"绕"：草稿只是一个普通 token（没有分布、没有 q），下一轮验证时按点质量处理
（59 关的 `NO_DRAFT_PROBS` 分支），采样分布仍然**精确**是 target 的分布；而 target 每轮多跑
的那一行正好让"特征"和"KV"落在同一批槽位上——不需要自己造一套寻址与释放逻辑。

它不继承 `SpecDecodeBaseProposer`（上游也没有）：没有 draft 前向、没有自回归循环、没有随机流、
没有输入工作区，唯一的常驻状态是特征缓冲与它自己的 KV 缓存。

**本仓库与上游的差异**（逐条见 docs/step64_alignment.md §3）：

- `disable_padded_drafter_batch` 的校验：本仓库没有这个开关（也没有 padded drafter batch 的
  异步路径，那是 70 关），所以构造函数里不重复报错。
- `_get_slot_mapping()` / `_slot_mapping_buffer`：上游把 target 的槽位拷进常驻 buffer 并在尾部
  补 `PADDING_SLOT_ID`（给 CUDA Graph 的 padding 用，69 关）；本仓库直接复用本轮 target 的
  **同一份** `slot_mapping`，不复制、不补尾。
- CUDA Graph / DP 协调 / EPLB：本仓库都没有（69/70/分布式关卡），所以 `propose()` 里没有
  `_determine_batch_execution_and_padding()` 与 `dummy_run()`。
- `prepare_next_token_ids_padded()`：上游用它给"异步调度下不落 CPU 的输入缓冲"补 token
  （`prev_sampled_token_ids`）。本仓库的输入来自已提交历史（`_bookkeeping_sync` 写入镜像），
  没有这个消费者，所以不实现（不写没有调用方的代码；70 关做异步调度时再说）。
"""

import torch

from ..attention.forward_context import set_forward_context
from ..models.extract_hidden_states import CacheOnlyAttentionBackend, CacheOnlyAttentionLayer
from .draft_model import _dtype    # 与 draft 提议者共用同一套「dtype 字符串 → torch dtype」映射


class ExtractHiddenStatesProposer:
    def __init__(self, vllm_config, device: str) -> None:
        spec_config = vllm_config.speculative_config
        if spec_config is None or not spec_config.uses_extract_hidden_states():
            raise ValueError(
                "ExtractHiddenStatesProposer 只能用于 method='extract_hidden_states'"
                "（上游同样在构造时 assert 方法/K，本仓库把 K 的校验提前到配置期）")
        # 上游：`assert self.num_speculative_tokens == 1`（配置期已校验，这里只做二次确认）
        self.num_speculative_tokens = spec_config.num_speculative_tokens
        if self.num_speculative_tokens != 1:
            raise ValueError(f"extract_hidden_states 只支持 K=1，收到 {self.num_speculative_tokens}")

        self.vllm_config = vllm_config
        self.device = device
        self.dtype = _dtype(vllm_config.model_config.dtype)

        # 模型与层名（`load_model()` 里初始化）
        self.model = None
        self.attn_layer_names: list[str] = []
        self.attn_metadata_builder = None
        self.kv_caches: dict[str, torch.Tensor] = {}
        # 本仓库只有一个 KV group（上游在这里记 gid，用来挑对应的 common_attn_metadata）
        self.kv_cache_gid = 0

        # 特征缓冲的容量（上游同款）：本轮的行数最多是 target 的 token 预算，再留一个批大小的
        # 富余（上游是给 cudagraph padding 留的；本仓库只是照抄这个口径，不额外加旋钮）
        max_batch_size = vllm_config.scheduler_config.max_num_seqs
        self.max_num_tokens = (
            vllm_config.scheduler_config.max_num_batched_tokens + max_batch_size)

        # cache-only 模型的配置：上游在 `SpeculativeConfig.__post_init__` 里派生（它持有 target
        # 配置），本仓库在提议者这里派生（它同时拿得到 target 与 cache 配置）
        self.draft_model_config = spec_config.derive_extract_hidden_states_config(
            vllm_config.model_config, vllm_config.cache_config)
        self.hf_config = self.draft_model_config.hf_config
        layer_ids = self.hf_config.get("eagle_aux_hidden_state_layer_ids")
        if not layer_ids:
            # 上游原话：eagle_aux_hidden_state_layer_ids must be set in the draft model config
            raise ValueError(
                "method='extract_hidden_states' 必须在 draft 配置里给 "
                "eagle_aux_hidden_state_layer_ids（要存哪几层特征）")
        self.num_hidden_states = len(layer_ids)
        self.hidden_size = int(vllm_config.model_config.hf_config["hidden_size"])
        # [T, L, H]：每行一个 token 的 L 层特征（写进 cache-only 层的缓存）
        self.hidden_states = torch.zeros(
            (self.max_num_tokens, self.num_hidden_states, self.hidden_size),
            dtype=self.dtype, device=device)

    # -------- 加载与校验 --------

    def load_model(self):
        """建 cache-only 模型（无权重）→ 收集层 → 分配它自己的缓存。"""
        from ..model_loader import get_model

        self.model = get_model(self.draft_model_config, self.device)
        self._collect_attn_layers()
        self._allocate_kv_caches()
        return self.model

    def _collect_attn_layers(self) -> None:
        """找出模型里的 cache-only 层（上游是"全部 attention 层 − target 的 attention 层"，
        因为上游的 draft 与 target 共用一个全局层表；本仓库直接从提议者自己的模型里按类型收集）。

        上游断言"恰好一层"（`ExtractHiddenStatesModel` 只该有一个 cache-only 层）：本仓库同样
        断言——多一层就意味着"特征写进了两份缓存"或者"有一层没人给 metadata"，都是静默错。
        """
        layers: dict[str, CacheOnlyAttentionLayer] = {}
        for name, module in self.model.named_modules():
            if not isinstance(module, CacheOnlyAttentionLayer):
                continue
            if module.layer_name != name:
                raise ValueError(
                    f"cache-only 层名与模块路径不一致：layer_name={module.layer_name!r}，"
                    f"实际路径={name!r}。forward 上下文按层名取 metadata，两者必须相同")
            layers[name] = module
        if len(layers) != 1:
            raise ValueError(
                f"ExtractHiddenStatesModel 应该恰好有 1 个 cache-only 层，找到 {len(layers)} 个："
                f"{sorted(layers)}")
        self.attn_layer_names = list(layers)
        self.attn_metadata_builder = CacheOnlyAttentionBackend.get_builder_cls()()

    def _allocate_kv_caches(self) -> None:
        """给 cache-only 层分配它自己的缓存：`[num_blocks, block_size, L, H]`，**没有 k/v 维**。

        块编号与 target 的块表**同一套**（和 draft 模型的 KV 一样：同一张逻辑块表、每层自己的
        物理张量）。于是"prefix 命中的块"在两边指向同一段位置：特征与 KV 一样是"token 与位置的
        确定函数"，所以共享的块内容对任何同前缀请求都成立。
        """
        cache_config = self.vllm_config.cache_config
        for name, module in self.model.named_modules():
            if not isinstance(module, CacheOnlyAttentionLayer):
                continue
            shape = module.get_kv_cache_shape(cache_config.num_gpu_blocks)
            cache = torch.zeros(shape, dtype=module.kv_cache_torch_dtype, device=self.device)
            module.kv_cache = cache
            self.kv_caches[name] = cache

    def validate_same_kv_cache_group(self, kv_cache_config) -> None:
        """上游：校验所有 cache-only 层属于**同一个 KV cache group**，并记下组号。

        上游的模型层是在 `initialize_kv_cache_tensors()` 里从 KV 规格树绑缓存的，所以它要按
        group 找层；本仓库只有一个 KV group、cache-only 层不走 `KVCacheManager` 的层列表
        （它复用 target 的逻辑块表，与 draft 同一套），所以这里的等价校验是"**缓存确实是按本轮
        规格分配的**"：块大小或块数对不上，写进去的 slot 就会落到别的块上（读出来时是别人的
        特征，不报错）。
        """
        if len(self.attn_layer_names) != 1:
            raise ValueError(f"cache-only 层必须恰好 1 个，实际 {self.attn_layer_names}")
        cache = self.kv_caches[self.attn_layer_names[0]]
        if (cache.shape[0] != kv_cache_config.num_gpu_blocks
                or cache.shape[1] != kv_cache_config.block_size):
            raise ValueError(
                f"cache-only 缓存的形状 {tuple(cache.shape[:2])} 与 KV 规格不符"
                f"（num_gpu_blocks={kv_cache_config.num_gpu_blocks}, "
                f"block_size={kv_cache_config.block_size}）：块编号与槽位对不上，"
                f"特征会写进别的块")

    # -------- 提议 --------

    @torch.inference_mode()
    def propose(self, num_speculative_tokens: int, sampled_token_ids,
                target_hidden_states, common_attn_metadata) -> torch.Tensor:
        """写特征 + 返回 target 采出的第一列（上游 `propose()` 的三步）。

        参数与上游同名同义（`slot_mappings` 那个"为了接口兼容而留着但不用"的参数本仓库没有，
        因为所有提议者的签名都是本仓库自己的）：
        - `sampled_token_ids`：本轮的采样结果；K=1 时宽度是 2（验证列 + bonus 列），
          **只返回第 0 列**；
        - `target_hidden_states`：本轮 target 每个 query 行的辅助层特征（本仓库的 target 把
          多层拼在最后一维，形状 `[T, L*H]`，这里按 `[T, L, H]` 重解释——与上游
          `torch.stack(list, dim=1)` 是同一份布局，用例里有逐值对照）；
        - `common_attn_metadata`：本轮 target 用的那份元数据（槽位就是它里面的 `slot_mapping`）。
        """
        if num_speculative_tokens != self.num_speculative_tokens:
            raise ValueError(
                f"本轮的草稿数 {num_speculative_tokens} 与配置的 "
                f"{self.num_speculative_tokens} 不一致：extract_hidden_states 固定 K=1")
        if self.model is None:
            raise RuntimeError("提议者还没 load_model()：cache-only 层与缓存都不存在")

        stacked_hidden_states = self._stack_hidden_states(target_hidden_states)
        num_tokens = int(stacked_hidden_states.shape[0])
        if num_tokens > self.max_num_tokens:
            raise RuntimeError(
                f"本轮有 {num_tokens} 行特征，超过特征缓冲的 {self.max_num_tokens} 行："
                f"缓冲口径是 max_num_batched_tokens + max_num_seqs，说明控制面/执行面的"
                f"输入预算口径不一致（不是模型问题）")
        self.hidden_states[:num_tokens] = stacked_hidden_states

        attn_metadata = self._build_attn_metadata(common_attn_metadata)
        with set_forward_context(attn_metadata, num_tokens=num_tokens):
            self.model(hidden_states=self.hidden_states[:num_tokens])

        # 上游：返回 target 自己采出的那一列当"草稿"（宽度可能大于 1，只取第 0 列）
        return sampled_token_ids[:, :1]

    def _stack_hidden_states(self, target_hidden_states) -> torch.Tensor:
        """把辅助层特征整理成 `[T, L, H]`。

        - 上游的形态是一个**列表**（每层一个 `[T, H]`）→ 逐字照抄 `torch.stack(..., dim=1)`；
        - 本仓库的 target 把多层**拼在最后一维**（63 关：draft 的 `combine_hidden_states` 按同样的
          顺序切块），形状是 `[T, L*H]` → `view(T, L, H)`。两者是同一份内存布局的重解释
          （第 l 段是第 l 层），不是"另算一份特征"。
        """
        if isinstance(target_hidden_states, (list, tuple)):
            return torch.stack(list(target_hidden_states), dim=1)
        if target_hidden_states.ndim != 2:
            raise ValueError(
                f"特征张量的形状应是 [T, L*H]（本仓库 target 的拼接布局），"
                f"收到 {tuple(target_hidden_states.shape)}")
        expected = self.num_hidden_states * self.hidden_size
        if target_hidden_states.shape[1] != expected:
            raise ValueError(
                f"特征宽度 {target_hidden_states.shape[1]} 与 L*H="
                f"{self.num_hidden_states}*{self.hidden_size}={expected} 不一致："
                f"层数或 hidden_size 与配置对不上（写进去的会错位）")
        return target_hidden_states.view(target_hidden_states.shape[0],
                                        self.num_hidden_states, self.hidden_size)

    def _build_attn_metadata(self, common_attn_metadata) -> dict:
        """每个 cache-only 层一份 metadata（同一份槽位）。

        **同源**：这里用的是本轮 target 真正用的那份元数据里的 `slot_mapping`——不是按位置公式
        重算的（64 关需求 §2：`slot_mapping` 与本轮 query 对齐）。上游同样直接吃
        `common_attn_metadata.slot_mapping`。
        """
        slot_mapping = getattr(common_attn_metadata, "slot_mapping", None)
        if slot_mapping is None:
            raise ValueError("本轮没有可用的 slot_mapping：extract_hidden_states 需要 target 的"
                             "本轮元数据（提议者靠它把特征写进与 KV 相同的槽位）")
        metadata = self.attn_metadata_builder.build(slot_mapping=slot_mapping)
        return {name: metadata for name in self.attn_layer_names}
