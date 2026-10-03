"""用一个**小模型**当提议者（对应 vLLM `v1/spec_decode/llm_base_proposer.py` +
`draft_model.py`）。

拆成两个类，位置与 vLLM 一致：

    SpecDecodeBaseProposer   提议的**步骤**：自己的输入缓冲、自己的 KV、自己的 attention 元数据
    DraftModelProposer       与"用哪个模型"有关的：造 draft 的 VllmConfig、加载、规格校验

### 它为什么不是"第二套引擎"

它没有调度器、没有请求状态、没有输出提交：只做一件事——**给定请求的已提交历史，提 K 枚草稿**。

    第一遍 forward：把"上一轮新提交的那段"喂给 draft 模型，把它的 KV 补到与 target 一致；
                    最后一行的 logits 顺便得到**第一枚**草稿
    自回归 K-1 步：每步一行（上一枚草稿当输入），得到其余草稿

第一遍为什么需要：draft 模型要提出**像样**的草稿，它的 KV 必须覆盖请求的已提交历史。
但我们不重算整段历史——上一轮提草稿时它已经算到那里了，本轮只需要补"新提交的 a+1 个 token"。

**有效边界**（199 §9 点名的一条）：上一轮为**被拒的草稿**写过的 KV 仍留在物理缓冲里，
但它们不是有效历史。所以每轮都按"已提交到哪"重建 positions 与 slot_mapping——
**位置从有效边界续，不从缓冲里残留的位置续**。

### KV：共用一个 group，但每层是自己的 tensor

draft 与 target 共用**逻辑块表**与分配生命周期（同一张块表、同一套 slot 编号），
但每个 Attention 层绑自己的物理 tensor。所以同一块号在两边代表同一段位置，**不是同一份 K/V**。
加载时校验规格（词表、dtype、KV head 数、head_size、block_size），不兼容就明确报错——
不假装"所有小模型都能配对"。

### 本关的简化（写清楚，不假装已实现）

- **有** `num_lookahead_tokens` 预留（204 §4 之后补上的）：草稿里"下一轮才验证"的那 K 枚写在
  target 本轮 query **之外**，所以 Scheduler 分配块时按 `num_lookahead_tokens=K` 多留 K 个槽位
  （`Scheduler.__init__` → `allocate_slots`）。预留块**不等于** token 已计算，发布 prefix 时不算。
  上下文快满时预留会被 `max_model_len` 截掉，所以自回归循环每写一枚前还要过
  逻辑上界 + `BlockTable.covers` 两道检查（205 §3），过不了就少提几枚。
- **没有预分配的定长输入缓冲**，也**没有** vLLM 的 `input_budget` / `max_num_new_slots_for_drafting`
  核算：每轮按"实际要补多少 token"现搭张量。它避免了定长缓冲的写越界，但**不等于**有了输入预算
  ——普通 draft 同样需要那份核算，只是本关用动态张量绕开了容量维度（对齐差异，见
  `docs/step57_alignment.md`）。代价是每轮重建张量，而且 **prefix 命中之后 draft 要整段重算**
  （命中省的是 target 的 prefill，draft 第一次同步仍要写满命中的位置）。
- 不做 EAGLE/MTP 的"左移一位"输入（本关只有普通自回归 draft）。
- draft 不共享 target 的 random stream：自己按 `seed` 建 generator（可复现），
  与 target 的采样流相互独立（vLLM 也把 draft 的随机数分开算）。
"""

import torch

from ..outputs import DraftTokenIds
from ..sample import Sampler
from ..sample.metadata import SAMPLING_EPS
from ..sample.ops.topk_topp_sampler import apply_top_k_top_p, random_sample
from ..attention import Attention, AttentionMetadataBuilder, set_forward_context


class SpecDecodeBaseProposer:
    """提议步骤的骨架：**没有调度、没有请求状态**，只有"历史进、草稿出"。"""

    def __init__(self, spec_config, vllm_config, device: str) -> None:
        self.spec_config = spec_config
        self.vllm_config = vllm_config
        self.device = device
        self.num_speculative_tokens = spec_config.num_speculative_tokens
        self.block_size = vllm_config.cache_config.block_size
        # **逻辑**上界。块表容量是按块向上取整的（10 个位置可能给 12 个槽位），
        # 所以"物理槽位够"不等于"模型允许写这个位置"——两个边界要分别检查（205 §3）。
        self.max_model_len = vllm_config.model_config.max_model_len
        self.model = None                     # 子类加载
        self.kv_caches: dict[str, torch.Tensor] = {}
        self.metadata_builder = AttentionMetadataBuilder(self.block_size)
        self.sampler = Sampler()
        # draft 模型自己的进度（"它的 KV 已经算到哪"）。与 target 的进度**分开维护**：
        # 恢复/新请求要重置，否则会拿旧物理编号上的 KV 当历史（199 §5）。
        # 它正常每轮都追平 target（中间 prefill 块也同步），所以不需要谁去夹发布边界——
        # 但要有一条用例盯着它（check_step57_draft_model.py §6）
        self._draft_computed: dict[str, int] = {}
        self._draft_generators: dict[str, torch.Generator] = {}
        self.num_drafts_proposed = 0

    # -------- 交给子类 --------

    def load_model(self) -> None:
        raise NotImplementedError

    # -------- 提议 --------

    def propose(self, req_ids: list[str], all_token_ids: dict[str, list[int]],
                num_computed_tokens: dict[str, int], input_batch,
                ready_req_ids: set[str] | None = None,
                reset_req_ids: set[str] | None = None) -> DraftTokenIds:
        """对每个被调度的请求都跑一遍：**同步 KV**，并给其中 ready 的那些提草稿。

        `num_computed_tokens[req_id]` 是"target 本轮之后算到哪"——draft 的 KV 必须覆盖到
        这个位置（通过第一遍前向补齐）。**中间 prefill 块也要同步**：同一个逻辑块在
        target/draft 的每一层都有各自的 tensor，draft 没写过的那一层会让这个块不能算
        "完整可复用"，而 target 的完整块这一轮就会发布出去（199 §9）。vLLM 的做法就是这样
        ——drafter 与 target 每步跑同一段位置，草案在同一轮里天然同步。

        草稿本身只给 `ready_req_ids` 里的请求提（None = 全提）：中间 prefill 块没有可验证的
        next token，提了 Scheduler 也会丢（vLLM 的 `update_draft_token_ids` 同款规则），
        这里更早一步不提，省掉 K 次前向。

        `input_batch` 是 target 的批状态：草稿要读**共享的块表**（同一套 slot 编号）与
        每条请求的采样参数。draft 的 KV 写在自己的 tensor 上，但位置与槽位由这里决定。

        **生命周期**（205 §4.4，与 Controller 的分工）：

        - 本轮没被调度的请求：它根本不进 `req_ids`，这里什么都不做——进度与随机流由
          `_draft_computed` / `_draft_generators` 保留（块仍然有效，"没排上"≠"结束"）；
        - 抢占恢复（`reset_req_ids`）：块表整表换过 → 进度作废重算，随机流**继续**；
        - 结束/abort：由 `remove_requests()` 显式删除（不在 batch 里不能当结束）。
        """

        self._reset_requests(reset_req_ids or set())
        ready = set(req_ids) if ready_req_ids is None else set(ready_req_ids)
        drafts: dict[str, list[int]] = {req_id: [] for req_id in req_ids}
        probs: dict[str, list[torch.Tensor]] = {req_id: [] for req_id in req_ids}

        # ---- 第一遍：把"target 算完、draft 还没算"的那段补进 draft 的 KV ----
        rows: list[tuple[str, int]] = []
        for req_id in req_ids:
            boundary = num_computed_tokens[req_id]
            start = min(self._draft_computed.get(req_id, 0), boundary)
            rows.extend((req_id, position) for position in range(start, boundary))
        if rows:
            hidden = self._forward(rows, all_token_ids, input_batch)
            # 进度对**所有**请求推进（包括中间 prefill 块）：它们这一轮的 KV 也写了
            for req_id, _ in rows:
                self._draft_computed[req_id] = num_computed_tokens[req_id]
            # 但只有 ready 的行才取第一枚草稿：其余行的草稿不会被采用（行是同请求连续的，
            # 所以"最后一行"就是该请求同步到的那一行）
            last_row: dict[str, int] = {}
            for index, (req_id, _) in enumerate(rows):
                if req_id in ready:
                    last_row[req_id] = index
            if last_row:
                self._sample(hidden, list(last_row.items()), input_batch, drafts, probs)

        # ---- 自回归补足 K 枚：上一枚当输入，位置接在它后面 ----
        #
        # 每一枚草稿都写在"已提交历史之后"（位置 = 已提交 + j - 1），那是 target 本轮 query
        # **之外**的位置：调度侧为此预留了 `num_lookahead_tokens` 个槽位（vLLM 同款规则）。
        # 但预留可能被 `max_model_len` 截掉、上下文也可能刚好走到尽头，所以每写一枚都要过
        # **两个**边界：模型自己的位置范围（逻辑）与块表覆盖（物理）。少一个就会在"10 个
        # 位置、3 个块（12 槽）"这种配置下写出 position=10（205 §3）。过不了就少提几枚——
        # 草稿只是候选，不写就不会越界，也不会让这一轮失败。
        while True:
            pending: list[str] = []
            for req_id in req_ids:
                if not 0 < len(drafts[req_id]) < self.num_speculative_tokens:
                    continue
                position = num_computed_tokens[req_id] + len(drafts[req_id]) - 1
                row = input_batch.req_id_to_index[req_id]
                if not 0 <= position < self.max_model_len:
                    continue
                if not input_batch.block_table.covers(row, position):
                    continue
                pending.append(req_id)
            if not pending:
                break
            rows = [(req_id, num_computed_tokens[req_id] + len(drafts[req_id]) - 1)
                    for req_id in pending]
            tokens = {req_id: list(all_token_ids[req_id][:num_computed_tokens[req_id]])
                      + drafts[req_id] for req_id in pending}
            hidden = self._forward(rows, tokens, input_batch)
            self._sample(hidden, [(req_id, index) for index, (req_id, _) in enumerate(rows)],
                         input_batch, drafts, probs)

        probs_rows = [row for req_id in req_ids for row in probs[req_id]]
        self.num_drafts_proposed += sum(len(drafts[req_id]) for req_id in req_ids)
        return DraftTokenIds(
            req_ids=list(req_ids),
            draft_token_ids=[drafts[req_id] for req_id in req_ids],
            draft_probs=torch.stack(probs_rows) if probs_rows else None)

    def _reset_requests(self, reset_req_ids: set[str]) -> None:
        """丢掉不再成立的 draft 侧**进度**：请求刚被抢占恢复（块表整表换过）。

        恢复之后旧物理编号上的 KV 已经不属于它了，**必须**从头补，不能接着用。
        随机流**不重置**：请求还活着，只是换了块，重新 seed 会改变它的采样序列
        （205 §4.4：抢占恢复 ≠ 新请求）。
        """
        for req_id in reset_req_ids:
            self._draft_computed.pop(req_id, None)

    def remove_requests(self, req_ids) -> None:
        """请求**结束/abort**：进度与随机流一并删除（草稿概率 q 存在 Runner 上、
        每轮整体重算，不需要按请求清）。

        只有控制端明确说"这条结束了"才调它。**不能**用"不在本轮的 req_ids 里"代替：
        预算不够没排上、被抢占等待恢复的请求都不在 batch 里，但它们的状态必须留着
        （205 §4.1/§4.3 就是这两种情况混在一起造成的）。
        0-token 的结束清理轮也要调——否则最后一条请求结束后，复用的 ID 会继承旧进度。
        """
        for req_id in req_ids:
            self._draft_computed.pop(req_id, None)
            self._draft_generators.pop(req_id, None)

    # -------- 单次 forward --------

    def _forward(self, rows: list[tuple[str, int]], token_ids_by_req, input_batch):
        """把 `(请求, 绝对位置)` 这批行喂给 draft 模型，返回 hidden states。"""
        max_model_len = self.vllm_config.model_config.max_model_len
        for req_id, position in rows:
            if not 0 <= position < max_model_len:
                raise RuntimeError(
                    f"{req_id!r} 的 draft 前向要算位置 {position}，超出了 "
                    f"max_model_len={max_model_len}：draft 侧的历史边界与 target 对不上了")
        block_table = input_batch.block_table
        device = self.device
        input_ids, positions, batch_rows = [], [], []
        for req_id, position in rows:
            input_ids.append(int(token_ids_by_req[req_id][position]))
            positions.append(int(position))
            batch_rows.append(input_batch.req_id_to_index[req_id])

        # 参与本轮的请求，按第一次出现的顺序（同一请求的行是连续的）
        unique_reqs = list(dict.fromkeys(req_id for req_id, _ in rows))
        counts = [sum(1 for req_id, _ in rows if req_id == unique) for unique in unique_reqs]
        query_start_loc = [0]
        for count in counts:
            query_start_loc.append(query_start_loc[-1] + count)
        last_position = {unique: max(position for req_id, position in rows if req_id == unique)
                         for unique in unique_reqs}

        positions_tensor = torch.tensor(positions, dtype=torch.int64, device=device)
        # 槽位映射在 **CPU** 上算：块表镜像是 CPU 结构（`.cpu`），索引也必须是 CPU 张量，
        # 否则会撞上 "Expected all tensors to be on the same device"。公式与 target 同一份实现。
        slots = block_table.compute_slot_mapping(
            input_batch.num_reqs, positions_tensor.cpu(),
            torch.tensor(batch_rows, dtype=torch.int64)).to(device)
        # 块表张量：只取参与的批行，顺序与 unique_reqs 一致
        max_blocks = max(block_table.num_blocks(row) for row in batch_rows)
        table_tensor = torch.zeros((len(unique_reqs), max(1, max_blocks)), dtype=torch.int64,
                                   device=device)
        for index, req_id in enumerate(unique_reqs):
            row = input_batch.req_id_to_index[req_id]
            count = block_table.num_blocks(row)
            table_tensor[index, :count] = torch.tensor(
                block_table.cpu[row, :count].tolist(), dtype=torch.int64, device=device)
        metadata = self.metadata_builder.build(
            query_start_loc=torch.tensor(query_start_loc, dtype=torch.int64, device=device),
            seq_lens=torch.tensor([last_position[req_id] + 1 for req_id in unique_reqs],
                                  dtype=torch.int64, device=device),
            block_table=table_tensor, slot_mapping=slots, num_reqs=len(unique_reqs))
        attn_metadata = {name: metadata for name in self.kv_caches}
        with set_forward_context(attn_metadata, num_tokens=len(rows)):
            return self.model(torch.tensor(input_ids, dtype=torch.int64, device=device),
                              positions_tensor)

    # -------- 采样草稿 --------

    def _sample(self, hidden: torch.Tensor, row_refs: list[tuple[str, int]], input_batch,
                drafts: dict, probs: dict) -> None:
        """对给定行各采一枚草稿，同时记下它来自的分布（q）。

        `q` 必须是**实际提议时用的分布**（199 §7），所以这里与 `Sampler.sample` 走同一条
        约束链（温度 → top-k/top-p → 指数竞赛），把 softmax 之后的整行留下来。
        贪心行的"分布"是草稿位置上的点质量（one-hot），与 `draft_probs=None` 的点质量提议同义。
        """
        # **先过 LM head**：hidden 是隐藏态，采样要的是词表上的 logits（与 Runner 同一条路：
        # `compute_logits` 只对需要的行做词表 GEMM）
        logits = self.model.compute_logits(
            hidden[torch.tensor([row for _, row in row_refs],
                                dtype=torch.int64, device=hidden.device)]).to(torch.float32)
        for index, (req_id, _) in enumerate(row_refs):
            parameter = input_batch.sampling_params[input_batch.req_id_to_index[req_id]]
            row_logits = logits[index]
            probs_row = self._row_probs(row_logits, parameter)
            token = int(row_logits.argmax()) if parameter.is_greedy else int(
                random_sample(probs_row.unsqueeze(0),
                              {0: self._generator(req_id, parameter)})[0])
            drafts[req_id].append(token)
            probs[req_id].append(probs_row)

    def _row_probs(self, row_logits: torch.Tensor, parameter) -> torch.Tensor:
        """一行的提议分布：贪心 → one-hot（点质量）；否则与普通采样同一条约束链。"""
        if parameter.is_greedy:
            one_hot = torch.zeros_like(row_logits)
            one_hot[int(row_logits.argmax())] = 1.0
            return one_hot
        logits = row_logits / max(parameter.temperature, SAMPLING_EPS)
        top_k = None if parameter.top_k in (-1, 0) else torch.tensor(
            [parameter.top_k], device=logits.device)
        top_p = None if parameter.top_p >= 1.0 else torch.tensor(
            [parameter.top_p], device=logits.device)
        logits = apply_top_k_top_p(logits.unsqueeze(0), top_k, top_p)
        return logits.softmax(dim=-1, dtype=torch.float32)[0]

    def _generator(self, req_id: str, parameter):
        """draft 自己的随机流：按请求的 seed 建一次，之后**每轮接着用**（不是每轮重置）。"""
        generator = self._draft_generators.get(req_id)
        if generator is None and parameter.seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(parameter.seed + 1)   # 与 target 的流分开，避免逐位重合
            self._draft_generators[req_id] = generator
        return generator

    def _attention_layer_names(self) -> list[str]:
        return sorted(self.kv_caches)



class DraftModelProposer(SpecDecodeBaseProposer):
    """用**另一个模型目录**当提议者（对应 vLLM `v1/spec_decode/draft_model.py`）。"""

    def __init__(self, spec_config, vllm_config, device: str) -> None:
        super().__init__(spec_config, vllm_config, device)
        if spec_config.draft_model_config is None:
            raise ValueError("draft_model 投机必须在 SpeculativeConfig 里给 draft_model_config")
        self.draft_model_config = spec_config.draft_model_config

    # -------- 加载与校验 --------

    def load_model(self) -> None:
        from ..model_loader import get_model

        self._validate_configs()
        self.model = get_model(self.draft_model_config, self.device)
        self._allocate_kv_caches()
        return self.model

    def _validate_configs(self) -> None:
        """词表与 KV 规格必须兼容（199 §9）：不满足就**明确报错**，不静默降级。"""
        target_config = self.vllm_config.model_config.hf_config or {}
        draft_config = self.draft_model_config.hf_config or {}
        target_vocab = target_config.get("vocab_size")
        draft_vocab = draft_config.get("vocab_size")
        if target_vocab != draft_vocab:
            raise ValueError(
                f"draft 与 target 的词表不一致（{draft_vocab} vs {target_vocab}）："
                f"草稿的 token 在 target 的词表里是另一个意思，验证没有意义")
        if str(self.draft_model_config.dtype) != str(self.vllm_config.model_config.dtype):
            raise ValueError("draft 与 target 的 dtype 必须一致（KV 缓存要放进同一个 group）")
        for key in ("num_key_value_heads", "head_dim", "max_position_embeddings"):
            if draft_config.get(key) != target_config.get(key):
                raise ValueError(
                    f"draft 与 target 的 {key} 不一致"
                    f"（{draft_config.get(key)} vs {target_config.get(key)}）："
                    f"本关只支持 KV 规格相同的 draft/target 共用一个 KV group")

    def _allocate_kv_caches(self) -> None:
        """给 draft 的每个 Attention 层分配**自己的**物理缓存。

        与 target 同形、同块数：块号在两边代表同一段位置。但**不是同一份 K/V**——
        target 写它自己的 tensor，draft 写自己的。
        """
        cache_config = self.vllm_config.cache_config
        dtype = next(self.model.parameters()).dtype
        self.kv_caches = {}
        for name, module in self.model.named_modules():
            if not isinstance(module, Attention):
                continue
            cache = torch.zeros(2, cache_config.num_gpu_blocks, self.block_size,
                                module.num_kv_heads, module.head_size, dtype=dtype,
                                device=self.device)
            module.kv_cache = cache
            self.kv_caches[name] = cache

