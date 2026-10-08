"""GPU 版 ngram 提议者（对应 vLLM `v1/spec_decode/ngram_proposer_gpu.py`）。

和 CPU 版的**匹配语义完全相同**（最长后缀 ngram、同长度取最早那处、`k` 的两个上限），区别只在
"历史放在哪、结果怎么表达"：

    历史        一直躺在显存里（`token_ids_gpu_tensor [max_num_reqs, max_model_len]`），
                每步只把本轮新采样的 token `scatter_` 进去（增量），**不把整段历史搬回 CPU**
    结果        `draft_tokens [B, K]`（**固定宽度**，尾部用 -1 填）+ `num_valid_draft_tokens [B]`
                —— 宽度是张量形状，有效个数才是语义；只看宽度就会把 -1 当成真 token

### 与上游的差异（逐条记在 `docs/step60_alignment.md`）

1. 上游的 `NgramGPUKernel` 用 `@support_torch_compile()` 编译（还有一整套 inductor 配置）；
   本机没有 torch.compile 支持，这里是**同样的 torch 张量运算**（`unfold` / `gather` / `argmax`），
   只是不编译。69 关做 CUDA Graph 时再谈固定形状与图捕获。
2. 上游用 pinned 缓冲 + 独立 stream/event 把 `[B]` 有效个数与 `[B, K]` 草稿**异步**拷回 CPU
   （`copy_num_valid_draft_tokens` / `_copy_draft_token_ids_to_cpu`）；本机是同步引擎，草稿最终
   必须落到 CPU 交给 Scheduler，所以这里直接同步拷（数据量是 B×K 个整数，不是"整段历史"）。
3. 上游在 **Runner 侧**裁"占位 → 有效"（`update_scheduler_for_invalid_drafts` 改的是 Runner 手里的
   `scheduler_output` 副本，因为异步调度让 Scheduler 的计划是乐观的）；本机没有异步调度，所以在
   **草稿交接给 Scheduler 时**裁一次（同一个函数、同样的效应），这样 `-1` 从不进入计划、预算与
   统计也不会把占位算成候选。
"""

import torch
from torch import nn

from ..outputs import DraftTokenIds
from .utils import TargetRows, update_scheduler_for_invalid_drafts  # noqa: F401（上游把这个函数放在本模块）


class NgramGPUKernel(nn.Module):
    """匹配 + 提取（上游同名类）：全程 torch 张量运算，一个 batch 一次算完。

    **上游的"kernel"也是 torch 算子**（不是手写 Triton）：它靠 `@support_torch_compile()` 让 inductor
    把几十个小算子融合成几个内核，并把批 padding 到 `max_num_reqs` 以拿到固定形状。本机没有编译基础设施，
    跑的是未融合版本 → **启动受限**：实测 B=32/历史 4096 时 127 次 CUDA 事件、约 2.2 ms/步，且与形状几乎无关；
    融合后是 8 次事件、约 0.13 ms/步（结果逐值相同）。见 docs/step60_alignment.md §3 第 2 条。
    """

    def __init__(self, vllm_config, prefix: str = "", device: str = "cuda") -> None:
        super().__init__()

        spec_config = vllm_config.speculative_config
        assert spec_config is not None
        assert spec_config.prompt_lookup_min is not None
        assert spec_config.prompt_lookup_max is not None

        self.min_n = spec_config.prompt_lookup_min
        self.max_n = spec_config.prompt_lookup_max
        self.k = spec_config.num_speculative_tokens
        self.max_model_len = vllm_config.model_config.max_model_len
        self.max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        self.device = device

    def _find_first_and_extract_all_n_parallel(self, token_ids: torch.Tensor,
                                               seq_lengths: torch.Tensor,
                                               min_ngram_len: int, max_ngram_len: int,
                                               num_draft_tokens: int) -> torch.Tensor:
        """找后缀 ngram 匹配并抄后面的 token（上游同名方法）。

        对每个 n 都做一遍"滑动窗口 == 后缀"的全量比较，取**最早**的匹配（`argmax` 在 bool 上
        取第一个 True），最后在"有匹配的 n 里取最大的那个"。返回值里 `-1` = 无效/没匹配。
        """
        batch_size = token_ids.shape[0]
        max_seq_len = token_ids.shape[1]
        device = token_ids.device
        num_ngram_sizes = max_ngram_len - min_ngram_len + 1

        ngram_lengths = torch.arange(min_ngram_len, max_ngram_len + 1, device=device)
        batch_indices = torch.arange(batch_size, device=device)

        # 每个 (请求, n) 的最早匹配位置；-1 = 没匹配
        first_match_positions = torch.full((batch_size, num_ngram_sizes), -1,
                                           dtype=torch.long, device=device)

        for i, ngram_len in enumerate(range(min_ngram_len, max_ngram_len + 1)):
            # 长度 ngram_len 的滑动窗口（unfold 是视图，不拷贝）
            search_windows = token_ids.unfold(1, ngram_len, 1)
            num_windows = search_windows.shape[1]

            # 末尾的 ngram_len 个 token（就是"后缀"，匹配它）
            suffix_starts = seq_lengths - ngram_len
            suffix_indices = suffix_starts.unsqueeze(1) + torch.arange(ngram_len,
                                                                      device=device)
            suffix_indices.clamp_(min=0)
            suffix = torch.gather(token_ids, 1, suffix_indices)

            matches = (search_windows == suffix.unsqueeze(1)).all(dim=-1)

            # 匹配点后面至少还要留一个 token（否则抄不到东西）
            max_valid_suffix_start = seq_lengths - ngram_len - 1
            window_positions = torch.arange(num_windows, device=device)
            valid_mask = window_positions <= max_valid_suffix_start.unsqueeze(1)
            final_matches = matches & valid_mask

            # 最早匹配（没匹配时 argmax 给 0，用 has_match 修正）
            first_match_idx = torch.argmax(final_matches.int(), dim=1)
            has_match = final_matches[batch_indices, first_match_idx]
            first_match_positions[:, i] = torch.where(has_match, first_match_idx, -1)

        # 在"有匹配的 n"里取最大的（flip 之后 argmax 取第一个 True）
        best_ngram_idx = (first_match_positions >= 0).int().flip(dims=[1]).argmax(dim=1)
        best_ngram_idx = num_ngram_sizes - 1 - best_ngram_idx
        best_match_pos = first_match_positions[batch_indices, best_ngram_idx]

        has_any_match = best_match_pos >= 0
        best_ngram_lengths = ngram_lengths[best_ngram_idx]

        # 匹配点之后开始抄；没匹配时 base 取 0（后面整个 mask 成 -1）
        draft_start = torch.where(has_any_match, best_match_pos + best_ngram_lengths,
                                  torch.zeros_like(best_match_pos))
        tokens_available = seq_lengths - draft_start

        draft_indices = draft_start.unsqueeze(1) + torch.arange(num_draft_tokens,
                                                                device=device)
        draft_indices.clamp_(min=0, max=max_seq_len - 1)

        draft_tokens = torch.gather(token_ids, 1, draft_indices)

        position_indices = torch.arange(num_draft_tokens, device=device).unsqueeze(0)
        valid_positions = position_indices < tokens_available.unsqueeze(1)
        draft_tokens = torch.where(valid_positions, draft_tokens,
                                   torch.full_like(draft_tokens, -1))
        # 没匹配 → 整行 -1
        draft_tokens = torch.where(has_any_match.unsqueeze(1), draft_tokens,
                                   torch.full_like(draft_tokens, -1))
        return draft_tokens

    def forward(self, num_tokens_no_spec: torch.Tensor, token_ids_gpu: torch.Tensor,
                combined_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 `(draft_tokens [B, k], num_valid_draft_tokens [B])`（都是 GPU 张量）。"""
        device = token_ids_gpu.device
        actual_batch_size = token_ids_gpu.shape[0]

        draft_tokens = torch.full((actual_batch_size, self.k), -1, dtype=torch.int32,
                                  device=device)
        results = self._find_first_and_extract_all_n_parallel(
            token_ids_gpu, num_tokens_no_spec, min_ngram_len=self.min_n,
            max_ngram_len=self.max_n, num_draft_tokens=self.k)
        draft_tokens = torch.where(combined_mask.unsqueeze(1), results, -1)

        # 每行**前导连续**有效 token 的个数：宽度是 k，有效数可能更小
        is_valid = draft_tokens != -1
        cum_valid = is_valid.int().cumsum(dim=1)
        positions = torch.arange(1, self.k + 1, device=device).unsqueeze(0)
        num_valid_draft_tokens = (cum_valid == positions).int().sum(dim=1)
        return draft_tokens, num_valid_draft_tokens

    def load_model(self, *args, **kwargs):
        """没有模型要装（上游同名空操作）。"""
        pass


class NgramProposerGPU:
    def __init__(self, vllm_config, device: torch.device, runner=None) -> None:
        spec_config = vllm_config.speculative_config
        assert spec_config is not None
        assert spec_config.prompt_lookup_min is not None
        assert spec_config.prompt_lookup_max is not None

        self.vllm_config = vllm_config
        self.min_n = spec_config.prompt_lookup_min
        self.max_n = spec_config.prompt_lookup_max
        self.k = spec_config.num_speculative_tokens
        self.max_model_len = vllm_config.model_config.max_model_len
        self.max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        # Runner 传进来的是字符串（"cuda"/"cpu"），这里统一成带序号的 `torch.device`：
        # `propose()` 里有一条 `token_ids_gpu.device == self.device` 的断言（上游同款），
        # 而 `torch.device("cuda") != torch.device("cuda:0")`——不带序号比不过。
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.runner = runner

        self.kernel = NgramGPUKernel(vllm_config=vllm_config, prefix="ngram_gpu_kernel",
                                     device=device)
        self.kernel.to(device)
        self.kernel.eval()
        self._dummy_run()

    # -------- 预热 --------

    def _dummy_run(self) -> None:
        """按最大批跑一次（上游用来触发 torch.compile；本机用来先验形状与显存）。"""
        token_ids, num_tokens, sampled_flags, valid_mask = self._generate_dummy_data(
            batch_size=self.max_num_seqs, max_seq_len=self.max_model_len,
            pattern_len=min(self.k, self.max_model_len), device=self.device)
        combined_mask = sampled_flags & valid_mask & (num_tokens >= self.min_n)
        for _ in range(3):
            self.kernel(num_tokens, token_ids, combined_mask)

    def _generate_dummy_data(self, batch_size: int, max_seq_len: int, pattern_len: int,
                             device="cuda"):
        token_ids = torch.zeros(batch_size, max_seq_len, dtype=torch.int32, device=device)
        num_tokens = torch.randint(pattern_len, max_seq_len, (batch_size,),
                                   dtype=torch.int32, device=device)
        sampled_flags = torch.ones(batch_size, dtype=torch.bool, device=device)
        valid_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
        return token_ids, num_tokens, sampled_flags, valid_mask

    # -------- 上游接口 --------

    def propose(self, num_speculative_tokens: int, num_tokens_no_spec: torch.Tensor,
                token_ids_gpu: torch.Tensor, valid_sampled_token_ids_gpu: torch.Tensor,
                valid_sampled_tokens_count: torch.Tensor,
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """把本轮新采样的 token 写进 GPU 历史，再跑 kernel 提议。
        `num_tokens_no_spec` 是**只读**的长度输入；返回 `(drafts [B,k], num_valid [B])`。
        """
        # 上游同款断言：K 是固定值。**71 关不放开它**——动态投机长度在配置期就把
        # `ngram_gpu` 明确拒绝（GPU 提议者返回固定宽度 `[B,k]`，逐轮改宽要改 kernel 的输出
        # 形状与有效个数缓冲）。需求 071 §3.6 明确要求保留这条断言、不宣称全方法支持。
        assert num_speculative_tokens == self.k
        assert token_ids_gpu.device == self.device
        assert num_tokens_no_spec.device == self.device

        batch_size = num_tokens_no_spec.shape[0]
        max_seq_len = token_ids_gpu.shape[1]
        max_new_tokens = valid_sampled_token_ids_gpu.shape[1]

        # 1) 新采样的 token 落进历史（就地 scatter，不搬整段历史）
        offsets = torch.arange(max_new_tokens, device=self.device)
        write_positions = num_tokens_no_spec.unsqueeze(1) + offsets.unsqueeze(0)
        valid_write_mask = offsets.unsqueeze(0) < valid_sampled_tokens_count.unsqueeze(1)
        in_bounds = write_positions < max_seq_len
        scatter_mask = valid_write_mask & (valid_sampled_token_ids_gpu != -1) & in_bounds

        write_positions.clamp_(max=max_seq_len - 1)
        write_positions_long = write_positions.long()
        existing_values = token_ids_gpu.gather(1, write_positions_long)
        tokens_cast = valid_sampled_token_ids_gpu.to(token_ids_gpu.dtype)
        tokens_to_scatter = torch.where(scatter_mask, tokens_cast, existing_values)
        token_ids_gpu.scatter_(1, write_positions_long, tokens_to_scatter)

        # 2) 本次匹配用的临时长度（**只读**给 kernel；不写回权威状态）
        num_tokens_tmp = (num_tokens_no_spec + valid_sampled_tokens_count).to(torch.int32)

        sampled_flags = valid_sampled_tokens_count > 0
        valid_mask = torch.ones(batch_size, dtype=torch.bool, device=self.device)
        combined_mask = sampled_flags & valid_mask & (num_tokens_tmp >= self.min_n)

        draft_tokens, num_valid_draft_tokens = self.kernel(num_tokens_tmp, token_ids_gpu,
                                                           combined_mask)
        return draft_tokens, num_valid_draft_tokens

    def update_token_ids_ngram(self, sampled_token_ids, gpu_input_batch,
                               token_ids_gpu: torch.Tensor,
                               num_tokens_no_spec: torch.Tensor,
                               discard_request_mask: torch.Tensor,
                               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """算出本轮"有效采样 token"（上游同名方法）：丢弃的请求整行 -1，越界 id 当无效。

        返回 `(next_token_ids, valid_sampled_tokens_count, valid_sampled_token_ids_gpu)`。
        第一个是"最后一个有效 token（没有就退回历史末尾那个）"，上游用它做异步调度的占位记账；
        本机没有异步调度，所以只用到后两个——但方法按上游原样保留与返回。
        """
        num_reqs = gpu_input_batch.num_reqs

        if isinstance(sampled_token_ids, list):
            # 参差的行要先用 -1 补齐才能进张量（本机 `sample_tokens` 交回的就是 list）
            max_len = max((len(sublist) for sublist in sampled_token_ids), default=0)
            max_len = max(max_len, 1)
            padded = [list(sublist) + [-1] * (max_len - len(sublist))
                      for sublist in sampled_token_ids]
            sampled_token_ids = torch.tensor(padded, dtype=torch.int32, device=self.device)

        # 历史末尾那个 token（本轮没有有效采样时回退用它）
        backup_indices = num_tokens_no_spec[:num_reqs] - 1
        backup_indices.clamp_(min=0)
        backup_next_token_ids = torch.gather(
            token_ids_gpu[:num_reqs], dim=1,
            index=backup_indices.long().unsqueeze(1)).squeeze(1)

        valid_sampled_token_ids_gpu = sampled_token_ids.clone()
        valid_sampled_token_ids_gpu.masked_fill_(
            discard_request_mask[:num_reqs].unsqueeze(1), -1)

        valid_mask = (valid_sampled_token_ids_gpu != -1) & (
            valid_sampled_token_ids_gpu < gpu_input_batch.vocab_size)
        valid_sampled_tokens_count = valid_mask.sum(dim=1).to(torch.int32)

        last_valid_indices = valid_sampled_tokens_count - 1
        has_valid_sample = last_valid_indices >= 0
        last_valid_indices.clamp_(min=0)
        selected_tokens = torch.gather(valid_sampled_token_ids_gpu, 1,
                                       last_valid_indices.unsqueeze(1)).squeeze(1)
        next_token_ids = torch.where(has_valid_sample, selected_tokens,
                                     backup_next_token_ids)
        return next_token_ids, valid_sampled_tokens_count, valid_sampled_token_ids_gpu

    def load_model(self, *args, **kwargs):
        self.kernel.load_model(*args, **kwargs)

    # -------- 本仓库统一的提议者协议 --------

    def propose_drafts(self, rows: list[TargetRows], all_token_ids: dict[str, list[int]],
                       input_batch=None, reset_req_ids: set[str] | None = None,
                       sampled_by_row: dict[int, list[int]] | None = None,
                       sample_rows: list[int] | None = None,
                       token_ids_gpu: torch.Tensor | None = None,
                       num_tokens_no_spec_gpu: torch.Tensor | None = None,
                       ) -> DraftTokenIds:
        """Runner 入口：把本轮采样结果按行摆成 `[B, K+1]`，走 `update_token_ids_ngram` +
        `propose`，返回**带占位**的草稿（尾部 -1）与每行的有效个数。

        `sampled_by_row` 是"批行号 → 本轮采样的 token"；不在 `sample_rows`（未 ready 的中间
        prefill 块）的行整行丢弃（`discard_request_mask`，上游同款）。两个 GPU 缓冲由 Runner
        传进来（上游也是显式传参），不从这里反查 Runner。
        """
        assert token_ids_gpu is not None and num_tokens_no_spec_gpu is not None, (
            "GPU ngram 需要 Runner 的历史缓冲（token_ids_gpu / num_tokens_no_spec_gpu）")
        num_reqs = len(rows)
        sampled_by_row = sampled_by_row or {}
        sample_rows = list(range(num_reqs)) if sample_rows is None else list(sample_rows)

        # 只取**批里这 num_reqs 行**：`propose()` 里的 `write_positions`/`in_bounds` 是按长度表的
        # 行数展开的，而采样矩阵只有批行那么多。批没填满（num_reqs < max_num_reqs）时两者对不上，
        # 广播要么报错（2 ≤ num_reqs < max）要么把第 0 行的采样数据广播到空闲行上（num_reqs==1）。
        # 切片是视图 → 就地 scatter 仍然写进真正的历史缓冲。
        token_ids_gpu = token_ids_gpu[:num_reqs]
        num_tokens_no_spec_gpu = num_tokens_no_spec_gpu[:num_reqs]

        sampled = [list(sampled_by_row.get(row, [])) for row in range(num_reqs)]
        discard_mask = torch.tensor([row not in set(sample_rows) for row in range(num_reqs)],
                                    dtype=torch.bool, device=self.device)

        # 上游同款：先算出"有效采样 token"（丢弃的行整行 -1），再提议
        _next_token_ids, counts, valid_ids = self.update_token_ids_ngram(
            sampled, input_batch, token_ids_gpu, num_tokens_no_spec_gpu, discard_mask)
        drafts, num_valid = self.propose(self.k, num_tokens_no_spec_gpu, token_ids_gpu,
                                         valid_ids, counts)

        padded = drafts.cpu().tolist()
        return DraftTokenIds(req_ids=[target.req_id for target in rows],
                             draft_token_ids=padded,
                             num_valid_draft_tokens=num_valid.cpu().tolist())

    def remove_requests(self, req_ids) -> None:
        """ngram_gpu 没有按请求的状态（历史缓冲按行索引，行由 Runner 维护）→ 空操作。"""


# ---------------------------------------------------------------------------
# 调度侧对齐 / 历史增量维护（上游模块级函数，60 关接进本仓库）
# ---------------------------------------------------------------------------


def update_ngram_gpu_tensors_incremental(input_batch, token_ids_gpu_tensor,
                                         num_tokens_no_spec_gpu, new_req_ids: set[str],
                                         prev_req_id_to_index: dict[str, int] | None,
                                         device) -> None:
    """增量维护 GPU 历史（上游同名函数）。

    三件事，顺序与上游一致：

        1. **行重排**：活下来的请求换行号了（`condense` 压实 / 别的请求结束）→ 把历史整行搬过去；
        2. **新请求/恢复**：从 CPU 权威历史**整段拷一次**（只有这一次是"全量"）；
        3. **长度同步**：把所有活跃行的 `num_tokens_no_spec` 从 CPU 真值拷过去（每步都做，很小）。

    上游用 pinned 缓冲省掉每步的小分配；本机直接建索引张量（长度是 `num_reqs` 级别，不是历史长度）。
    """
    curr_req_id_to_index = input_batch.req_id_to_index
    if not curr_req_id_to_index:
        return

    active_indices = list(curr_req_id_to_index.values())
    active_idx_gpu = torch.tensor(active_indices, dtype=torch.long, device=device)

    # 第一次：还没有上一轮的行号，全部整段拷一次
    if prev_req_id_to_index is None:
        for req_id, index in curr_req_id_to_index.items():
            num_tokens = int(input_batch.num_tokens_no_spec[index])
            if num_tokens > 0:
                token_ids_gpu_tensor[index, :num_tokens].copy_(
                    input_batch.token_ids_cpu[index, :num_tokens], non_blocking=True)
        _sync_num_tokens(input_batch, num_tokens_no_spec_gpu, active_indices, active_idx_gpu,
                         device)
        return

    reorder_src: list[int] = []
    reorder_dst: list[int] = []
    for req_id, curr_index in curr_req_id_to_index.items():
        if req_id in new_req_ids:
            continue
        prev_index = prev_req_id_to_index.get(req_id)
        if prev_index is not None and prev_index != curr_index:
            reorder_src.append(prev_index)
            reorder_dst.append(curr_index)

    if reorder_src:
        src = torch.tensor(reorder_src, dtype=torch.long, device=device)
        dst = torch.tensor(reorder_dst, dtype=torch.long, device=device)
        token_ids_gpu_tensor[dst] = token_ids_gpu_tensor[src].clone()
        num_tokens_no_spec_gpu[dst] = num_tokens_no_spec_gpu[src].clone()

    # 新请求 / 抢占恢复：整段拷一次（它的历史在 CPU 镜像里是完整的）
    for req_id in new_req_ids:
        index = curr_req_id_to_index.get(req_id)
        if index is None:
            continue
        num_tokens = int(input_batch.num_tokens_no_spec[index])
        if num_tokens > 0:
            token_ids_gpu_tensor[index, :num_tokens].copy_(
                input_batch.token_ids_cpu[index, :num_tokens], non_blocking=True)

    _sync_num_tokens(input_batch, num_tokens_no_spec_gpu, active_indices, active_idx_gpu, device)


def _sync_num_tokens(input_batch, num_tokens_no_spec_gpu, active_indices: list[int],
                     active_idx_gpu: torch.Tensor, device) -> None:
    """把活跃行的长度从 CPU 真值同步到 GPU（上游同名函数）。"""
    values = input_batch.num_tokens_no_spec[torch.tensor(active_indices,
                                                         dtype=torch.long)]
    # 本机 CPU 真值是 int64、显存缓冲是 int32（上游缓冲也是 int32）→ 同步时显式转一次
    num_tokens_no_spec_gpu.index_copy_(0, active_idx_gpu,
                                       values.to(device=device, dtype=torch.int32))
