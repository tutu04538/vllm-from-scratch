"""Suffix Decoding 提议者（需求 61）：接入 Arctic Inference 的官方实现。

对应上游 `vllm/v1/spec_decode/suffix_decoding.py`（本机 0.28.0，103 行）——**逐行对齐**：
上游自己不写后缀树，而是 import `arctic_inference.suffix_decoding.SuffixDecodingCache`
（Snowflake ArcticInference，论文 https://arxiv.org/pdf/2411.04975 的官方实现），
本关照做，只是把"传进去的 CPU 缓冲"从 vLLM 的 numpy 视图换成我们 InputBatch 的 torch 视图
（转换方式见 `_as_int32`，**不改变任何算法与候选**）。

它解决什么痛点（相对 ngram 提议者）：
    1. **跨请求复用**：ngram 只在**当前这条请求的历史**里找匹配。两条请求各自出现 "1 2 3 → 9 9"
       时，A 学到的 "1 2 3 后面跟 9 9" 对 B 毫无帮助；suffix decoding 有一条**全局树**，
       A 的输出会被写进去，B 只要结尾也是 "1 2 3" 就能直接猜 "9 9"。
    2. **候选长度动态**：ngram 每条请求固定猜 K 枚（不够就用 -1 补齐）；这里按"匹配到的后缀
       有多长、这个分支历史上出现过多少次"决定猜几枚（`max_spec_factor`/`min_token_prob`），
       匹配短就少猜、匹配长就多猜，返回的每条请求长度可以不同。

三个坐标系与 ngram 提议者完全一致：
    - `input_batch.req_ids[i]` / `req_id_to_index[req_id]`：请求身份 ↔ 批行
    - `token_ids_cpu[row, :]`：`[prompt][已提交输出][上轮草稿区]`，`num_tokens_no_spec[row]` 之后是草稿区
    - 返回 `list[list[int]]`：第 i 项就是批第 i 行的草稿（**不定长**，可以为空）

与上游的两点工程差异（都不改候选，逐条记录在 docs/step61_alignment.md §3）：
    a) 上游的 `token_ids_cpu`/`num_tokens_no_spec`/`num_prompt_tokens` 是 **int32 numpy 视图**，
       我们的是 **int64 torch tensor**，所以切片要转成连续 int32 ndarray
       （依赖包对 ndarray 只接受 1 维/连续/int32，否则退化到逐元素解析、慢且容易混类型）。
    b) 上游靠 `propose()` 末尾那次 `active_requests - req_id_to_index.keys()` 扫描来收尾；
       我们额外提供 `remove_requests()`（Runner 在请求**结束**时的统一入口），
       两者合起来才是完整生命周期（见 `remove_requests` 的说明）。
"""

import numpy as np
import torch

from ..outputs import DraftTokenIds
from .utils import TargetRows


def _as_int32(values) -> np.ndarray:
    """把 token 序列转成依赖包唯一接受的那种数组：1 维、连续、int32。

    上游传的是 `token_ids_cpu[index, :n]`——vLLM 里那本身就是 int32 numpy 视图。
    我们这里传 torch（int64）或 list，所以显式转换；`np.ascontiguousarray` 保证
    依赖包走的是"零拷贝 ndarray"重载，而不是逐元素解析。
    """
    if isinstance(values, torch.Tensor):
        values = values.numpy()
    array = np.asarray(values)
    return np.ascontiguousarray(array, dtype=np.int32)


class SuffixDecodingProposer:
    """Suffix Decoding 提议者（上游同名类）。

    与上游一致：`__init__` 里**懒加载**依赖包，`load_model()` 是空操作（没有模型要装）。
    """

    def __init__(self, vllm_config):
        config = vllm_config.speculative_config
        assert config is not None, "Speculative config must be set"
        self.num_speculative_tokens = config.num_speculative_tokens
        self.max_tree_depth = config.suffix_decoding_max_tree_depth
        self.max_spec_factor = config.suffix_decoding_max_spec_factor
        self.min_token_prob = config.suffix_decoding_min_token_prob
        self.max_model_len = vllm_config.model_config.max_model_len

        # Lazy import to avoid error when Suffix Decoding is not used.
        from arctic_inference.suffix_decoding import SuffixDecodingCache

        # 缓存对象负责：保存每条请求的输出、FIFO 淘汰旧请求、维护每条的 prompt 树。
        self.suffix_cache = SuffixDecodingCache(
            max_tree_depth=config.suffix_decoding_max_tree_depth,
            max_cached_requests=config.suffix_decoding_max_cached_requests,
        )

    # ------------------------------------------------------------------
    # 上游签名（逐行对齐）
    # ------------------------------------------------------------------

    def propose(
        self,
        num_speculative_tokens: int,
        input_batch,
        sampled_token_ids: list[list[int]],
        slot_mappings: dict[str, torch.Tensor]
        | list[dict[str, torch.Tensor]]
        | None = None,  # unused
    ) -> list[list[int]]:
        """为输入批里的每条请求提草稿（上游 `propose` 的六步顺序原样保留）。

        `sampled_token_ids[i]` 是**批第 i 行**本轮采出的 token：
            - 空列表 = 这一行本轮没采（中间 prefill 块 / 没被调度）→ 跳过，什么都不记；
            - 非空 = target 已经确认这一行的输出，先追加进树，再拿它去找后缀匹配。
        返回的 `draft_token_ids[i]` 是第 i 行的草稿，**长度不定、可以为空**。
        """
        assert num_speculative_tokens == self.num_speculative_tokens
        draft_token_ids: list[list[int]] = []
        for i, sampled_ids in enumerate(sampled_token_ids):
            if not sampled_ids:
                # 中间 prefill 块：没有已确认的输出可记，也没什么可猜
                draft_token_ids.append([])
                continue

            req_id = input_batch.req_ids[i]
            num_tokens = int(input_batch.num_tokens_no_spec[i])
            if num_tokens >= self.max_model_len:
                # 已经到上下文上限：再猜也没地方放（下一轮必然被截断）
                draft_token_ids.append([])
                continue

            index = input_batch.req_id_to_index[req_id]
            if req_id not in self.suffix_cache.active_requests:
                if req_id in self.suffix_cache.cached_requests:
                    # 同 ID 复用（旧请求结束过、全局树里还留着它的响应）：先清掉旧响应，
                    # 否则新请求会从上一轮的输出里猜出"看起来合理但与本 prompt 无关"的东西
                    self.suffix_cache.evict_cached_response(req_id)
                num_prompt_tokens = int(input_batch.num_prompt_tokens[index])
                prompt_token_ids = _as_int32(
                    input_batch.token_ids_cpu[index, :num_prompt_tokens])
                # 建这轮的 prompt 树（prompt 全量入树）
                self.suffix_cache.start_request(req_id, prompt_token_ids)

            # 本轮新确认的输出追加进（本地 + 若仍被缓存则全局）树。**一轮只追加一次**：
            # 这条路径由"本轮采样非空"把关，所以中间 prefill 块不会重复追加。
            self.suffix_cache.add_active_response(req_id, list(sampled_ids))

            # 只拿最近 max_tree_depth 个 token 当匹配模式（树本身也只有这么深）
            start = max(0, num_tokens - self.max_tree_depth)
            pattern = _as_int32(input_batch.token_ids_cpu[i, start:num_tokens])
            draft = self.suffix_cache.speculate(
                req_id,
                pattern,
                max_spec_tokens=min(
                    self.num_speculative_tokens, self.max_model_len - num_tokens - 1
                ),
                max_spec_factor=self.max_spec_factor,
                min_token_prob=self.min_token_prob,
            )

            draft_token_ids.append(draft.token_ids)

        # 收尾：本批**没出现**的活跃请求不再活跃（被抢占、暂停、或已经结束）。
        # 注意语义：这里只清"本地 prompt 树 + 活跃标记"，已经进全局树的响应**不删**
        # （请求重新被调度时会走上面 cached → evict → start 的分支）。
        for req_id in (
            self.suffix_cache.active_requests - input_batch.req_id_to_index.keys()
        ):
            self.suffix_cache.stop_request(req_id)

        return draft_token_ids

    def load_model(self, *args, **kwargs):
        # No model to load.
        pass

    # ------------------------------------------------------------------
    # 本仓库的 Runner 协议适配（与 ngram / draft_model 提议者同一套入口）
    # ------------------------------------------------------------------

    def propose_drafts(self, rows: list[TargetRows], all_token_ids,
                       input_batch=None, reset_req_ids: set[str] | None = None,
                       sampled_by_row: dict[int, list[int]] | None = None,
                       sample_rows=None) -> DraftTokenIds:
        """Runner 的入口：把"逐行的事实"翻译成上游 `propose` 的入参。

        `rows` 的顺序就是**批行序**（Runner 按 `input_batch.req_ids` 生成），所以这里
        直接用 `input_batch` 的缓冲与 `req_ids`，只把 `sampled_by_row`（`{批行: 采样结果}`）
        摊成上游要的"逐行 list"：没采样的行给空列表，`propose` 会跳过它们。

        `input_batch` 必须给：suffix decoding 的输入就是 InputBatch 的
        `token_ids_cpu` / `num_tokens_no_spec` / `num_prompt_tokens`，不像 ngram 那样能从
        `all_token_ids` 现场拼一份出来（拼出来的 `num_prompt_tokens` 会与真实 prompt 边界
        不一致，那会改变建树范围 → 改变候选，属于静默偏离）。
        `reset_req_ids` / `sample_rows` 在本提议者里用不到（上游同款：`slot_mappings` 也是 unused）。
        """
        if input_batch is None:
            raise ValueError(
                "suffix decoding 的 proposer 需要 input_batch（上游同样从 InputBatch 取 "
                "token_ids_cpu/num_tokens_no_spec/num_prompt_tokens）；"
                "只有 ngram 提议者支持从 all_token_ids 现场拼缓冲")
        num_reqs = len(input_batch.req_ids)
        sampled_token_ids: list[list[int]] = [[] for _ in range(num_reqs)]
        for row, tokens in (sampled_by_row or {}).items():
            if row < num_reqs:
                sampled_token_ids[row] = list(tokens)
        drafts = self.propose(self.num_speculative_tokens, input_batch, sampled_token_ids)
        return DraftTokenIds(req_ids=list(input_batch.req_ids), draft_token_ids=drafts)

    def remove_requests(self, req_ids) -> None:
        """请求**结束/abort**（Runner 的统一生命周期入口）：把它从活跃集合里摘掉。

        为什么 `propose` 末尾那次扫描不够：请求结束时 Runner 会调 `remove_requests`，
        但真正"把树清掉"必须发生在**下一轮**——因为结束的那一轮 Scheduler 可能已经
        不再把它放进 batch（扫描会处理），也可能还有本轮的草稿要收（不能提前删）。
        所以这里只是把已知结束的 ID 也用同一个 `stop_request` 摘掉：
        `stop_request` 只对**活跃**请求生效（不活跃时依赖包会抛 KeyError），因此这里先判活跃。
        重复调用安全：已经停过的 ID 不再活跃，跳过即可。
        """
        for req_id in req_ids:
            if req_id in self.suffix_cache.active_requests:
                self.suffix_cache.stop_request(req_id)
