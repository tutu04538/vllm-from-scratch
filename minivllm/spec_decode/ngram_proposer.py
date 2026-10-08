"""`NgramProposer`：从请求**自己的历史**里找重复片段当草稿（对应 vLLM
`v1/spec_decode/ngram_proposer.py`，CPU 版）。

不加载第二个模型、不做任何 forward：在已有 token 里找**最长**的后缀 ngram 匹配（匹配长度限定在
`[prompt_lookup_min, prompt_lookup_max]`），把匹配位置后面的 token 抄来当草稿。所以它是
**确定性**的——没有分布 q，用 `draft_probs=None` 表示点质量提议（验证时 `q[d] = 1`，
接受判定退化成 `p[d] >= u`）。

    tokens = [a b c d a b c]        末尾是 "a b c"
    历史里上一次出现 "a b c" 后面是 "d ..." → 草稿 = [d]

### 匹配规则（照源码；60 关校准的四条）

1. **最长优先**：`n` 从 `prompt_lookup_max` 往下找，取第一个有匹配的 n（等价于上游的反转 +
   KMP 里维护 `longest_ngram`）。
2. **同长度多处匹配取"最早"那处**：上游 `prev_lps >= longest_ngram` 时持续覆盖 `position`，
   拿到的是**原序列里最早**的出现位置（源码注释：*"we want to get the target n-gram of the
   earliest position in the original tokens"*）。这和"从后往前找最近一次"**不是**一回事，
   候选会不同（60 关之前本机就是后者）。
3. **`k` 的两个上限**：`k = min(k, max_model_len - total_token)`（不提超出上下文上限的草稿），
   再 `k = min(k, total_token - start_position)`（匹配点后面有多少就抄多少：给不满比给错好）。
4. **不匹配就不提**：`total_token < min_n` → 空；`longest_ngram < min_n` → 空。

### 与上游的差异（记在 `docs/step60_alignment.md`）

- 上游 `batch_propose` 用 numba（`@njit(parallel=True)`）多线程扫整个 batch；本机不引入 numba
  JIT 依赖，这里是**同一个匹配函数的逐请求循环**，语义逐值一致（测试与
  `benchmarks/check_step60_ngram.py` 里直接拿上游冻结的
  `_find_longest_matched_ngram_and_propose_tokens` / `batch_propose_numba` 做差分）。
- 上游 `propose()` 收 numpy 缓冲（`num_tokens_no_spec` / `token_ids_cpu`）+ 每行的采样结果；
  本机保留同名同签名的方法，另外给一个 `propose_drafts()` 适配本仓库统一的提议者协议
  （`rows` + `all_token_ids`），两个入口走同一套匹配代码。
"""

import numpy as np

from ..outputs import DraftTokenIds
from .utils import TargetRows


def _find_longest_matched_ngram_and_propose_tokens(
    origin_tokens: np.ndarray,
    min_ngram: int,
    max_ngram: int,
    max_model_len: int,
    k: int,
) -> np.ndarray:
    """上游同名函数（逐行照抄）：找 `[min_ngram, max_ngram]` 内最长的后缀匹配，抄后面 k 个。

    做法是"把序列反转 + KMP 的 lps（最长真前缀=后缀）数组"：反转之后"后缀匹配"变成
    "前缀匹配"，于是可以线性扫一遍。
    """
    # 历史比最短 ngram 还短 → 不提
    total_token = origin_tokens.shape[0]
    if total_token < min_ngram:
        return np.empty((0,), dtype=origin_tokens.dtype)

    # 不提议超出模型长度上限的草稿
    k = min(k, max_model_len - total_token)
    if k <= 0:
        return np.empty((0,), dtype=origin_tokens.dtype)

    tokens = origin_tokens[::-1]

    # lps[i] = max{v : tokens[0:v] == tokens[i+1-v:i+1]}
    # ngram 被 max_ngram 封顶，所以只存前 max_ngram 个前缀的 lps
    lps = np.zeros(max_ngram, dtype=np.int32)

    longest_ngram = 0
    position = 0

    prev_lps = 0
    i = 1
    while i < total_token:
        if tokens[prev_lps] == tokens[i]:
            prev_lps += 1
            # 不短于当前最长就更新 position：`>=` 让"反转序列里更靠后"（= 原序列里**更早**）
            # 的那处胜出——这是上游的 tie 规则，不能改成 `>`
            if prev_lps >= longest_ngram:
                longest_ngram = prev_lps
                position = i
            if i < max_ngram:
                lps[i] = prev_lps
            if prev_lps == max_ngram:
                # 达到上限就退回次长前缀，避免匹配到比 max_ngram 更长的 ngram
                prev_lps = lps[max_ngram - 1]
            i += 1
        elif prev_lps != 0:
            prev_lps = lps[prev_lps - 1]
        else:
            i += 1

    if longest_ngram < min_ngram:
        return np.empty((0,), dtype=origin_tokens.dtype)

    # 位置翻回原序列：匹配段是 [total-1-position, total-1-position+longest_ngram)
    start_position = total_token - 1 - position + longest_ngram
    k = min(k, total_token - start_position)
    return origin_tokens[start_position:start_position + k]


class NgramProposer:
    def __init__(self, vllm_config) -> None:
        """与上游同签名：参数全部从配置里取。"""
        spec_config = vllm_config.speculative_config
        assert spec_config is not None
        assert spec_config.prompt_lookup_min is not None
        assert spec_config.prompt_lookup_max is not None

        self.min_n = spec_config.prompt_lookup_min
        self.max_n = spec_config.prompt_lookup_max
        # 匹配点后面抄几枚（不足就少给）
        self.k = spec_config.num_speculative_tokens
        self.max_model_len = vllm_config.model_config.max_model_len

        # 上游为 numba 版预分配的结果缓冲；本机没有 numba，但保留同样的缓冲与
        # `batch_propose` 签名，方便逐行对照
        max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        self.valid_ngram_draft = np.zeros((max_num_seqs, self.k), dtype=np.int32)
        self.valid_ngram_num_drafts = np.zeros(max_num_seqs, dtype=np.int32)

    # -------- 上游接口 --------

    def propose(
        self,
        num_speculative_tokens: int,
        sampled_token_ids: list[list[int]],
        num_tokens_no_spec: np.ndarray,
        token_ids_cpu: np.ndarray,
        slot_mappings=None,            # 上游参数：ngram 用不到
    ) -> list[list[int]]:
        """上游同名方法：挑出本轮要提草稿的请求，再交给 `batch_propose`。

        两条跳过规则照源码：
        - 本轮**没采样出 token** 的请求跳过（中间 prefill 块：没有可验证的 next token）；
        - 已经到 `max_model_len` 的请求跳过（再给草稿也放不下）。
        """
        assert num_speculative_tokens <= self.k

        valid_ngram_requests = []
        for i, sampled_ids in enumerate(sampled_token_ids):
            if not len(sampled_ids):
                continue
            if num_tokens_no_spec[i] >= self.max_model_len:
                continue
            valid_ngram_requests.append(i)

        return self.batch_propose(len(sampled_token_ids), valid_ngram_requests,
                                  num_tokens_no_spec, token_ids_cpu,
                                  num_speculative_tokens)

    def batch_propose(
        self,
        num_requests: int,
        valid_ngram_requests: list,
        num_tokens_no_spec: np.ndarray,
        token_ids_cpu: np.ndarray,
        k: int,
    ) -> list[list[int]]:
        """上游同名方法（上游在这个函数里调 numba 并行版；本机是逐请求循环）。

        `token_ids_cpu` 是 `[batch, max_model_len]` 的历史缓冲，`num_tokens_no_spec[i]` 是第 i 行
        的有效长度；只有 `valid_ngram_requests` 里的行会被扫。
        """
        if len(valid_ngram_requests):
            self.valid_ngram_draft[:] = 0
            self.valid_ngram_num_drafts[:] = 0
            for i in valid_ngram_requests:
                num_tokens = int(num_tokens_no_spec[i])
                draft = _find_longest_matched_ngram_and_propose_tokens(
                    token_ids_cpu[i, :num_tokens], self.min_n, self.max_n,
                    self.max_model_len, k)
                self.valid_ngram_num_drafts[i] = len(draft)
                if len(draft):
                    self.valid_ngram_draft[i, :len(draft)] = draft

        draft_token_ids: list[list[int]] = []
        for i in range(num_requests):
            if i in valid_ngram_requests and self.valid_ngram_num_drafts[i] > 0:
                draft_token_ids.append(
                    self.valid_ngram_draft[i, : self.valid_ngram_num_drafts[i]].tolist())
            else:
                draft_token_ids.append([])
        return draft_token_ids

    def load_model(self, *args, **kwargs):
        """没有模型要装（上游同名空操作）。"""
        pass

    # -------- 本仓库统一的提议者协议 --------

    def propose_drafts(self, rows: list[TargetRows], all_token_ids: dict[str, list[int]],
                       input_batch=None,
                       reset_req_ids: set[str] | None = None,
                       num_speculative_tokens: int | None = None) -> DraftTokenIds:
        """Runner 的入口（与 draft_model 提议者同一套协议）。

        `rows` 的顺序就是**批行序**，第 i 项对应第 i 行；`input_batch` 给了就直接用它的
        `token_ids_cpu` / `num_tokens_no_spec` 两个缓冲（**切片是视图，不拷贝历史**，与上游
        传 `token_ids_cpu` 一样）。`rows` 里的 `history_end` 就是上游的 `num_tokens_no_spec`
        （本轮新采样的 token 已经记进历史、**不含**草稿），`ready` 就是"这一轮采样出了
        token"（中间 prefill 块不是 ready → 上游会跳过）。匹配只看 `[:history_end]`，不看上一轮
        遗留的草稿区——那部分可能刚被覆盖。

        71 关（动态投机长度）：`num_speculative_tokens` 是本轮要提几枚（`None` = 配置的 K）。
        ngram **不跑模型、不写 KV**，所以 K=0 就是"直接返回空草稿"——没有第一遍要同步
        （这与 `draft_model`/EAGLE 的 K=0 语义不同，需求 071 §3.4 的"仍要跑第一遍"只针对
        写 KV 的那几类提议者）。上游允许的 K 上界是 `self.k`（`assert K <= self.k`）。
        """
        if num_speculative_tokens is None:
            num_speculative_tokens = self.k
        if not 0 <= num_speculative_tokens <= self.k:
            raise ValueError(
                f"本轮 K={num_speculative_tokens} 超出 [0, {self.k}]：ngram 的草稿缓冲按最大 K "
                f"开，逐轮 K 只能是它的前缀（上游同一处是 `assert K <= self.k`）")
        if input_batch is not None:
            token_ids = input_batch.token_ids_cpu.numpy()          # [max_num_reqs, max_model_len]
            num_tokens_no_spec = input_batch.num_tokens_no_spec.numpy()
        else:
            # 不接批缓冲的调用方（单测）：按 `all_token_ids` 现场拼一份同样形状的数组
            token_ids = np.zeros((len(rows), self.max_model_len), dtype=np.int32)
            num_tokens_no_spec = np.zeros(len(rows), dtype=np.int32)
            for index, target in enumerate(rows):
                tokens = all_token_ids[target.req_id][:target.history_end]
                length = min(len(tokens), self.max_model_len)
                token_ids[index, :length] = tokens[:length]
                num_tokens_no_spec[index] = length

        sampled_token_ids: list[list[int]] = [[] for _ in rows]
        for index, target in enumerate(rows):
            if target.ready:
                # 只用来标记"本轮采样过"（内容不参与匹配，匹配只看 token_ids_cpu）
                sampled_token_ids[index] = [0]

        drafts = self.propose(num_speculative_tokens, sampled_token_ids,
                              num_tokens_no_spec, token_ids)
        return DraftTokenIds(req_ids=[target.req_id for target in rows],
                             draft_token_ids=drafts)

    def remove_requests(self, req_ids) -> None:
        """与 draft_model 提议者统一的接口：ngram 没有自己的进度与随机流，所以是空操作。

        留着它是为了**生命周期契约一致**：Runner 在"请求结束/abort"时只调这一个方法，
        不需要知道用的是哪种提议者。
        """
