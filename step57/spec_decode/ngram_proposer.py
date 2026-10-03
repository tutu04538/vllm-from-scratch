"""`NgramProposer`：从请求**自己的历史**里找重复片段当草稿（对应 vLLM
`v1/spec_decode/ngram_proposer.py` 的 CPU 版）。

不加载第二个模型、不做任何 forward：在已有 token 里找最长的后缀匹配，把匹配位置后面的
token 抄过来当草稿。所以它是**确定性**的——没有分布 q，用 `draft_probs=None` 表示点质量提议
（验证时 `q[d] = 1`，接受判定退化成 `p[d] >= u`）。

    tokens = [a b c d a b c]        末尾是 "a b c"
    历史里上一次出现 "a b c" 后面是 "d ..." → 草稿 = [d]

两层循环的意义：`n` 从大到小（匹配越长越可信），`k` 是往后抄几个。找到就返回，
抄不满 `k` 个（撞到末尾）就少给几个——**给不满比给错好**：草稿被拒只是白算，草稿"看着像"
却总被拒才是浪费。

**与 vLLM 的差异**：vLLM 在 GPU 上用 kernel 扫输入缓冲（一个 batch 一次），本关是纯 Python
逐请求扫它的 `all_token_ids`。这属于实现后端差异，语义（找最长后缀匹配、抄后续 token）一致。
"""

from ..outputs import DraftTokenIds


class NgramProposer:
    def __init__(self, num_speculative_tokens: int, max_ngram: int = 3) -> None:
        if num_speculative_tokens <= 0:
            raise ValueError(f"num_speculative_tokens 必须为正，收到 {num_speculative_tokens}")
        self.num_speculative_tokens = num_speculative_tokens
        self.max_ngram = max_ngram

    def propose(self, req_ids: list[str], all_token_ids: dict[str, list[int]],
                num_computed_tokens: dict[str, int], input_batch=None,
                ready_req_ids: set[str] | None = None,
                reset_req_ids: set[str] | None = None) -> DraftTokenIds:
        """给每条**ready** 请求提最多 `num_speculative_tokens` 枚草稿。

        `num_computed_tokens` 是"target 本轮之后算到哪"——**匹配只看这段历史**，不看上一轮
        留下的草稿区（那部分可能刚被覆盖，掺进来会让提议依赖上一轮的运气）。中间 prefill
        块拿到的边界只是"target 本轮算完的位置"，而且它们本来就不提（见下）。

        `ready_req_ids` 是**与 draft_model 提议器统一的接口**：ngram 不写 KV，没有"同步"这
        件事，所以只对 ready 的请求提；其余请求原样返回空列表。中间 prefill 块的草稿没有可
        验证的 next token，Scheduler 也会丢掉（vLLM 的 `update_draft_token_ids` 同款规则）。
        `input_batch` / `reset_req_ids` 同理用不到（ngram 没有块表，也没有自己的进度）。
        **生命周期**：请求没被调度时什么都不留；抢占恢复不需要重置；只有"结束/abort"由
        Runner 调 `remove_requests()`——ngram 无状态，那个方法是空操作（205 §4.4）。
        """
        ready = set(req_ids) if ready_req_ids is None else set(ready_req_ids)
        draft_token_ids: list[list[int]] = []
        for req_id in req_ids:
            tokens = all_token_ids[req_id][:num_computed_tokens[req_id]]
            draft_token_ids.append(self._propose_one(tokens) if req_id in ready else [])
        return DraftTokenIds(req_ids=list(req_ids), draft_token_ids=draft_token_ids)

    def remove_requests(self, req_ids) -> None:
        """与 draft_model 提议者统一的接口：ngram 没有自己的进度与随机流，所以是空操作。

        留着它是为了**生命周期契约一致**：Runner 在"请求结束/abort"时只调这一个方法，
        不需要知道用的是哪种提议者。
        """

    def _propose_one(self, tokens: list[int]) -> list[int]:
        for ngram_size in range(min(self.max_ngram, len(tokens)), 0, -1):
            suffix = tokens[-ngram_size:]
            # 从后往前找最近一次出现（越近越可能继续重复）
            for start in range(len(tokens) - ngram_size - 1, -1, -1):
                if tokens[start:start + ngram_size] == suffix:
                    draft = tokens[start + ngram_size:
                                   start + ngram_size + self.num_speculative_tokens]
                    if draft:
                        return list(draft)
        return []
