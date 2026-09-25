"""投机解码的两个纯函数：n-gram 提议 + 目标模型验证。

这一层不认识请求对象、不认识 KV 池、不写任何状态：输入是 token 序列与目标模型的
贪心结果，输出是「提议了什么 / 接受了几枚 / 提交哪些 token / 还要保留多少 KV」。
单独成模块是为了让这几条规则**能直接单测**，不必先跑起一个引擎。

第五十二关只做单请求、贪心、n-gram 提议的最小闭环；随机采样、批量、异步提议
都不在这里（见 docs/step52_ngram_speculative.md §5）。
"""

from dataclasses import dataclass


def propose_ngram(token_ids, n, k):
    """用历史末尾 n 个 token 去找最近一次相同片段，返回它后面最多 k 个已知 token。

    `token_ids` 可以是只读视图（ReadOnlyTokenList）：全程只用下标访问，
    **不把整段历史复制成 list**——那正是第五十一关消掉的开销。只有末尾的
    模式片段和命中的续写会被切片出来，两者都不超过 max(n, k) 个。

    找不到就返回空列表（这一轮按普通 1-token 路径走）。搜索从靠近末尾的位置
    往前，第一个命中的就是**最近**的那次出现；匹配片段本身绝不与末尾的
    n 个 token 重叠（起点上界是 len-n-1），否则会自己匹配自己。

    复杂度 O(len(history) * n)，本关先要正确性：n 通常很小（默认 2），
    历史长到几万时这个线性扫描仍在微秒量级，不值得先上哈希索引。
    """
    length = len(token_ids)
    if n <= 0 or k <= 0 or length <= n:
        return []
    pattern = token_ids[length - n:length]
    for start in range(length - n - 1, -1, -1):
        for offset in range(n):
            if token_ids[start + offset] != pattern[offset]:
                break
        else:
            tail_start = start + n
            return token_ids[tail_start:min(tail_start + k, length)]
    return []


@dataclass
class DraftVerification:
    """一次投机验证的结论：全是「提交之前」就能定下来的量。

    - `num_accepted`：前几枚草稿与目标模型的贪心结果逐个相同；
    - `committed_ids`：真正要提交的输出 token，已按 EOS / 输出上限截断；
    - `kept_inputs`：本轮输入的 KV 要保留多少个（其余是必须回滚的拒绝部分）；
    - `stopped`：是否因为 EOS 或输出上限提前停下（后面不再提交草稿/bonus）。
    """
    num_accepted: int
    committed_ids: list
    kept_inputs: int
    stopped: bool


def verify_drafts(draft_ids, greedy_ids, eos_token_ids, remaining_outputs):
    """拿目标模型一次 forward 的 K+1 行贪心结果验证 K 枚草稿。

    输入 `[x, d0..d_{K-1}]` 的三段结论（需求 §1 的表）：

        t0 != d0            提交 [t0]            输入只留 [x]
        t0 == d0, t1 != d1  提交 [d0, t1]        输入留 [x, d0]
        全部接受            提交 [d0.., tK]       输入留 [x, d0..]

    最后一枚 `tK` 是全部接受时的 bonus token——它**还没进过模型**，所以永远
    不在「要保留的输入」里；同理，被拒绝的草稿的 KV 必须回滚。

    被接受的草稿本身若是终止 token（EOS），它**之前的**草稿才需要留作下一轮
    输入：终止 token 结束这条请求，不会再进模型。
    """
    num_drafts = len(draft_ids)
    if len(greedy_ids) != num_drafts + 1:
        raise ValueError(f"验证需要 {num_drafts + 1} 行贪心结果（K 枚草稿 + 1 枚 bonus），"
                         f"收到 {len(greedy_ids)} 行")

    num_accepted = 0
    while num_accepted < num_drafts and greedy_ids[num_accepted] == draft_ids[num_accepted]:
        num_accepted += 1
    # 全部接受时 greedy_ids[num_accepted] 就是 bonus
    candidates = list(draft_ids[:num_accepted]) + [greedy_ids[num_accepted]]

    committed_ids, stopped, kept_drafts = [], False, num_accepted
    for index, token_id in enumerate(candidates):
        if len(committed_ids) >= remaining_outputs:
            stopped = True
            break
        committed_ids.append(token_id)
        if token_id in eos_token_ids:
            stopped = True
            if index < num_accepted:
                # 终止的就是草稿本身：它不必再作为下一轮的输入，留它前面的那些
                kept_drafts = index
            break
    # 本轮输入恒以最后一个真实 token x 开头，它一定还要用
    return DraftVerification(num_accepted, committed_ids, 1 + kept_drafts, stopped)
