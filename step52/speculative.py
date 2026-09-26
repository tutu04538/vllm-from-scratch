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
    - `committed_ids`：真正要提交的输出 token（遇到 EOS 就到此为止）；
    - `kept_inputs`：本轮输入的 KV 要保留多少个（其余是必须回滚的拒绝部分）；
    - `stopped`：是否因为 EOS 提前停下（后面不再提交草稿/bonus）。
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

    三种情况在代码里是**同一个式子**：提交 `draft_ids[:a] + [greedy_ids[a]]`，
    `a` 是逐枚比对下来接受了几枚。别把它读成「全接受时多给一枚 bonus、部分接受时
    换成纠正 token」两条分支——`greedy_ids[a]` 两种情况是同一枚东西：目标模型对
    位置 `p+a+1` 的预测，而且**都还没进过模型**（`a == K` 时那个位置压根没有输入行；
    `a < K` 时那个位置是刚被否掉的 `d_a`，它的 KV 马上要被回滚）。两种情况它都会
    成为下一轮的「最后一个真实 token」`x`。

    要分情况的是**回滚到哪儿**，也就是「本轮输入里还有哪些要留」：

    - `a == K`：整段输入都留，`kept_inputs = 1+K`——模型没被问过的位置无需回滚；
    - `a < K`：从 `d_a` 起的输入行连同它们的 KV 一起丢掉，`kept_inputs = 1+a`。

    被接受的草稿本身若是终止 token（EOS），它**之前的**草稿才需要留作下一轮
    输入：终止 token 结束这条请求，不会再进模型。

    `remaining_outputs`（这条请求还能再提交几个 token）是**前置条件**，不是截断阈值：
    函数要求 `K <= R-1`（`_plan_drafts()` 正是这么缩 K 的）。这里不做截断，
    因为截断出来的状态自相矛盾——`kept_inputs` 按接受的草稿数算，会比实际提交的
    token 数还多，于是 `cache.length` 要么**超过** `len(all_token_ids)`（KV 进度
    比历史还长），要么**正好相等**（就是「历史已算完却还不是 ready」那个状态，
    §1.8 刚把它的兜底删掉）。与其默默产出这种状态，不如直接报错。
    """
    num_drafts = len(draft_ids)
    if len(greedy_ids) != num_drafts + 1:
        raise ValueError(f"验证需要 {num_drafts + 1} 行贪心结果（K 枚草稿 + 1 枚 bonus），"
                         f"收到 {len(greedy_ids)} 行")
    if num_drafts > remaining_outputs - 1:
        raise ValueError(
            f"{num_drafts} 枚草稿要占 {num_drafts + 1} 个输出额度，但只剩 {remaining_outputs} 个；"
            f"调用方必须保证 K <= R-1（_plan_drafts() 就是这么缩的）")

    num_accepted = 0
    while num_accepted < num_drafts and greedy_ids[num_accepted] == draft_ids[num_accepted]:
        num_accepted += 1
    candidates = list(draft_ids[:num_accepted]) + [greedy_ids[num_accepted]]

    committed_ids, stopped, kept_drafts = [], False, num_accepted
    for index, token_id in enumerate(candidates):
        committed_ids.append(token_id)
        if token_id in eos_token_ids:
            stopped = True
            # 终止 token 的 KV 不必留作下一轮输入，保留它**之前**那些——要留几枚
            # 恰好就是 index。这里的两种情况都不用另写分支：
            #   index <  num_accepted：终止的是草稿 d_index，它的 KV 要退掉；
            #   index == num_accepted：终止的是 bonus（模型自己给的那枚），它从来
            #     没作为输入行进过模型，压根没有 KV 可退——而这时 index 正好就等于
            #     初值 num_accepted。
            kept_drafts = index
            break
    # 本轮输入恒以最后一个真实 token x 开头，它一定还要用
    return DraftVerification(num_accepted, committed_ids, 1 + kept_drafts, stopped)
