"""DSpark 串行 Markov 采样的纯 Torch 参考实现（**只给测试用**；生产路径不 import）。

为什么需要一份"冻"的参考：DSpark 有两条**不报错但会错**的路径——

    1. **条件分布**：第 i 步的 logits 必须是 `base_logits[:, i] + bias(上一步实际选中的 token)`。
       把 bias 丢掉（= 退回 DFlash 的独立采样）照样能跑、草稿照样合法，只是 q 与真实提议分布
       不一致 → 拒绝采样不再无损。这类错只能靠"逐值复算同一份数学"抓。
    2. **缩减词表 / top-k 的 q 宽度**：q 必须是 target 宽度、且**截断后**的那一份。少散射一列、
       或者让不存在的列继承上一轮的值，都不会报错（内核只断言 ndim）。

参考实现**不复制权重**：调用方把模型里的 `markov_w1` / `markov_w2` 权重直接交进来，所以它验的是
"生产路径的偏置 / 截断 / 散射顺序"，不是两套独立实现的数值巧合。

随机性由调用方提供：参考实现不自己抽 Gumbel（Triton 的 `tl.rand` 无法在 Torch 里逐位复现），
而是收一个 `choose(logits_step, step) -> draft 空间的 id` 回调。于是：

    greedy        `choose = argmax`（与生产路径的 greedy 分支逐位相同）
    probabilistic `choose = 生产路径**实际采出**的 id`（参考实现据此复算下一步的条件分布，
                  再去比"缓存下来的 q"——这正是需求 077 §4 第一条要钉的东西）
"""

import torch


def _scatter_to_target(logits: torch.Tensor, d2t: torch.Tensor | None,
                       target_vocab: int | None) -> torch.Tensor:
    """draft 空间的 logits → target 空间的一份新张量（其余列 `-inf`）。

    生产路径把这件事放在 `_draft_scatter_buf` 上（整块 `-inf` 只填一次、每步覆盖 draft 列）；
    参考实现每次都新建一份 —— 语义相同：**不存在的列永远是 -inf**。
    """
    if d2t is None:
        return logits
    assert target_vocab is not None
    out = logits.new_full((logits.shape[0], target_vocab), float("-inf"))
    out[:, torch.arange(d2t.shape[0], device=logits.device) + d2t] = logits
    return out


def reference_sequential(
    base_logits: torch.Tensor,
    anchor_tokens: torch.Tensor,
    markov_w1: torch.Tensor,
    markov_w2: torch.Tensor,
    choose,
    *,
    logit_scale: float = 1.0,
    topk: int | None = None,
    d2t: torch.Tensor | None = None,
    target_vocab: int | None = None,
):
    """DSpark 的串行 Markov 采样（参考实现）。

    参数（与生产路径的字段一一对应）：

        base_logits      `[B, K, V_draft]`，**已经过 logit_scale 的**主干 logits
                         （生产里是 `compute_draft_logits(hidden).view(B, K, V)`）
        anchor_tokens    `[B]`，每请求 step 0 的"前一个 token"（= anchor 行喂进去的 bonus token）
        markov_w1        `[V_target, r]`；`markov_w2`  `[V_draft, r]`
        choose           `(logits_step [B, V_draft], step) -> [B] draft 空间的 id`
        topk             只对 base logits 的 top-k 候选求偏置（其余 `-inf`）
        d2t              `[V_draft]` 偏移表（`target = draft + d2t[draft]`）；None = 全词表
        target_vocab     q 的宽度（缩减词表时必须给）

    返回 `(draft_tokens [B, K]（target 空间）, q_logits [B, K, V_target])`。
    `q_logits` 是"这一步实际用于采样的 logits"（= 生产路径缓存进 `draft_logits` 的那一份，
    温度缩放之前）。greedy 路径生产不缓存 q，调用方自行忽略。
    """
    batch, num_steps, vocab = base_logits.shape
    if topk is not None:
        base_values, cand_index = base_logits.topk(topk, dim=-1)

    prev = anchor_tokens
    tokens = []
    q_logits = []
    for step in range(num_steps):
        # Markov 偏置：只看"上一步实际选中的那个 token"（权重按 **target** id 取行）。
        markov_embed = markov_w1[prev]
        bias = logit_scale * (markov_embed @ markov_w2.t())
        if topk is None:
            logits_step = base_logits[:, step] + bias
        else:
            # 候选只选一次；每步只给候选列加偏置，其余列保持 -inf（= 生产路径的截断分布）。
            logits_step = torch.full((batch, vocab), float("-inf"),
                                     dtype=base_logits.dtype, device=base_logits.device)
            index = cand_index[:, step]
            corrected = base_values[:, step] + bias.gather(1, index)
            logits_step.scatter_(1, index, corrected)
        q_logits.append(_scatter_to_target(logits_step, d2t, target_vocab))
        draft_ids = choose(logits_step, step)
        target_ids = (draft_ids + d2t[draft_ids]) if d2t is not None else draft_ids
        tokens.append(target_ids)
        prev = target_ids
    return torch.stack(tokens, dim=1), torch.stack(q_logits, dim=1)
