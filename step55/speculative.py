"""投机解码的纯函数：n-gram 提议 + 两种目标模型验证（贪心 / 随机）。

这一层不认识请求对象、不认识 KV 池、不写任何状态：输入是 token 序列、目标模型
每行的**目标分布**（或贪心结果），输出是「接受了几枚 / 提交哪些 token / 还要保留
多少 KV」。单独成模块是为了让这几条规则**能直接单测**，不必先跑起一个引擎——
随机接受与纠正抽样还允许注入 `draw_uniform` / `draw_token` 两个回调，
分支用例可以完全确定地复现（见 benchmarks/check_step54_rejection.py）。

两条验证路径：

- `verify_drafts()`        贪心、无惩罚项。目标分布退化成 one-hot，接受与否就是
                           「argmax 是否等于草稿」，不需要随机数。
- `verify_drafts_random()` 一般情况：按目标分布 p 做拒绝采样。n-gram 是确定性提议，
                           所以 q(d)=1，接受概率就是 `min(1, p[d]/q[d]) = p[d]`，
                           不需要构造整张 q 矩阵。
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

    - `num_accepted`：前几枚草稿与目标模型的贪心结果逐个相同。**这个推不出来**：
      `committed_ids` 以 EOS 结尾时，「EOS 是草稿本身」与「EOS 是 bonus」会给出
      同样的 `(committed_ids, kept_inputs)`——比如接受 1 枚、bonus 是 EOS，与接受
      2 枚、第二枚草稿是 EOS，两者的可见结果完全一样；
    - `committed_ids`：真正要提交的输出 token；遇到 EOS 就到此为止，所以
      `committed_ids[-1] in eos_token_ids` 就是「因 EOS 停下」这个信号本身，
      不必再单开一个字段；
    - `kept_inputs`：本轮输入的 KV 要保留多少个（其余是必须回滚的拒绝部分）。
    """
    num_accepted: int
    committed_ids: list
    kept_inputs: int


def residual_probs(probs, q, token_id):
    """拒绝之后要用的**纠正分布**：`normalize(max(p - q, 0))`。

    `q` 是**抽出这枚草稿时用的实际提议分布**（一般 draft model 就传它）。为什么必须
    是 `max(p-q, 0)`：拒绝采样要保证「最终提交的第一个 token 服从 p」。按提议分布 q
    抽到 y 的概率是 `q[y]`，它以 `min(1, p[y]/q[y])` 被接受；被拒的那部分质量
    `Σ_x q[x]·(1 - min(1,p[x]/q[x]))` 全部落到纠正分布上。于是

        提交 y 的概率 = q[y]·min(1, p[y]/q[y]) + 拒绝质量 · residual[y]
                      = min(q[y], p[y]) + max(p[y] - q[y], 0) = p[y]

    ——两个 min/max 刚好互补。**换成别的纠正分布这条等式就断了**：第五十二到五十四关
    的 n-gram 是确定性提议（q 是 token 上的 one-hot），那时 `max(p-q,0)` 退化成
    「把该 token 挖掉再归一化」；对一般 q 继续只挖掉一个 token，得到的就不是 p 了
    （p=[0.6,0.3,0.1]、q=[0.2,0.5,0.3]、草稿 1 时，正确纠正分布是 [1,0,0]，
    只挖 token 1 会得到 [6/7,0,1/7]）。

    `q=None` 表示确定性提议：不建整张 one-hot，直接把那个 token 挖掉——数值上完全
    一样（`q[d]=1 > p[d]` 保证 `p[d]-q[d] < 0 → 0`），少一次词表大小的分配。
    """
    if q is None:
        residual = probs.clone()
        residual[token_id] = 0
    else:
        residual = (probs - q).clamp(min=0)     # 本模块不 import torch：只用张量自己的方法
    total = residual.sum()
    if float(total) <= 0.0:
        # 走到这里说明「被拒」与「有效质量」自相矛盾（例如 p 是个 one-hot 却判了拒绝、
        # 或 q 与 p 在数值上相等）。明确报错，绝不悄悄退回 argmax——那会把分布错误
        # 伪装成一次正常采样。**不按实际质量归一化以外的任何方式**（不加 epsilon 伪造）。
        raise ValueError(f"排除草稿 {token_id} 之后 max(p-q, 0) 没有剩余质量，"
                         f"无法抽纠正 token")
    return residual / total


def _finish_candidates(candidates, num_accepted, eos_token_ids):
    """EOS 截断 + 「本轮输入还要留几枚草稿」——两条验证路径共用的一份实现。

    保留几枚草稿恰好是终止 token 在候选里的**下标**：终止的是草稿 `d_index` 时它的
    KV 要退掉，终止的是末尾那枚（贪心的 argmax / 随机的 bonus）时它压根没有 KV
    （从没作为输入行进过模型），而那时 `index == num_accepted` 正好等于初值。
    """
    committed_ids, kept_drafts = [], num_accepted
    for index, token_id in enumerate(candidates):
        committed_ids.append(token_id)
        if token_id in eos_token_ids:
            kept_drafts = index
            break
    return committed_ids, 1 + kept_drafts


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
    committed_ids, kept_inputs = _finish_candidates(candidates, num_accepted, eos_token_ids)
    return DraftVerification(num_accepted, committed_ids, kept_inputs)


def verify_drafts_random(draft_ids, row_probs, eos_token_ids, remaining_outputs,
                         draw_uniform, draw_token, draft_probs=None):
    """随机采样下的验证：按目标分布做拒绝采样，返回与 `verify_drafts()` 同样的结论。

    一般形式（`draft_probs` 给出每枚草稿的提议分布 q，见第五十五关）：

        第 i 枚接受概率 = min(1, p_i[d_i] / q_i[d_i])
        被拒后的纠正分布 = normalize(max(p_i - q_i, 0))     （见 `residual_probs`）

    `draft_probs=None` 表示**确定性提议**（n-gram）：`q_i(d_i) = 1`、其余为 0，
    接受概率退化成 `p_i[d_i]`，纠正分布退化成「挖掉 d_i」——两条路径是同一个式子，
    EOS 与「保留多少 KV」的规则也只有一份（`_finish_candidates`）。

    规则（需求 §2）：

    1. 逐位置抽 `u ∈ [0,1)`，`u < p[d]` 就接受这一枚；**接受的这一枚若是终止 token，
       当场结束**——它后面的草稿不再验证，bonus 也不抽（需求 §4C）；
    2. 首次拒绝就停：从「挖掉 d 的纠正分布」抽一枚，本轮输出 = 已接受的草稿 + 它；
    3. 全部接受：从**最后一行的分布**抽一枚 bonus；
    4. 结果交回 `_finish_candidates()` 统一做 EOS 截断与「留多少 KV」。

    两个边界按约定处理，并在文档与测试里固定下来（判的是**接受概率** `min(1,p/q)`，
    `q=None` 时它就是 `p[d]`）：

    - 接受概率 `<= 0`（`p[d] == 0`）：必拒，**不消耗** uniform（抽了也是白抽）；
    - 接受概率 `>= 1`（`p[d] >= q[d]`，含 `p[d] == 1`）：必接受，同样不消耗——
      顺带避开「residual 全零」那条路。这条**一样要过终止检查**，必接受不是
      「跳过后面步骤」的捷径。

    `p == q` 时每一枚都落在「必接受」上：整轮全接受、不抽接受随机数，也不会走到
    residual。这正是「同结构同权重的 draft 模型」那条用例。

    提议分布必须**真的**是抽出草稿的那一个：`q_i[d_i] == 0` 是非法输入（从 q 里
    抽不出一个 q 质量为零的 token），这里明确报错，不除零、也不默默当成必接受。

    随机数的消费顺序（需求 §4C）：**逐位置先决定接受**，接受的当下就判终止；没在
    终止 token 上停下时，才是「首次拒绝抽纠正 / 全部接受抽 bonus」。顺序是约定的
    一部分，测试会数调用次数——接受终止 token 之后**一个随机数都不能再抽**，
    否则请求自己的随机流就与「没有投机时」错位了。

    `row_probs` 是 K+1 行的目标分布，行 j 的惩罚历史必须包含**前 j 枚草稿**
    （接受的那些）。调用方按「全都接受」构造即可：拒绝点之后的行根本不会被读到，
    而拒绝点之前的行历史恰好就是「前 j 枚都被接受」。
    """
    num_drafts = len(draft_ids)
    if len(row_probs) != num_drafts + 1:
        raise ValueError(f"验证需要 {num_drafts + 1} 行的目标分布（K 枚草稿 + 1 枚 bonus），"
                         f"收到 {len(row_probs)} 行")
    if draft_probs is not None and len(draft_probs) != num_drafts:
        raise ValueError(f"提议分布应有 {num_drafts} 行（每枚草稿一行），收到 {len(draft_probs)} 行")
    if num_drafts > remaining_outputs - 1:
        raise ValueError(
            f"{num_drafts} 枚草稿要占 {num_drafts + 1} 个输出额度，但只剩 {remaining_outputs} 个；"
            f"调用方必须保证 K <= R-1（_plan_drafts() 就是这么缩的）")

    num_accepted = 0
    while num_accepted < num_drafts:
        token = draft_ids[num_accepted]
        p_draft = float(row_probs[num_accepted][token])
        if draft_probs is None:
            # 确定性提议：q 是 token 上的 one-hot，接受概率就是 p[d]
            accept_prob = p_draft
        else:
            q_draft = float(draft_probs[num_accepted][token])
            if q_draft <= 0.0:
                raise ValueError(
                    f"第 {num_accepted} 枚草稿 {token} 的提议概率 q[d]={q_draft}："
                    f"从 q 里抽不出一个 q 质量为零的 token，这是调用方的错（提议必须"
                    f"真的从 q 抽样，不能 argmax 提议却拿 softmax q 来验证）")
            accept_prob = p_draft / q_draft
        if accept_prob >= 1.0:
            accepted = True                        # 必接受，不消耗随机数
        elif accept_prob <= 0.0:
            accepted = False                       # 必拒绝，同样不消耗
        else:
            accepted = draw_uniform() < accept_prob
        if not accepted:
            break
        num_accepted += 1
        # **接受的当下**就看它是不是终止 token：是则立即结束——它后面的草稿与 bonus
        # 既不提交、也不验证、更不抽随机数。「终止后不再消费随机数」是需求 §4C 的约定，
        # 所以这条检查必须在**两条**接受分支之后（必接受那条不是跳过它的捷径）。
        if token in eos_token_ids:
            break

    # 「最后一枚被接受的草稿是终止 token」等价于「因终止而停下」：循环一到终止 token
    # 就 break，不会再去看它后面的位置，所以拒绝不可能发生在终止 token 之后。
    if num_accepted > 0 and draft_ids[num_accepted - 1] in eos_token_ids:
        candidates = list(draft_ids[:num_accepted])
    elif num_accepted == num_drafts:
        # 全部接受：bonus 从**最后一行**的分布里抽
        candidates = list(draft_ids) + [int(draw_token(row_probs[num_drafts]))]
    else:
        # 首次拒绝：从 normalize(max(p-q, 0)) 里抽（q=None 时退化成「挖掉该草稿」）
        rejected = num_accepted
        candidates = list(draft_ids[:rejected]) + [
            int(draw_token(residual_probs(row_probs[rejected],
                                          None if draft_probs is None else draft_probs[rejected],
                                          draft_ids[rejected])))]

    committed_ids, kept_inputs = _finish_candidates(candidates, num_accepted, eos_token_ids)
    return DraftVerification(num_accepted, committed_ids, kept_inputs)
