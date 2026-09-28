"""拒绝验证的批量执行层：把本轮要验证的项整理成批 → 验证 → 集中回传结果。

第五十六关的新东西。职责**只到「产生验证结果」**：

    prepare_batch(logits, items, greedy)   CPU 整理元数据 + 在设备上构造每行的 p/q
    verify_batch(batch)                    后端分派：torch 参考路径 / triton GPU 批量
    materialize_results(outcome)           把后端产物变成 CPU 侧的结论（triton 在这里
                                           做**唯一一次**设备回传）

它不认识 `Scheduler`、不释放请求、也不提交 token——回滚、对齐、提交、收尾都是
`SampleRuntime` 的事，而且仍然严格按 `picked` 顺序做。

后端在**引擎创建时固定**，运行中不切换：

- `"torch"`：逐请求调用既有的 `verify_drafts()` / `verify_drafts_random()`。这是
  第五十二到五十五关的参考路径，**行为与随机流一字不改**（用每条请求自己的
  `torch.Generator`），CPU 也能跑；
- `"triton"`：CUDA 批量验证。counter-based RNG（`rejection_rng.py`）+ 分块纠正/bonus
  抽样，结果打包成一张固定容量张量、一次 `.cpu()` 拿回 CPU。

为什么要这一层：target 早就整批 forward 了，但验证还是「取一条请求的标量 → CPU 决定
→ 再取下一条」，CPU 反复等 GPU（验收方在 189 里数过：16 条请求 80 次
`aten::_local_scalar_dense`）。这里把决策搬进 GPU，只把「结论」拿回来。

**本关不消除**：draft 提议阶段的同步、概率构造的 Python 循环（逐行 `distribution()`）。
那些是下一步的事，别把局部优化说成全引擎没有同步。
"""

from dataclasses import dataclass, field
from types import SimpleNamespace

import torch

from .speculative import verify_drafts, verify_drafts_random

TORCH = "torch"
TRITON = "triton"
# 已经**验证通过**的后端：两个都在。triton 与 CPU oracle 的逐位对照见
# benchmarks/check_step56_gpu_rejection.py（确定性对照 + 分布/随机流 + 定向观测）。
BACKENDS = (TORCH, TRITON)

# 结果张量每一行几个字段：output_ids 占前 KMAX+1 列，后面是长度、接受数、保留输入数、
# 消费的随机事件数、错误码。字段名固定，CPU 侧按名取值。
PACKED_FIELDS = ("output_lengths", "num_accepted", "kept_inputs", "rng_consumed", "error_code")
SENTINEL_TOKEN = -1

# 错误码（triton 后端把「非法输入」编码进结果张量，`SampleRuntime` 整批检查后统一报错）
# 「接受前缀为什么停下来」——`kind` 的三个取值（内核写、下面几处读）。
# 与错误码是不同的命名空间：`kind == 1` 是「首拒绝」，`error == 1` 是「非法提议」。
#
#   0 KIND_ALL_ACCEPTED  K 枚全接受（K=0 也落在这一支）  -> 抽 bonus，权重取收尾行（第 K 行）
#   1 KIND_FIRST_REJECT  第 num_accepted 枚被拒          -> 抽纠正，权重取第 num_accepted 行，
#                                                          分布换 max(p-q,0) / 挖掉 d
#   2 KIND_ACCEPTED_EOS  刚接受的那枚是终止 token        -> 不抽，committed = 前 num_accepted 枚
KIND_ALL_ACCEPTED = 0
KIND_FIRST_REJECT = 1
KIND_ACCEPTED_EOS = 2

ERR_INVALID_PROPOSAL = 1       # q[d] = 0：从 q 里抽不出一个 q 质量为零的 token
ERR_NO_RESIDUAL_MASS = 2       # 纠正/bonus 的权重整行为零，抽不出 token
ERR_GREEDY_RANDOM_BRANCH = 3   # 贪心落进了「要抽随机数才决定接受」那一支：前提被破坏了
ERROR_MESSAGES = {
    ERR_INVALID_PROPOSAL: "非法提议：草稿的 q[d] = 0（提议必须真的从 q 抽样）",
    ERR_NO_RESIDUAL_MASS: "纠正/bonus 的权重整行为零，抽不出 token",
    ERR_GREEDY_RANDOM_BRANCH: "贪心请求落进了需要抽随机数的接受分支：贪心的目标分布"
                              "必须是 one-hot（ratio 只可能是 0 或 >= 1），"
                              "出现 (0,1) 说明喂进来的目标分布不是 one-hot——"
                              "拿不准就报错，不能悄悄当成必拒绝（那会给出一个看着合理、"
                              "其实偏掉的分布）",
}


@dataclass
class PackedResult:
    """triton 后端的产物：**一张**固定容量张量 + 各字段所在列。

    列的排布由 `columns_for(kmax)` 定：前 `kmax + 1` 列是 output_ids（未使用的位置是
    哨兵），之后依次是长度、接受数、保留输入数、消费的随机事件数、错误码。
    """
    tensor: object
    batch: list
    columns: dict

    @staticmethod
    def columns_for(kmax):
        return {"length": kmax + 1, "accepted": kmax + 2, "kept": kmax + 3,
                "consumed": kmax + 4, "error": kmax + 5}


@dataclass
class RejectionItem:
    """批里的一项：CPU 元数据 + 该行**留在设备上**的 p/q（不做任何标量回传）。

    `mode` 两种：

    - `"greedy_fast"`：贪心且无惩罚的投机项，目标模型的 K+1 枚贪心结果已经整批一次
      argmax 算好放在 `greedy_ids` 里，验证就是逐枚比对，不必构造分布；
    - `"distribution"`：一般路径——`row_probs` 是 K+1 行目标分布（设备上的张量），
      `draft_probs` 是每枚草稿的**实际**提议分布 q（ngram 确定性提议时为 None）。
    """

    plan: dict
    mode: str
    draft_ids: list
    remaining_outputs: int
    row_probs: list = field(default_factory=list)
    draft_probs: list = None
    greedy_ids: list = None


@dataclass
class ItemResult:
    """一项的验证结论（CPU 侧），与需求 §3 的结果契约逐项对应。

    - `committed_ids` 就是 `output_ids`（有效前缀；长度即 `output_lengths`）；
    - `num_accepted` 只数**真正验证接受**的草稿（不计纠正/bonus，不计 EOS 之后）；
    - `kept_inputs` 是本轮输入要保留几个位置（pending-token 语义）；
    - `rng_consumed` 是本项消费的随机事件数：triton 后端用它推进 counter，
      torch 后端由 generator 自己推进，恒为 0；
    - `error` 非空表示这项非法（如 `q[d]=0`、残差无质量）——**先检查完整批再提交**，
      不允许「报错前已经提交了半批」。
    """

    committed_ids: list
    num_accepted: int
    kept_inputs: int
    rng_consumed: int = 0
    error: str = None


class BatchedRejectionSampler:
    """批量拒绝验证：后端固定，输入按轮传进来。"""

    def __init__(self, backend, eos_token_ids, sampler, device=None):
        if backend not in BACKENDS:
            raise ValueError(f"未知或尚未实现的 rejection_backend={backend!r}，"
                             f"可选 {list(BACKENDS)}")
        self.backend = backend
        self.eos_token_ids = set(eos_token_ids)
        self.sampler = sampler
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        # 统计：用来证明「GPU 后端真的被调用过」，以及整批实际接受了多少草稿
        self.num_batches = 0
        self.num_items = 0
        self.num_rng_events = 0

    # -------- 1) 整理成批（两个后端共用；p/q 都是设备上的张量） --------

    def prepare_batch(self, logits, items, greedy):
        """把本轮要验证的项整理成一批。

        `items` 是 `picked` 里需要验证的项（有草稿的投机项；triton 后端还包括
        「计划要投机、实际 K=0」的回退项，见 `SampleRuntime.run()`）。
        `greedy` 是整批一次 argmax 的结果（这一轮没有快路径项时为 None）。
        """
        batch = []
        for plan in items:
            seq = plan["request"]
            draft_ids = list(plan["draft_ids"])
            remaining = seq.max_new_tokens - len(seq.output_ids)
            if (draft_ids and self.backend != TRITON and greedy is not None
                    and is_greedy_without_penalty(seq)):
                start = plan["sample_offset"]
                batch.append(RejectionItem(
                    plan=plan, mode="greedy_fast", draft_ids=draft_ids,
                    remaining_outputs=remaining,
                    greedy_ids=greedy[start:start + plan["num_sample_rows"]]))
                continue
            batch.append(RejectionItem(
                plan=plan, mode="distribution", draft_ids=draft_ids,
                remaining_outputs=remaining,
                row_probs=self._row_probs(logits, plan, draft_ids),
                # q 本来就该在设备上（提议层造出来时就在）；这里对齐一次，
                # 免得调用方塞进 CPU 张量后在 `gather` 那里才炸
                draft_probs=[q.to(logits.device) for q in plan["draft_probs"]]
                if plan["draft_probs"] else None))
        return batch

    def _row_probs(self, logits, plan, draft_ids):
        """逐行构造目标分布（K+1 行）。

        **每一行的惩罚历史不同**（需求 §3）：行 j 看到的是「真实已生成 + 前 j 枚草稿」。
        所以这里用一份**临时计数**——从真实计数复制一份，每接受一枚就往里加一枚；
        真实 `sampling_state` 在唯一提交入口之前一个字都不动。

        行按「全都接受」构造：拒绝点之后的行根本不会被读到，而拒绝点之前的行历史恰好
        就是「前 j 枚都被接受」。整个过程只有 Torch 张量操作，不回传任何标量。
        """
        seq = plan["request"]
        params = seq.sampling_params
        rows = logits[plan["sample_offset"]:plan["sample_offset"] + plan["num_sample_rows"]]
        temp = _row_history(seq)
        row_probs = []
        for index in range(len(draft_ids) + 1):
            row_probs.append(self.sampler.distribution(rows[index].to(torch.float32), params, temp))
            if index < len(draft_ids):
                token = draft_ids[index]
                temp.generated_counts[token] = temp.generated_counts.get(token, 0) + 1
        return row_probs

    # -------- 2) 验证（后端分派） --------

    def verify_batch(self, batch):
        """验证整批，返回后端自己的产物（`materialize_results()` 负责解释它）。"""
        self.num_batches += 1
        self.num_items += len(batch)
        if self.backend == TRITON:
            return self._triton_backend(batch)
        return self._torch_backend(batch)

    # -------- 3) 集中回传 --------

    def materialize_results(self, outcome):
        """把后端产物变成 CPU 侧的 `ItemResult` 列表。

        torch 后端的结果本来就在 CPU 上——原样返回；triton 后端在这里做唯一一次
        设备回传（`outcome.cpu()`）。
        """
        if isinstance(outcome, PackedResult):
            # **唯一一次**设备回传：一张固定容量张量，不逐字段、不逐请求
            host = outcome.tensor.cpu().tolist()
            plan = outcome.batch
            results = []
            for index in range(len(plan)):
                row, columns = host[index], outcome.columns
                length = int(row[columns["length"]])
                code = int(row[columns["error"]])
                self.num_rng_events += int(row[columns["consumed"]])
                # 报错项返回**空结论**：非法输入不进任何状态，调用方必须整批检查后
                # 才提交（`SampleRuntime.run()` 就是这么做的），空的 committed_ids
                # 让「有没有被误提交」在事后也查得出来。
                results.append(ItemResult(
                    committed_ids=[] if code else [int(t) for t in row[:length]],
                    num_accepted=0 if code else int(row[columns["accepted"]]),
                    kept_inputs=0 if code else int(row[columns["kept"]]),
                    rng_consumed=int(row[columns["consumed"]]),
                    error=None if code == 0 else ERROR_MESSAGES.get(code, f"拒绝验证错误码 {code}")))
            return results
        return outcome

    # -------- triton 后端：GPU 批量判定 + 分块抽样，结果打包成一张张量 --------

    def _triton_backend(self, batch):
        """一次批量验证：两个内核 + 向量化打包，**全程没有逐请求标量读取、没有 D2H**。

        数据流（每一段都只碰整批张量，没有一项一项的循环）：

            `_row_layout()`（CPU 侧元数据，只算整数）
              行坐标系：每项 K+1 行（K 枚草稿行 + 收尾行）首尾相接
              位置坐标系：只覆盖草稿位置（P = ΣK）
            → gather：每位置一个 `p[d]` / `q[d]` / `ratio` / `eos` / `invalid`
            → `verify_prefix_kernel`：接受前缀 / 停止类型 / 消费了几个接受事件
            → `_weight_rows()`：按停止类型取权重行（一次高级索引）
            → `sample_token_kernel`：词表分块指数竞赛，抽纠正 / bonus token
            → `_pack()`：打包成一张张量，仍留在 GPU 上
              （`materialize_results()` 才做唯一一次回传）

        **不做数据依赖的形状**：不按 `kind` 去筛子集（`mask[bool_mask]` 会把形状从
        设备拷回主机，是一次隐式同步），所以接受终止的项也照常参与抽样，结论随后丢掉。
        """
        # triton 本身不需要在这里 import：下面这句会导入 rejection_triton，
        # 缺 triton 时它自己就会失败（也更清楚地指出「是这个后端需要 triton」）
        from . import rejection_triton as rt
        from .rejection_rng import PHILOX_ROUNDS

        device = self.device
        num_items = len(batch)
        if num_items == 0:
            return PackedResult(tensor=torch.zeros((0, 1), dtype=torch.int64, device=device),
                                batch=[], columns=PackedResult.columns_for(1))
        lay = _row_layout(batch, device)
        eos_ids = torch.tensor(sorted(self.eos_token_ids), dtype=torch.long, device=device)

        # ---- 位置坐标系：p[d]、q[d]、ratio、eos、invalid（各算一次，不逐项读标量）----
        if lay.num_draft_positions:
            p_d = lay.target_rows[lay.draft_rows, lay.tokens]   # 每位置一个标量
            if lay.proposal_rows is None:
                # ngram：q 是 d 上的 one-hot，`q[d]` 就是 1，不必物化整行
                q_d = torch.ones_like(p_d)
            else:
                # ngram 的位置 `q_map` 是 -1（哨兵）：夹成 0 读到的是一行合法的 q，
                # 结果随即被 where 丢掉
                q_d = torch.where(lay.q_map >= 0,
                                  lay.proposal_rows[lay.q_map.clamp(min=0), lay.tokens],
                                  torch.ones_like(p_d))
            # 非法提议（q[d] = 0）不进除法：ratio 填 p[d]，错误由内核单独标记
            invalid = q_d <= 0
            ratio = torch.where(invalid, p_d,
                                p_d / torch.where(q_d > 0, q_d, torch.ones_like(q_d)))
            eoses = torch.isin(lay.tokens, eos_ids).to(torch.int32)
        else:
            ratio = torch.zeros(0, dtype=torch.float32, device=device)
            eoses = torch.zeros(0, dtype=torch.int32, device=device)
            invalid = torch.zeros(0, dtype=torch.bool, device=device)

        seeds = [int(entry.plan["request"].rejection_seed or 0) for entry in batch]
        counters = [int(entry.plan["request"].rejection_rng_counter) for entry in batch]
        greedy_flags = [entry.plan["request"].sampling_params.is_greedy for entry in batch]
        seed_lo = torch.tensor([value & 0xFFFFFFFF for value in seeds], dtype=torch.int64,
                               device=device)
        seed_hi = torch.tensor([value >> 32 for value in seeds], dtype=torch.int64,
                               device=device)
        counter_t = torch.tensor(counters, dtype=torch.int64, device=device)
        greedy_t = torch.tensor(greedy_flags, dtype=torch.bool, device=device)

        accepted = torch.zeros(num_items, dtype=torch.int32, device=device)
        kind = torch.full((num_items,), KIND_ALL_ACCEPTED, dtype=torch.int32,
                           device=device)
        consumed = torch.zeros(num_items, dtype=torch.int32, device=device)
        errors = torch.zeros(num_items, dtype=torch.int32, device=device)
        rt.verify_prefix_kernel[(num_items,)](
            ratio.contiguous(), eoses.contiguous(), invalid.to(torch.int32).contiguous(),
            greedy_t.to(torch.int32).contiguous(), lay.pos_offsets, lay.k_t,
            seed_lo, seed_hi, counter_t, accepted, kind, consumed, errors,
            # `KMAX` 直接给 kmax：整批一枚草稿都没有时 `tl.static_range(0)` 是合法的
            # （循环体不展开，四个输出保持初值 0），不必抬到 1。**下面结果张量的列布局
            # 不一样**——那里必须 pad 到 1，否则字段列的下标会和 `columns` 对不上。
            KMAX=lay.kmax, ROUNDS=PHILOX_ROUNDS)

        # ---- 抽样：首拒绝（KIND_FIRST_REJECT）用 max(p-q,0) / 挖掉 d，
        #      全接受（KIND_ALL_ACCEPTED）用收尾行 ----
        weights = self._weight_rows(lay, kind, accepted)
        # 贪心项不抽随机数：纠正/bonus 就是该行权重的 argmax（并列取小下标），
        # 也不消费 categorical 事件——与 torch 参考路径的 `draw_token = argmax` 一致。
        # 整批是否贪心是**CPU 侧就知道**的（`sampling_params`），拿它做分支不会同步。
        if all(greedy_flags):
            sampled, draw_errors = _greedy_draw(weights)
        else:
            sampled, draw_errors = self._sample(
                rt, weights, seed_lo, seed_hi, counter_t + consumed.to(torch.int64))
            if any(greedy_flags):
                greedy_draw, greedy_errors = _greedy_draw(weights)
                sampled = torch.where(greedy_t, greedy_draw, sampled)
                draw_errors = torch.where(greedy_t, greedy_errors, draw_errors)
        # 接受终止 token 的项**不抽样**：上面为整批算出来的结论对它没有意义，整段丢掉
        # （那一列会被长度掩码盖成哨兵）。不丢的话，一条 EOS 正常结束的请求会被
        # 「它的收尾行恰好没质量」这种与它无关的理由判成非法。
        # `KIND_ACCEPTED_EOS` 与接受内核的错误码 1 互斥（循环在任一个上都会停），所以
        # 这里覆盖 errors 不会吃掉真正的非法输入。
        stopped = kind == KIND_ACCEPTED_EOS
        drawn = torch.where(stopped, torch.zeros_like(sampled), sampled)
        draw_errors = torch.where(stopped, torch.zeros_like(draw_errors), draw_errors)
        errors = torch.where(errors != 0, errors, draw_errors)

        packed = self._pack(batch, lay.kmax, accepted, kind, consumed, drawn, errors,
                            greedy_t, device)
        return PackedResult(tensor=packed, batch=batch,
                            columns=PackedResult.columns_for(max(lay.kmax, 1)))

    def _weight_rows(self, lay, kind, accepted):
        """按停止类型取抽样用的权重行——**全向量化**，不逐项读 `kind` / `accepted`。

        行下标在设备上算（`accepted` 本身就在设备上，读它就要同步）：
        纠正用第 `num_accepted` 行（拒绝点那一行），bonus 用第 K 行（收尾行）。
        两者都在 `target_rows` 里，一次高级索引取整行。
        """
        accepted_l = accepted.to(torch.long)
        # 取哪一行：**收尾行**（全接受时用，= 起点 + K）或**拒绝点那一行**（首拒绝时用，
        # = 起点 + num_accepted）。这两个下标其实是**同一个表达式**——内核只在「没被拒、
        # 没遇到 EOS」时才停在 KIND_ALL_ACCEPTED，那时 `accepted == K`（K=0 时 0 == 0 也
        # 成立），所以 `起点 + accepted` 正好就是收尾行。因此这里不写 where：不变量由内核
        # 保证（有用例钉着），写出来反而像是在说「它可能是别的值」。
        chosen = lay.row_offsets + accepted_l
        weight = lay.target_rows[chosen]
        # 被拒的那枚草稿：位置 = 本项起点 + num_accepted。不是首拒绝时这个位置读到的是
        # 别人的草稿（全接受的项 `num_accepted == K`，正好越过本项），**必须夹住**：
        # 越界下标喂给下面的 `scatter_` 会直接触发 CUDA device-side assert。
        # 夹住之后读到的是一个合法 token id，而这条分支马上被 `KIND_FIRST_REJECT` 的 where 丢掉。
        if lay.num_draft_positions:
            rejected_pos = (lay.pos_offsets.to(torch.long) + accepted_l).clamp(
                max=lay.num_draft_positions - 1)
            rejected = lay.tokens[rejected_pos]
        else:
            rejected_pos, rejected = accepted_l, torch.zeros_like(accepted_l)
        # 纠正分布：ngram 用「挖掉被拒 token」，一般 q 用 `max(p-q,0)`（数值等价于
        # 挖掉 d 再归一化，但不物化 one-hot）。**两者都算、用 where 选**，不按项分支。
        dig_out = weight.clone()
        dig_out.scatter_(1, rejected.unsqueeze(1), 0.0)
        if lay.proposal_rows is None:
            deltas = dig_out
        else:
            # 被拒草稿自己的 q 行；没有 q 的项读到的是夹过的别处（随即被丢掉）
            # `q_map` 里的 -1 是哨兵（这一项没有 q）：夹成 0 只为索引合法，
            # 结果会被下面的 `has_q_item` 丢掉
            q_rejected = lay.proposal_rows[lay.q_map[rejected_pos].clamp(min=0)]
            deltas = torch.where(lay.has_q_item.unsqueeze(1),
                                 (weight - q_rejected).clamp(min=0.0), dig_out)
        return torch.where((kind == KIND_FIRST_REJECT).unsqueeze(1), deltas, weight)

    def _sample(self, rt, rows, seed_lo, seed_hi, event_index):
        from .rejection_rng import PHILOX_ROUNDS
        tokens = torch.zeros(rows.shape[0], dtype=torch.int64, device=rows.device)
        errs = torch.zeros(rows.shape[0], dtype=torch.int32, device=rows.device)
        rt.sample_token_kernel[(rows.shape[0],)](
            rows.contiguous(), seed_lo, seed_hi, event_index, tokens, errs,
            rows.shape[1], BLOCK_V=rt.BLOCK_V, ROUNDS=PHILOX_ROUNDS)
        return tokens, errs

    def _pack(self, batch, kmax, accepted, kind, consumed, drawn, errors, greedy, device):
        """向量化打包成一张固定容量张量：output_ids 前缀 + 五个字段。

        行内先铺草稿（用掩码，不逐项写），再把抽到的 token 散射到第 `num_accepted` 列
        ——接受 EOS 的项不抽样，那一列超出长度、会被哨兵盖掉——最后按长度掩码。
        """
        num_items = len(batch)
        # 输出列数按 `max(kmax, 1)` 算（与 `columns_for()` 同一套口径）：整批 K 全为 0
        # 时也要留一个哨兵列，否则字段列的下标会和 `columns` 对不上。
        pad = max(kmax, 1)
        max_len = pad + 1
        drafts = torch.full((num_items, max_len), SENTINEL_TOKEN, dtype=torch.int64,
                            device=device)
        if kmax:
            flat = [entry.draft_ids + [SENTINEL_TOKEN] * (kmax - len(entry.draft_ids))
                    for entry in batch]
            matrix = torch.tensor(flat, dtype=torch.int64, device=device)
            mask = (torch.arange(kmax, device=device).unsqueeze(0)
                    < torch.tensor([len(e.draft_ids) for e in batch], dtype=torch.long,
                                   device=device).unsqueeze(1))
            drafts[:, :kmax] = torch.where(mask, matrix, SENTINEL_TOKEN)
        lengths = accepted.to(torch.int64) + (kind != KIND_ACCEPTED_EOS).to(torch.int64)
        out = drafts.scatter(1, accepted.to(torch.int64).unsqueeze(1).clamp(max=max_len - 1),
                             drawn.unsqueeze(1))
        rows = torch.arange(max_len, device=device).unsqueeze(0)
        out = torch.where(rows < lengths.unsqueeze(1), out, SENTINEL_TOKEN)
        kept = 1 + accepted.to(torch.int64) - (kind == KIND_ACCEPTED_EOS).to(torch.int64)
        # 消费的随机事件数：接受事件（`consumed`，内核数的）+ 一次 categorical
        # （真的要抽纠正/bonus 时才有一个）——贪心两项都不加，**报错的项**也不加
        # categorical：它没有产出 token，报的必须是「真的抽掉的那几个接受事件」。
        rng = consumed.to(torch.int64) + ((kind != KIND_ACCEPTED_EOS) & ~greedy
                                          & (errors == 0)).to(torch.int64)
        fields = torch.stack([lengths, accepted.to(torch.int64), kept, rng,
                              errors.to(torch.int64)], dim=1)
        return torch.cat([out, fields], dim=1).to(torch.int64)

    # -------- torch 参考后端：逐请求调用既有验证函数（行为与第五十五关一致） --------

    def _torch_backend(self, batch):
        results = []
        for entry in batch:
            seq = entry.plan["request"]
            params, state = seq.sampling_params, seq.sampling_state
            if entry.mode == "greedy_fast":
                result = verify_drafts(entry.draft_ids, entry.greedy_ids, self.eos_token_ids,
                                       entry.remaining_outputs)
            else:
                if params.is_greedy:
                    # 贪心请求没有 generator（第五十二关起只给随机采样建），也确实不需要：
                    # 目标分布是 one-hot，一次 uniform 都不抽
                    def draw_uniform():
                        raise AssertionError("贪心路径不该抽接受随机数")

                    draw_token = lambda probs: int(torch.argmax(probs))
                else:
                    def draw_uniform():
                        return float(torch.rand((), generator=state.generator,
                                                device=state.generator.device))

                    draw_token = lambda probs: torch.multinomial(probs, num_samples=1,
                                                                 generator=state.generator)
                result = verify_drafts_random(entry.draft_ids, entry.row_probs,
                                              self.eos_token_ids, entry.remaining_outputs,
                                              draw_uniform, draw_token,
                                              draft_probs=entry.draft_probs)
            # `committed_ids` 不再复制一层：`_finish_candidates()` 每次都新建一个 list，
            # 且没有第二个持有者（triton 分支那边是同样的情况）
            results.append(ItemResult(committed_ids=result.committed_ids,
                                      num_accepted=result.num_accepted,
                                      kept_inputs=result.kept_inputs))
        return results


def _row_history(seq):
    """第 i 行的惩罚历史：真实已生成 + 前 i 枚草稿（只复制计数，不共享底层字典）。"""
    state = seq.sampling_state
    return SimpleNamespace(prompt_token_ids=state.prompt_token_ids,
                           generated_counts=dict(state.generated_counts))


def _row_layout(batch, device):
    """把一批项摊平成两个坐标系——全部是 CPU 侧整数，**不读任何 GPU 标量**。

    * **行坐标系**：每项 `K+1` 行（K 枚草稿行 + 收尾行）首尾相接，`row_offsets[i]`
      是第 i 项的起始行。收尾行必须留在这里：`kind == 0`（全部接受）时 bonus 就是
      从它抽的，取样权重时也要按下标取到它。`target_rows` 就是这些行（**所有**项都有，
      每项 `K+1` 行），一次 stack 出来，之后只做高级索引、不再复制。
    * **位置坐标系**：只覆盖**草稿位置**（P = ΣK），`pos_offsets[i]` 是第 i 项的起始
      位置，`num_draft_positions` 是它的总数（当下标上界用）。`ratio` / `eos` / `invalid`
      与接受前缀内核都按它排。

    `proposal_rows` 是 q 那侧的行，只收「真的有 q」的项（ngram 的提议是确定性的，`q` 是
    d 上的 one-hot，不必物化整行）：所以它是 `(ΣK_有q, V)`，一项都没有 q 时是 `None`——
    **不是**「所有行」，与 `target_rows` 的覆盖面不同。`q_map[j]` 是位置 j 在那张表里的
    行号，**没有 q 的位置是 -1**（哨兵，
    用到的时候再 clamp）——所以「这个位置有没有 q」不必单独存一张表，`q_map >= 0`
    就是它。`has_q_item[i]` 表示第 i 项走的是不是一般 q：这是**按项**的属性，
    与位置级别的那一份是两个粒度，不能混用。
    """
    row_offsets, pos_offsets, ks, has_q_item = [], [], [], []
    target_rows, tokens, draft_rows, proposal_rows, q_row_of_position = [], [], [], [], []
    cursor_rows = cursor_draft_positions = 0
    for entry in batch:
        k = len(entry.draft_ids)
        if len(entry.row_probs) != k + 1:
            raise ValueError(f"triton 后端需要每项 {k + 1} 行目标分布（K 枚草稿 + 1 枚 "
                             f"收尾行），收到 {len(entry.row_probs)} 行")
        row_offsets.append(cursor_rows)
        pos_offsets.append(cursor_draft_positions)
        ks.append(k)
        has_q_item.append(entry.draft_probs is not None)
        target_rows.extend(entry.row_probs)
        for index, token in enumerate(entry.draft_ids):
            tokens.append(token)
            draft_rows.append(cursor_rows + index)
            if entry.draft_probs is None:
                q_row_of_position.append(-1)
            else:
                q_row_of_position.append(len(proposal_rows))
                proposal_rows.append(entry.draft_probs[index])
        cursor_rows += k + 1
        cursor_draft_positions += k
    long = lambda values: torch.tensor(values, dtype=torch.long, device=device)   # noqa: E731
    return SimpleNamespace(
        kmax=max(ks, default=0), ks=ks,
        # 整批的**草稿位置**总数（P = ΣK）——注意不是行数 Σ(K+1)：行坐标系每项多一个
        # 收尾行。也别和 RoPE 的 positions 混（那个是 token 的逻辑位置）。
        num_draft_positions=cursor_draft_positions,
        # `target_rows` / `proposal_rows` 显式搬到设备：调用方给的 p/q 可能还在 CPU 上
        # （测试与工具脚本就这么干），在这里一次搬完，后面全是设备上的运算
        target_rows=torch.stack(target_rows).to(device=device, dtype=torch.float32),
        row_offsets=long(row_offsets),
        pos_offsets=torch.tensor(pos_offsets, dtype=torch.int32, device=device),
        k_t=torch.tensor(ks, dtype=torch.int32, device=device),
        tokens=long(tokens), draft_rows=long(draft_rows),
        # 只有「真的有 q」的项才有行；一项都没有时是 None（`_triton_backend` 与
        # `_weight_rows()` 都按 `is None` 分岔，不要改成空张量）
        proposal_rows=(torch.stack(proposal_rows).to(device=device, dtype=torch.float32)
                       if proposal_rows else None),
        # `q_map` **保留 -1 当哨兵**（用到的时候再 clamp）：这样「这个位置有没有 q」
        # 不必再单独存一张表，`q_map >= 0` 就是它——两张表存同一个事实最容易走偏。
        # `has_q_item` 是按项的那一份，仍然独立存着：它是唯一不依赖「夹过的下标」的、
        # 无条件正确的标志（见 `_weight_rows()` 里那些越界位置的讨论）。
        q_map=long(q_row_of_position),
        has_q_item=torch.tensor(has_q_item, dtype=torch.bool, device=device))


def _greedy_draw(weights):
    """贪心的纠正 / bonus：逐行 argmax（并列取最小下标），不抽任何随机数。

    `argmax` 返回**第一个**最大值，正好是下标最小的那个，与 CPU 侧的
    `weights.index(max(weights))` 一致。整行权重全为零时报错（与随机路径的
    「无剩余质量」同一个错误码），不返回一个 token 0 冒充结论。
    """
    return (weights.argmax(dim=1).to(torch.int64),
            (weights.max(dim=1).values <= 0).to(torch.int32))


def is_greedy_without_penalty(seq):
    params = seq.sampling_params
    return params.is_greedy and not params.has_penalty
