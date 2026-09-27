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
# 已经**验证通过**的后端。triton 的内核（counter RNG / 接受前缀 / 分块抽样）已经写好，
# 但还没通过「与 CPU oracle 逐位对照」那一关（现在跑会触发 CUDA device-side assert），
# 所以先不放进这里：构造时就明确拒绝，绝不留一条能走到但坏掉的路径。
# 对照测试在 benchmarks/check_step56_gpu_rejection.py，调通后把它加回来。
BACKENDS = (TORCH,)

# 结果张量每一行几个字段：output_ids 占前 KMAX+1 列，后面是长度、接受数、保留输入数、
# 消费的随机事件数、错误码。字段名固定，CPU 侧按名取值。
PACKED_FIELDS = ("output_lengths", "num_accepted", "kept_inputs", "rng_consumed", "error_code")
SENTINEL_TOKEN = -1


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
            for index, entry in enumerate(plan):
                row, columns = host[index], outcome.columns
                length = int(row[columns["length"]])
                code = int(row[columns["error"]])
                results.append(ItemResult(
                    committed_ids=[int(t) for t in row[:length]],
                    num_accepted=int(row[columns["accepted"]]),
                    kept_inputs=int(row[columns["kept"]]),
                    rng_consumed=int(row[columns["consumed"]]),
                    error=None if code == 0 else f"拒绝验证错误码 {code}"))
            return results
        return outcome

    # -------- triton 后端：GPU 批量判定 + 分块抽样，结果打包成一张张量 --------

    def _triton_backend(self, batch):
        """一次批量验证：两个内核 + 向量化打包，**全程没有逐请求标量读取、没有 D2H**。

        数据流：

            prepare（Torch，设备上）：每位置的 ratio / eos / invalid、每请求的 seed 与 counter
            → verify_prefix_kernel：接受前缀 / 停止类型 / 消费了几个接受事件
            → 按停止类型取**权重行**（Torch 高级索引，设备上）
            → sample_token_kernel：词表分块指数竞赛，一次抽一个 token
            → 向量化打包成一张张量，仍留在 GPU 上（`materialize_results` 才回传）
        """
        import triton

        from . import rejection_triton as rt
        from .rejection_rng import PHILOX_ROUNDS

        num_items = len(batch)
        device = self.device
        eos_ids = torch.tensor(sorted(self.eos_token_ids), dtype=torch.long, device=device)
        kmax = max((len(entry.draft_ids) for entry in batch), default=0)

        # ---- 逐位置：ratio / eos / invalid（拼成扁平张量，不逐项读标量）----
        row_offsets, draft_tokens, row_index, position_kinds = [], [], [], []
        position = 0
        for entry in batch:
            row_offsets.append(position)
            for index, token in enumerate(entry.draft_ids):
                draft_tokens.append(token)
                row_index.append(index)                 # 第 i 枚草稿用第 i 行
                position_kinds.append(1 if entry.draft_probs is not None else 0)
                position += 1
        row_offsets.append(position)                    # 收尾哨兵，方便切片

        if position:
            # 把每项的 K 行拼成 (total_positions, V) 再一次性 gather：
            # 这样 p[d]、q[d] 都是**一次** GPU 操作，没有 per-request 标量
            # q 一律拼成整张（ngram 的位置放 1.0 = 「q 是 d 上的 one-hot」）：
            # 这样 p[d]、q[d] 各是一次 gather，没有按项分支、也没有 GPU 标量读取
            # 摊平成「一行一位置」：K 枚草稿就用前 K 行，K=0 的项不贡献行
            rows_p = torch.stack([row for entry in batch
                                  for row in entry.row_probs[:len(entry.draft_ids)]]).to(torch.float32)
            rows_q = torch.stack([
                row for entry in batch
                for row in (entry.draft_probs if entry.draft_probs is not None
                            else [torch.ones_like(entry.row_probs[0])] * len(entry.draft_ids))
            ]).to(torch.float32)
            tokens_t = torch.tensor(draft_tokens, dtype=torch.long, device=device)
            p_d = rows_p.gather(1, tokens_t.unsqueeze(1)).squeeze(1)
            q_d = rows_q.gather(1, tokens_t.unsqueeze(1)).squeeze(1)
            invalid = q_d <= 0
            ratio = torch.where(invalid, p_d,
                                p_d / torch.where(q_d > 0, q_d, torch.ones_like(q_d)))
            eoses = torch.isin(tokens_t, eos_ids).to(torch.int32)
        else:
            rows_p = rows_q = torch.zeros(0, 0, device=device)
            ratio = torch.zeros(0, dtype=torch.float32, device=device)
            eoses = torch.zeros(0, dtype=torch.int32, device=device)
            invalid = torch.zeros(0, dtype=torch.bool, device=device)

        ks = [len(entry.draft_ids) for entry in batch]
        seeds = [int(entry.plan["request"].rejection_seed or 0) for entry in batch]
        counters = [int(entry.plan["request"].rejection_rng_counter) for entry in batch]
        start_t = torch.tensor(row_offsets[:-1], dtype=torch.int32, device=device)
        k_t = torch.tensor(ks, dtype=torch.int32, device=device)
        seed_lo = torch.tensor([v & 0xFFFFFFFF for v in seeds], dtype=torch.int64, device=device)
        seed_hi = torch.tensor([v >> 32 for v in seeds], dtype=torch.int64, device=device)
        counter_t = torch.tensor(counters, dtype=torch.int64, device=device)

        accepted = torch.zeros(num_items, dtype=torch.int32, device=device)
        kind = torch.zeros(num_items, dtype=torch.int32, device=device)
        consumed = torch.zeros(num_items, dtype=torch.int32, device=device)
        errors = torch.zeros(num_items, dtype=torch.int32, device=device)
        rt.verify_prefix_kernel[(num_items,)](
            ratio.contiguous(), eoses.contiguous(), invalid.to(torch.int32).contiguous(),
            start_t, k_t, seed_lo, seed_hi, counter_t, accepted, kind, consumed, errors,
            KMAX=max(kmax, 1), ROUNDS=PHILOX_ROUNDS)

        # ---- 抽样：纠正（kind==1）用 max(p-q,0)/挖掉 d，bonus（kind==0）用最后一行 ----
        # greedy 项不做随机抽样：纠正/bonus 就是该行权重的 argmax（并列取小下标），
        # 也**不消费 categorical 事件**——这与 CPU 参考路径的 `draw_token = argmax` 一致。
        greedy = torch.tensor([entry.plan["request"].sampling_params.is_greedy
                               for entry in batch], dtype=torch.bool, device=device)
        drawn = torch.zeros(num_items, dtype=torch.int64, device=device)
        if position:
            row_is_ngram = torch.tensor([kind_code == 0 for kind_code in position_kinds],
                                       dtype=torch.bool, device=device)
            weights = self._weight_rows(batch, rows_p, rows_q, row_is_ngram, tokens_t, ks,
                                        accepted, kind)
            if weights is not None:
                draw_rows, draw_index = weights
                draw_greedy = greedy[draw_index]
                if draw_greedy.all():
                    tokens = draw_rows.argmax(dim=1).to(torch.int64)
                    errs = torch.zeros(len(draw_index), dtype=torch.int32, device=device)
                else:
                    tokens, errs = self._sample(
                        rt, draw_rows, seed_lo[draw_index], seed_hi[draw_index],
                        counter_t[draw_index] + consumed[draw_index].to(torch.int64))
                    if draw_greedy.any():
                        tokens = torch.where(draw_greedy, draw_rows.argmax(dim=1), tokens)
                drawn.index_copy_(0, draw_index, tokens)
                errors.index_copy_(0, draw_index, errs)

        packed = self._pack(batch, kmax, accepted, kind, consumed, drawn, errors, greedy, device)
        return PackedResult(tensor=packed, batch=batch,
                            columns=PackedResult.columns_for(max(kmax, 1)))

    def _weight_rows(self, batch, rows_p, rows_q, row_is_ngram, tokens_t, ks, accepted, kind):
        """按停止类型取抽样用的权重行——**全向量化**，不逐项读 `kind` / `accepted`。

        行偏移在 CPU 侧算（都是 Python 整数，不是 GPU 标量），取行用一次高级索引。
        返回 `(权重张量, 需要抽样的项下标)`；没有要抽的项时返回 None。
        """
        offsets = []
        cursor = 0
        for size in ks:
            offsets.append(cursor)
            cursor += size
        device = accepted.device
        index = torch.arange(len(batch), device=device)
        rows_of_item = torch.tensor(offsets, dtype=torch.long, device=device)
        accepted_l = accepted.to(torch.long)
        # 纠正用的行：第 num_accepted 行（拒绝点）；bonus 用的行：第 K 行（最后一行）
        chosen = torch.where(kind == 0,
                             rows_of_item + torch.tensor(ks, dtype=torch.long, device=device),
                             rows_of_item + accepted_l)
        draws = (kind != 2)
        picked = chosen.clamp(min=0)
        # 被拒的那枚草稿：从扁平 token 张量里 gather（不让 Python 去索引 GPU 标量）
        rejected_tokens = tokens_t[picked]
        weight = rows_p[picked]
        # 纠正分布：一般 q 用 max(p-q,0)，ngram 用「挖掉被拒 token」（数值等价，
        # 但不物化 one-hot）。**两者都用向量化的 where 选**，不按项分支、不读标量。
        dig_out = weight.clone()
        dig_out.scatter_(1, rejected_tokens.unsqueeze(1), 0)
        deltas = torch.where(row_is_ngram.unsqueeze(1), dig_out,
                             (weight - rows_q[picked]).clamp(min=0))
        weight = torch.where((kind == 1).unsqueeze(1), deltas, weight)
        draw_index = index[draws]
        if draw_index.numel() == 0:
            return None
        return weight[draws].contiguous(), draw_index

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
        max_len = kmax + 1
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
        lengths = accepted.to(torch.int64) + (kind != 2).to(torch.int64)
        out = drafts.scatter(1, accepted.to(torch.int64).unsqueeze(1).clamp(max=max_len - 1),
                             drawn.unsqueeze(1))
        rows = torch.arange(max_len, device=device).unsqueeze(0)
        out = torch.where(rows < lengths.unsqueeze(1), out, SENTINEL_TOKEN)
        kept = 1 + accepted.to(torch.int64) - (kind == 2).to(torch.int64)
        rng = consumed.to(torch.int64) + ((kind != 2) & ~greedy).to(torch.int64)
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
            results.append(ItemResult(committed_ids=list(result.committed_ids),
                                      num_accepted=result.num_accepted,
                                      kept_inputs=result.kept_inputs))
        return results


def _row_history(seq):
    """第 i 行的惩罚历史：真实已生成 + 前 i 枚草稿（只复制计数，不共享底层字典）。"""
    state = seq.sampling_state
    return SimpleNamespace(prompt_token_ids=state.prompt_token_ids,
                           generated_counts=dict(state.generated_counts))


def is_greedy_without_penalty(seq):
    params = seq.sampling_params
    return params.is_greedy and not params.has_penalty
