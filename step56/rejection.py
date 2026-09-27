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
# 已经实现的后端。阶段 4 把 triton 加进来（在那之前构造时就拒绝它，不留半成品）
BACKENDS = (TORCH,)


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

    def __init__(self, backend, eos_token_ids, sampler):
        if backend not in BACKENDS:
            raise ValueError(f"未知或尚未实现的 rejection_backend={backend!r}，"
                             f"可选 {list(BACKENDS)}")
        self.backend = backend
        self.eos_token_ids = set(eos_token_ids)
        self.sampler = sampler
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
            if draft_ids and greedy is not None and is_greedy_without_penalty(seq):
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
                draft_probs=list(plan["draft_probs"]) if plan["draft_probs"] else None))
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
        return self._torch_backend(batch)

    # -------- 3) 集中回传 --------

    def materialize_results(self, outcome):
        """把后端产物变成 CPU 侧的 `ItemResult` 列表。

        torch 后端的结果本来就在 CPU 上——原样返回；triton 后端在这里做唯一一次
        设备回传（`outcome.cpu()`）。
        """
        return outcome

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
