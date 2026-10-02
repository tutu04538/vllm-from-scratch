"""`RejectionSampler`：验证草稿（对应 vLLM `v1/sample/rejection_sampler.py` 的 Torch 路径）。

输入是"target 模型在 K+1 行上的 logits"和"草稿 token 及其概率 q"，输出是**本轮真正提交的 token
序列**（每条请求 1 ~ K+1 个）。它不碰 Request、不判停止——截断与结束仍然是 Scheduler 的事。

### 算法（199 §7）

```text
greedy 行：逐位置比对草稿 == target 的 argmax
           第一个不同处 → 用 target 的 argmax 顶替，**后面全部丢掉**
           全部相同     → 追加 bonus token（target 在最后一行采出来的那个）
random 行：草稿 d 来自分布 q，接受概率 = min(1, p[d] / q[d])，用一次均匀随机数判定
           拒绝 → 用**修正分布** max(p - q, 0) 采一个 token 顶替（recovered token）
           全部接受 → 追加 bonus
```

`q` 必须是**实际提议时用的分布**：ngram/确定性草稿没有分布（点质量），用 `draft_probs=None`
表示，此时 `q[d] = 1`、接受判定退化成 `p[d] >= u` ✓（**不能**把"没有 q"当"按 p 随便采"）。
`q[d] == 0` 时按拒绝处理（199 §1 的修正：本机内核也是防御性拒绝）。

### 历史条件（199 §7 点名的一条）

第 j 个验证位置的 p 要按"已确定历史 + 草稿前缀 `[:j]`"应用惩罚与约束——因为**如果前面都接受**，
这行面对的就是那个历史。bonus 行则按"全部草稿都接受"的历史。被拒绝之后，后面预先算好的结果
直接丢掉。所以这里临时构造一份**假设历史**（`_combine_outputs_with_spec_tokens`），
不去动请求镜像里的权威历史。

### 随机数（199 §8）

按 draft 段预生成均匀随机数（K=0 的请求**不消耗**随机数）；recovered token 用指数竞赛采
（每个请求一行噪声）。bonus、recovered 都可能先算了没用上，**不回滚随机流**。测试可以注入
固定的 uniform / recovered 值，从而把"实现差异"与"算法错误"分开。

### 提议侧的 q 与验证侧的 p 不必一样（199 §7）

拒绝采样对**任何** q 都成立（接受概率 `min(1, p/q)` 保证边缘分布是 p），所以提议侧可以省掉
惩罚、top-k/top-p 这些约束——q 离 p 越远只是**接受率**越低，不改变输出分布。
**验证侧必须施加**：惩罚要按"已提交历史 + 草稿前缀"算（199 §7 的历史条件），
`min_tokens` 的停止 token 屏蔽也要有（否则草稿可能在 min_tokens 之前把停止 token 送进来）。

### 本关与 vLLM 的差异

| 差异 | 说明 |
|---|---|
| 不做 padding 输入 | vLLM 的 `SamplerOutput` 是 `[B, max_spec_len+1]`、无效位置填 -1；本关照做（同一个形状），但**不**为它准备 padded 的中间张量 |
| 没有 synthetic mode / fp64 Gumbel / logprobs | 前两个是实验与对照用，logprobs 不在 57E 范围 |
| 只有一条 Torch 路径 | vLLM 走 Triton 内核；本关按 199 §7"第一版先 Torch 可读实现，函数边界对应源码" |
"""

import torch

from ..outputs import SamplerOutput
from ..sample import SamplingMetadata
from ..sample.ops.penalties import apply_all_penalties
from ..sample.ops.topk_topp_sampler import SAMPLING_EPS, apply_top_k_top_p

# 无效位置的填充值（vLLM 同名常量）：`[B, max_spec_len+1]` 的右边部分填它
PLACEHOLDER_TOKEN_ID = -1


def expand_batch_to_tokens(x: torch.Tensor, num_tokens_per_req: list[int]) -> torch.Tensor:
    """`[B]` → `[P]`：第 i 条请求的 K_i 个验证行共用同一份参数。

    对应 vLLM 的 `expand_batch_to_tokens` 内核（例：x=[a,b,c]、K=[2,3,1] → [a,a,b,b,b,c]）。
    """
    counts = torch.tensor(num_tokens_per_req, dtype=torch.int64, device=x.device)
    return torch.repeat_interleave(x, counts, dim=0)


def combine_outputs_with_spec_tokens(output_token_ids: list[list[int]],
                                     spec_token_ids: list[list[int]]):
    """验证行的**假设历史**：第 j 行 = 已提交历史 + 草稿前缀 `[:j]`（各请求展开成 K 行）。

    对应 vLLM `RejectionSampler._combine_outputs_with_spec_tokens`（它的写法是"复制上一行再追加
    一个草稿"，本关直接切片，语义一样）。没有草稿的请求不产出任何行——它的 K=0，
    target 行本来就是空的。
    """
    result: list[list[int]] = []
    for out, spec in zip(output_token_ids, spec_token_ids):
        for index in range(len(spec)):
            result.append(list(out) + list(spec[:index]))
    return result


def bonus_histories(output_token_ids: list[list[int]], spec_token_ids: list[list[int]]):
    """bonus 行的历史 = 已提交历史 + **全部**草稿（只在全部接受时才走到它）。"""
    return [list(out) + list(spec) if spec else list(out)
            for out, spec in zip(output_token_ids, spec_token_ids)]


class RejectionSampler:
    def __init__(self, sampler) -> None:
        # 复用普通采样器做 bonus 与"修正分布"抽样：bonus 就是一次普通采样
        self.sampler = sampler

    # -------- 入口 --------

    def forward(self, metadata, logits: torch.Tensor, draft_probs: torch.Tensor | None,
                sampling_metadata: SamplingMetadata, *, uniforms: torch.Tensor | None = None,
                recoveries: torch.Tensor | None = None) -> SamplerOutput:
        """`logits` 是 `[P+B, V]`（`logits_indices` 取完之后的行），行序就是 metadata 的顺序。"""
        if metadata.num_draft_tokens_total == 0 and metadata.batch_size == 0:
            raise ValueError("空的投机批次：没有请求就不该走到 RejectionSampler")

        # **行序契约**（2026-10-02 补）：`sampling_metadata` 必须正好覆盖 spec metadata 里的
        # 那些请求、且顺序一致——验证行的历史、惩罚、min_tokens 全靠它逐请求摊到逐行。
        # 少了或换了顺序都不会报错，只会把 A 的参数用到 B 的行上（静默错），所以在这里挡住。
        if len(sampling_metadata.prompt_token_ids) != metadata.batch_size:
            raise ValueError(
                f"采样元数据有 {len(sampling_metadata.prompt_token_ids)} 行，"
                f"但投机元数据里有 {metadata.batch_size} 条请求：两者必须逐请求对应、顺序一致"
                f"（不能只对一部分行建元数据）")

        # ---- 1) bonus token：用"全部草稿都接受"的历史，走一次普通采样 ----
        bonus_logits = logits[metadata.bonus_logits_indices]
        bonus_metadata = self._with_histories(
            sampling_metadata,
            bonus_histories(sampling_metadata.output_token_ids,
                            sampling_metadata.spec_token_ids))
        bonus_token_ids = self.sampler.forward(bonus_logits, bonus_metadata).sampled_token_ids

        # ---- 2) 验证行的 p：历史按草稿前缀逐行不同，再应用惩罚与采样约束 ----
        target_logits = logits[metadata.target_logits_indices].to(torch.float32)
        target_metadata = self._with_histories(
            sampling_metadata,
            combine_outputs_with_spec_tokens(sampling_metadata.output_token_ids,
                                             sampling_metadata.spec_token_ids),
            num_tokens_per_req=metadata.num_draft_tokens)
        # 「会改变 argmax 的约束」也要施加（目前只有 min_tokens 的停止 token 屏蔽）：
        # 复用普通采样器的同一条逻辑，历史是**假设历史**（已提交 + 草稿前缀）
        target_logits = self.sampler.apply_logits_processors(target_logits, target_metadata)
        target_logits = self._apply_penalties(target_logits, target_metadata,
                                              metadata.num_draft_tokens)
        target_logits = self._apply_constraints(target_logits, target_metadata,
                                                metadata.num_draft_tokens)

        # ---- 3) 逐请求判定（greedy / random 两路），产出 padded [B, max_spec_len+1] ----
        return SamplerOutput(sampled_token_ids=self._verify(
            metadata, target_logits, target_metadata, draft_probs, bonus_token_ids,
            sampling_metadata=sampling_metadata, uniforms=uniforms, recoveries=recoveries))

    # -------- 结果合成 --------

    def _verify(self, metadata, target_logits: torch.Tensor,
                target_metadata: SamplingMetadata, draft_probs: torch.Tensor | None,
                bonus_token_ids: torch.Tensor, *, sampling_metadata: SamplingMetadata,
                uniforms: torch.Tensor | None,
                recoveries: torch.Tensor | None) -> torch.Tensor:
        batch_size = metadata.batch_size
        max_spec_len = metadata.max_spec_len
        device = target_logits.device
        output = torch.full((batch_size, max_spec_len + 1), PLACEHOLDER_TOKEN_ID,
                            dtype=torch.int64, device=device)
        if max_spec_len == 0:
            # 全批 K=0：等价于普通解码，最后一行（= bonus 行）就是答案
            return bonus_token_ids.to(torch.int64)

        target_probs = target_logits.softmax(dim=-1, dtype=torch.float32)
        target_argmax = target_logits.argmax(dim=-1)
        # 逐行的"这行是不是贪心"：全贪心批不用算（None = 整批贪心），全随机批给全 False；
        # 混批时把 [B] 的温度展开成 [P] 行（验证行是每请求 K 行）
        if target_metadata.all_greedy:
            is_greedy_rows = None
        elif target_metadata.all_random:
            is_greedy_rows = torch.zeros(target_logits.shape[0], dtype=torch.bool,
                                         device=device)
        else:
            # 注意用**逐请求**的 `sampling_metadata.temperature`（[B]）来展开成 [P] 行；
            # `target_metadata.temperature` 已经是展开过的 [P]，再 expand 一次会
            # "repeats.size(0) != input.size(0)"（K 不一致时当场报错，K 一致时静默算错）
            is_greedy_rows = expand_batch_to_tokens(
                sampling_metadata.temperature < SAMPLING_EPS, metadata.num_draft_tokens)

        if uniforms is None and is_greedy_rows is not None and not bool(is_greedy_rows.all()):
            uniforms = self._draw_uniforms(metadata, target_metadata, device)
        if recoveries is None and not bool(target_metadata.all_greedy):
            recoveries = self._draw_recoveries(metadata, target_metadata, target_probs,
                                               draft_probs, device)

        row_start = 0
        for req_index, num_draft in enumerate(metadata.num_draft_tokens):
            rejected = False
            for position in range(num_draft):
                row = row_start + position
                greedy = True if is_greedy_rows is None else bool(is_greedy_rows[row])
                if greedy:
                    # greedy：草稿必须等于 target 的 argmax，否则用 argmax 顶替并截断
                    token = int(target_argmax[row])
                    accepted = int(metadata.draft_token_ids[row]) == token
                else:
                    token, accepted = self._accept_or_recover(
                        row, metadata, target_probs, draft_probs, uniforms, recoveries)
                output[req_index, position] = token
                if not accepted:
                    rejected = True
                    break
            if not rejected:
                # 全部接受 → 追加 bonus
                output[req_index, num_draft] = int(bonus_token_ids[req_index])
            row_start += num_draft
        return output

    def _accept_or_recover(self, row: int, metadata, target_probs: torch.Tensor,
                           draft_probs: torch.Tensor | None, uniforms: torch.Tensor,
                           recoveries: torch.Tensor):
        """random 行的接受判定：`min(1, p[d]/q[d]) >= u`；拒绝就用 recovered token 顶替。"""
        draft_token = int(metadata.draft_token_ids[row])
        target_prob = float(target_probs[row, draft_token])
        if draft_probs is None:
            draft_prob = 1.0                     # 点质量提议（ngram）：q[d] = 1
        else:
            draft_prob = float(draft_probs[row, draft_token])
        # q[d] == 0 防御性拒绝（vLLM 内核同样处理，避免 p/q 得到 NaN）
        accepted = draft_prob > 0.0 and target_prob / draft_prob >= float(uniforms[row])
        token = draft_token if accepted else int(recoveries[row])
        return token, accepted

    # -------- 三个辅助：历史、惩罚、约束 --------

    @staticmethod
    def _with_histories(sampling_metadata: SamplingMetadata, histories: list[list[int]],
                        num_tokens_per_req: list[int] | None = None) -> SamplingMetadata:
        """复制一份元数据、换掉逐行的历史与逐请求参数（**不改原对象**）。

        `num_tokens_per_req` 给了就把 `[B]` 的参数展开成 `[P]`（验证行是"每请求 K 行"）。
        """
        import dataclasses

        # ---- 契约检查（2026-10-02 补）----
        # 这个方法把"逐请求"的参数摊成"逐验证行"，**行序必须与传入的 sampling_metadata 一致**，
        # 长度也必须对得上。原来什么都不查、还用 `zip()` 摊平——zip 会按短的那边**静默截断**：
        # 少给一项 min_tokens，那一行的停止 token 屏蔽就悄悄没了（不报错，只是行为变了）。
        num_rows = len(sampling_metadata.prompt_token_ids)
        for name in ("min_tokens", "stop_token_ids"):
            if len(getattr(sampling_metadata, name)) != num_rows:
                raise ValueError(f"元数据自相矛盾：{name} 有 "
                                 f"{len(getattr(sampling_metadata, name))} 项，"
                                 f"但 prompt_token_ids 有 {num_rows} 项")
        if num_tokens_per_req is None:
            if len(histories) != num_rows:
                raise ValueError(
                    f"历史有 {len(histories)} 行，元数据有 {num_rows} 行：逐行对应，必须相等")
        else:
            if len(num_tokens_per_req) != num_rows:
                raise ValueError(
                    f"num_tokens_per_req 有 {len(num_tokens_per_req)} 项，元数据有 {num_rows} 行："
                    f"前者是**逐请求**的草稿数，行序必须与元数据一致（不能只给一部分请求）")
            if len(histories) != sum(num_tokens_per_req):
                raise ValueError(
                    f"历史有 {len(histories)} 行，但按草稿数展开应该是 "
                    f"{sum(num_tokens_per_req)} 行（ΣK）：验证行数 = 各请求草稿数之和")

        updates = {"output_token_ids": histories}
        if num_tokens_per_req is not None:
            for name in ("temperature", "top_k", "top_p", "presence_penalties",
                         "frequency_penalties", "repetition_penalties"):
                tensor = getattr(sampling_metadata, name)
                if tensor is not None:
                    updates[name] = expand_batch_to_tokens(tensor, num_tokens_per_req)
            updates["prompt_token_ids"] = [
                prompt for prompt, count in zip(sampling_metadata.prompt_token_ids,
                                                num_tokens_per_req)
                for _ in range(count)]
            updates["min_tokens"] = [value for value, count in zip(
                sampling_metadata.min_tokens, num_tokens_per_req) for _ in range(count)]
            updates["stop_token_ids"] = [value for value, count in zip(
                sampling_metadata.stop_token_ids, num_tokens_per_req) for _ in range(count)]
            # `generators` **按请求下标留原样**（不展开、也不清空）：抽样按
            # `enumerate(num_draft_tokens)` 的下标取，与 vLLM 的
            # `generate_uniform_probs(..., generators, ...)` 同一套键。清空的话，
            # 有 seed 的请求会退化成用全局 RNG → 结果随全局种子变（验收方的独立探针抓到了）
        return dataclasses.replace(sampling_metadata, **updates)

    @staticmethod
    def _apply_penalties(logits: torch.Tensor, sampling_metadata: SamplingMetadata,
                         num_tokens_per_req: list[int]) -> torch.Tensor:
        if sampling_metadata.no_penalties:
            return logits
        return apply_all_penalties(
            logits, sampling_metadata.prompt_token_ids, sampling_metadata.output_token_ids,
            sampling_metadata.presence_penalties, sampling_metadata.frequency_penalties,
            sampling_metadata.repetition_penalties)

    @staticmethod
    def _apply_constraints(logits: torch.Tensor, sampling_metadata: SamplingMetadata,
                           num_tokens_per_req: list[int]) -> torch.Tensor:
        """温度 + top-k/top-p（贪心行不做温度缩放，与普通采样器同一条规则）。"""
        if sampling_metadata.all_greedy:
            return logits
        temperature = sampling_metadata.temperature
        safe = torch.where(temperature < SAMPLING_EPS, torch.ones_like(temperature),
                           temperature)
        logits = logits.div_(safe.unsqueeze(dim=1))
        return apply_top_k_top_p(logits, sampling_metadata.top_k, sampling_metadata.top_p)

    # -------- 随机数 --------

    @staticmethod
    def _draw_uniforms(metadata, target_metadata: SamplingMetadata,
                       device) -> torch.Tensor:
        """每个草稿位置一个均匀随机数。**K=0 的请求不消耗随机数**（199 §8）。

        用 float64：float32 下 `rand()` 有非零概率给出精确 0（PyTorch 的老问题），
        那样 `p/q >= 0` 会无条件接受，破坏分布。
        """
        uniforms = torch.rand((metadata.num_draft_tokens_total,), dtype=torch.float64,
                              device=device)
        start = 0
        for req_index, num_draft in enumerate(metadata.num_draft_tokens):
            if num_draft == 0:
                continue
            generator = target_metadata.generators.get(req_index)
            if generator is not None:
                uniforms[start:start + num_draft].uniform_(generator=generator)
            start += num_draft
        return uniforms

    @staticmethod
    def _draw_recoveries(metadata, target_metadata: SamplingMetadata,
                         target_probs: torch.Tensor, draft_probs: torch.Tensor | None,
                         device) -> torch.Tensor:
        """按 `max(p - q, 0)` 采 recovered token：指数竞赛（**不需要归一化**）。

        每个请求一行噪声（vLLM 同款），因为它是对整条词表采一次。
        """
        vocab_size = target_probs.shape[-1]
        noise = torch.empty((len(metadata.num_draft_tokens), vocab_size),
                            dtype=target_probs.dtype, device=device)
        noise.exponential_()
        for req_index, generator in target_metadata.generators.items():
            if metadata.num_draft_tokens[req_index] > 0:
                noise[req_index].exponential_(generator=generator)
        inv_noise = noise.reciprocal()

        if draft_probs is None:
            # 点质量提议：q 只在草稿 token 上为 1，所以修正分布 = p 且把草稿位置剔掉
            weights = target_probs.clone()
            rows = torch.arange(weights.shape[0], device=device)
            weights[rows, metadata.draft_token_ids.to(device)] = 0.0
        else:
            weights = torch.clamp_min(target_probs - draft_probs, 0.0)

        req_of_row = torch.repeat_interleave(
            torch.arange(len(metadata.num_draft_tokens), device=device),
            torch.tensor(metadata.num_draft_tokens, device=device))
        # 每个验证行用**它所属请求**的那一行噪声：指数竞赛对未归一化的权重也成立
        scores = weights * inv_noise[req_of_row]
        return scores.argmax(dim=-1).to(torch.int64)
