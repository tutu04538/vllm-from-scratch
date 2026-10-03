"""**参考实现**（只给测试）：Torch 逐候选版本的拒绝采样（对应 vLLM `v1/sample/rejection_sampler.py`
的算法，但**不是**本项目的生产路径）。

为什么要有它（059 §5）：

- 生产路径是 `minivllm/sample/rejection_sampler.py` 的 **Triton 批量内核**（与上游一致）；
  本文件只是"同一算法的人眼可读版本"，**不允许**被生产代码 import。
- 用途一：**CPU 上的算法验证**——内核需要 CUDA，而接受/恢复/截断这些语义在 CPU 上也能验。
- 用途二：与内核**逐值差分**（同一批张量、同一组随机数下必须给出同样的 token 序列），
  这样"内核写错了"和"算法理解错了"能分开。

它与生产内核的已知差异（都只影响速度，不影响结果）：

    逐候选 Python 循环 + 标量取值（每个候选位 4~5 次 D2H），内核是每请求一个 program
    约束（温度/top-k/top-p）用 Torch 的 repeat_interleave 展开，内核用 `expand_kernel`
    随机数可以注入（`uniforms=` / `recoveries=`），内核只能按 `generators` 抽

`uniforms` / `recoveries` 注入的意义（199 §8）：把"算法错"与"随机流不同"分开——测试注入
固定值，就能对着**独立 CPU 公式**逐值比对接受判定与恢复 token。
"""

import dataclasses

import torch

from ..outputs import SamplerOutput
from ..sample.metadata import SAMPLING_EPS, SamplingMetadata
from ..sample.ops.penalties import apply_all_penalties
from ..sample.ops.topk_topp_sampler import apply_top_k_top_p
from ..sample.rejection_sampler import PLACEHOLDER_TOKEN_ID


def torch_expand_batch_to_tokens(x: torch.Tensor, num_tokens_per_req: list[int]) -> torch.Tensor:
    """`[B]` → `[P]`：第 i 条请求的 K_i 个验证行共用同一份参数。

    对应生产路径的 `expand_batch_to_tokens`（Triton `expand_kernel`）与
    vLLM 的同名函数：x=[a,b,c]、K=[2,3,1] → [a,a,b,b,b,c]。
    """
    counts = torch.tensor(num_tokens_per_req, dtype=torch.int64, device=x.device)
    return torch.repeat_interleave(x, counts, dim=0)


def combine_outputs_with_spec_tokens(output_token_ids: list[list[int]],
                                     spec_token_ids: list[list[int]]):
    """验证行的**假设历史**：第 j 行 = 已提交历史 + 草稿前缀 `[:j]`（各请求展开成 K 行）。

    对应生产路径的 `RejectionSampler._combine_outputs_with_spec_tokens`。没有草稿的请求不产出
    任何行——它的 K=0，target 行本来就是空的。
    """
    result: list[list[int]] = []
    for out, spec in zip(output_token_ids, spec_token_ids):
        for index in range(len(spec)):
            result.append(list(out) + list(spec[:index]))
    return result


class TorchRejectionSampler:
    def __init__(self, sampler) -> None:
        # 复用普通采样器做 bonus 与"修正分布"抽样：bonus 就是一次普通采样
        self.sampler = sampler

    # -------- 入口（与生产 `RejectionSampler.forward` 同签名）--------

    def forward(self, metadata, draft_probs: torch.Tensor | None, logits: torch.Tensor,
                sampling_metadata: SamplingMetadata, *, uniforms: torch.Tensor | None = None,
                recoveries: torch.Tensor | None = None) -> SamplerOutput:
        """`logits` 是 `[P+B, V]`（`logits_indices` 取完之后的行），行序就是 metadata 的顺序。"""
        # 行序契约（与生产路径同一条检查）：元数据与采样参数必须逐请求对应
        if len(sampling_metadata.prompt_token_ids) != len(metadata.num_draft_tokens):
            raise ValueError(
                f"采样元数据有 {len(sampling_metadata.prompt_token_ids)} 行，"
                f"但投机元数据里有 {len(metadata.num_draft_tokens)} 条请求：两者必须逐请求对应")

        # ---- 1) bonus token：用"全部草稿都接受"的历史，走一次普通采样 ----
        bonus_token_ids = self.sampler.forward(
            logits[metadata.bonus_logits_indices], sampling_metadata,
            predict_bonus_token=True).sampled_token_ids

        # ---- 2) 验证行的 p：历史按草稿前缀逐行不同，再施加惩罚与采样约束 ----
        target_logits = logits[metadata.target_logits_indices].to(torch.float32)
        target_metadata = self._with_histories(
            sampling_metadata,
            combine_outputs_with_spec_tokens(sampling_metadata.output_token_ids,
                                             sampling_metadata.spec_token_ids),
            num_tokens_per_req=metadata.num_draft_tokens)
        target_logits = self.sampler.apply_logits_processors(target_logits, target_metadata)
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
        batch_size = len(metadata.num_draft_tokens)
        max_spec_len = metadata.max_spec_len
        device = target_logits.device
        output = torch.full((batch_size, max_spec_len + 1), PLACEHOLDER_TOKEN_ID,
                            dtype=torch.int32, device=device)
        if max_spec_len == 0:
            # 全批 K=0：等价于普通解码，最后一行（= bonus 行）就是答案
            return bonus_token_ids.to(torch.int32)

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
            # 注意用**逐请求**的 `sampling_metadata.temperature`（[B]）来展开成 [P] 行
            is_greedy_rows = torch_expand_batch_to_tokens(
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
        # q[d] == 0 防御性拒绝（生产内核同样处理，避免 p/q 得到 NaN）
        accepted = draft_prob > 0.0 and target_prob / draft_prob >= float(uniforms[row])
        token = draft_token if accepted else int(recoveries[row])
        return token, accepted

    # -------- 三个辅助：历史、约束、随机数 --------

    @staticmethod
    def _with_histories(sampling_metadata: SamplingMetadata, histories: list[list[int]],
                        num_tokens_per_req: list[int] | None = None) -> SamplingMetadata:
        """复制一份元数据、换掉逐行的历史与逐请求参数（**不改原对象**）。

        `num_tokens_per_req` 给了就把 `[B]` 的参数展开成 `[P]`（验证行是"每请求 K 行"）。
        """
        # ---- 契约检查 ----
        # 这个方法把"逐请求"的参数摊成"逐验证行"，**行序必须与传入的 sampling_metadata 一致**，
        # 长度也必须对得上。用 `zip()` 摊平会按短的那边**静默截断**：少给一项 min_tokens，
        # 那一行的停止 token 屏蔽就悄悄没了（不报错，只是行为变了）。
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
                    updates[name] = torch_expand_batch_to_tokens(tensor, num_tokens_per_req)
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
            # `generate_uniform_probs(..., generators, ...)` 同一套键
        return dataclasses.replace(sampling_metadata, **updates)

    @staticmethod
    def _apply_constraints(logits: torch.Tensor, sampling_metadata: SamplingMetadata,
                           num_tokens_per_req: list[int]) -> torch.Tensor:
        """温度 + top-k/top-p 的 Torch 版（对应生产 `apply_sampling_constraints`）。"""
        if sampling_metadata.all_greedy:
            return logits
        temperature = sampling_metadata.temperature
        safe = torch.where(temperature < SAMPLING_EPS, torch.ones_like(temperature),
                           temperature)
        logits = logits.div_(safe.unsqueeze(dim=1))
        top_k = (None if sampling_metadata.top_k is None else
                 torch_expand_batch_to_tokens(sampling_metadata.top_k, num_tokens_per_req))
        top_p = (None if sampling_metadata.top_p is None else
                 torch_expand_batch_to_tokens(sampling_metadata.top_p, num_tokens_per_req))
        return apply_top_k_top_p(logits, top_k, top_p)

    # -------- 随机数 --------

    @staticmethod
    def _draw_uniforms(metadata, target_metadata: SamplingMetadata,
                       device) -> torch.Tensor:
        """每个草稿位置一个均匀随机数。**K=0 的请求不消耗随机数**（199 §8）。

        用 float64：float32 下 `rand()` 有非零概率给出精确 0，那样 `p/q >= 0` 会无条件接受。
        """
        uniforms = torch.rand((metadata.draft_token_ids.shape[0],), dtype=torch.float64,
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
