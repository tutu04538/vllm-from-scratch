"""`RejectionSampler`：GPU 批量拒绝采样（对应 vLLM `v1/sample/rejection_sampler.py`）。

输入是"target 模型在 `logits_indices` 那些行上的 logits"和"草稿 token（+提议时的分布 q）"，
输出是**本轮真正提交的 token 序列**（每条请求 1 ~ K+1 个，padded 成 `[B, max_spec_len+1]`，
无效位置填 `PLACEHOLDER_TOKEN_ID = -1`）。它不碰 Request、不判停止——截断与结束是 Scheduler 的事。

### 算法（与上游一致，见 059 §3）

```text
greedy 行：逐位置比对草稿 == target 的 argmax
           第一个不同处 → 用 target 的 argmax 顶替，**后面全部丢掉**
           全部相同     → 追加 bonus token（target 在最后一行采出来的那个）
random 行：草稿 d 来自分布 q，接受概率 = min(1, p[d] / q[d])，用一次均匀随机数判定
           拒绝 → 用**修正分布** max(p - q, 0) 采一个 token 顶替（recovered token）
           全部接受 → 追加 bonus
```

`q` 必须是**实际提议时用的分布**（059 §3.5）：模型提议者提供 `draft_probs`；确定性提议
（ngram）没有分布（点质量），用 `draft_probs=None` 表示，内核走 `NO_DRAFT_PROBS` 分支——
此时 `q[d] = 1`、接受判定退化成 `p[d] >= u`（**不能**把"没有 q"当"按 p 随便采"）。
`q[d] == 0` 时内核防御性拒绝（否则 `p/q` 出 NaN）。

### 为什么是内核而不是逐候选 Python 循环（059 §1）

验证是**每一步都要付**的固定开销，而且批越大、K 越大、接受率越高（越该赢）开销越大。
逐候选取标量意味着每个候选位 4~5 次 D2H 同步（读 argmax / p[d] / u / recovered / 写回），
整条流水线每次都被 flush。实测（RTX 5090 Laptop、V=1000、B=32、K=5、全接受）：
Torch 逐候选 29.2 ms/步（837 次 D2H），上游同层 Triton 内核 0.079 ms/步，且几乎不随批大小变。
所以本关把验证搬到 GPU：**CPU 只在 `parse_output()` 交付边界上出现一次**。

### 职责划分（059 §2，与上游同名同顺序）

    forward()                  组织：bonus 采样 → target logits 处理 → 拒绝采样
    apply_logits_processors()  惩罚 + min_tokens 屏蔽（按"假设历史"逐行展开）
    _get_logprobs_tensors()    logprobs（68 关；本关没有 logprobs，调用即报错）
    parse_output()             `[B, K+1]` → `list[list[int]]`（丢掉 -1 与越界 id）

`bonus` 走**普通采样器**（所以 top-k/top-p 生效，草稿验证不走）；验证行的温度/top-k/top-p 由
`apply_sampling_constraints()` 按请求展开后施加。**只有 `standard`**：`synthetic` / V2 `block`
在 75 关独立验收（059 §5），本关遇到就明确报错，不静默降级。

### 设备

拒绝采样是 Triton 内核（上游同样只有 GPU 路径）。CPU 上的**算法**对照用
`minivllm/testing/torch_rejection_sampler.py` 的参考实现（只给测试，生产路径不 import）。
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

from ..config import PROCESSED_LOGPROBS_MODES
from ..outputs import LogprobsTensors, SamplerOutput
from .metadata import SamplingMetadata
from .ops.penalties import apply_all_penalties
from .ops.topk_topp_sampler import apply_top_k_top_p
from .sampler import Sampler

if TYPE_CHECKING:
    # 只在类型标注里用到：`spec_decode/__init__` 会 import 提议者（提议者又要 import 本包），
    # 运行时 import 会成环。上游没有这个问题是因为它的 `v1/spec_decode/__init__.py` 是空的。
    from ..spec_decode.metadata import SpecDecodeMetadata

# 无效位置的填充值（上游同名常量）：`[B, max_spec_len+1]` 的右边部分填它
PLACEHOLDER_TOKEN_ID = -1
# 温度等于它 = 贪心（上游用同一个阈值做 `temperature == 0` 的判定）
GREEDY_TEMPERATURE = 0
# 单条请求一轮允许的最大草稿数（上游同名常量，用于避免 `expand_kernel` 反复重编译）
MAX_SPEC_LEN = 128


class RejectionSampler:
    """
    The implementation strictly follows the algorithm described in
        https://arxiv.org/abs/2211.17192.
    However, we want to clarify the terminology used in the implementation:
    accepted tokens:
        tokens that are accepted based on the relationship between the "raw"
        draft and target probabilities.
    recovered tokens:
        tokens that are sampled based on the adjusted probability
        distribution, which is derived from both the draft and target
        probabilities.
    bonus tokens:
        If all proposed tokens are accepted, the bonus token is added to the
        end of the sequence. The bonus token is only sampled from the target
        probabilities.
    output tokens:
        output tokens = accepted tokens + recovered tokens + bonus tokens
    """

    def __init__(self, sampler: Sampler, spec_config=None) -> None:
        # 复用普通采样器做 bonus 与"修正分布"抽样：bonus 就是一次普通采样
        self.sampler = sampler
        # 68 关：logprobs 的四种模式决定"要交付的那一份"是原始的还是处理过的。
        # 两个判据与上游同名属性一一对应（`PROCESSED_LOGPROBS_MODES` / 是否是 logits 模式）。
        self.logprobs_mode = sampler.logprobs_mode
        self.is_processed_logprobs_mode = self.logprobs_mode in PROCESSED_LOGPROBS_MODES
        self.is_logits_logprobs_mode = self.logprobs_mode in ("raw_logits",
                                                             "processed_logits")
        # 本关只接 standard（059 §5）。上游同一处按 `rejection_sample_method` 分
        # synthetic / block；那两条在 75 关验收，这里遇到就报错而不是悄悄按 standard 跑。
        method = getattr(spec_config, "rejection_sample_method", "standard")
        if method != "standard":
            raise ValueError(
                f"拒绝采样方法 {method!r} 本关不支持（只接 'standard'）：synthetic 与 V2 block "
                f"验证按需求顺序在 75 关实现；不要用 standard 的结果冒充它们")

    def __call__(self, metadata: SpecDecodeMetadata, draft_probs: torch.Tensor | None,
                 logits: torch.Tensor, sampling_metadata: SamplingMetadata) -> SamplerOutput:
        return self.forward(metadata, draft_probs, logits, sampling_metadata)

    # -------- 入口 --------

    def forward(self, metadata: SpecDecodeMetadata, draft_probs: torch.Tensor | None,
                logits: torch.Tensor, sampling_metadata: SamplingMetadata) -> SamplerOutput:
        """`logits` 是 `[P+B, V]`（`logits_indices` 取完之后的行），行序就是 metadata 的顺序。

        行序契约（2026-10-02 补）：`sampling_metadata` 必须正好覆盖 spec metadata 里的那些请求、
        且顺序一致——验证行的历史、惩罚、min_tokens 全靠它按请求摊到逐行。少了或换了顺序都不会
        报错，只会把 A 的参数用到 B 的行上（静默错），所以在这里挡住。
        """
        # 上游逐字：超过 MAX_SPEC_LEN 的批次直接断言失败（内核按最大长度编译）
        assert metadata.max_spec_len <= MAX_SPEC_LEN
        if len(sampling_metadata.prompt_token_ids) != len(metadata.num_draft_tokens):
            raise ValueError(
                f"采样元数据有 {len(sampling_metadata.prompt_token_ids)} 行，"
                f"但投机元数据里有 {len(metadata.num_draft_tokens)} 条请求："
                f"两者必须逐请求对应、顺序一致（不能只对一部分行建元数据）")

        # ---- 1) bonus token：用"全部草稿都接受"的历史，走一次普通采样 ----
        # `predict_bonus_token=True` 让采样器把惩罚的历史算成"已提交 + 全部草稿"——能走到
        # bonus 就说明草稿都被接受了（上游同名参数）。
        bonus_logits = logits[metadata.bonus_logits_indices]
        bonus_sampler_output = self.sampler.forward(
            bonus_logits, replace(sampling_metadata, max_num_logprobs=-1),
            predict_bonus_token=True,
            # 68 关（上游同款）：bonus 这一行的 logprobs 要与候选行拼在一起再统一算，
            # 所以这里强制它交回 **logits**（`raw_logits` / `processed_logits`），
            # 而不是它自己那份 logprobs。
            logprobs_mode_override=("processed_logits"
                                    if self.is_processed_logprobs_mode else "raw_logits"))
        bonus_token_ids = bonus_sampler_output.sampled_token_ids

        # ---- 2) 验证行的 p：先按"草稿前缀"逐行算历史，再施加惩罚/约束 ----
        # 索引出的张量有独立存储（`logits[...]` 不是视图），所以下面的原地操作不会污染原 logits
        raw_target_logits = logits[metadata.target_logits_indices]
        # 用 float32 算概率（上游同款：fp16 的 logits 在 softmax 前升精度）
        raw_target_logits = raw_target_logits.to(torch.float32)
        target_logits = raw_target_logits
        if not self.is_processed_logprobs_mode:
            # `apply_logits_processors` 会**原地**改 logits；raw_* 模式要交付"改之前"的那一份，
            # 所以先复制一份留底（上游同款注释）。
            target_logits = target_logits.clone()
        target_logits = self.apply_logits_processors(target_logits, sampling_metadata, metadata)
        # 温度 / top-k / top-p：按请求展开到每个验证行（上游在这里原地改 logits）
        target_logits = apply_sampling_constraints(
            target_logits, metadata.cu_num_draft_tokens, sampling_metadata)

        # ---- 3) 批量拒绝采样（greedy 内核 + random 内核 + recovered token）----
        output_token_ids = rejection_sample(
            metadata.draft_token_ids,
            metadata.num_draft_tokens,
            metadata.max_spec_len,
            metadata.cu_num_draft_tokens,
            draft_probs,
            target_logits,
            bonus_token_ids,
            sampling_metadata,
        )

        # ---- 4) logprobs（68 关）----
        logprobs_tensors = None
        if sampling_metadata.max_num_logprobs is not None:
            logprobs_tensors = self._get_logprobs_tensors(
                sampling_metadata.max_num_logprobs,
                metadata,
                logits,
                # processed 模式交付处理过的 logits，raw 模式交付留底的那一份（上游同款三元）
                target_logits if self.is_processed_logprobs_mode else raw_target_logits,
                bonus_sampler_output.logprobs_tensors.logprobs,
                output_token_ids,
            )
        return SamplerOutput(sampled_token_ids=output_token_ids,
                             logprobs_tensors=logprobs_tensors)

    # -------- logprobs（68 关）--------

    def _get_logprobs_tensors(self, max_num_logprobs: int, metadata: SpecDecodeMetadata,
                              logits: torch.Tensor, target_logits: torch.Tensor,
                              bonus_logits: torch.Tensor,
                              sampled_token_ids: torch.Tensor) -> LogprobsTensors:
        """投机下的 logprobs（上游同名方法，逐行对应）。

        ### 行索引怎么来的（068 §3.5 的"各自索引正确"）

        一轮里交付的 token 可能是"接受的候选 + 恢复 token + bonus"，每个位置该读**哪一行**：

            第 j 个位置（j < 本请求排了几个候选位）→ 紧凑行号 `start + j`（候选 j 的 target 行）
            拒绝发生在第 a 个位置 → 恢复 token 也读 `start + a`（同一个位置的 p）
            全部接受 → bonus 位读 `start + K`（bonus 行）

        `final_logits` 就是按这个坐标铺的（上游同款）：候选行放**处理过**的 target logits、
        bonus 行放 bonus 采样器交回的那一份。于是"读第 j 行"天然满足上面的规则，
        **不需要**知道这一轮到底接受了几个。

        ### 为什么"多算"却不算错

        上游注释：为了避免 CPU-GPU 同步（要等 valid_mask 才知道每请求几个有效位），这里对
        **所有**候选位（含被拒绝的）都算一份 logprobs，多出来的部分在 `parse_output` 里由
        **同一张 valid_mask** 滤掉。所以"截断后的尾部不能漏出"靠的不是少算，而是**同一把尺子**。

        ### 与上游的两处差异（都写进 docs/step68_alignment.md）

        1. `max_num_logprobs == -1`（全词表）在投机下**上游会运行期报错**：它把 -1 直接传给
           `torch.topk`（实测 `RuntimeError: selected index k out of range`）。本仓库在进
           topk 之前就明确拒绝，报错信息说清"这条组合不支持"，语义与上游一致（都是不支持），
           只是失败点更靠前、原因更直白。
        2. 全词表在**非投机**路径上的行为见 sampler.py（上游把整份分布交出去，输出处理阶段
           又把 top-k 那一列当成"名次"来解释）。本仓库照抄，不在这里"顺手修好"。
        """
        if max_num_logprobs == -1:
            raise NotImplementedError(
                "投机 + logprobs=-1（全词表）上游同样不支持：它把 -1 直接交给 torch.topk，"
                "会在采样时抛 RuntimeError('selected index k out of range')。本仓库提前拒绝，"
                "请改用 logprobs=k（k >= 1）。三态矩阵里这条属「上游不支持」")

        # 每请求的起始行号：`cu_num_sampled_tokens` 是累积末端，往左挪一格就是起点
        cu_num_sampled_tokens = torch.zeros_like(metadata.cu_num_sampled_tokens)
        cu_num_sampled_tokens[1:] = metadata.cu_num_sampled_tokens[:-1]

        bonus_logits_indices = metadata.bonus_logits_indices
        target_logits_indices = metadata.target_logits_indices
        final_logits = torch.zeros_like(logits, dtype=torch.float32)
        final_logits[target_logits_indices] = target_logits.to(torch.float32)
        final_logits[bonus_logits_indices] = bonus_logits.to(torch.float32)

        logit_start_indices = cu_num_sampled_tokens
        offsets = torch.arange(sampled_token_ids.shape[-1],
                               device=logit_start_indices.device,
                               dtype=logit_start_indices.dtype)
        accepted_logit_indices = (logit_start_indices.unsqueeze(1)
                                  + offsets.unsqueeze(0)).flatten()
        # 越界兜底（上游同款）：padding 位会算出 `start + K` 之类的下标，夹到最后一行为止；
        # 它们的结果随后就被 valid_mask 丢掉，不会交付
        accepted_logit_indices.clamp_(max=final_logits.shape[0] - 1)
        accepted_tokens = sampled_token_ids.clone().flatten()
        # -1（PLACEHOLDER）不能拿去 gather：换成 0 让下标合法（上游同款注释）
        accepted_tokens[accepted_tokens == PLACEHOLDER_TOKEN_ID] = 0

        accepted_logits = final_logits[accepted_logit_indices]
        accepted_logprobs = (accepted_logits if self.is_logits_logprobs_mode
                            else self.sampler.compute_logprobs(accepted_logits))
        return self.sampler.gather_logprobs(accepted_logprobs, max_num_logprobs,
                                            accepted_tokens.to(torch.int64))

    # -------- 输出解析（CPU 交付边界）--------

    @staticmethod
    def parse_output(output_token_ids: torch.Tensor, vocab_size: int,
                     discard_req_indices=(), logprobs_tensors=None,
                     ) -> tuple[list[list[int]], "LogprobsLists | None"]:
        """把 `[B, max_spec_len+1]` 解析成 `list[list[int]]`（上游同名静态方法）。

        被拒绝的位置由内核填了 `PLACEHOLDER_TOKEN_ID = -1`，这里一次性过滤掉。**这是整条
        验证路径唯一一次 D2H**（`cpu().numpy()`）：不在每个候选位置上取标量（059 §3.6）。

        `discard_req_indices` 里的行整行丢掉（中间 prefill 块：它的 logits 有效，但这一轮
        不该产出 token）。上游还用 `id < vocab_size` 兜住脏数据，这里保持一致。

        68 关：`logprobs_tensors` 用**同一张 `valid_mask`** 过滤，并给出每请求的有效位置数
        （`cu_num_tokens`）。这是"截断后的尾部不能漏出"的落点（068 §3.5）——被拒绝的候选位
        在 token 与 logprobs 两边**同时**消失，不会出现"token 少了一个、logprobs 还留着它"。
        """
        output_token_ids_np = output_token_ids.cpu().numpy()
        valid_mask = (output_token_ids_np != PLACEHOLDER_TOKEN_ID) & (
            output_token_ids_np < vocab_size)
        output_logprobs = None
        if logprobs_tensors is not None:
            # 每请求的有效位置数 → 行偏移；`slice_request` 靠它切（投机时每请求 0~K+1 个）
            cu_num_tokens = [0] + valid_mask.sum(axis=1).cumsum().tolist()
            filtered_tensors = logprobs_tensors.filter(valid_mask.flatten())
            output_logprobs = filtered_tensors.tolists(cu_num_tokens)

        if len(discard_req_indices) > 0:
            valid_mask[list(discard_req_indices)] = False
        outputs = [row[valid_mask[index]].tolist()
                   for index, row in enumerate(output_token_ids_np)]
        return outputs, output_logprobs

    # -------- logits 处理 --------

    def apply_logits_processors(self, logits: torch.Tensor,
                                sampling_metadata: SamplingMetadata,
                                metadata: SpecDecodeMetadata) -> torch.Tensor:
        """验证行的 logits 处理：惩罚 → min_tokens（顺序与上游一致）。

        历史是**假设历史**：第 j 个验证行按"已提交历史 + 草稿前缀 `[:j]`"算——因为如果前面
        都接受，这行面对的就是那个历史（059 §3.5）。被拒绝之后预先算好的行直接丢掉。
        """
        has_penalties = not sampling_metadata.no_penalties
        output_token_ids = sampling_metadata.output_token_ids
        if has_penalties:
            output_token_ids = self._combine_outputs_with_spec_tokens(
                output_token_ids, sampling_metadata.spec_token_ids)
            logits = self.apply_penalties(logits, sampling_metadata, metadata,
                                          output_token_ids)
        # min_tokens 的停止 token 屏蔽（上游：非 argmax 不变的 logits processor，在惩罚之后）
        logits = self.sampler.apply_min_tokens_for_spec_decode(
            logits, sampling_metadata, metadata.num_draft_tokens)
        return logits

    @staticmethod
    def apply_penalties(logits: torch.Tensor, sampling_metadata: SamplingMetadata,
                        metadata: SpecDecodeMetadata,
                        output_token_ids: list[list[int]]) -> torch.Tensor:
        """按请求把惩罚参数与历史展开到每个验证行，再调惩罚算子。

        上游用 `repeat_indices = arange(B).repeat_interleave(num_draft_tokens)` 去索引
        **GPU 上的** token/惩罚张量；本机的 `SamplingMetadata` 把历史存成 list（57D 的结构），
        所以这里按同一个展开规则在 CPU 侧重复（行序一致，值逐行相同）。
        惩罚值本身是张量，走 `expand_batch_to_tokens` 内核展开（与上游同一套展开方式）。
        """
        if sampling_metadata.no_penalties:
            return logits
        num_tokens = logits.shape[0]
        # 逐请求参数 → 逐验证行（每请求 K_i 行）
        prompt_token_ids = [prompt
                            for prompt, count in zip(sampling_metadata.prompt_token_ids,
                                                     metadata.num_draft_tokens)
                            for _ in range(count)]
        cu_num_draft_tokens = metadata.cu_num_draft_tokens
        return apply_all_penalties(
            logits, prompt_token_ids, output_token_ids,
            expand_batch_to_tokens(sampling_metadata.presence_penalties,
                                   cu_num_draft_tokens, num_tokens),
            expand_batch_to_tokens(sampling_metadata.frequency_penalties,
                                   cu_num_draft_tokens, num_tokens),
            expand_batch_to_tokens(sampling_metadata.repetition_penalties,
                                   cu_num_draft_tokens, num_tokens))

    @staticmethod
    def _combine_outputs_with_spec_tokens(
            output_token_ids: list[list[int]],
            spec_token_ids: list[list[int]] | None = None) -> list[list[int]]:
        """验证行的历史 = 已提交历史 + 草稿前缀（每个草稿位置一行，K=0 的请求不产出行）。

        与上游同名方法逐字一致（它也是"跳过没有草稿的请求"——那类请求在验证行里占 0 行）。
        注意：**Sampler 里那个同名方法是另一回事**（它对 bonus 行做 `out + 全部草稿`、行数不变）。
        """
        if spec_token_ids is None:
            return output_token_ids

        result = []
        for out, spec in zip(output_token_ids, spec_token_ids):
            if len(spec) == 0:
                continue
            result.append(out)
            for i in range(len(spec) - 1):
                result.append([*result[-1], spec[i]])
        return result


# ---------------------------------------------------------------------------
# 批量入口：组织 greedy / random 两条内核路径
# ---------------------------------------------------------------------------


def rejection_sample(
    # [num_tokens]
    draft_token_ids: torch.Tensor,
    # [batch_size]
    num_draft_tokens: list[int],
    max_spec_len: int,
    # [batch_size]
    cu_num_draft_tokens: torch.Tensor,
    # [num_tokens, vocab_size]
    draft_probs: torch.Tensor | None,
    # [num_tokens, vocab_size]
    target_logits: torch.Tensor,
    # [batch_size, 1]
    bonus_token_ids: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor:
    """上游同名函数（去掉 synthetic 模式与 fp64 gumbel 两个实验分支）。

    上游签名里还有 `synthetic_mode` / `synthetic_conditional_rates` / `use_fp64_gumbel`：
    前两个是 75 关的 V2 块验证，第三个是实验性的 fp64 指数竞赛（本机采样器没有这个开关），
    都不在本关范围（059 §5），所以这里不保留"传了也不生效"的参数。
    """
    assert draft_token_ids.ndim == 1
    assert draft_probs is None or draft_probs.ndim == 2
    assert cu_num_draft_tokens.ndim == 1
    assert target_logits.ndim == 2

    if not target_logits.is_cuda:
        raise NotImplementedError(
            f"拒绝采样走 Triton 内核，只支持 CUDA 张量（收到 {target_logits.device}）："
            "上游同样只在 GPU 执行路径上提供它。CPU 上的算法对照请用 "
            "minivllm/testing/torch_rejection_sampler.py 的参考实现")

    batch_size = len(num_draft_tokens)
    num_tokens = draft_token_ids.shape[0]
    vocab_size = target_logits.shape[-1]
    device = target_logits.device
    assert draft_token_ids.is_contiguous()
    assert draft_probs is None or draft_probs.is_contiguous()
    assert bonus_token_ids.is_contiguous()
    assert target_logits.shape == (num_tokens, vocab_size)

    # 输出缓冲：`[B, K+1]`，被拒绝的位置保持 PLACEHOLDER_TOKEN_ID
    output_token_ids = torch.full(
        (batch_size, max_spec_len + 1),
        PLACEHOLDER_TOKEN_ID,
        dtype=torch.int32,          # 与 SamplerOutput.sampled_token_ids 一致（上游同为 int32）
        device=device,
    )

    if sampling_metadata.all_greedy:
        is_greedy = None
    else:
        # 按请求的"是不是贪心"分流：两条内核各自早退不属于自己的请求
        is_greedy = sampling_metadata.temperature == GREEDY_TEMPERATURE

    # 均匀随机数只给非贪心批生成（贪心内核不需要它）
    uniform_probs: torch.Tensor | None = None
    if not sampling_metadata.all_greedy:
        uniform_probs = generate_uniform_probs(
            num_tokens, num_draft_tokens, sampling_metadata.generators, device)

    if not sampling_metadata.all_random:
        # 贪心验证：草稿 == target argmax 才接受，否则用 argmax 顶替并截断
        target_argmax = target_logits.argmax(dim=-1)
        rejection_greedy_sample_kernel[(batch_size,)](
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            target_argmax,
            bonus_token_ids,
            is_greedy,
            max_spec_len,
        )
        if sampling_metadata.all_greedy:
            return output_token_ids

    # 概率分布从 target logits 算（`apply_sampling_constraints` 已经把温度/top-k/top-p 施加过）
    target_probs = target_logits.softmax(dim=-1, dtype=torch.float32)
    assert target_probs.is_contiguous()

    # 每个验证位置先算好"被拒绝时用哪个 token"（`max(p-q, 0)` 的指数竞赛）
    recovered_token_ids = sample_recovered_tokens(
        max_spec_len,
        num_draft_tokens,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        sampling_metadata,
        device,
    )

    assert uniform_probs is not None
    rejection_random_sample_kernel[(batch_size,)](
        output_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        bonus_token_ids,
        recovered_token_ids,
        uniform_probs,
        is_greedy,
        max_spec_len,
        vocab_size,
        NO_DRAFT_PROBS=draft_probs is None,
    )
    return output_token_ids


def apply_sampling_constraints(
    logits: torch.Tensor,               # [num_tokens, vocab_size]
    cu_num_draft_tokens: torch.Tensor,  # [batch_size]
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor:
    """温度缩放 + top-k/top-p（上游同名函数：只有随机行要做，贪心行直接返回原 logits）。

    参数按请求展开到每个验证行：`expand_batch_to_tokens`。温度等于 0 的行（贪心行）在展开时
    换成 1.0，避免除以 0。
    """
    assert logits.ndim == 2
    assert cu_num_draft_tokens.ndim == 1
    if sampling_metadata.all_greedy:
        return logits

    num_tokens = logits.shape[0]
    temperature = expand_batch_to_tokens(
        sampling_metadata.temperature,
        cu_num_draft_tokens,
        num_tokens,
        replace_from=GREEDY_TEMPERATURE,
        replace_to=1,
    )
    # 原地改：目标 logits 是刚索引出来的独立张量，改它不影响调用者的原张量
    logits.div_(temperature.unsqueeze(-1))

    top_k = None
    if sampling_metadata.top_k is not None:
        top_k = expand_batch_to_tokens(
            sampling_metadata.top_k, cu_num_draft_tokens, num_tokens)
    top_p = None
    if sampling_metadata.top_p is not None:
        top_p = expand_batch_to_tokens(
            sampling_metadata.top_p, cu_num_draft_tokens, num_tokens)
    return apply_top_k_top_p(logits, top_k, top_p)


def expand_batch_to_tokens(
    x: torch.Tensor,                 # [batch_size]
    cu_num_tokens: torch.Tensor,     # [batch_size]
    num_tokens: int,
    replace_from: int = 0,
    replace_to: int = 0,
) -> torch.Tensor:
    """`[batch_size]` → `[num_tokens]`：按每请求的 token 数展开（上游同名函数）。

    例：x=[a,b,c]、cu_num_tokens=[2,5,6]、num_tokens=6 → expanded=[a,a,b,b,b,c]。
    `replace_from/replace_to` 用来把"贪心行的温度 0"换成 1（避免除以 0）。
    """
    batch_size = x.shape[0]
    assert cu_num_tokens.shape[0] == batch_size
    expanded_x = x.new_empty(num_tokens)
    expand_kernel[(batch_size,)](
        expanded_x,
        x,
        cu_num_tokens,
        replace_from,
        replace_to,
        MAX_NUM_TOKENS=MAX_SPEC_LEN,   # 固定常量，避免按实际长度反复重编译（上游同款）
    )
    return expanded_x


def generate_uniform_probs(
    num_tokens: int,
    num_draft_tokens: list[int],
    generators: dict[int, torch.Generator],
    device: torch.device,
) -> torch.Tensor:
    """每个验证位置一个 `[0, 1)` 的均匀随机数；有 seed 的请求走它自己的 generator。

    用 float64：float32 下 `rand()` 有非零概率给出精确 0（PyTorch 的老问题），
    那样 `p/q >= 0` 会无条件接受，破坏分布（上游同款注释）。

    **K=0 的请求不消耗随机数**（上游同款）——否则"这条请求这一轮没草稿"也会推进它的随机流，
    同一个 seed 下别人复现不出结果。
    """
    uniform_probs = torch.rand((num_tokens,), dtype=torch.float64, device=device)
    start_idx = 0
    for req_idx, n in enumerate(num_draft_tokens):
        if n == 0:
            continue
        end_idx = start_idx + n
        generator = generators.get(req_idx)
        if generator is not None:
            uniform_probs[start_idx:end_idx].uniform_(generator=generator)
        start_idx = end_idx
    return uniform_probs


def sample_recovered_tokens(
    max_spec_len: int,
    num_draft_tokens: list[int],
    # [batch_size]
    cu_num_draft_tokens: torch.Tensor,
    # [num_tokens]
    draft_token_ids: torch.Tensor,
    # [num_tokens, vocab_size]
    draft_probs: torch.Tensor | None,
    # [num_tokens, vocab_size]
    target_probs: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    device: torch.device,
) -> torch.Tensor:
    """按 `max(p - q, 0)` 采 recovered token：指数竞赛（**不需要归一化**）。

    "每条请求一行噪声"（上游同款）——它是对整条词表采一次，所以噪声是请求级的；
    行与请求的对应靠 `cu_num_draft_tokens` 在内核里切。
    """
    batch_size = len(num_draft_tokens)
    vocab_size = target_probs.shape[-1]
    q = torch.empty((batch_size, vocab_size), dtype=torch.float32, device=device)
    q.exponential_()
    for i, generator in sampling_metadata.generators.items():
        # K=0 的请求不采随机数（可复现性；上游同款）
        if num_draft_tokens[i] > 0:
            q[i].exponential_(generator=generator)

    inv_q = q.reciprocal()

    recovered_token_ids = torch.empty_like(draft_token_ids)
    BLOCK_SIZE = 8192
    sample_recovered_tokens_kernel[(batch_size, max_spec_len)](
        recovered_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        inv_q,
        vocab_size,
        BLOCK_SIZE,
        NO_DRAFT_PROBS=draft_probs is None,
    )
    return recovered_token_ids


# ---------------------------------------------------------------------------
# Triton 内核（与上游同名内核逐行对应；去掉 synthetic 分支）
# ---------------------------------------------------------------------------


@triton.jit
def rejection_greedy_sample_kernel(
    output_token_ids_ptr,        # [batch_size, max_spec_len + 1]
    cu_num_draft_tokens_ptr,     # [batch_size]
    draft_token_ids_ptr,         # [num_tokens]
    target_argmax_ptr,           # [num_tokens]
    bonus_token_ids_ptr,         # [batch_size]
    is_greedy_ptr,               # [batch_size] or None
    max_spec_len,
):
    """贪心验证：一个 program 处理一条请求，逐位置比对草稿与 target argmax。

    第一个不匹配的位置写入 target 的 argmax 并**停止**（后面的候选不再提交，输出缓冲里
    保持 -1）；全部匹配则追加 bonus token。
    """
    req_idx = tl.program_id(0)
    is_greedy = True if is_greedy_ptr is None else tl.load(is_greedy_ptr + req_idx)
    if not is_greedy:
        # 随机采样的请求由另一条内核处理
        return

    start_idx = (
        tl.zeros([], dtype=cu_num_draft_tokens_ptr.dtype.element_ty)
        if req_idx == 0
        else tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    )
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx

    rejected = False
    for pos in range(num_draft_tokens):
        if not rejected:
            draft_token_id = tl.load(draft_token_ids_ptr + start_idx + pos)
            target_argmax_id = tl.load(target_argmax_ptr + start_idx + pos).to(tl.int32)
            token_id = target_argmax_id
            rejected = draft_token_id != target_argmax_id
            tl.store(
                output_token_ids_ptr + req_idx * (max_spec_len + 1) + pos,
                token_id,
            )

    if not rejected:
        # 全部接受 → 追加 bonus token
        bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx)
        tl.store(
            output_token_ids_ptr + req_idx * (max_spec_len + 1) + num_draft_tokens,
            bonus_token_id,
        )


# NOTE: 上游同款：不做特化，避免 max_spec_len 变化时反复重编译
@triton.jit(do_not_specialize=["max_spec_len"])
def rejection_random_sample_kernel(
    output_token_ids_ptr,        # [batch_size, max_spec_len + 1]
    cu_num_draft_tokens_ptr,     # [batch_size]
    draft_token_ids_ptr,         # [num_tokens]
    draft_probs_ptr,             # [num_tokens, vocab_size] or None
    target_probs_ptr,            # [num_tokens, vocab_size]
    bonus_token_ids_ptr,         # [batch_size]
    recovered_token_ids_ptr,     # [num_tokens]
    uniform_probs_ptr,           # [num_tokens]
    is_greedy_ptr,               # [batch_size]
    max_spec_len,
    vocab_size,
    NO_DRAFT_PROBS: tl.constexpr,
):
    """随机验证：`min(1, p[d]/q[d]) >= u` 接受，拒绝处写 recovered token 并停止。"""
    req_idx = tl.program_id(0)
    is_greedy = tl.load(is_greedy_ptr + req_idx)
    if is_greedy:
        # 贪心的请求由另一条内核处理
        return

    start_idx = (
        tl.zeros([], dtype=cu_num_draft_tokens_ptr.dtype.element_ty)
        if req_idx == 0
        else tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    )
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx

    rejected = False
    for pos in range(num_draft_tokens):
        if not rejected:
            draft_token_id = tl.load(draft_token_ids_ptr + start_idx + pos)
            uniform_prob = tl.load(uniform_probs_ptr + start_idx + pos)
            if draft_token_id < 0:
                # -1 = 被 padding 的草稿位，必须直接判拒（不能拿它去索引概率）
                accepted = False
            else:
                if NO_DRAFT_PROBS:
                    # 点质量提议（ngram）：q[d] = 1
                    draft_prob = 1
                else:
                    draft_prob = tl.load(
                        draft_probs_ptr
                        + (start_idx + pos) * vocab_size
                        + draft_token_id
                    )
                target_prob = tl.load(
                    target_probs_ptr + (start_idx + pos) * vocab_size + draft_token_id
                )
                # q[d] 理论上是 0 的概率极低，但检查一下避免 NaN；真为 0 就拒绝
                accepted = draft_prob > 0 and target_prob / draft_prob >= uniform_prob
            if accepted:
                token_id = draft_token_id
            else:
                rejected = True
                token_id = tl.load(recovered_token_ids_ptr + start_idx + pos)
            tl.store(
                output_token_ids_ptr + req_idx * (max_spec_len + 1) + pos, token_id
            )

    if not rejected:
        # 全部接受 → 追加 bonus token
        bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx)
        tl.store(
            output_token_ids_ptr + req_idx * (max_spec_len + 1) + num_draft_tokens,
            bonus_token_id,
        )


# NOTE: 上游同款：replace_from/replace_to 不特化
@triton.jit(do_not_specialize=["replace_from", "replace_to"])
def expand_kernel(
    output_ptr,            # [num_tokens]
    input_ptr,             # [batch_size]
    cu_num_tokens_ptr,     # [batch_size]
    replace_from,
    replace_to,
    MAX_NUM_TOKENS: tl.constexpr,
):
    """`[B]` → `[P]`：每个请求的值复制 `K_i` 份（用累积长度切区间）。"""
    req_idx = tl.program_id(0)
    if req_idx == 0:
        start_idx = tl.zeros([], dtype=cu_num_tokens_ptr.dtype.element_ty)
    else:
        start_idx = tl.load(cu_num_tokens_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_tokens_ptr + req_idx)
    num_tokens = end_idx - start_idx

    src_val = tl.load(input_ptr + req_idx)
    src_val = tl.where(src_val == replace_from, replace_to, src_val)
    offset = tl.arange(0, MAX_NUM_TOKENS)
    tl.store(output_ptr + start_idx + offset, src_val, mask=offset < num_tokens)


@triton.jit
def sample_recovered_tokens_kernel(
    output_token_ids_ptr,        # [num_tokens]
    cu_num_draft_tokens_ptr,     # [batch_size]
    draft_token_ids_ptr,         # [num_tokens]
    draft_probs_ptr,             # [num_tokens, vocab_size] or None
    target_probs_ptr,            # [num_tokens, vocab_size]
    inv_q_ptr,                   # [batch_size, vocab_size]
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    NO_DRAFT_PROBS: tl.constexpr,
):
    """按 `max(p - q, 0) * (1/noise)` 取 argmax（指数竞赛；不需要归一化）。

    每个 (请求, 草稿位置) 一个 program；词表按 BLOCK_SIZE 分块扫，块内先取局部最大值。
    """
    req_idx = tl.program_id(0)
    start_idx = (
        tl.zeros([], dtype=cu_num_draft_tokens_ptr.dtype.element_ty)
        if req_idx == 0
        else tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    )
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx

    # 超出该请求草稿数的位置直接退出（网格是按 max_spec_len 开的）
    pos = tl.program_id(1)
    if pos >= num_draft_tokens:
        return

    token_idx = start_idx + pos

    if NO_DRAFT_PROBS:
        draft_token_id = tl.load(draft_token_ids_ptr + token_idx)

    max_val = tl.full((), float("-inf"), tl.float32)
    recovered_id = 0
    for v in range(0, vocab_size, BLOCK_SIZE):
        vocab_offset = v + tl.arange(0, BLOCK_SIZE)
        vocab_mask = vocab_offset < vocab_size

        if NO_DRAFT_PROBS:
            # 点质量：修正分布 = p 且把草稿位置本身剔掉
            prob = tl.load(
                target_probs_ptr + token_idx * vocab_size + vocab_offset,
                mask=(vocab_mask & (vocab_offset != draft_token_id)),
                other=0.0,
            )
        else:
            draft_prob = tl.load(
                draft_probs_ptr + token_idx * vocab_size + vocab_offset,
                mask=vocab_mask,
                other=0.0,
            )
            target_prob = tl.load(
                target_probs_ptr + token_idx * vocab_size + vocab_offset,
                mask=vocab_mask,
                other=0.0,
            )
            prob = tl.maximum(target_prob - draft_prob, 0.0)
            # 不需要 `prob / sum(prob)`：argmax 只关心相对大小

        inv_q = tl.load(
            inv_q_ptr + req_idx * vocab_size + vocab_offset,
            mask=vocab_mask,
            other=0.0,
        )

        score = prob * inv_q
        # 越界项打成 -inf，防止最后一块全是 0 概率时选出 >= vocab_size 的下标
        score = tl.where(vocab_mask, score, float("-inf"))
        local_max, local_id = tl.max(score, axis=0, return_indices=True)

        if local_max > max_val:
            max_val = local_max
            recovered_id = v + local_id

    recovered_id = tl.minimum(recovered_id, vocab_size - 1)
    tl.store(output_token_ids_ptr + token_idx, recovered_id)
