"""75 关的 Torch 数学参考：**块验证**（Sun et al. 2024）在 V2 张量布局上的逐位参考实现。

只给测试用（`minivllm/testing/` 的约定：生产包不留没有调用方的代码）。

### 为什么这份参考必须"照着源码递推写"，不能照着论文公式写

需求 075 §3.2 的原话是"`p_tau` 按源码递推，不能凭变量名自行替换成简单联合概率乘积"，§5 又写
"这一关必须以源码递推为实现基准"。源码递推指的是
`vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py` 里的三件事：

1. `_compute_cumulative_log_p_kernel`：`log_p_i = min(log_p_{i-1} + (log p(d_i) - log q(d_i)), 0)`
   —— **上限加在"累计值"上**（某一步比值 > 1 能补前面的亏空），不是每步各自 `min(1, p/q)` 再累乘。
   实测差别很大：K=3/σ=1.0 的合成分布下，按"每步 min(1,p/q) 再累乘"建模得 E[τ]=1.03，
   内核实测 1.75（差 70%）。
2. `_compute_local_residual_mass_kernel` + `_compute_global_residual_mass`：第 i 步的阈值
   `h_i = M / (M + 1 - p_i)`，其中
   `M = Σ_x max(p_i * M_b(x | 第 i+1 行) - M_s(x | 第 i+1 行), 0)`
   —— 残差质量取自**下一个位置**的行；`p_i = 1` 且 `M = 0`（草稿与 target 完全一致）时
   `h_i = 1.0`（`denom > 0` 的分支兜底）。一步草稿（one-hot q）时闭式化为 `p_i * (1 - M_b(d_{i+1}))`。
3. `_rejection_kernel` 的块分支：**不早停**——每一位都掷硬币，接受长度取**最后**一个
   `u_i <= h_i` 的位置；最后一位（`i == K-1`，没有"下一个位置"）的阈值就是 `h = p_i`。
4. `_resample_kernel` 的块分支：拒绝位的恢复分布是 `max(p_tau * M_b - M_s, 0)` 归一化，
   `p_tau = exp(cumulative_log_p[rejected_idx - 1])`（τ=0 时 `p_tau = 1`）；one-hot 草稿下
   `p_tau` 在归一化里约掉，所以内核直接"target 去掉被拒草稿 token"；`τ == K`（全接受）时
   恢复分布就是 bonus 行的 target 分布。

### 与内核的差别（有意为之）

- **随机源不同**：内核用 `tl_rand32(seed, pos)` 与 gumbel 竞赛；这里接受一组**显式给定的
  `u`**，并直接给出"交付 token 的精确分布"。所以测试里比的是**分布**（总变差 / 卡方），
  不是逐值相等（总纲 §5：比较采样系统时比分布）。
- 这里只建模**非贪心**请求：内核里贪心请求在 `_rejection_kernel` 的 `is_greedy` 分支就返回了，
  块验证的阈值/残差一个都不参与（`_compute_cumulative_log_p_kernel` 对 `temp == 0` 也是早退）。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

#: 词表维度的记号：本例把"一条请求"组织成 `[K+1, V]` 的行块
#: （第 0..K-1 行是草稿验证行，第 K 行是 bonus 行），与 V2 扁平布局的对应关系见
#: `tests/step75/spec75_helpers.py`。


@dataclass
class BlockCase:
    """一条请求的块验证输入（全部是 CPU 张量，形状写成具名含义）。

    target_logits  [K+1, V]  **已经按温度缩放过的** target logits（V2 的
                             `apply_sampling_params()` 在进内核前就除过温度）
    draft_logits   [K, V]    草稿分布的**原始** logits（未除温度；内核里现除，上游同款）
                    None     = 点质量草稿（one-hot，q(d)=1）
    draft_tokens   list[int] 长度 K；`-1` = padding 占位（内核判为"必拒"并终止块）
    temperature    float     该请求的温度（非 0；0 = 贪心，本参考不建模）
    """

    target_logits: torch.Tensor
    draft_logits: torch.Tensor | None
    draft_tokens: list[int]
    temperature: float

    def __post_init__(self) -> None:
        if self.temperature <= 0.0:
            raise ValueError("块验证的 Torch 参考只建模非贪心请求（temp > 0）")
        k = len(self.draft_tokens)
        assert self.target_logits.shape[0] == k + 1, self.target_logits.shape
        if self.draft_logits is not None:
            assert self.draft_logits.shape[0] == k, self.draft_logits.shape
        assert all(-1 <= t < self.target_logits.shape[1] for t in self.draft_tokens)

    @property
    def num_drafts(self) -> int:
        return len(self.draft_tokens)

    @property
    def vocab_size(self) -> int:
        return self.target_logits.shape[1]

    # ---------------- 概率 ----------------

    def target_probs(self) -> torch.Tensor:
        """`[K+1, V]`：target 分布（内核 `_compute_global_logsumexp` 的结果）。"""
        return torch.softmax(self.target_logits.double(), dim=-1)

    def draft_probs(self) -> torch.Tensor | None:
        """`[K, V]` 或 None：草稿分布 = softmax(draft_logits / temperature)（内核同款）。"""
        if self.draft_logits is None:
            return None
        return torch.softmax(self.draft_logits.double() / self.temperature, dim=-1)

    # ---------------- 1) 前缀联合比 p_i ----------------

    def prefix_joint_ratio(self) -> torch.Tensor:
        """`[K]`：`p_i = exp(cumulative_log_p[start + i])`（内核第 1 件事）。

        递推里 `-1` 占位**跳过归约并沿用上一个累计值**（内核注释：placeholders can never be
        accepted, skip their reductions and carry the last valid cumulative value）。
        """
        tp = self.target_probs()
        dp = self.draft_probs()
        log_p = 0.0
        out = []
        for i, token in enumerate(self.draft_tokens):
            if token >= 0:
                log_target = torch.log(tp[i, token]) if tp[i, token] > 0 else -torch.inf
                if dp is None:
                    log_draft = torch.zeros((), dtype=torch.float64)
                else:
                    log_draft = torch.log(dp[i, token]) if dp[i, token] > 0 else -torch.inf
                log_p = min(log_p + float(log_target - log_draft), 0.0)
            out.append(log_p)
        return torch.exp(torch.tensor(out, dtype=torch.float64))

    # ---------------- 2) 阈值 h_i ----------------

    def residual_mass(self, i: int) -> float:
        """第 i 步用的残差质量 `M`（取自**第 i+1 行**，与内核 `logit_idx + 1` 一致）。

        - 有草稿分布：`Σ_x max(p_i * M_b(x) - M_s(x), 0)`（`_compute_local_residual_mass_kernel`）。
        - 点质量草稿：闭式 `p_i * (1 - M_b(d_{i+1}))`（`_compute_global_residual_mass` 的
          `HAS_DRAFT_LOGITS=False` 分支）。
        - 下一步的草稿是 `-1` 占位 → 该位置的残差质量没被算过（内核早退），调用方不该问。
        """
        p_i = float(self.prefix_joint_ratio()[i])
        tp = self.target_probs()
        dp = self.draft_probs()
        if dp is None:
            next_token = self.draft_tokens[i + 1]
            assert next_token >= 0
            return p_i * (1.0 - float(tp[i + 1, next_token]))
        partial = torch.clamp(p_i * tp[i + 1] - dp[i + 1], min=0.0)
        return float(partial.sum())

    def thresholds(self) -> torch.Tensor:
        """`[K]`：块验证每一位的接受阈值 `h_i`（第 3 步的判定用它）。

        `i == K-1`（最后一位，没有"下一行"）→ `h_i = p_i`；否则
        `h_i = M / (M + 1 - p_i)`，`denom <= 0` 时取 1.0（内核的 `tl.where` 兜底）。
        """
        p = self.prefix_joint_ratio()
        out = []
        for i in range(self.num_drafts):
            if i == self.num_drafts - 1 or self.draft_tokens[i + 1] < 0:
                # 下一位不存在（最后一位）或下一位是占位：内核取 h = p_i
                # （占位那条：`next_draft_token >= 0` 不成立 → else 分支）
                out.append(float(p[i]))
                continue
            m = self.residual_mass(i)
            denom = m + 1.0 - float(p[i])
            out.append(m / denom if denom > 0.0 else 1.0)
        return torch.tensor(out, dtype=torch.float64)

    # ---------------- standard 的逐位接受概率（对照用） ----------------

    def standard_accept_probs(self) -> torch.Tensor:
        """`[K]`：standard 的逐位接受概率（内核标准分支 `p(d) > u * q(d)`）。

        `c_i = min(1, p(d_i)/q(d_i))`；点质量草稿时 `q=1` → `c_i = p(d_i)`；
        `d_i` 是占位 → 必拒（0）。
        """
        tp = self.target_probs()
        dp = self.draft_probs()
        out = []
        for i, token in enumerate(self.draft_tokens):
            if token < 0:
                out.append(0.0)
                continue
            p = float(tp[i, token])
            q = 1.0 if dp is None else float(dp[i, token])
            out.append(min(1.0, p / q) if q > 0 else 0.0)
        return torch.tensor(out, dtype=torch.float64)

    # ---------------- 3) 接受长度 ----------------

    def acceptance_length(self, u: torch.Tensor) -> int:
        """给定每个草稿位一个均匀随机数 `u`（`[K]`，∈ (0,1)〕，返回接受长度 τ。

        块验证：τ = **最后**一个 `u_i <= h_i` 的位置 + 1（都不满足则 0）——**不早停**。
        占位之前/之后的行为也照抄内核：`verifying &= is_valid_draft`，占位处一旦出现，
        后面的位不再参与（`h` 也不再被检查）。
        """
        h = self.thresholds()
        accepted = 0
        for i in range(self.num_drafts):
            if self.draft_tokens[i] < 0:
                break  # 内核：verifying &= is_valid_draft → 这里之后的位不参与
            if float(u[i]) <= float(h[i]):
                accepted = i + 1
        return accepted

    def standard_acceptance_length(self, u: torch.Tensor) -> int:
        """standard：第一次 `u_i > c_i` 就停（早停），返回接受长度。"""
        c = self.standard_accept_probs()
        accepted = 0
        for i in range(self.num_drafts):
            if self.draft_tokens[i] < 0 or float(u[i]) > float(c[i]):
                break
            accepted += 1
        return accepted

    def acceptance_length_distribution(self, block: bool = True) -> torch.Tensor:
        """`[K+1]`：τ 的精确分布（u 独立均匀）。

        块验证有闭式：`P(τ = n) = h_{n-1} * Π_{j>=n} (1 - h_j)`（n ≥ 1），
        `P(τ = 0) = Π_j (1 - h_j)`；standard 是连乘后一次失败。
        """
        if block:
            h = [float(x) for x in self.thresholds()]
            out = [1.0] * (self.num_drafts + 1)
            tail = 1.0
            for n in range(self.num_drafts, 0, -1):
                out[n] = h[n - 1] * tail
                tail *= 1.0 - h[n - 1]
            out[0] = tail
        else:
            c = [float(x) for x in self.standard_accept_probs()]
            out = [0.0] * (self.num_drafts + 1)
            prefix = 1.0
            for n in range(self.num_drafts):
                out[n] = prefix * (1.0 - c[n])
                prefix *= c[n]
            out[self.num_drafts] = prefix
        return torch.tensor(out, dtype=torch.float64)

    def mean_acceptance_length(self, block: bool = True) -> float:
        dist = self.acceptance_length_distribution(block=block)
        return float((dist * torch.arange(len(dist), dtype=torch.float64)).sum())

    # ---------------- 4) 恢复/bonus 分布 ----------------

    def residual_probs(self, tau: int, block: bool = True) -> torch.Tensor:
        """`[V]`：接受长度为 `tau` 时，内核在"被拒/bonus 位"上**实际采样**的分布。

        四种情形（顺序照抄 `_resample_kernel`）：

        1. `tau == K`（全接受）→ bonus 位就是 target 分布（`is_bonus` 分支：直接用 target logits）。
        2. `draft_tokens[tau] < 0`（占位）→ 也直接用 target 分布
           （`is_valid_rejected_draft=False` 分支）。
        3. 有草稿分布 → `max(p_tau * M_b - M_s, 0)`，`p_tau = p_{tau-1}`（τ=0 时 1.0）。
           `block=False` 时 `p_tau` 恒为 1（standard 的 `max(M_b - M_s, 0)`）。
        4. 点质量草稿 → "target 去掉被拒草稿 token"（块验证的 `p_tau` 在归一化里约掉）。
        """
        tp = self.target_probs()
        dp = self.draft_probs()
        if tau >= self.num_drafts or self.draft_tokens[tau] < 0:
            return tp[tau]
        if dp is None:
            out = tp[tau].clone()
            out[self.draft_tokens[tau]] = 0.0
            return out / out.sum()
        p_tau = 1.0 if (tau == 0 or not block) else float(self.prefix_joint_ratio()[tau - 1])
        residual = torch.clamp(p_tau * tp[tau] - dp[tau], min=0.0)
        return residual / residual.sum()

    def delivered_distribution(self, block: bool = True) -> torch.Tensor:
        """`[V]`：**交付 token 的精确分布**（对 τ 的分布做混合）。

        交付序列 = `d_0..d_{τ-1}` + 一枚"被拒位恢复出来"或"bonus"的 token，所以测试可以直接
        比这枚尾巴 token 的分布（比"只比接受率"强：它同时约束恢复分布的公式）。
        """
        dist = self.acceptance_length_distribution(block=block)
        out = torch.zeros(self.vocab_size, dtype=torch.float64)
        for tau in range(self.num_drafts + 1):
            weight = float(dist[tau])
            if weight == 0.0:
                continue
            out += weight * self.residual_probs(tau, block=block)
        return out


def total_variation(p: torch.Tensor, q: torch.Tensor) -> float:
    """两个分布的总变差距离 `0.5 * Σ|p - q|`（测试里用它比分布，不用逐值相等）。

    张量可以先搬到 CPU（GPU 上的经验直方图与这份参考都在 CPU 上比，避免设备不匹配）。
    """
    p = p.detach().to(device="cpu", dtype=torch.float64)
    q = q.detach().to(device="cpu", dtype=torch.float64)
    return float(0.5 * (p - q).abs().sum())
