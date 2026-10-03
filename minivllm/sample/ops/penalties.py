"""三种惩罚算子（对应 vLLM `v1/sample/ops/penalties.py` + `model_executor/layers/utils.py`）。

它们**改变 argmax**（所以必须在 greedy 之前应用，见 `sampler.py` 的顺序说明）：

    repetition_penalty  出现过的 token：logits 为正就除以它、为负就乘以它
    frequency_penalty   出现过的 token：减去 `频率 × 出现次数`
    presence_penalty    出现过的 token：减去 `presence`（不管出现几次，都只减一次）

"出现过"= prompt + **已提交**的 output（草稿不算，57E）。

实现上照抄 vLLM 的两步：先把 token 打成"每行每 token 出现几次"的 bin counts（`scatter_add_`
到一个多出一列的缓冲里，那一列专门吸收 padding 值 `vocab_size`），再按公式改 logits。
`repetition` 的写法与 vLLM 的 `apply_repetition_penalties_torch` 逐行一致（它是 CUDA 算子的
参考实现）：

    penalties = where(prompt_mask | output_mask, repetition_penalty, 1.0)
    logits *= where(logits > 0, 1 / penalties, penalties)

**本关的差异**：vLLM 在 `InputBatch` 里维护 CPU 张量、按需传到 GPU，这里按行传 list、在算子
内部拼成 padded 张量（批量小，且这样更好逐值对照）。padding 值用 `vocab_size` —— 它不对应
任何合法 token，正好落在那个多出来的一列上。
"""

import torch


def token_bin_counts_and_mask(tokens: torch.Tensor, vocab_size: int,
                             num_seqs: int) -> tuple[torch.Tensor, torch.Tensor]:
    """`[num_seqs, max_len]` 的 token 表 → 每个 token 出现几次 + 出现过没有。

    缓冲多开一列（`vocab_size + 1`）：padding 值就是 `vocab_size`，让它落进那一列、
    统计完再切掉——省掉一次 masked 处理。
    """
    bin_counts = torch.zeros((num_seqs, vocab_size + 1), dtype=torch.long,
                             device=tokens.device)
    bin_counts.scatter_add_(1, tokens, torch.ones_like(tokens))
    bin_counts = bin_counts[:, :vocab_size]
    return bin_counts, bin_counts > 0


def _to_padded(rows: list[list[int]], vocab_size: int, device) -> torch.Tensor:
    """逐行的 token 列表 → `[num_rows, max_len]`，短的行用 `vocab_size` 补齐。"""
    max_len = max((len(row) for row in rows), default=0)
    padded = torch.full((len(rows), max_len), vocab_size, dtype=torch.int64, device=device)
    for index, row in enumerate(rows):
        if row:
            padded[index, :len(row)] = torch.tensor(row, dtype=torch.int64, device=device)
    return padded


def apply_all_penalties(logits: torch.Tensor, prompt_token_ids: list[list[int]],
                        output_token_ids: list[list[int]], presence_penalties: torch.Tensor,
                        frequency_penalties: torch.Tensor,
                        repetition_penalties: torch.Tensor) -> torch.Tensor:
    """就地把三种惩罚应用到 logits（`[num_rows, vocab_size]`）。"""
    num_seqs, vocab_size = logits.shape
    device = logits.device
    prompt_tokens = _to_padded(prompt_token_ids, vocab_size, device)
    output_tokens = _to_padded(output_token_ids, vocab_size, device)
    _, prompt_mask = token_bin_counts_and_mask(prompt_tokens, vocab_size, num_seqs)
    output_bin_counts, output_mask = token_bin_counts_and_mask(output_tokens, vocab_size,
                                                               num_seqs)

    # repetition：出现过的 token 按"正负分两侧"缩放，没出现过的用 1.0（无操作）
    penalty = repetition_penalties.unsqueeze(dim=1)
    scaling = torch.where(prompt_mask | output_mask, penalty, torch.ones_like(penalty))
    logits *= torch.where(logits > 0, 1.0 / scaling, scaling)

    # frequency / presence：按 OpenAI 的定义，都作用在**输出** token 上
    logits -= frequency_penalties.unsqueeze(dim=1) * output_bin_counts
    logits -= presence_penalties.unsqueeze(dim=1) * output_mask
    return logits
