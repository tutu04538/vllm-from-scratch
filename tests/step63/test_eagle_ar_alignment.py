"""step63 补充：EAGLE / draft 提议者**自回归步的起点**必须与上游同源（2026-10-08 复核修复）。

### 这条测试钉的是什么（2026-10-08 两轮修复）

第一轮修的是**自回归步的起点**（位置/上下文长度的公式，见下面 R1/R2）；
第二轮修的是**锚点（采样行）本身选错了行**——EAGLE 把新采出的 token 打在"块的最后一行"，
而上游（padded 通路，`prepare_inputs_padded` 的 `token_indices_to_sample`）打在
**最后一枚有效行** `块起点 + target_rows − 1 − num_rejected` 上。两轮的关系是：
锚点行错的时候，R1/R2 也会"自洽地错"，症状只有一条能识破——**自回归行掉到自己的上下文之外**
（`position != seq_len − 1`），所以本文件把这条不变量单独钉住（R3）。


上游 `SpecDecodeBaseProposer.propose()`（`llm_base_proposer.py:634`）：

    positions = self.positions[token_indices_to_sample]      # ← **采样行自己的 position**
    ...
    for token_index in range(K - 1):
        positions = self._update_positions_dependent_metadata(positions, ...)   # 每步 position + 1
        ...  # 上下文长度也在同一个内核里 +1

也就是两条规则：

    R1  第 k 枚草稿的**位置** = 第一遍采样行的 position + k        (k = 1..K-1)
    R2  第 k 枚草稿的**上下文长度** = (第一遍 seq_lens − 被拒行数) + k

本仓库此前用的是**反推公式** `position = history_end + k − 1`、`seq_len = start + num_valid + 1`：
对 draft 布局（采样行 = 尾部扩容行）恰好等价，但对 EAGLE 布局（采样行 = target 行块的**最后一行**）
差 `1 − 被拒数` 格——被拒 3 枚时自回归行会落回第一遍刚写过的行，把它刚写的 KV 覆盖掉
（实测：修复前 positions 是 5、6，修复后是 7、8；seq_lens 从 6、7 变成 5、6）。

**为什么必须由布局自己给出采样位置**：两种布局的采样行不是同一行（EAGLE 复用 target 行块的最后一行、
draft 在尾部新加扩容行），任何"用历史长度反推"的公式都只对其中一种成立。

### 这条测试怎么取数

不改生产代码，只用 spy 读提议者每一步真实用掉的位置/长度（`set_inputs_first_pass()` 的产物 +
`_set_autoregressive_inputs()` 的入参），再按 R1/R2 逐轮验算。要求两种情形都出现：
`被拒 == 0`（全接受那一类）与 `被拒 > 0`（首拒/中拒那一类）——否则这条测试证明不了"被拒时也对"。
"""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "step69"))

from spec69_helpers import make_engine, run_to_end, tiny_eagle3_dir, tiny_model  # noqa: E402
from minivllm import SamplingParams  # noqa: E402

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="投机验证走 Triton 拒绝采样内核（59 关起只在 CUDA 上），CPU 上待验")


class ArRecorder:
    """记录每一轮第一遍的采样行位置/长度，以及随后每个自回归步的位置/长度。"""

    def __init__(self, runner) -> None:
        self.runner = runner
        proposer = runner.proposer
        self.proposer = proposer
        self.rounds: list[dict] = []
        self._patched = []
        self._last_rows: list = []

        original_first = proposer.set_inputs_first_pass

        def first(*args, **kwargs):
            plan = original_first(*args, **kwargs)
            self._last_rows = list(args[0]) if args else []
            self.rounds.append({
                "rows": [(int(target.target_rows), int(target.num_rejected))
                         for target in self._last_rows],
                "target_row_start": self._row_starts(plan),
                "sample_rows": list(plan.sample_rows),
                "sample_positions": list(plan.sample_positions),
                "first_pass_seq_lens": list(plan.seq_lens),
                "ar": [],
            })
            return plan

        proposer.set_inputs_first_pass = first
        self._patched.append(("set_inputs_first_pass", original_first))

        if hasattr(proposer, "_set_autoregressive_inputs"):
            original_ar = proposer._set_autoregressive_inputs

            def ar(pending, drafts, input_batch):
                result = original_ar(pending, drafts, input_batch)
                self.rounds[-1]["ar"].append({
                    "positions": [int(position) for _, position in pending],
                    "seq_lens": [int(x) for x in
                                 proposer.seq_lens_cpu[:len(pending)]],
                    "num_drafted": [len(drafts[target.req_id]) for target, _ in pending],
                    "rejected": [int(target.num_rejected) for target, _ in pending],
                })
                return result

            proposer._set_autoregressive_inputs = ar
            self._patched.append(("_set_autoregressive_inputs", original_ar))

    def _row_starts(self, plan) -> list[int]:
        """每条请求在**工作区**里的块起点（= Σ 前面请求的 target 行数）。"""
        starts, cursor = [], 0
        for target in self._last_rows:
            starts.append(cursor)
            cursor += int(target.target_rows)
        return starts

    def restore(self) -> None:
        for name, original in self._patched:
            setattr(self.proposer, name, original)


def _run_and_record(*, method: str, spec_k: int, max_tokens: int = 6, budget: int = 32):
    """跑一小段真引擎（tiny 权重），返回 (recorder 记录, recorder, engine)。"""
    import json

    target_dir, hf = tiny_model("tiny_gqa")
    draft_dir, draft_hf = target_dir, hf
    if method == "eagle3":
        draft_dir = tiny_eagle3_dir("tiny_gqa")
        draft_hf = json.loads((Path(draft_dir) / "config.json").read_text())
    engine, _core, runner = make_engine(tiny_dir=target_dir, hf_config=hf, spec_k=spec_k,
                                        draft_dir=draft_dir, draft_config=draft_hf,
                                        method=method, mode="none",
                                        max_num_seqs=1, budget=budget)
    recorder = ArRecorder(runner)
    engine.add_request("A", [1, 2, 3, 4, 5, 6],
                       SamplingParams(max_tokens=max_tokens, temperature=0.0,
                                      eos_token_id=999))
    run_to_end(engine)
    return recorder, engine


def _check_rules(recorder) -> tuple[int, int]:
    """逐轮验算 R1/R2/R3；返回 (被拒>0 的轮数, 被拒==0 的轮数)。"""
    zero_rounds = rejected_rounds = 0
    checked_steps = 0
    for round_index, round_ in enumerate(recorder.rounds):
        # R3a：锚点行 = 最后一枚有效行（上游 padded 通路的索引）。
        # **只对"锚点写在 target 行块内部"的布局成立**（EAGLE 串行：net=0）；
        # draft_model / 并行提议的第一遍是 kernel 语义——锚点打在有效行之后的**新行**上
        # （`块起点 + target_rows`），那一行不属于 target 的块，不能用这个公式比。
        anchor_inside_block = (not recorder.proposer.needs_extra_input_slots
                               and not recorder.proposer.parallel_drafting)
        for index, (target_rows, rejected) in enumerate(round_["rows"]):
            if not anchor_inside_block:
                break
            if not round_["sample_rows"] or index >= len(round_["sample_rows"]):
                continue
            want_row = round_["target_row_start"][index] + target_rows - 1 - rejected
            assert round_["sample_rows"][index] == want_row, (
                f"第 {round_index} 轮请求 {index} 的锚点行 {round_['sample_rows'][index]} "
                f"!= 块起点 {round_['target_row_start'][index]} + {target_rows} − 1 − 被拒 "
                f"{rejected} = {want_row}（上游 padded 通路）")
        if not round_["ar"]:
            continue                     # K=1：没有自回归步
        for step in round_["ar"]:
            for index in range(len(step["positions"])):
                anchor = round_["sample_positions"][index]
                base_seq = round_["first_pass_seq_lens"][index] - step["rejected"][index]
                k = step["num_drafted"][index]
                assert step["positions"][index] == anchor + k, (
                    f"第 {round_index} 轮第 {k} 枚草稿的位置 {step['positions'][index]} "
                    f"!= 采样行位置 {anchor} + {k}（上游 R1）")
                assert step["seq_lens"][index] == base_seq + k, (
                    f"第 {round_index} 轮第 {k} 枚草稿的上下文 {step['seq_lens'][index]} "
                    f"!= (第一遍 {round_['first_pass_seq_lens'][index]} − 被拒 "
                    f"{step['rejected'][index]}) + {k}（上游 R2）")
                # R3b（真正能识破"锚点选错行"的那条）：自回归行必须在自己的上下文里。
                # 锚点行偏了 rejected 格时，position 会落在 seq_len 之外 —— 不报错，只是草稿变差。
                assert step["positions"][index] == step["seq_lens"][index] - 1, (
                    f"第 {round_index} 轮第 {k} 枚草稿：position={step['positions'][index]} 但 "
                    f"seq_len={step['seq_lens'][index]}（应当 position == seq_len − 1）——"
                    f"自回归行掉到自己的上下文之外，说明锚点行/上下文起点选错了")
                checked_steps += 1
            if step["rejected"][0] > 0:
                rejected_rounds += 1
            else:
                zero_rounds += 1
    assert checked_steps > 0, "一次自回归步都没跑到：这条测试没有验到任何东西"
    return rejected_rounds, zero_rounds


@requires_cuda
def test_eagle3_autoregressive_steps_follow_upstream_positions():
    """EAGLE3：自回归步的位置/上下文长度必须等于"采样行位置/第一遍长度 − 被拒"再 +k。

    EAGLE 的采样行是 target 行块的**最后一行**（不是尾部新加的那一行），所以这条以前会差
    `1 − 被拒数` 格——修复前实测：propmt 3 token、K=3 时第一轮自回归位置是 4、5（应为 3、4），
    第二轮被拒 3 枚时是 5、6（应为 7、8，且 5、6 正是第一遍刚写过的行）。
    """
    recorder, engine = _run_and_record(method="eagle3", spec_k=3)
    try:
        rejected_rounds, zero_rounds = _check_rules(recorder)
    finally:
        recorder.restore()
        engine.shutdown()
    assert rejected_rounds > 0 and zero_rounds > 0, (
        f"两种情形都要出现才算验到了：被拒>0 的轮 {rejected_rounds}、被拒==0 的轮 {zero_rounds}")


@requires_cuda
def test_draft_model_autoregressive_steps_follow_the_same_rule():
    """普通 draft 布局走同一条规则（回归：它的采样行是**尾部扩容行**，修复前后都成立）。"""
    recorder, engine = _run_and_record(method="draft_model", spec_k=3)
    try:
        rejected_rounds, zero_rounds = _check_rules(recorder)
    finally:
        recorder.restore()
        engine.shutdown()
    assert rejected_rounds + zero_rounds > 0
