"""74 关：融合多步 decode 的"选哪条路"与两条路的**逐步等价性**（需求 074 §3、§4）。

覆盖：

    G1  capability 门：K=1 恒不融合；K>1 时 **完全由后端的 capability 决定**
        （支持 → fused，不支持 → fallback），两个"后端"各跑一遍
    G2  两条路的逐轮状态逐位相同：草稿 token / positions / seq_lens / slot_mapping / 草稿 KV
    G3  "多步真的在图里"：融合路径下一轮跑完 K−1 步而草稿模型的 Python forward **一次都没被调**
        （逐步路径是 K−1 次）——只数"捕获了几张图"是分不出来的
    G4  K=1 不捕获 decode 图；K=2/4 捕获
    G5  `advance_draft_positions=False` 的特殊路径：源码规则是**强制融合**（位置不动时逐步重建
        反而会把 seq_lens 推着走），单测钉住
"""

import pytest
import torch

from spec74_helpers import (DEFAULT_SIZES, DraftForwardCounter,  # noqa: E402
                            DraftSampleCounter, RoundRecorder,  # noqa: E402
                            make_config,  # noqa: E402
                            force_draft_metadata_capability, greedy, make_engine,
                            requires_cuda, run_steps)

from minivllm import SamplingParams  # noqa: E402

PROMPTS = [("A", [1, 2, 3, 4, 5, 6]), ("B", [7, 8, 9]), ("C", [10, 9, 8, 7])]


def _engine_with(monkeypatch, *, spec_k, supports_update, mode="full_decode_only",
                 capture_sizes=DEFAULT_SIZES, max_num_seqs=3):
    force_draft_metadata_capability(monkeypatch, supports_update=supports_update)
    return make_engine(spec_k=spec_k, mode=mode, capture_sizes=capture_sizes,
                       max_num_seqs=max_num_seqs)


@requires_cuda
@pytest.mark.parametrize("spec_k", [1, 2, 4])
@pytest.mark.parametrize("supports_update", [True, False])
def test_gate_selects_fused_or_fallback_by_backend_capability(monkeypatch, spec_k,
                                                              supports_update):
    """G1：融合/回退**只**由 (K, 后端 capability) 决定，不猜、也不伪造。"""
    engine, _core, runner = _engine_with(monkeypatch, spec_k=spec_k,
                                         supports_update=supports_update)
    try:
        speculator = runner.speculator
        expected = supports_update and spec_k > 1
        assert speculator.use_fused_multi_step_decode is expected, (
            f"K={spec_k} supports_update={supports_update} 时选错了路径："
            f"use_fused={speculator.use_fused_multi_step_decode}")
        # K=1 的早退发生在**捕获之前**：`capture()` 直接返回，所以 decode 图一张都不会有
        # （候选表仍然建着——那是"形状表"，不是"已经录了图"）。
        if spec_k == 1:
            assert speculator.decode_cudagraph_manager.graphs == {}
            assert speculator.prefill_cudagraph_manager.graphs, "prefill 图仍然要捕获"
        else:
            assert speculator.decode_cudagraph_manager.graphs, "K>1 必须有 decode 图"
    finally:
        engine.shutdown()


@requires_cuda
@pytest.mark.parametrize("spec_k", [1, 2, 4])
@pytest.mark.parametrize("supports_update", [True, False])
def test_decode_graph_replay_count_equals_steps_or_one(monkeypatch, spec_k,
                                                       supports_update):
    """G1+G4：一轮里草稿 decode 图重放了几次 = 融合 1 次 / 逐步 K−1 次（K=1 时 0 次）。"""
    engine, _core, runner = _engine_with(monkeypatch, spec_k=spec_k,
                                         supports_update=supports_update)
    rec = RoundRecorder(runner)
    try:
        for req_id, prompt in PROMPTS[:1]:
            engine.add_request(req_id, list(prompt),
                               SamplingParams(max_tokens=10, temperature=0.0,
                                              eos_token_id=18))
        run_steps(engine, runner)
    finally:
        engine.shutdown()

    decode_rounds = [r for r in rec.rounds if r["num_reqs"] > 0]
    assert decode_rounds
    expected = 0 if spec_k == 1 else (1 if supports_update else spec_k - 1)
    for entry in decode_rounds:
        assert entry["decode_replays"] == expected, (
            f"K={spec_k} supports_update={supports_update} 时这一轮重放了 "
            f"{entry['decode_replays']} 次 decode 图，应该是 {expected} 次")
        assert entry["current_draft_step"] == spec_k - 1


@requires_cuda
def test_fused_multi_step_replay_runs_everything_inside_the_graph(monkeypatch):
    """G3：融合路径下"重放**一次**"就走完 K−1 个草稿步，且全程不执行 Python。

    这是"不是只捕获最后一次 forward"的直接判据，三个量必须同时成立（K=4）：

        融合：每轮 decode 图 replay 次数 == 1，Python forward 调用 == 0，
              Python 采样调用 == 0，而 `current_draft_step` 已经到 K−1、K 列草稿都写满了
        逐步：同样的 K 与 workload，replay 次数 == K−1（每步一张图重放一次）
    """
    observed = {}
    for supports_update in (True, False):
        engine, _core, runner = _engine_with(monkeypatch, spec_k=4,
                                             supports_update=supports_update)
        speculator = runner.speculator
        forward_counter = DraftForwardCounter(speculator)
        sample_counter = DraftSampleCounter(speculator)
        rec = RoundRecorder(runner)
        try:
            engine.add_request("A", [1, 2, 3, 4, 5, 6],
                               SamplingParams(max_tokens=8, temperature=0.0,
                                              eos_token_id=18))
            engine.step()          # 第一轮（prefill）本身也要跑一次前向，先把它排除
            forward_counter.count = 0
            sample_counter.count = 0
            rec.rounds.clear()
            run_steps(engine, runner)
        finally:
            engine.shutdown()
        decode_rounds = [r for r in rec.rounds if r["num_reqs"] > 0]
        assert decode_rounds, "没有 decode 轮，用例失去意义"
        observed[supports_update] = {
            "replays": [r["decode_replays"] for r in decode_rounds],
            "forwards": forward_counter.count,
            "samples": sample_counter.count,
            "final_step": decode_rounds[-1]["current_draft_step"],
            "columns_filled": bool((decode_rounds[-1]["draft_tokens"] > 0).all()),
        }

    fused, fallback = observed[True], observed[False]
    assert set(fused["replays"]) == {1}, f"融合路径每轮应当只重放一次：{fused['replays']}"
    assert set(fallback["replays"]) == {3}, f"逐步路径每轮应当重放 K−1=3 次：{fallback['replays']}"
    # 两条路的草稿步都在**图里**跑：Python 一次都没被调（这正是"捕获的是整个循环"的判据）
    assert fused["forwards"] == fallback["forwards"] == 0
    assert fused["samples"] == fallback["samples"] == 0
    assert fused["final_step"] == fallback["final_step"] == 3
    assert fused["columns_filled"], "融合路径这一轮必须把 K 列草稿都写出来"


@requires_cuda
@pytest.mark.parametrize("mode", ["none", "full_decode_only"])
def test_fused_and_fallback_agree_on_every_round(monkeypatch, mode):
    """G2：同一 workload 下两条路的**逐轮**草稿 token / positions / seq_lens / 槽位 / KV 全等。

    用同一条请求序列、同一随机源（greedy），两条路各自跑一遍；比较每一轮的快照。
    中间状态必须逐位相同——只比最终 token 会漏掉"某步写错槽位但后来被覆盖"这类错。

    两种图模式都跑：`none` 比的是**纯 eager** 的两条路（融合循环 vs 逐步循环本身），
    `full_decode_only` 比的是**图里**的两条路（顺带验证捕获没有把某一步录丢）。
    """
    snapshots = {}
    for supports_update in (True, False):
        engine, _core, runner = _engine_with(monkeypatch, spec_k=3,
                                             supports_update=supports_update, mode=mode)
        rec = RoundRecorder(runner)
        try:
            for req_id, prompt in PROMPTS:
                engine.add_request(req_id, list(prompt),
                                   SamplingParams(max_tokens=10, temperature=0.0,
                                                  eos_token_id=18))
            run_steps(engine, runner)
        finally:
            engine.shutdown()
        snapshots[supports_update] = rec.rounds

    fused, fallback = snapshots[True], snapshots[False]
    assert len(fused) == len(fallback) > 1, "两条路跑的轮数不同，比较没有意义"
    for index, (a, b) in enumerate(zip(fused, fallback)):
        assert a["num_reqs"] == b["num_reqs"]
        assert torch.equal(a["draft_tokens"], b["draft_tokens"]), f"第 {index} 轮草稿不同"
        assert torch.equal(a["positions"], b["positions"]), f"第 {index} 轮 positions 不同"
        assert torch.equal(a["seq_lens"], b["seq_lens"]), f"第 {index} 轮 seq_lens 不同"
        assert torch.equal(a["slot_mappings"], b["slot_mappings"]), (
            f"第 {index} 轮 slot_mapping 不同")
        assert a["kv_checksum"] == b["kv_checksum"], f"第 {index} 轮草稿 KV 不同"


@requires_cuda
def test_fused_and_fallback_agree_with_probabilistic_draft(monkeypatch):
    """G2 的随机采样版：`draft_sample_method="probabilistic"`（gumbel + 持久 draft logits）。

    为什么单独测：概率草稿把温度、种子、位置**都**用上了（`draft_logits` 还按 `draft_step`
    分列缓存），而 greedy 只用 argmax——两条路在"位置/步号写错"时 greedy 可能照样相同。

    这一份比较走 **eager**（`mode="none"`）：图路径下采样在图里跑，Python 侧没有逐步的采样
    回调可比（那正是 G3 要证明的事）；图路径用"逐轮草稿 token 快照"来比，见
    `test_fused_and_fallback_agree_on_every_round`。这里比两样：每一步**写进持久缓存的
    草稿 logits 列**（q 的来源）与每一步采出来的 token。
    """
    logs, tokens_per_step = {}, {}
    for supports_update in (True, False):
        force_draft_metadata_capability(monkeypatch, supports_update=supports_update)
        engine, _core, runner = make_engine(spec_k=2, mode="none",
                                            draft_sample="probabilistic")
        speculator = runner.speculator
        assert speculator.draft_logits is not None
        cache, steps = [], []
        original = speculator.sample_draft

        def patched(hidden_states, positions, idx_mapping, temperature, seeds,
                    draft_step, draft_logits, _orig=original, _cache=cache, _steps=steps):
            out = _orig(hidden_states, positions, idx_mapping, temperature, seeds,
                        draft_step, draft_logits)
            # 这一步写进持久缓存的**那一列**就是本步的草稿 logits（q 的来源）
            _cache.append(draft_logits[:, int(draft_step.item())].clone().cpu())
            _steps.append(out.clone().cpu())
            return out

        speculator.sample_draft = patched
        try:
            # 显式给 seed：`SamplingParams.seed=None` 时上游/本仓库都从**全局随机源**抽一个，
            # 那样两次运行本来就不可比（概率草稿的 gumbel 噪声按"种子 + 位置"生成）。
            engine.add_request("A", [1, 2, 3, 4, 5, 6],
                               SamplingParams(max_tokens=8, temperature=0.7, seed=1234,
                                              eos_token_id=18))
            for _ in range(400):
                if not engine.has_unfinished_requests():
                    break
                engine.step()
        finally:
            engine.shutdown()
        logs[supports_update] = cache
        tokens_per_step[supports_update] = steps

    fused, fallback = logs[True], logs[False]
    assert fused and len(fused) == len(fallback) > 1, (
        f"两步以上的草稿步数不同：{len(fused)} vs {len(fallback)}")
    for index, (a, b) in enumerate(zip(fused, fallback)):
        assert torch.equal(a, b), f"第 {index} 步的草稿 logits（q 的来源）不同"
        assert torch.equal(tokens_per_step[True][index], tokens_per_step[False][index]), (
            f"第 {index} 步采出来的草稿 token 不同")


@requires_cuda
@pytest.mark.parametrize("sample_method", ["greedy", "probabilistic"])
def test_graph_path_matches_eager_path(monkeypatch, sample_method):
    """G6：开图与不开图，**逐轮草稿 token** 与最终输出都一样（图不改语义，只改谁提交 kernel）。

    概率草稿也跑一遍：它的 gumbel 噪声取"位置 + 步号"，而图会把 `current_draft_step` 的
    写入录进去——两者错位的话 token 立刻不同。
    """
    snapshots, finals = {}, {}
    for mode in ("none", "full_decode_only"):
        force_draft_metadata_capability(monkeypatch, supports_update=True)
        engine, _core, runner = make_engine(spec_k=3, mode=mode,
                                            draft_sample=sample_method)
        rec = RoundRecorder(runner)
        try:
            for req_id, prompt in PROMPTS:
                engine.add_request(req_id, list(prompt),
                                   SamplingParams(max_tokens=10, temperature=0.7, seed=1234,
                                                  eos_token_id=18))
            final = {}
            for _ in range(400):
                if not engine.has_unfinished_requests():
                    break
                for out in engine.step():
                    final[out.request_id] = list(out.token_ids)
        finally:
            engine.shutdown()
        snapshots[mode] = rec.rounds
        finals[mode] = final

    assert finals["none"] == finals["full_decode_only"]
    assert len(snapshots["none"]) == len(snapshots["full_decode_only"]) > 1
    for index, (a, b) in enumerate(zip(snapshots["none"], snapshots["full_decode_only"])):
        assert torch.equal(a["draft_tokens"], b["draft_tokens"]), (
            f"第 {index} 轮草稿 token 不同（{sample_method}）")


@requires_cuda
def test_k1_captures_no_decode_graph(monkeypatch):
    """G4：K=1 时 decode 图管理器一个候选都没有（早退发生在捕获之前）。"""
    force_draft_metadata_capability(monkeypatch, supports_update=True)
    engine, _core, runner = make_engine(spec_k=1)
    try:
        speculator = runner.speculator
        assert speculator.num_speculative_steps == 1
        assert speculator.decode_cudagraph_manager.graphs == {}
        assert speculator.prefill_cudagraph_manager.graphs, "prefill 图仍然要捕获"
    finally:
        engine.shutdown()


def test_advance_draft_positions_false_forces_fused(monkeypatch):
    """G5：不推进 positions 的特殊路径**强制融合**（源码规则，不是优化开关）。

    位置不动的模型（Q-only 的 MTP，如 Gemma4）每一步的槽位完全相同；逐步重建会把
    `seq_lens` 推着走并重建元数据，算出来的注意力上下文就错了。
    """
    from minivllm.worker.gpu.spec_decode.autoregressive.speculator import (
        AutoRegressiveSpeculator,
    )


    class Stub(AutoRegressiveSpeculator):
        """只为"选路"这一件事的替身：不加载模型（`load_draft_model` 是抽象方法）。"""

        def load_draft_model(self, target_model):
            raise NotImplementedError

    config = make_config(spec_k=3)
    speculator = Stub(config, torch.device("cpu"))
    # 直接喂一组"声明不支持原地更新"的组：按 capability 应当回退……
    speculator.attn_groups = __import__(
        "minivllm.testing.fake_attn_backend", fromlist=["draft_attn_groups"]
    ).draft_attn_groups(["layer0"], 4, supports_update=False)
    speculator._configure_fused_multi_step_decode()
    assert speculator.use_fused_multi_step_decode is False

    # ……但只要"位置不推进"，就必须融合（这条压过 capability）
    class NoAdvance(Stub):
        @property
        def advance_draft_positions(self) -> bool:
            return False

    no_advance = NoAdvance(config, torch.device("cpu"))
    no_advance.attn_groups = speculator.attn_groups
    no_advance._configure_fused_multi_step_decode()
    assert no_advance.use_fused_multi_step_decode is True

    # K=1 时连"强制融合"都不成立（没有第二步可融合）
    config1 = make_config(spec_k=1)
    one = NoAdvance(config1, torch.device("cpu"))
    one.attn_groups = speculator.attn_groups
    one._configure_fused_multi_step_decode()
    assert one.use_fused_multi_step_decode is False
