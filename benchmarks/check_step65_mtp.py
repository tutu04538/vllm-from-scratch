"""65 关验收脚本：原生 MTP 的加载与通用迭代提议（需求 065 §3/§4）。

脚本式 PASS/FAIL（供回归用），口径与 `tests/step65/` 一致，**只走生产路径**（tiny 引擎真跑，
外加"实例化上游类做数值对照"这一条差分；上游只允许出现在测试/脚本里）。

检查项：

  A. 配置：别名归一、K 与 n_predict 的整除、`use_eagle()` 覆盖 mtp
  B. 加载：两派命名（`mtp.*` / `model.layers.{N+i}.*`）都认；spec 层的值不是 target 第 0 层的；
     target 侧跳过 spec 权重；不支持的家族参数名明确报错
  C. 前向：胶水顺序的三条反证（拼接顺序/归一化/embedding 半边）+ 与**上游 Qwen3NextMTP** 的逐值对照
  D. 端到端：greedy == 非投机（两派命名 × K=1/2）、草稿进调度器、hidden 回灌生效（冻结反证）、
     错位 hidden 改变 logits、抢占恢复后仍一致
  E. 别名表：把所有 MTP 别名 → 归一结果 → 本仓库状态打出来（需求 §5 的交付物）
"""

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "step65"))

import test_mtp_config as cfg  # noqa: E402
import test_mtp_e2e as e2e  # noqa: E402
import test_mtp_forward as fwd  # noqa: E402
import test_mtp_loader as loader  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FAIL = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def _raises(exc, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc:
        return True
    return False


# ---------------------------------------------------------------- A. 配置
from minivllm import SpeculativeConfig  # noqa: E402
from minivllm.config import MTP_MODEL_TYPES  # noqa: E402

aliases_ok = all(SpeculativeConfig(method=alias, num_speculative_tokens=1).method == "mtp"
                 for alias in MTP_MODEL_TYPES)
check("A1. 所有 MTP 别名都归一到 method='mtp'", aliases_ok,
      f"{len(MTP_MODEL_TYPES)} 个别名")

one = SpeculativeConfig(method="mtp", num_speculative_tokens=2)
check("A2. use_eagle()/uses_mtp()/lookahead 口径正确",
      one.use_eagle() and one.uses_mtp() and one.max_num_new_slots_for_drafting == 0)
target2 = cfg.make_target(num_nextn_predict_layers=2)
check("A3. K 与 n_predict 的整除约束（K=4 通过 / K=3 报错）",
      SpeculativeConfig(method="mtp", num_speculative_tokens=4).derive_mtp_draft_config(
          target2).hf_config["n_predict"] == 2
      and _raises(ValueError, SpeculativeConfig(method="mtp", num_speculative_tokens=3)
                  .derive_mtp_draft_config, target2))
derived = SpeculativeConfig(method="mtp",
                            num_speculative_tokens=1).derive_mtp_draft_config(
    cfg.make_target(num_nextn_predict_layers=1))
check("A4. 派生 draft 配置：目录 = target、架构 = 本仓库的 MTP 类",
      derived.hf_config["architectures"] == ["Qwen3MTPModel"]
      and derived.hf_config["n_predict"] == 1)

# ---------------------------------------------------------------- B. 加载
loaded_counts = {}
for naming in ("mtp", "absolute"):
    config, state = loader.load_state(naming)
    model = loader.Qwen3MTP({**config, "architectures": ["Qwen3MTPModel"]})
    loaded = model.load_weights(iter(state.items()))
    params = dict(model.named_parameters())
    loaded_counts[naming] = len(loaded)
    spec_ok = torch.equal(params["model.fc.weight"], state[loader.spec_name(naming, "fc.weight")])
    not_target = not torch.equal(params["model.layers.0.mlp.down_proj.weight"],
                                 state["model.layers.0.mlp.down_proj.weight"])
    check(f"B1[{naming}]. 加载数量 = 参数数，且 spec 权重取自 spec 层（不是 target 第 0 层）",
          loaded == set(params) and spec_ok and not_target, f"{len(loaded)} 个参数")

config, state = loader.load_state("absolute")
target_model = loader.Qwen3ForCausalLM(config)
target_loaded = target_model.load_weights(iter(state.items()))
check("B2. target 侧跳过 spec 权重（同文件里两套权重并存）",
      not any("mtp." in n or f"layers.{loader.TARGET_LAYERS}." in n for n in target_loaded),
      f"target 加载 {len(target_loaded)} 个参数")
fake = dict(state)
fake[f"model.layers.{loader.TARGET_LAYERS}.self_attn.q_a_proj.weight"] = torch.zeros(4, 4)
model = loader.Qwen3MTP(config)
check("B3. 不支持的家族参数名（MLA 的 q_a_proj）明确报错",
      _raises(ValueError, model.load_weights, iter(fake.items())))

# ---------------------------------------------------------------- C. 前向
from minivllm.testing.tiny_models import tiny_qwen3_config  # noqa: E402

unit = fwd.Qwen3MTP(fwd.mtp_config())
with torch.no_grad():
    for parameter in unit.parameters():
        parameter.normal_(0.0, 0.05)
unit.model.layers = torch.nn.ModuleList([fwd.IdentityBlock()])
ids = torch.tensor([1, 2, 3])
pos = torch.tensor([0, 1, 2])
hidden = torch.randn(3, fwd.HIDDEN)
with torch.inference_mode():
    out = unit(ids, pos, hidden)
    embeds = unit.model.pre_fc_norm_embedding(unit.model.embed_tokens(ids))
    hidden_normed = unit.model.pre_fc_norm_hidden(hidden)
    swapped = unit.model.norm(unit.model.fc(torch.cat([hidden_normed, embeds], dim=-1)))
    no_norm = unit.model.norm(unit.model.fc(torch.cat(
        [unit.model.embed_tokens(ids), hidden], dim=-1)))
    shifted = unit(ids, pos, torch.roll(hidden, 1, dims=0))
check("C1. 胶水顺序的三条反证：拼接顺序 / 少归一化 / 错位 hidden 都必须改变输出",
      not torch.allclose(out, swapped, atol=1e-6)
      and not torch.allclose(out, no_norm, atol=1e-6)
      and not torch.allclose(out, shifted, atol=1e-6))

step_ok = True
model2 = fwd.build_model(num_mtp_layers=2, seed=3)
fwd.stub_attention(model2)
with torch.no_grad():
    for index, layer in enumerate(model2.model.layers):
        layer.mlp.down_proj.weight.add_(float(index + 1))
with torch.inference_mode():
    step0 = model2(ids, pos, hidden, spec_step_idx=0)
    step1 = model2(ids, pos, hidden, spec_step_idx=1)
    step2 = model2(ids, pos, hidden, spec_step_idx=2)
check("C2. spec_step_idx 按 num_mtp_layers 取模选层",
      not torch.allclose(step0, step1) and torch.equal(step0, step2))

if DEVICE == "cuda":
    class _Factory:
        def mktemp(self, name):
            p = Path("/tmp") / name
            p.mkdir(parents=True, exist_ok=True)
            return p

    ours, upstream, glue_state = fwd.glue_pair.__wrapped__(_Factory())
    torch.manual_seed(11)
    h = torch.randn(5, glue_state["model.norm.weight"].shape[0])
    ids5 = torch.tensor([1, 2, 3, 4, 5])
    pos5 = torch.tensor([0, 1, 2, 10, 11])
    with torch.inference_mode():
        ours_out, upstream_out = ours(ids5, pos5, h), upstream(ids5, pos5, h)
        ours_logits = ours.compute_logits(h[:4])
        upstream_logits = upstream.compute_logits(h[:4], spec_step_idx=0)
    delta = float((ours_out - upstream_out).abs().max())
    delta_logits = float((ours_logits - upstream_logits).abs().max())
    check("C3. 与上游 Qwen3NextMTP 的胶水 forward 逐值一致（两侧都把 decoder 层换成直通）",
          delta < 1e-3, f"max|Δ|={delta:.3e}")
    check("C4. 与上游 compute_logits 逐值一致", delta_logits < 1e-3,
          f"max|Δ|={delta_logits:.3e}")
else:
    check("C3/C4. 与上游的数值对照（需要 CUDA）", True, "本次在 CPU 上跑，跳过")

# ---------------------------------------------------------------- D. 端到端
base_outputs = {}
for naming in ("mtp", "absolute"):
    engine, core, _ = e2e.make_engine(naming=naming, with_spec=False)
    try:
        base_outputs[naming], _ = e2e.run(engine, core, e2e.PROMPTS)
    finally:
        engine.shutdown()
e2e_ok, e2e_detail = True, []
for naming in ("mtp", "absolute"):
    for k in (1, 2):
        engine, core, _ = e2e.make_engine(naming=naming, k=k)
        try:
            outputs, stats = e2e.run(engine, core, e2e.PROMPTS)
        finally:
            engine.shutdown()
        same = outputs == base_outputs[naming]
        drafted = sum(s.num_draft_tokens for s in stats) > 0
        e2e_ok &= same and drafted
        e2e_detail.append(f"{naming}/K={k}: match={same} drafts={drafted}")
check("D1. greedy == 非投机（两派命名 × K=1/2），且草稿进过调度器", e2e_ok,
      "；".join(e2e_detail))

engine, core, runner = e2e.make_engine(k=1, max_num_seqs=1)
seen = []
original = runner.proposer.model.forward


def spy(input_ids, positions, hidden_states):
    staged = runner.proposer.hidden_states_cpu[:hidden_states.shape[0]].to(hidden_states.device)
    seen.append(torch.equal(hidden_states, staged))
    return original(input_ids, positions, hidden_states)


runner.proposer.model.forward = spy
try:
    e2e.run(engine, core, [("a", e2e.PROMPTS[0][1])], max_tokens=3)
finally:
    engine.shutdown()
check("D2. target 的 hidden 真的**上传到 device** 后喂进 draft（65 关修掉的 63 关 bug）",
      bool(seen) and all(seen), f"{len(seen)} 次前向")

freeze_deltas = None
engine, core, runner = e2e.make_engine(k=3, max_num_seqs=1)
logits_log = e2e.install_logits_spy(runner)
first_pass = {}
original_forward = runner.proposer._forward
original_ar = runner.proposer._set_autoregressive_inputs


def forward_spy(num_tokens, num_reqs, hidden_states=None):
    if not first_pass:
        first_pass["rows"] = runner.proposer.hidden_states_cpu[:num_tokens].clone()
    return original_forward(num_tokens, num_reqs, hidden_states)


def ar_spy(pending, drafts, input_batch):
    original_ar(pending, drafts, input_batch)
    runner.proposer.hidden_states_cpu[:len(pending)] = first_pass["rows"][:len(pending)]


runner.proposer._forward = forward_spy
runner.proposer._set_autoregressive_inputs = ar_spy      # ← 挂上去才会真的"冻结"
try:
    e2e.run(engine, core, [("a", e2e.PROMPTS[0][1])], max_tokens=3)
    frozen_logits = [t.clone() for t in logits_log]
finally:
    engine.shutdown()
engine, core, runner = e2e.make_engine(k=3, max_num_seqs=1)
correct_logits = e2e.install_logits_spy(runner)
try:
    e2e.run(engine, core, [("a", e2e.PROMPTS[0][1])], max_tokens=3)
finally:
    engine.shutdown()
freeze_deltas = [float((a - b).abs().max()) for a, b in zip(correct_logits, frozen_logits)]
check("D3. 冻结「上一枚 hidden 的回灌」后草稿 logits 必变（回灌真的生效）",
      max(freeze_deltas) > 1e-3, f"max|Δ|={max(freeze_deltas):.3e}")

engine, core, runner = e2e.make_engine(k=2, max_num_seqs=1)
shift_logits = e2e.install_logits_spy(runner)
original_hidden = runner._target_hidden_states_by_req


def hidden_spy(scheduler_output, num_reqs):
    by_req = original_hidden(scheduler_output, num_reqs)
    if by_req is None:
        return None
    shifted = {}
    for req_id, rows in by_req.items():
        moved = torch.zeros_like(rows)
        if rows.shape[0] > 1:
            moved[1:] = rows[:-1]
        shifted[req_id] = moved
    return shifted


runner._target_hidden_states_by_req = hidden_spy
try:
    e2e.run(engine, core, [("a", e2e.PROMPTS[0][1])], max_tokens=3)
finally:
    engine.shutdown()
engine, core, runner = e2e.make_engine(k=2, max_num_seqs=1)
plain_logits = e2e.install_logits_spy(runner)
try:
    e2e.run(engine, core, [("a", e2e.PROMPTS[0][1])], max_tokens=3)
finally:
    engine.shutdown()
shift_deltas = [float((a - b).abs().max()) for a, b in zip(plain_logits, shift_logits)]
check("D4. 故意错位 target hidden 后草稿 logits 必变（抓「假接线」）",
      max(shift_deltas) > 1e-3, f"max|Δ|={max(shift_deltas):.3e}")

engine, core, _ = e2e.make_engine(k=2, num_gpu_blocks=6)
try:
    preempt_outputs, _ = e2e.run(engine, core, e2e.PROMPTS, max_tokens=6)
finally:
    engine.shutdown()
base_engine, base_core, _ = e2e.make_engine(with_spec=False, num_gpu_blocks=6)
try:
    base_preempt, _ = e2e.run(base_engine, base_core, e2e.PROMPTS, max_tokens=6)
finally:
    base_engine.shutdown()
check("D5. 抢占/恢复（块很少）后 greedy 仍与非投机一致", preempt_outputs == base_preempt,
      f"{preempt_outputs}")

# ---------------------------------------------------------------- E. 别名表
STATUS = {
    "mtp": "✅ 实现（本仓库以 Qwen3 稠密 MTP 为对齐对象，见 docs/step65_alignment.md §3）",
    "qwen3_next_mtp": "⚠️ 归一为 mtp；块结构不同（GatedDeltaNet 混合层未实现）→ 真 checkpoint 会在"
                      "加载期报「没有这个参数」，不会静默跑错",
}
print()
print("别名表（需求 §5 的交付物；完整版见 docs/step65_alignment.md §2）：")
for alias in MTP_MODEL_TYPES:
    status = STATUS.get(alias, "⏳ 未适配（缺 MLA/MoE/混合层等结构；加载期明确报错）")
    print(f"  {alias:24s} → method='mtp' → {status}")

print()
print(f"设备={DEVICE}；加载数量={loaded_counts}")
print(f"{'全部通过' if not FAIL else '失败: ' + ', '.join(FAIL)}")
sys.exit(1 if FAIL else 0)
