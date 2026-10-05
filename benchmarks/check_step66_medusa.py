"""66 关验收脚本：Medusa 多头提议 + MLP 支持缺口（需求 066 §2/§3/§4）。

脚本式 PASS/FAIL（供回归用），口径与 `tests/step66/` 一致，**只走生产路径**（tiny 引擎真跑）；
上游对照那一条会实例化 site-packages 里真的 `Medusa` / `MedusaProposer`（只允许出现在
测试/脚本里，生产路径不 import 上游）。

检查项：

  A. 配置：旧 checkpoint 的 key 改名 / model_type / architectures、默认值对上游 MedusaConfig、
     **K 就是 head 数**、与 target 对齐词表、社区版 architectures 明确拒绝、
     `num_lookahead_tokens`/`draft_slots` 都是 0
  B. 加载：三种命名同值、bias 的加载与记账、K 与 head 数不一致（裁掉 vs 报错）、未知名报错、
     共享 lm_head + token_map 截断词表、dummy_run 与 EPLB 组合
  C. 上游逐值对照：每个 head 的 blocks/logits、候选列顺序、配置字段
  D. 行选择：`select_target_hidden_states()` 的算式、上游 stride 在混合 prefill 批次的**反证**
  E. 端到端：K=1/2/3 与三种命名下 greedy 与非投机一致、草稿按请求各 K 枚且不带 q、
     行选错草稿必变（反证）、混合 prefill、小块数（抢占恢复）
  F. MLP 缺口：上游"配置层认得、模型层没有"的四条证据 + 本仓库"不写自创实现"
"""

import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "step66"))

import test_medusa as medusa  # noqa: E402
import test_mlp_support_boundary as mlp  # noqa: E402

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
    except Exception:                     # noqa: BLE001 —— 别的异常不算"预期失败"
        return False
    return False


def _pytest_raises(exc, fn, match=None):
    try:
        with pytest.raises(exc, match=match):
            fn()
        return True
    except Exception:                     # noqa: BLE001
        return False


# ---------------------------------------------------------------- A. 配置
from minivllm import ModelConfig, SamplingParams, SpeculativeConfig  # noqa: E402
from minivllm.config import MLP_SPECULATOR_MODEL_TYPE, medusa_hf_config  # noqa: E402

K = medusa.K
target = medusa.target_config()
target_model_config = target.model_config

config = medusa_hf_config(medusa.OLD_CHECKPOINT_CONFIG, K)
from vllm.transformers_utils.configs.medusa import MedusaConfig  # noqa: E402

upstream_defaults = MedusaConfig()
defaults_ok = all(config[field] == getattr(upstream_defaults, field) for field in
                  ("hidden_size", "vocab_size", "num_hidden_layers", "max_paths", "topk",
                   "max_seq_len", "truncated_vocab_size"))
check("A1. 旧 config.json 的 key 改名 + 缺省值逐个对上上游 MedusaConfig",
      config["num_hidden_layers"] == 1 and config["model_type"] == "medusa"
      and config["architectures"] == ["MedusaModel"] and defaults_ok,
      f"medusa_num_layers→num_hidden_layers={config['num_hidden_layers']}")

check("A2. K 就是 head 数（checkpoint 写 2、K=5 → 建 5 个 head）",
      medusa_hf_config({"medusa_num_heads": 2}, 5)["num_heads"] == 5
      and medusa.medusa_spec(medusa.draft_dir(), k=2)
      .draft_model_config.hf_config["num_heads"] == 2)

derived = medusa.medusa_spec(medusa.draft_dir()).derive_medusa_draft_config(target_model_config)
check("A3. 与 target 对齐词表（缺省 32001 → target 的 11，truncated 同步）",
      derived.hf_config["vocab_size"] == 11
      and derived.hf_config["truncated_vocab_size"] == 11,
      f"vocab={derived.hf_config['vocab_size']}")

check("A4. 社区版 architectures（抄了基座）明确报错 + hidden_size 不一致报错",
      _raises(ValueError, medusa_hf_config,
              {**medusa.draft_hf(medusa.draft_dir()), "architectures": ["Qwen2ForCausalLM"]}, K)
      and _raises(ValueError, medusa.medusa_spec(
          medusa.draft_dir(), cfg={**medusa.draft_hf(medusa.draft_dir()),
                                   "hidden_size": 64}).derive_medusa_draft_config,
          target_model_config))

from minivllm import SchedulerConfig  # noqa: E402
from minivllm.core.sched.scheduler import Scheduler  # noqa: E402


class _FakeKVCacheManager:
    enable_caching = False


spec = medusa.medusa_spec(medusa.draft_dir())
scheduler = Scheduler(SchedulerConfig(), _FakeKVCacheManager(), max_model_len=64,
                      speculative_config=spec)
check("A5. Medusa 不预留 KV lookahead、也不吃额外 draft 输入行（都是 0，上游同款）",
      scheduler.num_lookahead_tokens == 0 and scheduler.draft_slots == 0
      and not spec.use_eagle() and spec.uses_medusa())

# ---------------------------------------------------------------- B. 加载
from safetensors.torch import load_file  # noqa: E402

naming_params, naming_coverage = {}, {}
for naming in ("old", "medusa_heads", "vllm"):
    directory = medusa.draft_dir(naming=naming)
    model = medusa.medusa_model(directory)
    state = load_file(str(Path(directory) / "model.safetensors"))
    loaded = model.load_weights(iter(state.items()))
    params = {name: tensor.clone() for name, tensor in model.named_parameters()}
    naming_params[naming] = params
    naming_coverage[naming] = loaded == set(params)
reference_params = naming_params["old"]
same_values = all(torch.equal(reference_params[name], params[name])
                  for naming, params in naming_params.items() for name in reference_params)
check("B1. 三种权重命名（真实旧格式 / medusa_heads 前缀 / 本模型名字）加载结果逐位相同",
      all(naming_coverage.values()) and same_values,
      f"{len(reference_params)} 个参数")

bias_dir = medusa.draft_dir(fc_bias=True)
bias_state = load_file(str(Path(bias_dir) / "model.safetensors"))
without_bias = medusa.medusa_model(bias_dir, cfg={
    key: value for key, value in medusa.draft_hf(bias_dir).items() if key != "medusa_fc_bias"})
without_bias.load_weights(iter(bias_state.items()))
with_bias = medusa.medusa_model(bias_dir, extra={"medusa_fc_bias": True})
with_bias.load_weights(iter(bias_state.items()))
check("B2. `medusa_fc_bias`：开了就加载 bias，没开就把检查点里的 bias 显式记账后丢掉",
      "blocks.0.layers.0.bias" not in dict(without_bias.named_parameters())
      and "blocks.0.layers.0.bias" in without_bias.dropped_weights
      and torch.equal(dict(with_bias.named_parameters())["blocks.0.layers.0.bias"],
                      bias_state["0.0.linear.bias"]),
      f"记账 {len(without_bias.dropped_weights)} 项")

many = medusa.draft_dir(num_heads=5)
many_state = load_file(str(Path(many) / "model.safetensors"))
trimmed = medusa.medusa_model(many, k=2)
trimmed.load_weights(iter(many_state.items()))
too_many = medusa.medusa_model(many, k=6)          # 检查点只有 5 个 head，K=6 缺权重
check("B3. K<head：只建 K 个 head、多余的记账裁掉；K>head：缺权重当场报错",
      len(trimmed.blocks) == 2 and len(trimmed.dropped_weights) == 6
      and _raises(ValueError, too_many.load_weights, iter(many_state.items())),
      f"裁掉 {len(trimmed.dropped_weights)} 项，K=6 时缺权重报错")

unknown_state = dict(load_file(str(Path(medusa.draft_dir()) / "model.safetensors")))
unknown_state["0.5.linear.weight"] = torch.zeros(32, 32)
check("B4. 认不出的参数名当场报错（上游静默丢；这是「块没实现」的唯一信号）",
      _raises(ValueError, medusa.medusa_model(medusa.draft_dir()).load_weights,
              iter(unknown_state.items())))

truncated_dir = medusa.draft_dir(original_lm_head=True, truncated_vocab=5)
truncated_state = load_file(str(Path(truncated_dir) / "model.safetensors"))
truncated = medusa.medusa_model(truncated_dir)
truncated_loaded = truncated.load_weights(iter(truncated_state.items()))
logits = truncated.compute_logits(truncated(torch.randn(2, 32)))
token_map = truncated_state["token_map"]
outside = torch.ones(11, dtype=torch.bool)
outside[token_map] = False
check("B5. 共享 lm_head + token_map：截断词表映射回原词表（其余 -inf，argmax 落在 token_map 内）",
      "lm_head.weight" in dict(truncated.named_parameters())
      and truncated_loaded == set(dict(truncated.named_parameters())) | {"token_map"}
      and all(bool(torch.all(row[:, outside] == -torch.inf)
                   and torch.isin(row.argmax(-1), token_map).all()) for row in logits))

proposer = medusa.MedusaProposer(medusa.target_config(spec=medusa.medusa_spec(
    medusa.draft_dir())), DEVICE)
proposer.load_model()
proposer.dummy_run(num_tokens=8)
proposer.model.is_mixture_of_experts = True


class _ParallelConfig:
    enable_eplb = True


_proposer_config = proposer.vllm_config
object.__setattr__(_proposer_config, "parallel_config", _ParallelConfig())
eplb_rejected = _raises(ValueError, proposer._reject_unsupported_eplb)
object.__delattr__(_proposer_config, "parallel_config")
check("B6. `dummy_run()` 跑得通；MoE+EPLB 的组合明确拒绝（上游 assert）",
      proposer.model is not None and eplb_rejected)

# ---------------------------------------------------------------- C. 上游逐值对照
ours, upstream, upstream_proposer, upstream_state = medusa.build_upstream_pair()
torch.manual_seed(0)
hidden = torch.randn(5, ours.config["hidden_size"], dtype=torch.float32)
with torch.inference_mode():
    ours_logits = ours.compute_logits(ours(hidden))
    upstream_logits = upstream.compute_logits(upstream(hidden))
    ours_columns = torch.stack([row.argmax(-1) for row in ours_logits], dim=1)
    upstream_columns = upstream_proposer.propose(K, hidden, None)
logits_delta = max(float((a - b).abs().max()) for a, b in zip(ours_logits, upstream_logits))
check("C1. 每个 head 的 logits 与上游逐位相同", logits_delta == 0.0,
      f"max|Δ|={logits_delta:.1e}（{K} 个 head）")
check("C2. 最终候选的**列顺序**与上游相同", torch.equal(ours_columns, upstream_columns),
      f"{upstream_columns.tolist()[:2]} …")
from vllm.config import ModelConfig as UpModelConfig, ParallelConfig  # noqa: E402
from vllm.config import SpeculativeConfig as UpSpeculativeConfig  # noqa: E402
from minivllm.testing.tiny_models import tiny_qwen3_config, tiny_qwen3_dir  # noqa: E402

upstream_spec = UpSpeculativeConfig(
    model=medusa.draft_dir(num_layers=2), method="medusa", num_speculative_tokens=K,
    target_model_config=UpModelConfig(model=tiny_qwen3_dir("tiny_gqa"), dtype="float32",
                                      max_model_len=64),
    target_parallel_config=ParallelConfig())
upstream_hf = upstream_spec.draft_model_config.hf_config
ours_hf = medusa.medusa_spec(medusa.draft_dir(num_layers=2)).derive_medusa_draft_config(
    medusa.ModelConfig(model=tiny_qwen3_dir("tiny_gqa"), dtype="float32", max_model_len=64,
                       hf_config=tiny_qwen3_config("tiny_gqa"))).hf_config
check("C3. 配置归一结果与上游 MedusaConfig 的同一组字段逐个相同",
      all(ours_hf[field] == getattr(upstream_hf, field) for field in
          ("hidden_size", "vocab_size", "truncated_vocab_size", "num_heads",
           "num_hidden_layers", "max_paths", "topk"))
      and upstream_hf.architectures == ["MedusaModel"])

# ---------------------------------------------------------------- D. 行选择
from minivllm.spec_decode.medusa import MedusaProposer  # noqa: E402

rows = torch.arange(24, dtype=torch.float32).reshape(6, 4)
selected = MedusaProposer.select_target_hidden_states(rows, [3, 3], [3, 1])
one_row = torch.arange(8, dtype=torch.float32).reshape(2, 4)
check("D1. 行选择：块内第 `采样数-1` 行；无草稿批退化成 0..B-1（上游第一条分支同值）",
      torch.equal(selected, rows[[2, 3]])
      and torch.equal(MedusaProposer.select_target_hidden_states(one_row, [1, 1], [1, 1]),
                      one_row))

mixed = torch.arange(44, dtype=torch.float32).reshape(11, 4)
mixed_selected = MedusaProposer.select_target_hidden_states(mixed[5:], [3, 3], [3, 1])
upstream_indices, offset = [], 0
for num_draft, tokens in zip([0, 2, 2], [1, 3, 1]):
    upstream_indices.append(offset + tokens - 1)
    offset += num_draft + 1
check("D2. 反证：上游 `offset += num_draft+1` 在混合 prefill 批次下会整体错位",
      upstream_indices == [0, 3, 4] and not torch.equal(mixed[upstream_indices[1:]],
                                                        mixed_selected),
      f"上游行号 {upstream_indices}，本仓库 {[7, 8]}")
check("D3. 中间 prefill 块（没采到 token）不允许进这套算式；行数口径不一致也报错",
      _raises(ValueError, MedusaProposer.select_target_hidden_states, torch.zeros(4, 4),
              [1, 3], [1, 0])
      and _raises(RuntimeError, MedusaProposer.select_target_hidden_states,
                  torch.zeros(4, 4), [1, 1], [1, 1]))

# ---------------------------------------------------------------- E. 端到端
baseline_engine, _, _ = medusa.make_engine(spec=None)
try:
    baseline = medusa.run(baseline_engine, medusa.PROMPTS)
finally:
    baseline_engine.shutdown()

greedy_ok = True
for k in (1, 2, 3):
    engine, _, _ = medusa.make_engine(spec=medusa.medusa_spec(medusa.draft_dir(num_heads=k), k=k))
    try:
        greedy_ok &= medusa.run(engine, medusa.PROMPTS) == baseline
    finally:
        engine.shutdown()
check("E1. K=1/2/3 下 greedy 输出与非投机逐 token 相同", greedy_ok)
naming_ok = True
for naming in ("old", "medusa_heads", "vllm"):
    engine, _, _ = medusa.make_engine(spec=medusa.medusa_spec(medusa.draft_dir(naming=naming)))
    try:
        naming_ok &= medusa.run(engine, medusa.PROMPTS) == baseline
    finally:
        engine.shutdown()
check("E2. 三种权重命名下 greedy 也与非投机一致", naming_ok)

engine, _, runner = medusa.make_engine()
drafts_log = medusa.install_draft_spy(runner)
calls = medusa.install_propose_spy(runner)
try:
    outputs = medusa.run(engine, medusa.PROMPTS)
finally:
    engine.shutdown()
width_ok = all(len(tokens) in (0, K) for entry in drafts_log for tokens in entry["drafts"])
prob_free = all(not entry["has_probs"] for entry in drafts_log)
mapped = [entry for entry in drafts_log if all(len(t) == K for t in entry["drafts"])]
row_ok = bool(mapped) and all(entry["drafts"][0] != entry["drafts"][1] for entry in mapped)
recompute_ok = True
for call in calls:
    with torch.inference_mode():
        recomputed = torch.stack([row.argmax(-1) for row in
                                  runner.proposer.model.compute_logits(
                                      runner.proposer.model(call["hidden"]))], dim=1)
    recompute_ok &= torch.equal(call["tokens"].cpu(), recomputed.cpu())
check("E3. 草稿：每条请求 K 枚、不带 q（argmax=点质量）、行映射不串、与按该 hidden 重算一致",
      outputs == baseline and width_ok and prob_free and row_ok and recompute_ok,
      f"{len(drafts_log)} 轮草稿")

correct_engine, _, correct_runner = medusa.make_engine()
correct_log = medusa.install_draft_spy(correct_runner)
shift_engine, _, shift_runner = medusa.make_engine()
shift_log = medusa.install_draft_spy(shift_runner)
original_select = MedusaProposer.select_target_hidden_states


def shifted(hidden_states, rows_per_request, num_sampled_tokens):
    indices, cursor = [], 0
    for num_rows, _ in zip(rows_per_request, num_sampled_tokens):
        indices.append(min(cursor + num_rows, hidden_states.shape[0] - 1))
        cursor += num_rows
    return hidden_states[torch.tensor(indices, dtype=torch.int64)]


try:
    medusa.run(correct_engine, medusa.PROMPTS)
    MedusaProposer.select_target_hidden_states = staticmethod(shifted)
    medusa.run(shift_engine, medusa.PROMPTS)
finally:
    MedusaProposer.select_target_hidden_states = staticmethod(original_select)
    correct_engine.shutdown()
    shift_engine.shutdown()
check("E4. 反证：把行选择整体错开一位，草稿必变（那份 hidden 真在用）",
      any(a["drafts"] != b["drafts"] for a, b in zip(correct_log, shift_log)))

long_prompts = [("a", [1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 7]), ("b", [5, 6, 5, 6])]
small_engine, _, _ = medusa.make_engine(spec=None, budget=8)
try:
    small_baseline = medusa.run(small_engine, long_prompts)
finally:
    small_engine.shutdown()
mix_engine, _, mix_runner = medusa.make_engine(budget=8)
mix_log = medusa.install_draft_spy(mix_runner)
try:
    mix_outputs = medusa.run(mix_engine, long_prompts)
finally:
    mix_engine.shutdown()
preempt_engine, _, _ = medusa.make_engine(num_gpu_blocks=6)
try:
    preempt_outputs = medusa.run(preempt_engine, medusa.PROMPTS)
finally:
    preempt_engine.shutdown()
check("E5. 混合 prefill 批次（预算 8）greedy 一致、prefill 轮不提草稿；小块数抢占后也一致",
      mix_outputs == small_baseline
      and any(len(tokens) == 0 for entry in mix_log for tokens in entry["drafts"])
      and preempt_outputs == baseline)

# E6. 首拒 / 中拒 / 全接受：三种模式都走 59 关那套 verifier（注入 oracle 草稿稳定造出来）
verifier_ok, verifier_detail = True, []
for break_at, expected in ((None, 3), (0, 0), (1, 1)):
    prompt = [medusa.PROMPTS[0]]
    base_engine, _, _ = medusa.make_engine(spec=None)
    try:
        base_tokens = medusa.run(base_engine, prompt)
    finally:
        base_engine.shutdown()
    engine, core, _runner = medusa.make_engine(
        spec=medusa.medusa_spec(medusa.draft_dir(num_heads=3), k=3))
    injected = medusa.install_oracle_drafts(_runner, base_tokens, break_at=break_at)
    first_full, outputs = None, {}
    try:
        for req_id, tokens in prompt:
            engine.add_request(req_id, list(tokens), SamplingParams(max_tokens=6,
                                                                   temperature=0.0,
                                                                   eos_token_id=999))
        for _ in range(200):
            if not engine.has_unfinished_requests():
                break
            for out in engine.step():
                outputs[out.request_id] = list(out.token_ids)
            stats = core.scheduler.spec_decoding_stats
            if stats is not None and first_full is None and stats.num_draft_tokens == 3:
                first_full = (stats.num_draft_tokens, stats.num_accepted_tokens)
    finally:
        engine.shutdown()
    ok = (outputs == base_tokens and first_full == (3, expected)
          and any(any(tokens) for tokens, _ in injected))
    verifier_ok &= ok
    verifier_detail.append(f"break_at={break_at}→{first_full}")
check("E6. 首拒/中拒/全接受三种模式都走同一 verifier，greedy 输出仍与基线一致",
      verifier_ok, "；".join(verifier_detail))

# ---------------------------------------------------------------- F. MLP 缺口
check("F1. 本仓库：method='mlp_speculator' 明确报「版本缺口」（不静默降级）",
      _pytest_raises(NotImplementedError, lambda: SpeculativeConfig(
          method=MLP_SPECULATOR_MODEL_TYPE, num_speculative_tokens=3,
          draft_model_config=ModelConfig(model="/nonexistent", dtype="float32",
                                         max_model_len=64, hf_config=mlp.MLP_CONFIG)),
          match="版本缺口"))
from vllm.model_executor.models.registry import ModelRegistry  # noqa: E402
from vllm.transformers_utils.configs.mlp_speculator import MLPSpeculatorConfig  # noqa: E402

import tempfile  # noqa: E402

mlp_dir = Path(tempfile.mkdtemp(prefix="step66_mlp_probe_"))
(mlp_dir / "config.json").write_text(json.dumps(mlp.MLP_CONFIG, indent=2) + "\n")
mlp_config = MLPSpeculatorConfig.from_pretrained(str(mlp_dir))
check("F2. 上游配置层认得它（MLPSpeculatorConfig 能解析；n_predict=num_lookahead_tokens）",
      mlp_config.model_type == "mlp_speculator"
      and mlp_config.n_predict == mlp.MLP_CONFIG["n_predict"]
      and mlp_config.num_lookahead_tokens == mlp.MLP_CONFIG["n_predict"])
from vllm.config import ModelConfig as UpModelConfigForProbe  # noqa: E402
import pydantic  # noqa: E402

check("F3. 上游注册表没有模型类（那行是注释掉的）→ ModelConfig 当场 ValidationError",
      ModelRegistry._try_inspect_model_cls("MLPSpeculatorPreTrainedModel") is None
      and _pytest_raises(pydantic.ValidationError,
                         lambda: UpModelConfigForProbe(model=str(mlp_dir), dtype="float32",
                                                       max_model_len=64),
                         match="not supported for now"))
import ast  # noqa: E402
import vllm  # noqa: E402

vllm_dir = Path(vllm.__file__).parent
mlp_src = (vllm_dir / "model_executor" / "models" / "mlp_speculator.py").read_text("utf-8")
mlp_tree = ast.parse(mlp_src)
mlp_methods = {node.name for node in
               next(node for node in mlp_tree.body
                    if isinstance(node, ast.ClassDef) and node.name == "MLPSpeculator").body
               if isinstance(node, ast.FunctionDef)}
runner_src = (vllm_dir / "v1" / "worker" / "gpu_model_runner.py").read_text("utf-8")
check("F4. 上游模型类没有 forward（只剩 __init__/load_weights）+ Runner 里没有它的分派",
      "forward" not in mlp_methods and "mlp_speculator" not in runner_src
      and "Unknown speculative decoding method" in runner_src,
      f"methods={sorted(mlp_methods)}")
mentions = {path.relative_to(ROOT).as_posix() for path in (ROOT / "minivllm").rglob("*.py")
            if "mlp_speculator" in path.read_text("utf-8")}
check("F5. 生产包没有自创的 MLP 实现（只有 config.py 里那条版本缺口报错）",
      mentions == {"minivllm/config.py"}, str(sorted(mentions)))

# 清掉探针目录（它只是给上游 config 解析用的临时 config.json）
import shutil  # noqa: E402

shutil.rmtree(mlp_dir, ignore_errors=True)

# ---------------------------------------------------------------- G. 差异账本
print()
print("64/65/66 的一点口径说明（完整版见 docs/step66_alignment.md §3）：")
print("  · Medusa 的候选是**线性链**（每个 head 一个 argmax），不是论文里的树；")
print("    `max_paths` / `topk` 在本机 V1 里没有任何读取点。")
print("  · 上游把「取哪一行 hidden」写在 Runner 里、stride 用 `num_draft + 1`；")
print("    本仓库用调度快照的行数，混合 prefill 批次不会错位（D2 就是这条反证）。")
print("  · 上游对「检查点里有、模型里没有」的名字一律静默丢弃；本仓库只丢登记过的那几类，")
print("    其余当场报错（B3/B4）。")
print(f"设备={DEVICE}；K={K}；上游 logits max|Δ|={logits_delta:.1e}")
print()
print(f"{'全部通过' if not FAIL else '失败: ' + ', '.join(FAIL)}")
sys.exit(1 if FAIL else 0)
