"""62 关验收脚本：自定义 Proposer 的接入与配置分派边界。

分段（对应需求 062 §3/§4）：
  A. 方法推断：只在配置期判一次（点号路径 / ngram / URL / HF 名 / 显式 method）
  B. 接口：`create_custom_proposer` 返回实例本身（不套壳），构造参数只有 VllmConfig
  C. 错误分类：无点号 / 模块不存在 / 类不存在 / 构造失败 / propose 缺失 / propose 不可调用
  D. Runner 接线：调用参数与冻结的上游分支一致；定长缓冲契约；插件拿不到运行时状态
  E. 行为：空 / 全错 / 变长候选都不改 greedy 答案；行数不匹配立刻报错
  F. 端到端：命令行入口（demo 的 --spec-model）能把这个类接进真实链路
"""

import importlib
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "step59"))
sys.path.insert(0, str(ROOT / "tests" / "step62"))

from minivllm import (CacheConfig, DeviceConfig, LLMEngine, ModelConfig, SamplingParams,
                      SchedulerConfig, SpeculativeConfig, UniProcExecutor, VllmConfig, Worker)
from minivllm.spec_decode.custom_class_proposer import create_custom_proposer
from minivllm.testing.tiny_models import tiny_qwen3_config, tiny_qwen3_dir

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EXAMPLE = "examples.custom_proposer.RepeatLastTokenProposer"
FAIL = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def make_engine(spec, *, max_num_seqs=2, budget=64, max_model_len=64):
    tiny_dir, hf_config = tiny_qwen3_dir("tiny_gqa"), tiny_qwen3_config("tiny_gqa")
    config = VllmConfig(
        model_config=ModelConfig(model=tiny_dir, dtype="float32", max_model_len=max_model_len,
                                 hf_config=hf_config),
        cache_config=CacheConfig(block_size=4, num_gpu_blocks=32),
        scheduler_config=SchedulerConfig(max_num_seqs=max_num_seqs,
                                         max_num_batched_tokens=budget),
        device_config=DeviceConfig(device=DEVICE), speculative_config=spec)
    engine = LLMEngine(config, UniProcExecutor(config, Worker(config)))
    core = engine.engine_core.engine_core
    return engine, core, core.model_executor.driver_worker.model_runner


def run(engine, core, requests, *, max_tokens=6):
    for req_id, prompt in requests:
        engine.add_request(req_id, list(prompt),
                           SamplingParams(max_tokens=max_tokens, temperature=0.0,
                                          eos_token_id=999))
    outputs, stats = {}, []
    for _ in range(300):
        if not engine.has_unfinished_requests():
            break
        for out in engine.step():
            outputs[out.request_id] = list(out.token_ids)
        stats.append(core.scheduler.spec_decoding_stats)
    return outputs, stats


# ================================================ A. 方法推断
print("== A. 方法推断（配置期一次判定）==")
inferred = {
    EXAMPLE: "custom_class",
    "pkg.sub.Class": "custom_class",
    "ngram": "ngram",
    "[ngram]": "ngram",
    "Qwen/Qwen3-0.6B": "draft_model",
    "https://huggingface.co/a.b": "draft_model",
    "single": "draft_model",
    "my-module.Class": "draft_model",
}
got = {model: SpeculativeConfig(model=model).method for model in inferred}
check("A1. 点号路径 → custom_class；ngram/URL/HF 名/单词 → 上游默认", got == inferred, f"{got}")
check("A2. 显式 method 优先于推断",
      SpeculativeConfig(method="ngram", model="a.b.C").method == "ngram"
      and SpeculativeConfig(method="custom_class", model="a.b.C").method == "custom_class")
try:
    SpeculativeConfig(method="custom_class")
    missing_model_ok = False
except ValueError as error:
    missing_model_ok = "requires 'model'" in str(error)
check("A3. 显式 custom_class 但没给 model → 配置期 ValueError", missing_model_ok)
try:
    SpeculativeConfig(method="medusa")     # 63 关起 eagle/eagle3 已支持，换仍未实现的 medusa
    unknown_ok = False
except ValueError:
    unknown_ok = True
check("A4. 未支持的方法在配置期拒绝（不回退成 ngram）", unknown_ok)
runner_source = (ROOT / "minivllm" / "worker" / "gpu_model_runner.py").read_text()
check("A5. 分派边界：Runner 只按 config.method 分派，不再自己判点号路径",
      "_is_custom_proposer_path" not in runner_source
      and 'config.method == "custom_class"' in runner_source)
custom_spec = SpeculativeConfig(method="custom_class", model="a.b.C", num_speculative_tokens=4)
check("A6. 派生量与 ngram 同档（不写 KV、不占输入槽位）",
      not custom_spec.uses_draft_model() and not custom_spec.use_ngram_gpu()
      and custom_spec.max_num_new_slots_for_drafting == 0)


# ================================================ B. 接口
print("\n== B. 接口（返回实例本身）==")
def cfg_for(path, k=4):
    return VllmConfig(
        model_config=ModelConfig(model="fake/tiny", dtype="float32", max_model_len=64),
        speculative_config=SpeculativeConfig(method="custom_class", model=path,
                                             num_speculative_tokens=k))


example_class = importlib.import_module("examples.custom_proposer").RepeatLastTokenProposer
instance = create_custom_proposer(cfg_for(EXAMPLE))
check("B1. 合法类：返回的就是该类的实例（type 相同，无套壳）",
      type(instance) is example_class and callable(instance.propose),
      f"{type(instance).__name__}")
check("B2. 构造参数只有 VllmConfig，且它不暴露任何运行时对象",
      all(not hasattr(cfg_for(EXAMPLE), name)
          for name in ("scheduler", "kv_cache_manager", "requests", "input_batch", "worker")))


# ================================================ C. 错误分类
print("\n== C. 错误分类（含异常链）==")
def error_of(path):
    try:
        create_custom_proposer(cfg_for(path))
        return None
    except Exception as error:  # noqa: BLE001 - 就是要分类
        return error


cases = [
    ("NoDots", ValueError, "full module path"),
    ("no_such_module_xyz.Klass", ImportError, "Cannot import module"),
    ("examples.custom_proposer.NoSuchClass", AttributeError, "has no attribute"),
    ("test_custom_proposer.BoomOnInitProposer", RuntimeError, "must accept VllmConfig"),
    ("test_custom_proposer.NoProposeProposer", AttributeError, "must have a 'propose'"),
    ("test_custom_proposer.NotCallableProposeProposer", AttributeError, "not callable"),
]
for path, expected_type, needle in cases:
    error = error_of(path)
    ok = isinstance(error, expected_type) and needle in str(error)
    check(f"C. {path} → {expected_type.__name__}", ok,
          f"实际 {type(error).__name__}: {str(error)[:60]}")
cause_ok = isinstance(getattr(error_of("no_such_module_xyz.Klass"), "__cause__", None), ImportError) \
    and isinstance(getattr(error_of("test_custom_proposer.BoomOnInitProposer"), "__cause__", None),
                   RuntimeError)
check("C7. 异常链保留原始原因（`raise ... from e`）", cause_ok)


# ================================================ D. Runner 接线
print("\n== D. Runner 接线 ==")
recorder = importlib.import_module("test_custom_proposer")
recorder.RecordingProposer.calls.clear()
engine, core, runner = make_engine(
    SpeculativeConfig(method="custom_class",
                      model="test_custom_proposer.RecordingProposer", num_speculative_tokens=4))
rows_at_call, identity_at_call = [], []
original_propose = runner.proposer.propose


def spy(sampled, num_tokens_no_spec, token_ids_cpu, slot_mappings=None):
    rows_at_call.append(len(runner.input_batch.req_ids))
    identity_at_call.append((num_tokens_no_spec is runner.input_batch.num_tokens_no_spec,
                             token_ids_cpu is runner.input_batch.token_ids_cpu))
    return original_propose(sampled, num_tokens_no_spec, token_ids_cpu, slot_mappings)


runner.proposer.propose = spy
outputs, stats = run(engine, core, [("a", [1, 2, 3, 4, 1, 2, 3, 4])])
engine.shutdown()
check("D1. 合法实例就是 Runner 持有的对象（不是包装类）",
      type(runner.proposer) is recorder.RecordingProposer)
check("D2. 三个位置参数就是 InputBatch 自己的缓冲（身份相同，不拷贝/不换格式）",
      bool(identity_at_call) and all(identity_at_call))
check("D3. 行对齐：`sampled_token_ids` 行数 == 调用当刻的批行数",
      len(rows_at_call) == len(recorder.RecordingProposer.calls)
      and all(len(call["rows"]) == rows
              for rows, call in zip(rows_at_call, recorder.RecordingProposer.calls)),
      f"{rows_at_call}")
check("D4. 缓冲是定长的（长度 = max_num_reqs），只有前 N 行有效",
      all(len(call["num_tokens"]) == runner.input_batch.max_num_reqs
          for call in recorder.RecordingProposer.calls),
      f"max_num_reqs={runner.input_batch.max_num_reqs}")
upstream_path = (Path("/home/user/anaconda3/envs/vllm-omni-dev/lib/python3.12/site-packages/vllm")
                 / "v1" / "worker" / "gpu_model_runner.py")
upstream = upstream_path.read_text()
marker = 'elif spec_config.method == "custom_class":'
snippet = upstream[upstream.index(marker):upstream.index(marker) + 400]
check("D5. 调用参数与冻结的上游分支一致（3 个位置参数 + slot_mappings=）",
      all(token in snippet for token in ("propose(", "sampled_token_ids,", "num_tokens_no_spec,",
                                         "token_ids_cpu,", "slot_mappings=slot_mappings,")))

engine, core, runner = make_engine(SpeculativeConfig(model=EXAMPLE, num_speculative_tokens=4))
check("D6. 插件不需要实现 `remove_requests`（只有 propose 是契约）",
      not hasattr(runner.proposer, "remove_requests"))
outputs, _ = run(engine, core, [("a", [1, 2, 3, 4, 1, 2, 3, 4])], max_tokens=4)
list(engine.step())          # 结束清理轮
engine.shutdown()
check("D7. 请求结束那一轮不因缺少可选钩子而崩", bool(outputs["a"]), f"{outputs}")


# ================================================ E. 行为
print("\n== E. 行为（候选质量不改变答案）==")
prompts = [("a", [1, 2, 3, 4, 1, 2, 3, 4]), ("b", [5, 6, 5, 6, 5, 6])]
baseline_engine, baseline_core, _ = make_engine(None)
baseline, _ = run(baseline_engine, baseline_core, prompts)
baseline_engine.shutdown()
for label, path, expect_tokens in (("常规", EXAMPLE, True),
                                   ("空候选", "test_custom_proposer.EmptyProposer", False),
                                   ("全错候选", "test_custom_proposer.AllWrongProposer", True),
                                   ("变长候选", "test_custom_proposer.VariableLengthProposer", True)):
    recorder.EmptyProposer.calls = 0
    engine, core, runner = make_engine(
        SpeculativeConfig(method="custom_class", model=path, num_speculative_tokens=3))
    outputs, stats = run(engine, core, prompts)
    engine.shutdown()
    hits = [entry for entry in stats if entry is not None]
    draft_tokens = sum(entry.num_draft_tokens for entry in hits)
    check(f"E. {label}：greedy 输出与非投机逐 token 相同", outputs == baseline, f"{outputs}")
    check(f"E. {label}：链路真的走过（草稿枚数={draft_tokens}）",
          draft_tokens > 0 if expect_tokens else recorder.EmptyProposer.calls > 0)

for path in ("test_custom_proposer.TooFewRowsProposer", "test_custom_proposer.TooManyRowsProposer"):
    engine, core, runner = make_engine(
        SpeculativeConfig(method="custom_class", model=path, num_speculative_tokens=2))
    try:
        run(engine, core, [("a", [1, 2, 3, 4, 1, 2, 3, 4])], max_tokens=3)
        loud = False
    except RuntimeError as error:
        loud = "必须按批行逐行返回" in str(error)
    finally:
        engine.shutdown()
    check(f"E. {path.split('.')[-1]}：行数不匹配当场报错", loud)


# ================================================ F. 端到端（命令行入口）
print("\n== F. 端到端（demo 的 --spec-model）==")
import subprocess  # noqa: E402

demo_model = ROOT / "models" / "Qwen3-1.7B"
if not demo_model.is_dir():
    check("F1. demo 入口（--spec-method custom_class --spec-model 点号路径）", True,
          "跳过：本机没有 models/Qwen3-1.7B（同一路径已由 D 段的 tiny 模型用例覆盖）")
elif DEVICE != "cuda":
    check("F1. demo 入口（--spec-method custom_class --spec-model 点号路径）", True,
          "跳过：投机验证只在 CUDA 上（拒绝采样是 Triton 内核，与上游一致）")
else:
    result = subprocess.run(
        [sys.executable, str(ROOT / "minivllm" / "demo.py"), "--device", "cuda",
         "--max-new-tokens", "4", "--spec-method", "custom_class",
         "--spec-model", EXAMPLE, "--spec-k", "3", "测试"],
        capture_output=True, text=True, timeout=900)
    check("F1. demo 入口（--spec-method custom_class --spec-model 点号路径）",
          result.returncode == 0 and "RepeatLastTokenProposer" in result.stdout,
          [line for line in result.stdout.splitlines() if "投机" in line]
          or result.stderr.strip().splitlines()[-1:])

# F2：只用 --spec-model（不给 --spec-method）→ 靠方法推断走 custom_class
demo_source = (ROOT / "minivllm" / "demo.py").read_text()
check("F2. 教学入口的推断：只给 --spec-model 时交给 SpeculativeConfig 推断",
      "SpeculativeConfig(method=args.spec_method, model=args.spec_model" in demo_source)

print()
print(f"设备={DEVICE}")
print(f"{'全部通过' if not FAIL else '失败: ' + ', '.join(FAIL)}")
sys.exit(1 if FAIL else 0)
