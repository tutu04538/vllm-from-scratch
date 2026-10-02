"""57F 对照（B）：与**真实 vLLM**（本机 0.28.0）的定向对照。

四个对照，从"纯数学"到"整条链路"：

  A. **模型 logits**：同一份小权重（`fixtures/step30_qwen3/tiny_gqa`）、同一 prompt，
     逐位置比 top-5 logprobs（我们的 `compute_logits` vs vLLM 的 `prompt_logprobs`）——
     这条覆盖 GQA、q/k norm、RoPE、以及"绝对位置"。
  B. **增量位置**：vLLM greedy 生成几步的 `logprobs` vs 我们**逐 token decode**（每步一行的
     KV 复用路径）算出的 logprobs——覆盖"增量位置与全量位置一致"。
  C. **拒绝采样**：固定 p/q、ragged K，比**接受率**与 **recovered 分布**（统计对照；
     199/200 明确不要求碰巧同 seed，随机输入不同就比算法结果或统计）。
  D. **端到端 greedy 短输出**：本机真实 Qwen3-1.7B，同一 prompt，比 token 序列。

### 运行环境说明（200 §B 要求记录）

WSL2 下 vLLM 默认关掉 pinned memory，进而 `is_uva_available()` 为假、引擎起不来，
所以本脚本必须带两个环境变量：

    VLLM_WSL2_ENABLE_PIN_MEMORY=1   # 打开 WSL2 的 pinned memory（内核 6.6 ✓ 支持）
    VLLM_ENABLE_V1_MULTIPROCESSING=0 # 引擎留在本进程（否则子进程无法 re-exec `<stdin>`）

**显存**：A/B/C 用 tiny 模型（几十 MB）；D 用 1.7B，两个引擎**顺序**跑并在中间释放，
避免同时驻留（本机 24 GB，实测够，但没必要冒险）。
"""

import json
import os
import sys

sys.path.insert(0, "/home/user/proj/vllm-from-scratch")

import torch

FAIL = []
TINY = "fixtures/step30_qwen3/tiny_gqa"
TINY_CONFIG = json.load(open(f"{TINY}/config.json"))
REAL = "models/Qwen3-1.7B"
PROMPT = [1, 2, 3, 5, 7, 9, 0, 1, 4]


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def logprobs_of(logits: torch.Tensor, token_ids) -> dict[int, float]:
    """一组 logits → {token: logprob}（只取给定的 token，够我们比对了）。"""
    logp = logits.to(torch.float32).log_softmax(dim=-1)
    return {int(token): float(logp[token]) for token in token_ids}


# ============================================ 我们这边：直接驱动 Runner 的真实前向路径

def our_logits_by_position(token_ids, chunks=None, model_dir=TINY, hf_config=None,
                           dtype="float32", device="cpu", num_gpu_blocks=8,
                           block_size=4, max_model_len=64):
    """用 Runner 的真实路径跑一段 token，返回 `{位置: logits}`。

    `chunks=None` → 一次算完；给了 chunks → 分块（chunked prefill）；每个 chunk 一行 → decode。
    与 `check_step57_model_logits.py` 同一套驱动方式（只喂协议包）。
    """
    hf_config = hf_config or TINY_CONFIG
    from step57.config import (CacheConfig, DeviceConfig, ModelConfig, SchedulerConfig,
                               VllmConfig)
    from step57.core.sched.output import (CachedRequestData, NewRequestData, SchedulerOutput)
    from step57.sampling_params import SamplingParams
    from step57.worker.gpu_model_runner import GPUModelRunner

    config = VllmConfig(
        model_config=ModelConfig(model=model_dir, dtype=dtype, max_model_len=max_model_len,
                                 hf_config=hf_config),
        cache_config=CacheConfig(block_size=block_size, num_gpu_blocks=num_gpu_blocks),
        scheduler_config=SchedulerConfig(max_num_seqs=2,
                                         max_num_batched_tokens=2 * len(token_ids)),
        device_config=DeviceConfig(device=device))
    runner = GPUModelRunner(config, device)
    runner.load_model()
    runner.initialize_kv_cache(config.cache_config)

    chunks = chunks or [list(token_ids)]
    blocks = list(range((len(token_ids) + block_size - 1) // block_size))
    params = SamplingParams(temperature=0.0)
    by_position = {}
    computed = 0
    for index, chunk in enumerate(chunks):
        if index == 0:
            packet = SchedulerOutput(
                scheduled_new_reqs=[NewRequestData(req_id="r", prompt_token_ids=list(token_ids),
                                                   sampling_params=params,
                                                   block_ids=(list(blocks),),
                                                   num_computed_tokens=0)],
                scheduled_cached_reqs=CachedRequestData.make_empty(),
                num_scheduled_tokens={"r": len(chunk)},
                total_num_scheduled_tokens=len(chunk), scheduled_spec_decode_tokens={},
                finished_req_ids=set())
        else:
            packet = SchedulerOutput(
                scheduled_new_reqs=[],
                scheduled_cached_reqs=CachedRequestData(
                    req_ids=["r"], resumed_req_ids=set(), new_block_ids=[None],
                    num_computed_tokens=[computed], num_output_tokens=[0], all_token_ids={}),
                num_scheduled_tokens={"r": len(chunk)},
                total_num_scheduled_tokens=len(chunk), scheduled_spec_decode_tokens={},
                finished_req_ids=set())
        runner._update_states(packet)
        inputs = runner._prepare_inputs(packet)
        with torch.no_grad():
            logits = runner.model.compute_logits(runner._run_model(inputs))
        for offset, position in enumerate(range(computed, computed + len(chunk))):
            by_position[position] = logits[offset]
        computed += len(chunk)
    return by_position


# ============================================ A. 模型 logits（逐位置 top-5）

def vllm_tiny():
    from vllm import LLM

    return LLM(model=TINY, dtype="float32", max_model_len=64, gpu_memory_utilization=0.2,
               enforce_eager=True, disable_log_stats=True, enable_prefix_caching=False,
               seed=0)


def compare_logprobs(name, prompt, vllm_prompt_logprobs, ours_by_position, expected_positions):
    """vLLM 的 prompt_logprobs（位置 i 给 token i 的分布）对应我们位置 i-1 的 logits。"""
    worst = 0.0
    mismatched = []
    for position in expected_positions:
        reference = vllm_prompt_logprobs[position]      # 预测 prompt[position]
        if reference is None:
            continue
        our_logits = ours_by_position[position - 1]
        our_top = int(our_logits.argmax())
        vllm_top = max(reference.items(), key=lambda item: item[1].logprob)[0]
        if our_top != vllm_top:
            mismatched.append((position, our_top, vllm_top))
        for token, entry in reference.items():
            difference = abs(logprobs_of(our_logits, [token])[token] - entry.logprob)
            worst = max(worst, difference)
    check(f"{name}：每个位置的 argmax 与 vLLM 一致",
          not mismatched, f"不一致 {mismatched[:3]}")
    check(f"{name}：vLLM 给的每个 (token, logprob) 与我们逐值一致",
          worst < 2e-4, f"最大差 {worst:.2e}")


print("=== A. 模型 logits（tiny 权重，逐位置对照 vLLM 的 prompt_logprobs）")
from vllm import SamplingParams as VllmSamplingParams   # noqa: E402  （真实 vLLM 的采样参数）

ours_full = our_logits_by_position(PROMPT)
llm = vllm_tiny()
out = llm.generate([{"prompt_token_ids": PROMPT}],
                   VllmSamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=5),
                   use_tqdm=False)[0]
compare_logprobs("A. 全量 prefill", PROMPT, out.prompt_logprobs, ours_full, range(1, len(PROMPT)))

# 同一条 prompt 分 3 块（chunked prefill）与逐 token（decode）：位置必须与全量一致
ours_chunked = our_logits_by_position(PROMPT, chunks=[PROMPT[:3], PROMPT[3:6], PROMPT[6:]])
ours_decoded = our_logits_by_position(PROMPT, chunks=[[token] for token in PROMPT])
worst_chunked = max((ours_chunked[p] - ours_full[p]).abs().max().item()
                    for p in range(len(PROMPT)))
worst_decoded = max((ours_decoded[p] - ours_full[p]).abs().max().item()
                    for p in range(len(PROMPT)))
check("A. 分 3 块 / 逐 token decode 的位置 logits 与全量一致（增量位置）",
      worst_chunked < 1e-5 and worst_decoded < 1e-5,
      f"分块 {worst_chunked:.2e}、逐 token {worst_decoded:.2e}")

# B. 增量位置直接对 vLLM：让它 greedy 生成几步（每步都给 logprobs）
print("\n=== B. 增量位置（vLLM 逐步生成的 logprobs vs 我们逐 token decode）")
out = llm.generate([{"prompt_token_ids": PROMPT}],
                   VllmSamplingParams(max_tokens=6, temperature=0.0, logprobs=5),
                   use_tqdm=False)[0]
generated = list(out.outputs[0].token_ids)
step_logprobs = out.outputs[0].logprobs            # 第 j 步：预测 generated[j]
worst = 0.0
argmax_mismatch = []
extended = PROMPT + generated
ours_extended = our_logits_by_position(extended, chunks=[[t] for t in extended])
for step, entries in enumerate(step_logprobs):
    # 第 step 步的输入是 PROMPT + generated[:step]，预测 generated[step]：即位置 len(PROMPT)+step-1 的 logits
    our_logits = ours_extended[len(PROMPT) + step - 1]
    if int(our_logits.argmax()) != max(entries.items(), key=lambda item: item[1].logprob)[0]:
        argmax_mismatch.append(step)
    for token, entry in entries.items():
        worst = max(worst, abs(logprobs_of(our_logits, [token])[token] - entry.logprob))
check("B. 生成阶段每一步的 top-5 logprobs 与 vLLM 逐值一致（增量位置）",
      not argmax_mismatch and worst < 2e-4,
      f"argmax 不一致的步={argmax_mismatch}、最大差 {worst:.2e}")

# A/B 用的 tiny vLLM 会按 gpu_memory_utilization 预留几 GiB 的 KV：**明确释放**，
# 否则后面的 D 子进程起不来（父进程还占着显存）
del llm
import gc as _gc

_gc.collect()
torch.cuda.empty_cache()

# ============================================ C. 拒绝采样：与 vLLM 内核对统计
print("\n=== C. 拒绝采样（固定 p/q + ragged K，统计对照 vLLM 内核）")


def _rejection_case():
    """固定的一组 p/q 与 ragged K（K=[2,1]），两边共用。

    注意**两个接口收的东西不一样**（这正是第一版对不上的原因）：
      - vLLM 的 `rejection_sample` 收 **target-only** 的 `[P, V]`（bonus 行在调用前已经切走了）
      - 我们的 `RejectionSampler.forward` 收**紧凑 [P+B, V]**（两个坐标系见 metadata.py）
    所以下面用 metadata 自己的索引把同一组 logits 摆成各自的形状，避免手摆出错。
    """
    vocab = 6
    drafts = {"r0": [0, 0], "r1": [0]}
    scheduled = {"r0": 3, "r1": 2}
    from step57.spec_decode.metadata import SpecDecodeMetadata

    meta = SpecDecodeMetadata.from_scheduled(drafts, scheduled, ["r0", "r1"])
    draft_probs = torch.zeros(3, vocab)
    draft_probs[:, 0], draft_probs[:, 1] = 0.6, 0.4
    target_rows = torch.zeros(3, vocab)
    target_rows[:, 0], target_rows[:, 1], target_rows[:, 2] = 1.0, 1.0, 0.5
    bonus_rows = torch.zeros(2, vocab)
    bonus_rows[:, 2] = 5.0
    return meta, draft_probs, target_rows, bonus_rows


def _summarise(draws, step):
    """跑 `draws` 次，统计：平均接受数、首位置是否被拒、以及"首位置被拒时提交的 token"分布。"""
    accepted_counts, first_rejected, first_accepted = [], 0, 0
    recovered = torch.zeros(6)
    for _ in range(draws):
        token_ids, lengths = step()
        for row, count in enumerate(lengths):
            accepted_counts.append(count)
        if lengths[0] < 2:                      # r0 的 K=2：接受的不足 2 枚 = 位置 0 或 1 被拒
            if token_ids[0] == 0:               # 草稿 token 就是 0 → 位置 0 被接受了
                first_accepted += 1
            else:
                first_rejected += 1
                recovered[token_ids[0]] += 1
    distribution = recovered / max(1.0, recovered.sum())
    return {"mean": sum(accepted_counts) / len(accepted_counts),
            "rejected_first": first_rejected, "accepted_first": first_accepted,
            "recovered": distribution}


def vllm_rejection_statistics(draws=600):
    from vllm.v1.sample.logits_processor import LogitsProcessors
    from vllm.v1.sample.metadata import SamplingMetadata
    from vllm.v1.sample.rejection_sampler import rejection_sample

    meta, draft_probs, target_rows, bonus_rows = _rejection_case()
    device = "cuda"
    num_draft_tokens = meta.num_draft_tokens
    draft_token_ids = meta.draft_token_ids.to(torch.int32).to(device)
    cu_num_draft = meta.cu_num_draft_tokens.to(torch.int32).to(device)
    draft_probs = draft_probs.to(device)
    target_logits = target_rows.to(device)              # ← target-only（vLLM 的接口）
    # bonus 行：vLLM 只用来在"全接受"时追加，取一个固定 token 即可
    bonus = torch.full((2, 1), 2, dtype=torch.int32, device=device)
    sampling_metadata = SamplingMetadata(
        temperature=torch.ones(2, device=device), all_greedy=False, all_random=True,
        top_p=None, top_k=None, generators={}, max_num_logprobs=None, no_penalties=True,
        prompt_token_ids=None, frequency_penalties=torch.zeros(2, device=device),
        presence_penalties=torch.zeros(2, device=device),
        repetition_penalties=torch.ones(2, device=device),
        output_token_ids=[[], []], allowed_token_ids_mask=None, bad_words_token_ids={},
        logitsprocs=LogitsProcessors())

    def step():
        out = rejection_sample(draft_token_ids, num_draft_tokens,
                               max(num_draft_tokens), cu_num_draft, draft_probs,
                               target_logits.clone(), bonus, sampling_metadata).cpu()
        rows = out.tolist()
        lengths = [len([t for t in row if t != -1]) - 1 for row in rows]
        return rows[0], lengths

    return _summarise(draws, step)


def our_rejection_statistics(draws=600):
    from step57.sample import Sampler
    from step57.sample import SamplingMetadata as OurMetadata
    from step57.spec_decode.rejection_sampler import RejectionSampler

    meta, draft_probs, target_rows, bonus_rows = _rejection_case()
    # 摆成**紧凑 [P+B, V]**：用 metadata 自己的索引，不手摆
    logits = torch.zeros(meta.num_draft_tokens_total + meta.batch_size, 6)
    logits[meta.target_logits_indices] = target_rows
    logits[meta.bonus_logits_indices] = bonus_rows
    sampling_metadata = OurMetadata(
        temperature=torch.ones(2), all_greedy=False, all_random=True, top_k=None, top_p=None,
        generators={}, no_penalties=True, prompt_token_ids=[[], []], output_token_ids=[[], []],
        min_tokens=[0, 0], stop_token_ids=[[], []], spec_token_ids=[[0, 0], [0]])
    sampler = RejectionSampler(Sampler())

    def step():
        out = sampler.forward(meta, logits, draft_probs, sampling_metadata).sampled_token_ids
        rows = out.tolist()
        lengths = [len([t for t in row if t != -1]) - 1 for row in rows]
        return rows[0], lengths

    return _summarise(draws, step)


theirs = vllm_rejection_statistics()
ours = our_rejection_statistics()

# 解析值：接受概率 = min(1, p[d]/q[d])；平均接受数 = a0 + a0·a1（r0）与 a0（r1）
_, _, analytic_target_rows, _ = _rejection_case()
probability = analytic_target_rows.softmax(-1)[0].tolist()
accept_first = min(1.0, probability[0] / 0.6)
accept_second = min(1.0, probability[0] / 0.6)
analytic = ((accept_first + accept_first * accept_second) + accept_first) / 2
check("C. 平均接受数与解析值 / vLLM 一致（同一 p/q、同一 ragged K）",
      abs(ours["mean"] - analytic) < 0.06 and abs(theirs["mean"] - analytic) < 0.06,
      f"解析 {analytic:.3f}｜vLLM {theirs['mean']:.3f}｜我们 {ours['mean']:.3f}")
check("C. 首位置接受次数一致（`min(1, p/q)` 的判定）",
      abs(theirs["accepted_first"] - ours["accepted_first"]) < 0.12 * 600,
      f"vLLM {theirs['accepted_first']}/600 vs 我们 {ours['accepted_first']}/600"
      f"（解析 {(1 - accept_first) * 600:.0f}？反了：接受应为 {accept_first * 600:.0f}）")

# recovered 分布 ∝ max(p − q, 0)（这里 q 只在 token 0/1 上有质量）
weights = torch.clamp_min(torch.tensor(probability) - torch.tensor([0.6, 0.4, 0, 0, 0, 0]), 0)
expected = weights / weights.sum()
for label, stats in (("vLLM", theirs), ("我们", ours)):
    check(f"C. {label}：位置 0 被拒后 recovered 分布 ∝ max(p−q,0)",
          (stats["recovered"] - expected).abs().max().item() < 0.05,
          f"实测 {[round(x, 3) for x in stats['recovered'].tolist()]} vs "
          f"解析 {[round(x, 3) for x in expected.tolist()]}")

# ============================================ D. 端到端（真实 1.7B，greedy）
#
# 四个引擎（vLLM/我们 × bf16/fp32）如果都留在同一个进程里，CUDA 缓存会把显存吃光
# （实测跑完 fp32 之后只剩 7.8 GiB，vLLM 起不来）。所以**每种 dtype 用一个子进程**：
# 子进程内跑两个引擎并把结论打成一行 JSON，父进程只做比较。
if "--phase" in sys.argv:
    dtype = sys.argv[sys.argv.index("--phase") + 1]
    import gc

    from transformers import AutoTokenizer

    from step57 import (CacheConfig, DeviceConfig, LLMEngine, ModelConfig, SamplingParams,
                        SchedulerConfig, UniProcExecutor, VllmConfig, Worker)

    real_config = json.load(open(f"{REAL}/config.json"))
    tokenizer = AutoTokenizer.from_pretrained(REAL, local_files_only=True)
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "用一句话解释什么是 KV cache。"}],
        add_generation_prompt=True, enable_thinking=False, tokenize=True)
    prompt_ids = list(rendered["input_ids"] if hasattr(rendered, "keys") else rendered)
    max_new = 16

    from vllm import LLM

    free, total = torch.cuda.mem_get_info()
    print(f"[{dtype}] 子进程开始前：空闲 {free / 2**30:.1f} / {total / 2**30:.1f} GiB", flush=True)
    # fp32 的 1.7B 权重就要 6.8 GiB，utilization 给小了会连 KV 块都分不出来
    llm = LLM(model=REAL, dtype=dtype, max_model_len=512, enforce_eager=True,
              gpu_memory_utilization=0.6, disable_log_stats=True,
              enable_prefix_caching=False, seed=0)
    vllm_out = llm.generate([{"prompt_token_ids": prompt_ids}],
                            VllmSamplingParams(max_tokens=max_new, temperature=0.0, logprobs=5),
                            use_tqdm=False)[0].outputs[0]
    vllm_tokens = list(vllm_out.token_ids)
    vllm_step_logprobs = [
        {int(token): entry.logprob for token, entry in step.items()}
        for step in (vllm_out.logprobs or [])]
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    config = VllmConfig(
        model_config=ModelConfig(model=REAL, dtype=dtype, max_model_len=512,
                                 hf_config=real_config),
        cache_config=CacheConfig(block_size=16, num_gpu_blocks=64),
        scheduler_config=SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=64),
        device_config=DeviceConfig(device="cuda"))
    engine = LLMEngine(config, UniProcExecutor(config, Worker(config)))
    engine.add_request("r", prompt_ids, SamplingParams(
        max_tokens=max_new, temperature=0.0, eos_token_id=real_config["eos_token_id"]))
    our_tokens = []
    while engine.has_unfinished_requests():
        for output in engine.step():
            our_tokens = list(output.token_ids)
    engine.shutdown()
    del engine
    gc.collect()
    torch.cuda.empty_cache()

    # 在同一段输入（vLLM 生成的前缀）上比每步 logprobs，量出**数值差**与分歧点的候选间隔
    prefix = 0
    while prefix < min(len(our_tokens), len(vllm_tokens)) and \
            our_tokens[prefix] == vllm_tokens[prefix]:
        prefix += 1
    report = {"dtype": dtype, "ours": our_tokens, "vllm": vllm_tokens, "prefix": prefix,
              "worst": None, "gap": None, "vllm_gap": None, "step": None}
    if vllm_step_logprobs:
        probe = prompt_ids + vllm_tokens[:prefix + 1]
        ours_by_position = our_logits_by_position(
            probe, model_dir=REAL, hf_config=real_config, dtype=dtype, device="cuda",
            num_gpu_blocks=8, block_size=16, max_model_len=512)
        worst = 0.0
        for step in range(prefix):
            our_logits = ours_by_position[len(prompt_ids) + step - 1]
            for token, logprob in vllm_step_logprobs[step].items():
                worst = max(worst, abs(logprobs_of(our_logits, [token])[token] - logprob))
        report["worst"] = worst
        if prefix < min(len(our_tokens), len(vllm_tokens)):
            step = prefix
            our_logits = ours_by_position[len(prompt_ids) + step - 1]
            ours_logprobs = logprobs_of(our_logits, [our_tokens[step], vllm_tokens[step]])
            report["step"] = step
            report["gap"] = abs(ours_logprobs[our_tokens[step]] - ours_logprobs[vllm_tokens[step]])
            entries = sorted(vllm_step_logprobs[step].values(), reverse=True)
            report["vllm_gap"] = entries[0] - entries[1] if len(entries) > 1 else 0.0
    print("RESULT " + json.dumps(report))
    sys.exit(0)

print("\n=== D. 端到端（真实 Qwen3-1.7B，greedy；fp32 是对照组，bf16 用来量化数值敏感度）")

if not os.path.isdir(REAL):
    check("D. 本机没有 Qwen3-1.7B → 跳过端到端对照（不下载新模型）", True, REAL)
else:
    import subprocess

    def run_phase(dtype):
        environment = dict(os.environ, VLLM_WSL2_ENABLE_PIN_MEMORY="1",
                           VLLM_ENABLE_V1_MULTIPROCESSING="0")
        process = subprocess.run([sys.executable, __file__, "--phase", dtype],
                                 capture_output=True, text=True, env=environment)
        for line in process.stdout.splitlines():
            if line.startswith("RESULT "):
                return json.loads(line[len("RESULT "):])
        raise RuntimeError(f"{dtype} 子进程没有给出结果：\n{process.stdout[-2000:]}\n"
                           f"{process.stderr[-2000:]}")

    fp32, bf16 = run_phase("float32"), run_phase("bfloat16")

    check("D1. fp32 下 greedy 短输出与 vLLM **逐 token 一致**（端到端对照的正式验收）",
          fp32["ours"] == fp32["vllm"],
          f"公共前缀 {fp32['prefix']}/{len(fp32['vllm'])}；"
          f"我们 {fp32['ours'][:8]}…｜vLLM {fp32['vllm'][:8]}…")

    if bf16["prefix"] < min(len(bf16["ours"]), len(bf16["vllm"])):
        check("D2. bf16 下分歧点是**近似并列**：两个候选在我们这边的 logprob 差，"
              "不超过同一次运行里观测到的数值差量级 → 可由 bf16 的 logits 网格解释",
              bf16["gap"] <= max(bf16["worst"], 0.25),
              f"第 {bf16['step'] + 1} 步分歧：我们选 {bf16['ours'][bf16['step']]}"
              f"（在我们这边两候选差 {bf16['gap']:.2f}）｜"
              f"vLLM 选 {bf16['vllm'][bf16['step']]}（它的 top1-top2 差 {bf16['vllm_gap']:.2f}）｜"
              f"共同前缀上的逐值最大差 {bf16['worst']:.2f}")
        check("D3. bf16 下的公共前缀仍占多数（数值噪声只影响个别近并列位置）",
              bf16["prefix"] >= 5,
              f"公共前缀 {bf16['prefix']}/{min(len(bf16['ours']), len(bf16['vllm']))}")
    else:
        check("D2. bf16 下也逐 token 一致", True, f"{len(bf16['ours'])} 个 token")
        check("D3. bf16 下的公共前缀", bf16["prefix"] >= 5, str(bf16["prefix"]))

print()
print(f"{'全部通过' if not FAIL else '失败: ' + ', '.join(FAIL)}")
sys.exit(1 if FAIL else 0)
