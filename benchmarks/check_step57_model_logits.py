"""57B 验收（对应需求里的 `test_model_logits.py`）：模型数学与三种切分的一致性。

四段，越往前越"定点"，越往后越"整体"：

  1. **注意力块的参考实现对照**：用公式手写一遍（QKV 投影 → 按 head 的 q/k RMSNorm →
     RoPE → 绝对位置 causal attention → GQA 复制 → o_proj），逐值比对我们的 `Qwen3Attention`。
     GQA、q/k norm、RoPE、causal mask 四件事在这段里各查一遍：参考实现只用需求里的公式，
     不调用被测代码，而且要**对 GQA 敏感**（扰动第二个 KV head 必须让参考跟着变，否则这段
     对照可能是个恒等式）。
  2. **与 HF（transformers 的 Qwen3）整体对照**：同一份权重、同一个 config，逐位置 logits
     相等 → 证明"没有漏掉任何一处"。再做三次**消融**：q/k norm 换成非平凡值、rope_theta 换掉、
     扰动第二个 KV head 的 k_proj——每处都两边同步改，仍要相等。
  3. **残差与 tied embedding**：最终 hidden 与 HF 的 `hidden_states[-1]` 逐值相等（残差链错一
     步就对不上）；tie=True 时两边都共享词嵌入，logits 仍然相等。
  4. **full / chunked / decode 三种切分**：同一段 token 一次算完、分 3 块算、逐 token 算，
     同一位置的 logits 必须一致（causal 的含义就是这个）。这条**走 Runner 的真实路径**：
     块表镜像、positions、slot_mapping 都由 `_update_states` / `_prepare_inputs` 算出来，
     测试只提供协议包——第一版这里手搭 `PreparedInputs`，绕过了 runner 自己的块表镜像，
     于是读 KV 时用的是全 0 的块表，三种切分当然对不上。**测试不能自己发明一条输入路径。**

模型 forward 拿不到 Request 这条边界也在这里查（签名里只有 input_ids / positions）。
"""

import inspect
import json
import sys

sys.path.insert(0, "/home/user/proj/vllm-from-scratch")

import torch

from step57.attention import AttentionMetadataBuilder, set_forward_context
from step57.config import CacheConfig, DeviceConfig, ModelConfig, SchedulerConfig, VllmConfig
from step57.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput
from step57.model_loader import get_model, iter_weights
from step57.models import Qwen3ForCausalLM
from step57.request import Request
from step57.sampling_params import SamplingParams
from step57.worker import GPUModelRunner

FAIL = []
TINY_DIR = "fixtures/step30_qwen3/tiny_gqa"
TINY_CONFIG = json.load(open(f"{TINY_DIR}/config.json"))
CHECKPOINT = dict(iter_weights(TINY_DIR))
PROMPT = [1, 2, 3, 5, 7, 9, 0, 1, 4]
BLOCK_SIZE = 4
HEAD_DIM = TINY_CONFIG["head_dim"]
NUM_HEADS = TINY_CONFIG["num_attention_heads"]
NUM_KV_HEADS = TINY_CONFIG["num_key_value_heads"]


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def build_tiny(tie=False):
    """直接构造模型并装权重（绕开加载器，专测数学）。"""
    config = dict(TINY_CONFIG, tie_word_embeddings=tie)
    model = Qwen3ForCausalLM(config)
    model.load_weights(iter(CHECKPOINT.items()))
    return model.eval()


def tiny_vllm_config(hf_config=None):
    return VllmConfig(
        model_config=ModelConfig(model=TINY_DIR, dtype="float32", max_model_len=64,
                                 hf_config=hf_config or TINY_CONFIG),
        cache_config=CacheConfig(block_size=BLOCK_SIZE, num_gpu_blocks=8),
        scheduler_config=SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=16),
        device_config=DeviceConfig(device="cpu"))


def run_chunks(model, token_ids, chunks):
    """用 Runner 的真实路径跑一段 token（可切成多块），返回 `{位置: (logits, hidden)}`。

    只喂**协议包**（`NewRequestData` / `CachedRequestData`），块表、positions、slot_mapping
    全部由 Runner 自己算——这样测的就是生产路径，而不是测试另写的一套。
    """
    vllm_config = tiny_vllm_config()
    runner = GPUModelRunner(vllm_config, "cpu", model=model)
    runner.initialize_kv_cache(vllm_config.cache_config)
    block_ids = list(range((len(token_ids) + BLOCK_SIZE - 1) // BLOCK_SIZE))
    params = SamplingParams(temperature=0.0)
    by_position = {}
    computed = 0
    for index, chunk in enumerate(chunks):
        num_scheduled = {"r": len(chunk)}
        if index == 0:
            packet = SchedulerOutput(
                scheduled_new_reqs=[NewRequestData(
                    req_id="r", prompt_token_ids=list(token_ids), sampling_params=params,
                    block_ids=(list(block_ids),), num_computed_tokens=0)],
                scheduled_cached_reqs=CachedRequestData.make_empty(),
                num_scheduled_tokens=num_scheduled,
                total_num_scheduled_tokens=len(chunk), scheduled_spec_decode_tokens={},
                finished_req_ids=set())
        else:
            packet = SchedulerOutput(
                scheduled_new_reqs=[],
                scheduled_cached_reqs=CachedRequestData(
                    req_ids=["r"], resumed_req_ids=set(), new_block_ids=[None],
                    num_computed_tokens=[computed], num_output_tokens=[0], all_token_ids={}),
                num_scheduled_tokens=num_scheduled,
                total_num_scheduled_tokens=len(chunk), scheduled_spec_decode_tokens={},
                finished_req_ids=set())
        runner._update_states(packet)
        inputs = runner._prepare_inputs(packet)
        with torch.no_grad():
            hidden = runner._run_model(inputs)
            logits = runner.model.compute_logits(hidden)
        for offset, position in enumerate(range(computed, computed + len(chunk))):
            by_position[position] = (logits[offset], hidden[offset])
        computed += len(chunk)
    return by_position


def ours_logits(model, token_ids=None):
    token_ids = PROMPT if token_ids is None else token_ids
    return torch.stack([run_chunks(model, token_ids, [token_ids])[position][0]
                        for position in range(len(token_ids))])


# ------------------------------------------------ 1. 注意力块的参考实现

model = build_tiny()
layer = model.model.layers[0]
num_tokens = 8
token_ids = torch.tensor([1, 2, 3, 5, 7, 9, 0, 1])
positions = torch.arange(num_tokens)
hidden = layer.input_layernorm(model.model.embed_input_ids(token_ids))

# 手工给这一层搭好上下文与物理缓存：这段是**单元**测试，只问"这一层的数学对不对"
attention = layer.self_attn.attn
attention.kv_cache = torch.zeros(2, 4, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM)
metadata = AttentionMetadataBuilder(BLOCK_SIZE).build(
    query_start_loc=torch.tensor([0, num_tokens]), seq_lens=torch.tensor([num_tokens]),
    block_table=torch.tensor([[0, 1]]), slot_mapping=torch.arange(num_tokens), num_reqs=1)
with set_forward_context({attention.layer_name: metadata}, num_tokens=num_tokens):
    ours_block = layer.self_attn(positions, hidden)


def reference_attention(hidden, positions, weights):
    """按公式手写一遍（不调用被测代码）：只用 q/k/v/o 投影与两个 qk norm 的权重。"""
    def rms_norm(x, weight, eps=1e-6):
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        return x / torch.sqrt(variance + eps) * weight

    q = hidden @ weights["q_proj"].T
    k = hidden @ weights["k_proj"].T
    v = hidden @ weights["v_proj"].T
    # 按 head 切开做 q/k norm（Qwen3 特有：作用在 head_dim 上，不是 hidden 上）
    q = rms_norm(q.view(-1, NUM_HEADS, HEAD_DIM), weights["q_norm"]).view(-1, NUM_HEADS * HEAD_DIM)
    k = rms_norm(k.view(-1, NUM_KV_HEADS, HEAD_DIM), weights["k_norm"]).view(
        -1, NUM_KV_HEADS * HEAD_DIM)

    # RoPE：自己算频率表（rope_theta 从 config 读），不用被测代码的 cos_sin_cache
    theta = TINY_CONFIG["rope_parameters"]["rope_theta"]
    inv_freq = 1.0 / (theta ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM))
    freqs = torch.outer(positions.to(torch.float32), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos, sin = emb.cos().unsqueeze(1), emb.sin().unsqueeze(1)

    def rotate_half(x):
        first, second = x.chunk(2, dim=-1)
        return torch.cat((-second, first), dim=-1)

    q = q.view(num_tokens, NUM_HEADS, HEAD_DIM)
    k = k.view(num_tokens, NUM_KV_HEADS, HEAD_DIM)
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin

    # GQA：第 i 个 query 头用第 i // (num_heads / kv_heads) 个 KV 头（repeat_interleave 的次序）
    repeat = NUM_HEADS // NUM_KV_HEADS
    k = k.repeat_interleave(repeat, dim=1).transpose(0, 1)
    v = v.view(num_tokens, NUM_KV_HEADS, HEAD_DIM).repeat_interleave(repeat, dim=1).transpose(0, 1)
    q = q.transpose(0, 1)

    scores = q @ k.transpose(-1, -2) / (HEAD_DIM ** 0.5)
    scores = scores.masked_fill(positions[None, :] > positions[:, None], float("-inf"))
    out = torch.softmax(scores, dim=-1) @ v
    return out.transpose(0, 1).reshape(num_tokens, NUM_HEADS * HEAD_DIM) @ weights["o_proj"].T


def layer0_weights(k_proj=None):
    prefix = "model.layers.0.self_attn."
    return {"q_proj": CHECKPOINT[prefix + "q_proj.weight"],
            "k_proj": CHECKPOINT[prefix + "k_proj.weight"] if k_proj is None else k_proj,
            "v_proj": CHECKPOINT[prefix + "v_proj.weight"],
            "o_proj": CHECKPOINT[prefix + "o_proj.weight"],
            "q_norm": CHECKPOINT[prefix + "q_norm.weight"],
            "k_norm": CHECKPOINT[prefix + "k_norm.weight"]}


reference = reference_attention(hidden, positions, layer0_weights())
attention_diff = (ours_block - reference).abs().max().item()
check("1. 注意力块与手写参考逐值一致（GQA + q/k norm + RoPE + 绝对位置 causal 一次到位）",
      attention_diff < 1e-5, f"最大差 {attention_diff:.2e}")

perturbed_k = CHECKPOINT["model.layers.0.self_attn.k_proj.weight"].clone()
perturbed_k[HEAD_DIM:] += 0.5                      # 只动第二个 KV head
perturbed = reference_attention(hidden, positions, layer0_weights(perturbed_k))
check("1. 扰动第二个 KV head 会改变参考输出（说明这段对照对 GQA 敏感，不是恒等式）",
      (perturbed - reference).abs().max().item() > 1e-3,
      f"扰动后变化 {(perturbed - reference).abs().max().item():.2e}")

# ------------------------------------------------ 2. 与 HF 整体对照 + 三次消融

from transformers import AutoConfig, AutoModelForCausalLM   # noqa: E402


def hf_model(config=None):
    return AutoModelForCausalLM.from_pretrained(
        TINY_DIR, config=config or AutoConfig.from_pretrained(TINY_DIR)).eval()


def hf_logits(model):
    with torch.no_grad():
        return model(torch.tensor([PROMPT])).logits[0]


ours = build_tiny()
hf = hf_model()
diff = (ours_logits(ours) - hf_logits(hf)).abs().max().item()
check("2. 与 HF 的 Qwen3 逐位置 logits 一致（同一份权重 → 没有漏掉任何一处）",
      diff < 1e-4, f"最大差 {diff:.2e}")

# 消融一：q/k norm 换成非平凡权重——两边都换，仍要相等（漏了 q_norm 会在这里崩）
qk_ours, qk_hf = build_tiny(), hf_model()
pattern = (torch.arange(HEAD_DIM, dtype=torch.float32) + 1.0) / HEAD_DIM
for index in range(TINY_CONFIG["num_hidden_layers"]):
    for target, value in ((qk_ours, pattern), (qk_hf, pattern)):
        target.model.layers[index].self_attn.q_norm.weight.data.copy_(value)
        target.model.layers[index].self_attn.k_norm.weight.data.copy_(value * 0.5 + 0.25)
diff = (ours_logits(qk_ours) - hf_logits(qk_hf)).abs().max().item()
check("2. 消融：q/k norm 换成非平凡权重后仍与 HF 一致（q_norm/k_norm 确实参与了计算）",
      diff < 1e-4, f"最大差 {diff:.2e}")

# 消融二：换 rope_theta——权重没变、只有位置编码变了（漏了 RoPE 或读错 theta 会崩）
rope_config = dict(TINY_CONFIG, rope_parameters={"rope_theta": 5000.0, "rope_type": "default"})
rope_ours = get_model(ModelConfig(model=TINY_DIR, dtype="float32", hf_config=rope_config), "cpu")
hf_config = AutoConfig.from_pretrained(TINY_DIR)
hf_config.rope_parameters = {"rope_theta": 5000.0, "rope_type": "default"}
diff = (ours_logits(rope_ours) - hf_logits(hf_model(hf_config))).abs().max().item()
check("2. 消融：rope_theta 换成 5000 后仍与 HF 一致（RoPE 真的按 config 算）",
      diff < 1e-4, f"最大差 {diff:.2e}")

# 消融三：扰动第二个 KV head 的 k_proj——两边同步改（GQA 头对应错就会崩）
gqa_ours, gqa_hf = build_tiny(), hf_model()
delta = torch.full((HEAD_DIM, TINY_CONFIG["hidden_size"]), 0.1)
for index in range(TINY_CONFIG["num_hidden_layers"]):
    with torch.no_grad():
        gqa_ours.model.layers[index].self_attn.qkv_proj.weight[
            NUM_HEADS * HEAD_DIM + HEAD_DIM:NUM_HEADS * HEAD_DIM + 2 * HEAD_DIM] += delta
        gqa_hf.model.layers[index].self_attn.k_proj.weight[HEAD_DIM:] += delta
diff = (ours_logits(gqa_ours) - hf_logits(gqa_hf)).abs().max().item()
check("2. 消融：扰动第二个 KV head 后仍与 HF 一致（GQA 的头对应与 HF 相同，不是全用 0 号）",
      diff < 1e-4, f"最大差 {diff:.2e}")

# ------------------------------------------------ 3. 残差与 tied embedding

with torch.no_grad():
    # HF 的 hidden_states[-1] 就是"过完最终 norm、喂给 lm_head 的那份"（第一个维度是 batch）
    hf_hidden = hf(torch.tensor([PROMPT]), output_hidden_states=True).hidden_states[-1][0]
ours_chunks = run_chunks(ours, PROMPT, [PROMPT])
ours_hidden = torch.stack([ours_chunks[position][1] for position in range(len(PROMPT))])
hidden_diff = (ours_hidden - hf_hidden).abs().max().item()
check("3. 残差：最终 hidden 逐位置与 HF 的 hidden_states[-1] 一致（残差链错一步就对不上）",
      hidden_diff < 1e-4, f"最大差 {hidden_diff:.2e}")
check("3. 这份 hidden 就是 lm_head 的输入（用 HF 自己的 lm_head 复算能得到它的 logits）",
      (hf.lm_head(ours_hidden) - hf_logits(hf)).abs().max().item() < 1e-4)

# tiny_gqa 的检查点 tie=False（lm_head 与词嵌入是两份权重）。两边都改成 tie=True：
# 我们跳过检查点里的 lm_head.weight 用词嵌入，HF 也 tie 掉 → 只剩一份来源，仍要相等。
# tiny_gqa 的检查点 tie=False，而且 lm_head.weight 与 embed_tokens.weight **数值不同**，
# 所以 tie 与否是有区别的（这一点先确认，否则下面的对照是空转）：
#   - tie=False：用检查点里那份 lm_head → 与 HF 一致；
#   - tie=True ：跳过检查点里的 lm_head，改用词嵌入 → 与"HF 手工 tie 之后"一致。
untied_ours = build_tiny(tie=False)
untied_diff = (ours_logits(untied_ours) - hf_logits(hf)).abs().max().item()
tied_ours = build_tiny(tie=True)
tied_hf = hf_model()
with torch.no_grad():
    tied_hf.lm_head.weight.copy_(tied_hf.model.embed_tokens.weight)     # 手工 tie
tied_diff = (ours_logits(tied_ours) - hf_logits(tied_hf)).abs().max().item()
untied_vs_tied = (ours_logits(untied_ours) - ours_logits(tied_ours)).abs().max().item()
check("3. tie=False 时用检查点里那份 lm_head，与 HF 一致",
      untied_diff < 1e-4 and untied_ours.lm_head.weight is not untied_ours.model.embed_tokens.weight,
      f"最大差 {untied_diff:.2e}")
check("3. tie=True 时共享词嵌入（跳过检查点里的 lm_head），与手工 tie 后的 HF 一致",
      tied_ours.lm_head.weight is tied_ours.model.embed_tokens.weight and tied_diff < 1e-4,
      f"最大差 {tied_diff:.2e}")
check("3. 两种 tie 的结果确实不同（说明上面两组对照不是在测同一件事）",
      untied_vs_tied > 1e-3, f"两者相差 {untied_vs_tied:.2e}")

# ------------------------------------------------ 4. full / chunked / decode（走 Runner 的真实路径）

full = run_chunks(ours, PROMPT, [PROMPT])
chunked = run_chunks(ours, PROMPT, [PROMPT[:3], PROMPT[3:6], PROMPT[6:]])
decoded = run_chunks(ours, PROMPT, [[token] for token in PROMPT])
hf_reference = hf_logits(hf)


def max_diff(left, right):
    return max((left[position][0] - right[position][0]).abs().max().item()
               for position in range(len(PROMPT)))


full_vs_chunked = max_diff(full, chunked)
full_vs_decoded = max_diff(full, decoded)
full_vs_hf = max((full[position][0] - hf_reference[position]).abs().max().item()
                 for position in range(len(PROMPT)))
check("4. full 一次算完 vs 分 3 块 chunked prefill：每个位置的 logits 一致",
      full_vs_chunked < 1e-5, f"最大差 {full_vs_chunked:.2e}")
check("4. full vs 逐 token decode：每个位置的 logits 一致",
      full_vs_decoded < 1e-5, f"最大差 {full_vs_decoded:.2e}")
check("4. 三种切分与 HF 参考也一致（不只是自洽）", full_vs_hf < 1e-4,
      f"最大差 {full_vs_hf:.2e}")
# 切分方式不同，逐位置的 hidden 也应该逐个位置对得上（不只最后一个位置）
hidden_by_position = max((chunked[position][1] - full[position][1]).abs().max().item()
                         for position in range(len(PROMPT)))
check("4. chunked 与 full 的 hidden 也逐位置一致（不是只有末尾对得上）",
      hidden_by_position < 1e-5, f"最大差 {hidden_by_position:.2e}")

# ------------------------------------------------ 5. 模型 forward 的边界

signature = inspect.signature(Qwen3ForCausalLM.forward)
check("5. 模型 forward 的签名里只有 input_ids / positions（拿不到 Request / KV 池 / 采样参数）",
      list(signature.parameters) == ["self", "input_ids", "positions"],
      str(list(signature.parameters)))

seen = []
original_forward = model.forward
model.forward = lambda input_ids, positions: (
    seen.append((type(input_ids).__name__, type(positions).__name__)),
    original_forward(input_ids, positions))[1]
run_chunks(model, PROMPT, [PROMPT])
model.forward = original_forward
check("5. 实际调用时也只收到两个张量（没有活对象能溜进来）",
      seen == [("Tensor", "Tensor")], str(seen))
check("5. `Request` 不是模型的入参类型（处理请求的是 `_prepare_inputs` 那一层）",
      not any(parameter.annotation is Request for parameter in signature.parameters.values()))

print()
print(f"{'全部通过' if not FAIL else '失败: ' + ', '.join(FAIL)}")
sys.exit(1 if FAIL else 0)
