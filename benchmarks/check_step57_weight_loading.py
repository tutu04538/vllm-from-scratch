"""57B 验收（对应需求里的 `test_weight_loading.py`）：权重读取、打包路由与严格检查。

三层分开查，出问题时能直接定位到哪一层：

  1. **文件层**（`weight_utils`）：单文件 / index 分片两种布局；index 点名的 shard 缺失、
     张量放错 shard、名字缺失、重名、前缀不连续——都要**明确报错**（不能静默少加载）；
  2. **路由层**（`AutoWeightsLoader` + `WeightsMapper`）：q/k/v → qkv_proj 的分段区间逐值正确、
     gate/up → gate_up_proj 正确；未知名字、指到单个参数里面的名字、给非打包参数塞 shard_id
     都要报错；**每个参数都必须被加载**（覆盖检查，替代 strict=False）；
  3. **模型层**：tied embedding（共享 Parameter + 跳过检查点里的 lm_head）、真实 Qwen3-1.7B
     分片检查点的完整加载。

数值方向的对照（full/chunk/decode 与 HF 参考）在 `check_step57_model_logits.py`。
"""

import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, "/home/user/proj/vllm-from-scratch")

import torch
from safetensors.torch import save_file

from minivllm.config import ModelConfig
from minivllm.model_loader import (AutoWeightsLoader, DefaultModelLoader, WeightsMapper,
                                 get_model, get_model_loader, iter_weights)
from minivllm.model_loader.weight_utils import _check_grouping
from minivllm.models import Qwen3ForCausalLM

FAIL = []
TINY_DIR = "fixtures/step30_qwen3/tiny_gqa"
TINY_CONFIG = json.load(open(f"{TINY_DIR}/config.json"))
REAL_DIR = "models/Qwen3-1.7B"


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def expect_error(name, fn, *types, contains=""):
    try:
        fn()
    except types as exc:                       # noqa: B902
        detail = str(exc).splitlines()[0]
        ok = contains in str(exc)
        check(name, ok, detail if ok else f"报错内容里没有 {contains!r}：{detail}")
        return
    except Exception as exc:                   # noqa: BLE001  类型不对也要看得见
        check(name, False, f"抛了 {type(exc).__name__}: {exc}")
        return
    check(name, False, "没有报错")


def tiny_model(tie=False, **overrides):
    config = dict(TINY_CONFIG, **overrides)
    if tie:
        config["tie_word_embeddings"] = True
    return Qwen3ForCausalLM(config)


# ------------------------------------------------ 1. 文件层

names = [name for name, _ in iter_weights(TINY_DIR)]
check("1. 单文件布局：读出全部权重（tiny_gqa 是 25 个）",
      len(names) == 25 and names[0] == "lm_head.weight", f"{len(names)} 个")

workdir = tempfile.mkdtemp(prefix="step57_weights_")
try:
    # 造一个 3 个权重、两个 shard 的检查点，用来试各种坏情况
    tensors = {f"model.layers.0.w{i}": torch.arange(4.0) + i for i in range(3)}
    save_file({"model.layers.0.w0": tensors["model.layers.0.w0"],
               "model.layers.0.w1": tensors["model.layers.0.w1"]},
              os.path.join(workdir, "model-00001-of-00002.safetensors"))
    # 第二个 shard 里**故意也放一份 w0**：真实检查点里"同一个名字落在两个 shard"
    # 就是这么坏掉的（index 是 dict，表达不了重复，只能从文件内容里发现）
    save_file({"model.layers.0.w0": tensors["model.layers.0.w0"],
               "model.layers.0.w2": tensors["model.layers.0.w2"]},
              os.path.join(workdir, "model-00002-of-00002.safetensors"))

    def write_index(weight_map):
        with open(os.path.join(workdir, "model.safetensors.index.json"), "w") as handle:
            json.dump({"metadata": {}, "weight_map": weight_map}, handle)

    good_map = {"model.layers.0.w0": "model-00001-of-00002.safetensors",
                "model.layers.0.w1": "model-00001-of-00002.safetensors"}
    write_index(good_map)
    check("1. index 布局：正常检查点能读出来",
          [name for name, _ in iter_weights(workdir)] == ["model.layers.0.w0", "model.layers.0.w1"])

    write_index(dict(good_map, **{"model.layers.0.w2": "model-00003-of-00003.safetensors"}))
    expect_error("1. index 点名的 shard 不存在 → FileNotFoundError", lambda: list(iter_weights(workdir)),
                 FileNotFoundError, contains="目录里没有")

    write_index({"model.layers.0.w0": "model-00002-of-00002.safetensors",
                 "model.layers.0.w1": "model-00001-of-00002.safetensors"})
    expect_error("1. 张量在别的 shard 里（检查点与索引对不上）→ 报错",
                 lambda: list(iter_weights(workdir)), ValueError, contains="索引说它在")

    write_index(dict(good_map, **{"model.layers.0.w9": "model-00001-of-00002.safetensors"}))
    expect_error("1. index 里的名字在 shard 里读不到 → 报错（漏加载不能静默）",
                 lambda: list(iter_weights(workdir)), ValueError, contains="一个都没读到")

    write_index({"model.layers.0.w0": "model-00001-of-00002.safetensors",
                 "model.layers.0.w1": "model-00001-of-00002.safetensors",
                 "model.layers.0.w2": "model-00002-of-00002.safetensors"})
    expect_error("1. 同一个名字出现在两个 shard → 报错（重复 shard）",
                 lambda: list(iter_weights(workdir)), ValueError, contains="出现了两次")

    # 前缀不连续：分组器只认相邻的同一组，交错会让同一层被加载两遍
    expect_error("1. 权重名没有按模块分组（交错）→ 报错",
                 lambda: _check_grouping(["a.b.w", "c.d.w", "a.e.w"]), ValueError, contains="分组")
    _check_grouping(["a.b.w", "a.b.bias", "a.c.w", "b.w"])
    check("1. 正常顺序（同组相邻）不误报", True)

    empty = os.path.join(workdir, "empty")
    os.makedirs(empty)
    open(os.path.join(empty, "pytorch_model.bin"), "w").close()
    expect_error("1. 只有 .bin（本关不支持的格式）→ NotImplementedError",
                 lambda: get_model_loader(ModelConfig(model=empty)), NotImplementedError,
                 contains="未实现")
    expect_error("1. 目录不存在 → FileNotFoundError",
                 lambda: get_model_loader(ModelConfig(model="/nonexistent/model")),
                 FileNotFoundError, contains="不存在")
finally:
    shutil.rmtree(workdir, ignore_errors=True)

# ------------------------------------------------ 2. 路由层

model = tiny_model()
loaded = model.load_weights(iter_weights(TINY_DIR))
params = {name for name, _ in model.named_parameters()}
# 19 个参数：共享的 lm_head/embed 各自一个（tiny 模型 tie=False），每层 8 个 ×2 层，加最终 norm
check("2. 覆盖检查：每个参数都被加载（等价于 strict=True，但错误信息更具体）",
      loaded == params and len(params) == 19, f"加载了 {len(loaded)} / 参数 {len(params)}")

checkpoint = dict(iter_weights(TINY_DIR))
layer0 = "model.layers.0.self_attn"
q_proj = checkpoint[f"{layer0}.q_proj.weight"]
k_proj = checkpoint[f"{layer0}.k_proj.weight"]
v_proj = checkpoint[f"{layer0}.v_proj.weight"]
qkv = model.model.layers[0].self_attn.qkv_proj.weight
q_size, kv_size = 4 * 16, 2 * 16          # num_heads * head_dim / num_kv_heads * head_dim
check("2. packed qkv：q 在 [0:64]、k 在 [64:96]、v 在 [96:128]（GQA 不能三等分）",
      torch.equal(qkv[:q_size], q_proj)
      and torch.equal(qkv[q_size:q_size + kv_size], k_proj)
      and torch.equal(qkv[q_size + kv_size:], v_proj),
      f"q_size={q_size}、kv_size={kv_size}、目标形状={tuple(qkv.shape)}")
gate = checkpoint["model.layers.0.mlp.gate_proj.weight"]
up = checkpoint["model.layers.0.mlp.up_proj.weight"]
gate_up = model.model.layers[0].mlp.gate_up_proj.weight
check("2. packed gate_up：gate 在 [0:48]、up 在 [48:96]",
      torch.equal(gate_up[:48], gate) and torch.equal(gate_up[48:], up),
      f"目标形状={tuple(gate_up.shape)}")

# 覆盖检查真的会抓漏加载：把 q_norm.weight 从流里去掉
def without(name_to_drop):
    return ((name, weight) for name, weight in iter_weights(TINY_DIR) if name != name_to_drop)


expect_error("2. 少加载一个参数（q_norm.weight）→ 报错，不留下随机初始化的参数",
             lambda: tiny_model().load_weights(without("model.layers.1.self_attn.q_norm.weight")),
             ValueError, contains="q_norm.weight")

unknown = [("model.layers.0.self_attn.q_proj.bias", torch.zeros(4)),
           ("model.layers.0.self_attn.attn.cache", torch.zeros(4))]
expect_error("2. 检查点里有模型没有的名字 → 报错，并列出可用的参数名",
             lambda: tiny_model().load_weights(iter(unknown)), ValueError, contains="没有名为")

nested = [("model.norm.weight.extra", torch.zeros(4))]
expect_error("2. 名字指到单个参数**里面** → 报错（参数没有子结构）",
             lambda: tiny_model().load_weights(iter(nested)), ValueError, contains="单个参数")

wrong_mapper = WeightsMapper(orig_to_new_stacked={".k_proj": (".o_proj", "k")})
expect_error("2. 打包映射指到非打包参数（shard_id 落到 RowParallelLinear）→ 报错",
             lambda: AutoWeightsLoader(tiny_model()).load_weights(
                 iter_weights(TINY_DIR), mapper=wrong_mapper),
             ValueError, contains="不是打包参数")

# skip_substrs 的真实用途：老检查点里带 rotary_emb.inv_freq 这类"模型里根本没有的参数"
# （本关的 RoPE 表是算出来的、不是参数）。不跳过就会撞上"没有这个名字"的报错。
with_legacy = [("model.layers.0.self_attn.rotary_emb.inv_freq", torch.zeros(8))] + \
    list(iter_weights(TINY_DIR))
expect_error("2. 老检查点的 rotary_emb.inv_freq（模型里没有）→ 默认报错",
             lambda: tiny_model().load_weights(iter(with_legacy)), ValueError,
             contains="没有名为")
skipped = AutoWeightsLoader(tiny_model(), skip_substrs=["rotary_emb."]).load_weights(
    iter(with_legacy))
check("2. skip_substrs 声明之后正常加载，覆盖检查也不受影响（豁免名单生效）",
      any(name.endswith("rotary_emb.inv_freq") is False for name in skipped)
      and len(skipped) == 19, f"加载了 {len(skipped)} 个")

# 整层跳过**不行**：被跳过的那一层参数没人填，覆盖检查会拦下来。
# （vLLM 没有覆盖检查，所以它允许；本关认为"漏一层"必须显式失败，见差异账本）
expect_error("2. 跳掉一整层解码器 → 覆盖检查拦下来（不做静默的半加载）",
             lambda: AutoWeightsLoader(tiny_model(), skip_prefixes=["model.layers.1."])
             .load_weights(iter_weights(TINY_DIR)),
             ValueError, contains="没有被加载")

# ------------------------------------------------ 3. 模型层：tied embedding 与真实检查点

tied = tiny_model(tie=True)
tied_lm_head = tied.lm_head
check("3. tie_word_embeddings=True：lm_head 与 embed_tokens 共享**同一个** Parameter",
      tied.lm_head.weight is tied.model.embed_tokens.weight
      and tied.lm_head is tied_lm_head,
      f"同一对象={tied.lm_head.weight is tied.model.embed_tokens.weight}")

# 检查点里同时有 lm_head.weight 与 embed_tokens.weight（tiny 模型的检查点就是），
# tie=True 时必须跳过 lm_head，不跳过就是同一份权重被覆盖两遍
tied_loaded = tied.load_weights(iter_weights(TINY_DIR))
check("3. tie=True 时跳过检查点里的 lm_head.weight（装载集合里没有它）",
      "lm_head.weight" not in tied_loaded and "model.embed_tokens.weight" in tied_loaded)

# 用一个"lm_head 与 embed_tokens 故意不同"的检查点验证跳过真的生效：
# 跳过之后 lm_head 的数值必须还是 embed_tokens 的，而不是检查点里那份
different = dict(checkpoint)
different["lm_head.weight"] = checkpoint["model.embed_tokens.weight"] + 1.0
tied2 = tiny_model(tie=True)
tied2.load_weights(iter(different.items()))
check("3. 跳过之后 lm_head 用的是词嵌入那份权重（不是检查点里另一份）",
      torch.equal(tied2.lm_head.weight, checkpoint["model.embed_tokens.weight"])
      and tied2.compute_logits(torch.zeros(1, TINY_CONFIG["hidden_size"]))
      is not None)

if os.path.isdir(REAL_DIR):
    real_config = json.load(open(f"{REAL_DIR}/config.json"))
    real = get_model(ModelConfig(model=REAL_DIR, dtype="bfloat16", hf_config=real_config), "cpu")
    real_params = {name for name, _ in real.named_parameters()}
    check("3. 真实 Qwen3-1.7B（2 个 shard + index，bf16）整模型加载：参数一个不少",
          real.lm_head.weight is real.model.embed_tokens.weight
          and len(real_params) == 226 and real.config["tie_word_embeddings"] is True,
          f"{len(real_params)} 个参数（检查点里有 311 个名字：分离的 q/k/v 与 gate/up "
          f"合并成打包参数，lm_head 与词嵌入共享）、dtype={real.model.embed_tokens.weight.dtype}")
    qkv_real = real.model.layers[0].self_attn.qkv_proj.weight
    with_shard = dict((name, tensor) for name, tensor in iter_weights(REAL_DIR))
    check("3. 真实检查点的打包区间逐值一致（q/k/v 与 gate/up）",
          torch.equal(qkv_real[:2048], with_shard["model.layers.0.self_attn.q_proj.weight"])
          and torch.equal(qkv_real[3072:], with_shard["model.layers.0.self_attn.v_proj.weight"])
          and torch.equal(real.model.layers[0].mlp.gate_up_proj.weight[6144:],
                          with_shard["model.layers.0.mlp.up_proj.weight"]),
          f"qkv 形状={tuple(qkv_real.shape)}")
    del real, with_shard
else:
    check("3. 真实 Qwen3-1.7B 不在本机 → 跳过（不下载新模型）", True, "models/Qwen3-1.7B 不存在")

print()
print(f"{'全部通过' if not FAIL else '失败: ' + ', '.join(FAIL)}")
sys.exit(1 if FAIL else 0)
