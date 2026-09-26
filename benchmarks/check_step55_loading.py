"""第 55 关：分片权重与双模型目录加载（需求 §6C）。

两块：

1. **分片 safetensors**：把一个 native 目录拆成两个分片 + 索引，走完整的加载路径
   读回来；再把「缺文件 / 缺参数 / 未声明的参数 / 放错分片 / 非法路径」五种坏索引
   逐个造出来，确认每一种都明确报错（不静默、不猜）。
2. **双目录加载**：两个**结构不同**的模型各存一个 native 目录，由
   `Engine.from_model_dir(target_dir, draft_model_dir=...)` 装起来真的跑一遍；
   顺带确认 draft 的结构来自它自己的目录，不是套用 target 的。
"""

import json
import pathlib
import shutil
import sys
import tempfile

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, "/home/user/proj/vllm-from-scratch")

import step55
from step55 import Engine
from step55.formats import read_raw_weights
from step55.formats.native import save_model

FAIL = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def raises(fn):
    try:
        fn()
    except (ValueError, FileNotFoundError) as exc:
        return str(exc)
    return None


TARGET_DIMS = dict(vocab_size=64, d_model=32, num_q_heads=4, num_kv_heads=2,
                   num_layers=3, intermediate_size=64, head_dim=8, max_seq_len=128,
                   eos_token_ids=[63])
DRAFT_DIMS = dict(vocab_size=64, d_model=16, num_q_heads=2, num_kv_heads=1,
                  num_layers=1, intermediate_size=32, head_dim=8, max_seq_len=128,
                  eos_token_ids=[63])

work = pathlib.Path(tempfile.mkdtemp(prefix="step55_loading_"))
target_dir, draft_dir = work / "target", work / "draft"

# ------------------------------------------------ 1. 双目录加载

torch.manual_seed(1)
target = step55.TinyCausalLM(device="cpu", attention_backend="torch", max_num_query_tokens=16,
                             **TARGET_DIMS)
torch.manual_seed(2)
draft = step55.TinyCausalLM(device="cpu", attention_backend="torch", max_num_query_tokens=16,
                            **DRAFT_DIMS)
save_model(target, target_dir)
save_model(draft, draft_dir)

engine = Engine.from_model_dir(
    target_dir, draft_model_dir=draft_dir, device="cpu", attention_backend="torch",
    max_num_seqs=2, max_num_batched_tokens=16, block_size=4, num_kv_blocks=32,
    draft_num_kv_blocks=16, enable_prefix_caching=False, speculative_mode="draft_model",
    num_speculative_tokens=2)
check("双目录加载：两个模型的结构各自来自自己的目录（draft 只有 1 层、hidden 16）",
      engine.model.num_layers == 3 and engine.model.d_model == 32
      and engine.draft_model.num_layers == 1 and engine.draft_model.d_model == 16,
      f"target {engine.model.num_layers} 层/d{engine.model.d_model}、"
      f"draft {engine.draft_model.num_layers} 层/d{engine.draft_model.d_model}")

# 权重真的装进去了：与保存前的 state_dict 逐张量相同
loaded_target = load_file(target_dir / "model.safetensors")
same_target = all(torch.equal(loaded_target[name], tensor)
                  for name, tensor in engine.model.state_dict().items())
check("双目录加载：target 权重逐张量等于保存前（strict 装入、没有随机残留）", same_target)

out = {}
engine.on_token = lambda ev: out.setdefault(ev["request_id"], []).append(ev["token_id"])
engine.add_request({"request_id": "A", "prompt_ids": [1, 2, 3, 1, 2, 3], "max_new_tokens": 6})
engine.add_request({"request_id": "B", "prompt_ids": [5, 6, 5, 6], "max_new_tokens": 6})
steps = 0
while engine.has_unfinished_requests():
    engine.step()
    steps += 1
    assert steps < 60, "疑似活锁"
check("双目录加载：装出来的双模型引擎真的跑得完，两条请求各提交 6 个 token",
      len(out.get("A", [])) == 6 and len(out.get("B", [])) == 6,
      str({k: len(v) for k, v in out.items()}))
check("双目录加载：结束时两套池子的活动引用都归零",
      all(u == 0 for u in engine.kv_cache_pool.block_usage)
      and all(u == 0 for u in engine.draft_kv_pool.block_usage))
check("双目录加载：draft 池是按 draft 自己的形状建的（1 层、1 个 KV 头、hidden 16）",
      engine.draft_kv_pool.num_layers == 1 and engine.draft_kv_pool.num_kv_heads == 1
      and engine.draft_kv_pool.head_dim == 8,
      f"层 {engine.draft_kv_pool.num_layers}、kv 头 {engine.draft_kv_pool.num_kv_heads}")

# ------------------------------------------------ 2. 分片权重

shard_dir = work / "sharded"
shard_dir.mkdir()
weights = load_file(target_dir / "model.safetensors")
names = sorted(weights)
first, second = names[:len(names) // 2], names[len(names) // 2:]
idx = shard_dir / "model.safetensors.index.json"
save_file({n: weights[n] for n in first}, shard_dir / "model-00001-of-00002.safetensors")
save_file({n: weights[n] for n in second}, shard_dir / "model-00002-of-00002.safetensors")
idx.write_text(json.dumps({"metadata": {"total_size": 1},
                           "weight_map": {**{n: "model-00001-of-00002.safetensors" for n in first},
                                          **{n: "model-00002-of-00002.safetensors" for n in second}}}))
shutil.copy(target_dir / "config.json", shard_dir / "config.json")

read = read_raw_weights(shard_dir, (torch.float32,))
check("分片加载：按 weight_map 合并后的权重与单文件目录逐张量相同",
      set(read) == set(weights) and all(torch.equal(read[n], weights[n]) for n in names),
      f"{len(read)} 个参数、{len(first)} + {len(second)} 两片")

# 模型本身也能从分片目录装起来
from step55.loading import load_model_from_dir

sharded_model = load_model_from_dir(shard_dir, torch.device("cpu"), "torch", 16, False,
                                    torch.float32)
check("分片加载：整条 from_model_dir 路径可用（配置 + 分片权重 -> 装好的模型）",
      all(torch.equal(sharded_model.state_dict()[n], weights[n]) for n in names))

# ---- 五种坏索引，每一种都要明确报错 ----

bad = work / "bad"
bad.mkdir()


def with_index(mapping, mutate=None, keep_files=True):
    """造一个只改了 weight_map 的目录，返回异常信息（None 表示没报错）。"""
    shutil.rmtree(bad, ignore_errors=True)
    bad.mkdir()
    for name in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"):
        if keep_files:
            shutil.copy(shard_dir / name, bad / name)
    index = {"metadata": {}, "weight_map": mapping}
    if mutate:
        index = mutate(index)
    (bad / "model.safetensors.index.json").write_text(json.dumps(index))
    shutil.copy(shard_dir / "config.json", bad / "config.json")
    return raises(lambda: read_raw_weights(bad, (torch.float32,)))


good_map = json.loads(idx.read_text())["weight_map"]
message = with_index(good_map, keep_files=False)
check("坏索引：分片文件缺失 -> 明确报错（不去猜别的文件名）",
      message is not None and "不存在" in message, str(message))

message = with_index({n: s for n, s in good_map.items() if n != names[0]})
check("坏索引：索引漏声明一个真实存在的参数 -> 明确报错（不静默多装一个）",
      message is not None and "没有出现在索引的 weight_map 里" in message, str(message))

mapping = dict(good_map)
mapping["model.not_a_real_weight"] = "model-00001-of-00002.safetensors"
message = with_index(mapping)
check("坏索引：声明了一个分片里没有的参数 -> 报「缺少声明的参数」",
      message is not None and "缺少索引为它声明的参数" in message, str(message))

mapping = dict(good_map)
mapping[names[0]] = "model-00002-of-00002.safetensors"      # 声明反了
message = with_index(mapping)
check("坏索引：参数被声明到另一个分片 -> 报「重复或放错分片」",
      message is not None and "却出现在分片" in message, str(message))

mapping = dict(good_map)
mapping[names[0]] = "../../etc/passwd.safetensors"
message = with_index(mapping)
check("坏索引：分片名是路径穿越 -> 明确拒绝（拼接前就挡住）",
      message is not None and "目录内的文件名" in message, str(message))

shutil.rmtree(work, ignore_errors=True)
print()
print(f"{'全部通过' if not FAIL else '失败: ' + ', '.join(FAIL)}")
sys.exit(1 if FAIL else 0)
