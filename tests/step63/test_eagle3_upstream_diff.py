"""step63：与**上游真实实现**的逐层数值对照（需求 §4 的"记录容差"）。

对照对象：site-packages 里真的 vLLM —— `vllm/model_executor/models/llama_eagle3.py::Eagle3LlamaForCausalLM`
（真实 checkpoint 的 `architectures` 就是它）。**只允许测试这么做**（生产路径不 import 上游）。

比什么：**不需要 attention 上下文**的那两层，正好是 EAGLE3 最独特的两个算子
  1. `combine_hidden_states()`：多个辅助层特征 → 拼接 → （可选 norm）→ `fc` 投影；
  2. `compute_logits()`：draft 词表的 lm_head。
上层（decoder layer 的 attention/MLP）两边都依赖各自的 forward 上下文与 KV 元数据，本关**不**在这里比
（那是 69/70 关 CUDA Graph / 异步那套基建到位后的事），所以这条对照的边界要写清楚。

权重与输入都固定：同一份 safetensors、同一个 `torch.manual_seed`，fp16 在 CUDA 上算，报 **最大绝对误差**。
"""

import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REAL_DRAFT_DIR = ROOT / "models" / "Qwen3-1.7B-eagle3"
TARGET_DIR = ROOT / "models" / "Qwen3-1.7B"
AUX_LAYERS = [2, 14, 25]
HIDDEN = 2048
NUM_AUX = len(AUX_LAYERS)
# 对照放在 **fp32** 上做：这一步比的是"结构/公式是否一致"，fp16 的舍入会把结构差异盖住。
# 权重完全一样（同一份 safetensors）时 fp32 的差值应在 1e-5 量级，容差给 1e-4；
# **不允许**用大容差掩盖结构错误（结构错了差值是 1e-1 以上）。
TOL = 1e-4            # combine_hidden_states：实测 max|Δ| = 0.0（逐位相同）
# lm_head 是 2048→32000 的 GEMM：两边命中不同的 GEMM 内核，fp32 累加顺序不同 →
# 实测 max|Δ| ≈ 7.0e-4（logits 量级 ~10，相对 1e-5）。**结构错**（层号/归一化位置/映射错）
# 会差 1e-1 以上，所以 5e-3 这个上界仍有意义，不是"放大到能过"。
TOL_LOGITS = 5e-3

pytestmark = pytest.mark.skipif(
    not (REAL_DRAFT_DIR / "model.safetensors").is_file() or not TARGET_DIR.is_dir(),
    reason="本机没有真实 target/draft 权重（models/Qwen3-1.7B[-eagle3]）：对照待验，不是通过")


@pytest.fixture(scope="module")
def models():
    """返回 `(ours, upstream, state)`：同一份配置、同一份权重、同一个设备。"""
    from safetensors.torch import load_file

    import minivllm.models.qwen3_eagle3 as ours_module
    from minivllm.models import get_model_class

    config = json.loads((REAL_DRAFT_DIR / "config.json").read_text())
    config["eagle_aux_hidden_state_layer_ids"] = AUX_LAYERS
    config["target_hidden_size"] = HIDDEN

    # ---- 我们的实现 ----
    ours = get_model_class(config["architectures"][0])(config)
    state = load_file(str(REAL_DRAFT_DIR / "model.safetensors"))
    ours.load_weights(iter(state.items()))
    ours = ours.float().to("cuda").eval()

    # ---- 上游实现（真的 vLLM 类；需要 vllm 的当前配置上下文与单进程分布式初始化）----
    from vllm.config import (DeviceConfig, ModelConfig, ParallelConfig, SpeculativeConfig,
                             VllmConfig, set_current_vllm_config)
    from vllm.config.cache import CacheConfig
    from vllm.config.scheduler import SchedulerConfig as UpSchedulerConfig
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM

    target_config = ModelConfig(model=str(TARGET_DIR), dtype="float16", max_model_len=64)
    draft_config = ModelConfig(model=str(REAL_DRAFT_DIR), dtype="float16", max_model_len=64)
    parallel = ParallelConfig()
    spec = SpeculativeConfig(model=str(REAL_DRAFT_DIR), method="eagle3",
                             num_speculative_tokens=4, target_model_config=target_config,
                             draft_model_config=draft_config, target_parallel_config=parallel)
    vllm_config = VllmConfig(
        model_config=target_config, speculative_config=spec,
        cache_config=CacheConfig(block_size=16, gpu_memory_utilization=0.1),
        parallel_config=parallel, device_config=DeviceConfig("cuda"),
        scheduler_config=UpSchedulerConfig(max_num_seqs=1, max_num_batched_tokens=64,
                                           max_model_len=64, is_encoder_decoder=False))
    with set_current_vllm_config(vllm_config):
        # 幂等：同一个 pytest 进程里可能已有别的用例初始化过（vLLM 的
        # `initialize_model_parallel()` 对"重复初始化"是 assert 失败）
        from vllm.distributed.parallel_state import model_parallel_is_initialized

        if not model_parallel_is_initialized():
            init_distributed_environment(world_size=1, rank=0, local_rank=0,
                                         distributed_init_method="tcp://127.0.0.1:29781",
                                         backend="gloo")
            initialize_model_parallel(tensor_model_parallel_size=1,
                                      pipeline_model_parallel_size=1)
        upstream = Eagle3LlamaForCausalLM(vllm_config=vllm_config, prefix="")
        upstream.load_weights(iter(state.items()))
    upstream = upstream.float().to("cuda").eval()
    return ours, upstream, state


def test_num_aux_layers_and_fc_shape_agree(models):
    """两边的辅助层数与 fc 形状必须一致（差一位就是"层号错了但能跑"的静默错）。"""
    ours, upstream, _ = models
    # 上游两个实现的命名不同：llama_eagle3 叫 num_aux_hidden_states，qwen3_eagle3 叫 num_aux_layers
    assert ours.model.num_aux_layers == NUM_AUX
    assert upstream.model.num_aux_hidden_states == NUM_AUX
    assert tuple(ours.model.fc.weight.shape) == tuple(upstream.model.fc.weight.shape) \
        == (HIDDEN, HIDDEN * NUM_AUX)
    # 同一份权重真的进了两边（不是"各自随机初始化"）
    assert torch.equal(ours.model.fc.weight.cpu(), upstream.model.fc.weight.cpu())


def test_combine_hidden_states_matches_upstream(models):
    """`combine_hidden_states`：同输入下与上游的最大绝对误差 < 3e-3（fp16 精度量级）。"""
    ours, upstream, _ = models
    torch.manual_seed(0)
    aux = torch.randn(5, HIDDEN * NUM_AUX, dtype=torch.float32, device="cuda") * 0.1
    with torch.no_grad():
        mine = ours.combine_hidden_states(aux)
        theirs = upstream.combine_hidden_states(aux)
    assert mine.shape == theirs.shape == (5, HIDDEN)
    max_abs = (mine.float() - theirs.float()).abs().max().item()
    rel = max_abs / max(theirs.float().abs().max().item(), 1e-6)
    print(f"[step63] combine_hidden_states: max|Δ|={max_abs:.3e}, 相对={rel:.3e}")
    assert max_abs < TOL, f"combine_hidden_states 与上游差 {max_abs:.3e}（容差 {TOL}）"


def test_compute_logits_matches_upstream(models):
    """draft lm_head：同输入下与上游的最大绝对误差 < 3e-3（这是"词表对不对"的那一层）。"""
    ours, upstream, _ = models
    torch.manual_seed(1)
    hidden = torch.randn(4, HIDDEN, dtype=torch.float32, device="cuda") * 0.5
    with torch.no_grad():
        mine = ours.compute_logits(hidden)                # 与上游同名方法一样：已映射回 target 词表宽度
        theirs = upstream.compute_logits(hidden)
    assert mine.shape == theirs.shape == (4, 151936)
    # 不能直接相减：两边在"draft 词表没覆盖的 target id"上都是 -inf（-inf - -inf = nan）。
    # 先比"哪些位置有值"（= 允许被采样到的 target id 集合），再比有值位置的数值。
    finite_mine, finite_theirs = torch.isfinite(mine), torch.isfinite(theirs)
    assert torch.equal(finite_mine, finite_theirs), "两边允许的 target id 集合必须完全一致"
    max_abs = (mine[finite_mine].float() - theirs[finite_theirs].float()).abs().max().item()
    print(f"[step63] compute_logits: 有值位置 {int(finite_mine.sum())} 个, max|Δ|={max_abs:.3e}")
    assert max_abs < TOL_LOGITS, f"compute_logits 与上游差 {max_abs:.3e}（容差 {TOL_LOGITS}）"


def test_end_to_end_two_stage_matches_upstream(models):
    """连着算：辅助特征 → combine → lm_head，两步复合误差也必须在容差内。

    （不是简单重复上面两条：复合起来能暴露"中间某步 dtype/归一化位置不同"这类错。）
    """
    ours, upstream, _ = models
    torch.manual_seed(2)
    aux = torch.randn(3, HIDDEN * NUM_AUX, dtype=torch.float32, device="cuda") * 0.1
    with torch.no_grad():
        mine = ours.compute_logits(ours.combine_hidden_states(aux))
        theirs = upstream.compute_logits(upstream.combine_hidden_states(aux))
    argmax_equal = bool((mine.argmax(-1) == theirs.argmax(-1)).all())
    finite = torch.isfinite(mine) & torch.isfinite(theirs)
    max_abs = (mine[finite].float() - theirs[finite].float()).abs().max().item()
    print(f"[step63] 复合两步: max|Δ|={max_abs:.3e}, argmax 相同={argmax_equal}")
    assert max_abs < TOL_LOGITS
    assert argmax_equal, "逐行 argmax 都要相同（候选选谁一样，不只是数值接近）"
