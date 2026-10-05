"""step65：65 关修掉的一个**老 bug** 的回归——target 的 hidden states 必须真的上传到 device。

背景（这一关的错位反证用例抓到的）：`SpecDecodeBaseProposer._upload()` 上传了
`input_ids / positions / slot_mapping / mask / query_start_loc / seq_lens / block_table`，
**唯独漏了 `hidden_states`**。于是：

- **CPU**：staging 与 device 是**同一份张量**（`_buffer()` 里 `return cpu, cpu`），
  所以模型照样拿到正确特征 → 只在 CPU 上跑测试永远发现不了；
- **CUDA**：`_forward()` 传进模型的是 device 侧那份从没被写过的缓冲（全零）→
  **EAGLE3 / MTP 的草稿完全没吃到 target 的 hidden**。它不报错、草稿照样出，
  只是 EAGLE3 退化成"只用 token 的 draft"、MTP 退化成"token + 常数 hidden"。

这个 bug 在 63 关的验收里没被发现：当时的用例比的是 **CPU staging 缓冲**（`hidden_states_cpu`）
与特征对齐，而"模型实际收到的是什么"没人看过。所以这里补两条**在 CUDA 上比 device 侧**的用例，
EAGLE3 与 MTP 各一条。
"""

import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "step63"))
sys.path.insert(0, str(ROOT / "tests" / "step65"))

import test_eagle_e2e as eagle  # noqa: E402
import test_mtp_e2e as mtp  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
pytestmark = pytest.mark.skipif(
    DEVICE == "cpu",
    reason="这个 bug 只在 CPU/device 缓冲分离时出现（CPU 上两者是同一份张量）")


def test_eagle3_features_reach_the_model():
    """EAGLE3：draft 模型收到的特征 == staging 里的那份（不是全零的 device 缓冲）。"""
    engine, core, runner = eagle.make_engine(spec_k=1, max_num_seqs=1)
    proposer = runner.proposer
    seen = []
    original = proposer.model.forward

    def spy(input_ids, positions, hidden_states):
        # 在**调用当场**比：staging 里这一刻的内容就该是模型收到的内容
        # （跑完再看 CPU 缓冲比的是最后一轮，会拿到别的轮的 staging）
        staged = proposer.hidden_states_cpu[:hidden_states.shape[0]].to(hidden_states.device)
        seen.append((hidden_states.clone(), staged))
        return original(input_ids, positions, hidden_states)

    proposer.model.forward = spy
    try:
        eagle.run(engine, core, [("a", eagle.PROMPTS[0][1])], max_tokens=3)
    finally:
        engine.shutdown()
    assert seen, "必须真的跑过 draft 前向"
    assert all(torch.equal(received, staged) for received, staged in seen), \
        "draft 模型收到的特征与 staging 不一致：`_upload()` 漏了 hidden_states"
    assert all(int(torch.count_nonzero(received)) > 0 for received, _ in seen), \
        "模型收到的特征全是 0（device 缓冲从没被写过）"


def test_mtp_features_reach_the_model():
    """MTP：同上（MTP 吃的是 target 的最后一层 hidden，漏上传就退化成"常数特征"）。"""
    engine, core, runner = mtp.make_engine(k=1, max_num_seqs=1)
    proposer = runner.proposer
    seen = []
    original = proposer.model.forward

    def spy(input_ids, positions, hidden_states):
        staged = proposer.hidden_states_cpu[:hidden_states.shape[0]].to(hidden_states.device)
        seen.append((hidden_states.clone(), staged))
        return original(input_ids, positions, hidden_states)

    proposer.model.forward = spy
    try:
        mtp.run(engine, core, [("a", mtp.PROMPTS[0][1])], max_tokens=3)
    finally:
        engine.shutdown()
    assert seen
    assert all(torch.equal(received, staged) for received, staged in seen)
    assert all(int(torch.count_nonzero(received)) > 0 for received, _ in seen)


def test_staging_and_device_buffers_differ_on_cuda():
    """把"为什么只在 CUDA 上暴露"钉在**真实代码路径**上：CPU 上两者是同一份张量，CUDA 上不是。"""
    from minivllm import (CacheConfig, DeviceConfig, ModelConfig, SchedulerConfig,
                          SpeculativeConfig, VllmConfig)
    from minivllm.spec_decode.eagle import EagleProposer
    from minivllm.testing.tiny_models import (tiny_eagle3_dir, tiny_qwen3_config,
                                              tiny_qwen3_dir)

    target_dir = tiny_qwen3_dir("tiny_gqa")
    draft_dir = tiny_eagle3_dir("tiny_gqa", num_aux_layers=2, aux_layers=(0, 1))
    draft_hf = json.loads((Path(draft_dir) / "config.json").read_text())
    spec = SpeculativeConfig(method="eagle3", num_speculative_tokens=1,
                             draft_model_config=ModelConfig(
                                 model=draft_dir, dtype="float32", max_model_len=64,
                                 hf_config=draft_hf))
    config = VllmConfig(
        model_config=ModelConfig(model=target_dir, dtype="float32", max_model_len=64,
                                 hf_config=tiny_qwen3_config("tiny_gqa")),
        cache_config=CacheConfig(block_size=4, num_gpu_blocks=8),
        scheduler_config=SchedulerConfig(max_num_seqs=1, max_num_batched_tokens=32),
        device_config=DeviceConfig(device="cpu"), speculative_config=spec)

    on_cpu = EagleProposer(spec, config, "cpu")
    assert on_cpu.hidden_states_cpu is on_cpu.hidden_states, \
        "CPU 上 staging 与 device 必须是同一份（所以这个 bug 在 CPU 上测不出来）"
    on_cuda = EagleProposer(spec, config, "cuda")
    assert on_cuda.hidden_states_cpu is not on_cuda.hidden_states, \
        "CUDA 上必须是两份：少了上传，模型收到的是没写过的 device 缓冲"
