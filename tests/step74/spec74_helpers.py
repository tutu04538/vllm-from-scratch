"""step74 测试夹具：V2 提议者的两张图 + "融合 vs 逐步"两条路。

与 73 关同一套开关（环境变量 `VLLM_USE_V2_MODEL_RUNNER=1` 必须在建 `VllmConfig` 之前设好；
UVA 需要 `VLLM_WSL2_ENABLE_PIN_MEMORY=1`），外加三样本关专属的东西：

    `force_draft_metadata_capability()`   把"草稿后端支持不支持原地更新元数据"换一档
                                          （生产路径只有支持的那一档，替身在 `minivllm/testing/`）
    `RoundRecorder`                       每轮快照：草稿 token / positions / seq_lens /
                                          slot_mapping / 草稿 KV / 这一轮草稿图被重放了几次
    `fused_or_fallback()`                 同一个 workload 跑两遍（支持/不支持各一遍）

**为什么快照要在 `propose()` 外面包一层**：验收要的不是"最终输出一样"，而是"两条路在**每一步**
的中间状态都一样"（草稿 token、positions、slot mapping、KV）。只比最终 token 会漏掉"某一步写错
了槽位但后来被覆盖"这类错。
"""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("VLLM_WSL2_ENABLE_PIN_MEMORY", "1")

import pytest  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minivllm import (CacheConfig, DeviceConfig, LLMEngine, ModelConfig,  # noqa: E402
                      SamplingParams, SchedulerConfig, SpeculativeConfig,
                      UniProcExecutor, VllmConfig, Worker)
from minivllm.config import CompilationConfig  # noqa: E402
from minivllm.testing.fake_attn_backend import draft_attn_groups  # noqa: E402
from minivllm.testing.tiny_models import (tiny_eagle3_dir, tiny_qwen3_config,  # noqa: E402
                                          tiny_qwen3_dir)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="V2 Model Runner 需要 CUDA（Triton 内核 + UVA 常驻状态 + CUDA Graph）")

TINY = tiny_qwen3_dir("tiny_gqa")
HF = tiny_qwen3_config("tiny_gqa")
DRAFT = tiny_eagle3_dir("tiny_gqa")
DRAFT_HF = json.loads((Path(DRAFT) / "config.json").read_text())
VOCAB = int(HF["vocab_size"])
EOS = VOCAB + 7
#: tiny 词表的默认档位表（显式给，才能让 1/2/3 条请求的统一 decode 批都命中图：
#: 1+K=4 的 prefill 图要 4/8 行，decode 图要 1/2/3 行）。
DEFAULT_SIZES = [1, 2, 3, 4, 8]


def make_config(*, spec_k: int = 3, mode: str = "full_decode_only",
                capture_sizes=DEFAULT_SIZES, budget: int = 64, blocks: int = 64,
                max_num_seqs: int = 3, max_model_len: int = 64,
                draft_sample: str = "greedy", attention_backend: str = "torch"):
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    spec = None
    if spec_k is not None:
        spec = SpeculativeConfig(
            method="eagle3", num_speculative_tokens=spec_k,
            draft_sample_method=draft_sample,
            draft_model_config=ModelConfig(model=DRAFT, dtype="float32",
                                           max_model_len=max_model_len, hf_config=DRAFT_HF))
    return VllmConfig(
        model_config=ModelConfig(model=TINY, dtype="float32", max_model_len=max_model_len,
                                 hf_config=HF),
        cache_config=CacheConfig(block_size=4, num_gpu_blocks=blocks),
        scheduler_config=SchedulerConfig(max_num_seqs=max_num_seqs,
                                         max_num_batched_tokens=budget),
        device_config=DeviceConfig(device=DEVICE), speculative_config=spec,
        compilation_config=CompilationConfig(
            cudagraph_mode=mode,
            cudagraph_capture_sizes=None if capture_sizes is None else list(capture_sizes)),
        attention_backend=attention_backend,
    )


def make_engine(**kwargs):
    config = make_config(**kwargs)
    engine = LLMEngine(config, UniProcExecutor(config, Worker(config)), tokenizer=None)
    core = engine.engine_core.engine_core
    return engine, core, core.model_executor.driver_worker.model_runner


def force_draft_metadata_capability(monkeypatch, *, supports_update: bool) -> None:
    """把草稿注意力组的 capability 换成 `supports_update`（测试接缝，见 `testing/`）。

    替换的是 `_init_attn_groups()`（上游"换一个注意力后端"的等价物），生产那条门
    （`_configure_fused_multi_step_decode()`）与两条执行路径都**原样执行**。
    """
    from minivllm.worker.gpu.spec_decode.autoregressive.speculator import (
        AutoRegressiveSpeculator,
    )

    def _init_attn_groups(self):
        return draft_attn_groups(
            self.draft_attn_layer_names,
            self.vllm_config.cache_config.block_size,
            supports_update=supports_update,
        )

    monkeypatch.setattr(AutoRegressiveSpeculator, "_init_attn_groups", _init_attn_groups)


class RoundRecorder:
    """每轮快照 V2 提议者的输入/输出（观测，不改变行为）。

    抓的都是"两条路必须逐位相同"的量：

        draft_tokens      本轮交出去的 K 枚草稿（`[num_reqs, K]`）
        positions         草稿步结束时的 positions（步进是否正确）
        seq_lens          草稿侧的上下文长度（`target_seq_len - num_rejected + step`）
        slot_mappings     草稿最后一步写 KV 用的槽位（位置 + 块表 → 物理槽位）
        kv_checksum       草稿 KV 缓存的逐层校验和（真的写进去了、且写在同一个地方）
        decode_replays    这一轮草稿 decode 图被重放了几次（融合 = 1，逐步 = K−1）
    """

    def __init__(self, runner) -> None:
        self.runner = runner
        speculator = runner.speculator
        self.rounds: list[dict] = []
        self._original = speculator.propose

        def patched(*args, **kwargs):
            before = {
                "decode": speculator.decode_cudagraph_manager.num_replays,
                "prefill": speculator.prefill_cudagraph_manager.num_replays,
            }
            out = self._original(*args, **kwargs)
            input_batch = args[0]
            num_reqs = input_batch.num_reqs
            self.rounds.append({
                "num_reqs": num_reqs,
                "draft_tokens": out.clone().cpu(),
                "positions": speculator.input_buffers.positions[:num_reqs].clone().cpu(),
                "seq_lens": speculator.input_buffers.seq_lens[:num_reqs].clone().cpu(),
                "slot_mappings": runner.block_tables.slot_mappings[:, :num_reqs]
                                    .clone().cpu(),
                "kv_checksum": draft_kv_checksum(speculator),
                "decode_replays": self._replays(speculator, "decode", before["decode"]),
                "prefill_replays": self._replays(speculator, "prefill", before["prefill"]),
                "current_draft_step": int(speculator.current_draft_step.item()),
            })
            return out

        speculator.propose = patched

    @staticmethod
    def _replays(speculator, which: str, since: int) -> int:
        """这一轮真的 `replay()` 了几次（融合 = 1 次走完 K−1 步；逐步 = K−1 次）。"""
        manager = getattr(speculator, f"{which}_cudagraph_manager")
        return manager.num_replays - since

    def modes(self, which: str = "decode") -> list[str]:
        manager = getattr(self.runner.speculator, f"{which}_cudagraph_manager")
        return [entry["mode"] for entry in manager.selections]


def draft_kv_checksum(speculator) -> list[float]:
    """草稿 KV 缓存的逐层校验和（**跳过 0 号块**，见下）。

    为什么跳过 0 号块：它是**垃圾桶**（`BlockPool(reserve_null_block=True)`，69 关），
    捕获期的假输入（槽位全是哨兵）在图内被 `clamp_min(0)` 写进去，写的是"假批的垃圾值"；
    两条路的假批内容不同（融合路一次跑 K−1 步、逐步路每步一次），所以垃圾桶里的垃圾天然不同
    ——那**不是**被测的行为。真实请求拿到的块永远 ≥ 1（0 号块留白），所以"跳过 0 号块"不会漏掉
    任何一个有效槽位。

    用"和"而不是完整张量：测试只关心"两条路写的是不是同一块地方、同样的值"，和不同就一定不同。
    """
    out = []
    for name, module in sorted(speculator.model.named_modules()):
        if getattr(module, "kv_cache", None) is None:
            continue
        out.append(round(float(module.kv_cache[:, 1:].sum().item()), 4))
    return out


def draft_kv_snapshot(speculator) -> dict[str, torch.Tensor]:
    return {
        name: module.kv_cache.clone().cpu()
        for name, module in sorted(speculator.model.named_modules())
        if getattr(module, "kv_cache", None) is not None
    }


class DraftSampleCounter:
    """数"草稿采样被 **Python** 调了几次"（判据见 `DraftForwardCounter`）。"""

    def __init__(self, speculator) -> None:
        self.count = 0
        original = speculator.sample_draft

        def patched(*args, **kwargs):
            self.count += 1
            return original(*args, **kwargs)

        speculator.sample_draft = patched


class DraftForwardCounter:
    """数"草稿模型 forward 被 **Python** 调了几次"。

    这是"多步执行真的在图里"的判据：图重放**不会**执行 Python，所以融合路径下"一轮里跑完 K−1
    个草稿步"时这个计数应当**一次都不涨**（K−1 步全在录好的图里）；逐步路径则涨 K−1 次
    （每步一次 Python 调用 + 一次重放）。只看 `graphs` 的数量是分不出来的：两条路都只捕获
    "每个档位一张图"。
    """

    def __init__(self, speculator) -> None:
        self.count = 0
        model = speculator.model
        original = model.forward

        def patched(*args, **kwargs):
            self.count += 1
            return original(*args, **kwargs)

        model.forward = patched


def run_steps(engine, runner, *, max_tokens: int = 10, after=None, limit: int = 400):
    """跑引擎到结束；每轮结束后回调 `after(engine, runner)`（观测用）。"""
    for _ in range(limit):
        if not engine.has_unfinished_requests():
            break
        for out in engine.step():
            pass
        if after is not None:
            after(engine, runner)
    return engine


def greedy(requests, *, max_tokens=10, spec_k=3, mode="full_decode_only",
           capture_sizes=DEFAULT_SIZES, budget=64, max_num_seqs=3, blocks=64,
           max_model_len=64, after=None):
    engine, _core, runner = make_engine(
        spec_k=spec_k, mode=mode, capture_sizes=capture_sizes, budget=budget,
        max_num_seqs=max_num_seqs, blocks=blocks, max_model_len=max_model_len)
    try:
        for req_id, prompt in requests:
            engine.add_request(req_id, list(prompt),
                               SamplingParams(max_tokens=max_tokens, temperature=0.0,
                                              eos_token_id=EOS))
        final = {}
        for _ in range(600):
            if not engine.has_unfinished_requests():
                break
            for out in engine.step():
                final[out.request_id] = list(out.token_ids)
            if after is not None:
                after(engine, runner)
        return final, runner
    finally:
        engine.shutdown()
