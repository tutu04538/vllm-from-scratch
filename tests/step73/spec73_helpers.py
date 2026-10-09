"""step73 测试夹具：V1/V2 两条执行路径的 tiny 引擎 + 接受/草稿观测。

V2 的开关是**环境变量**（与上游同名同义）：`VLLM_USE_V2_MODEL_RUNNER=1`。
必须在**建 `VllmConfig` 之前**设好——支持矩阵在配置期校验（`validate_v2_model_runner`），
`Worker.load_model()` 也按它分派 Runner 类。

另外：V2 的常驻状态走 **UVA**，而本机（WSL2）的判定与上游一致——pinned memory 默认关，
要 `VLLM_WSL2_ENABLE_PIN_MEMORY=1`。所以这个文件在 import `minivllm` 之前就把它设上
（`is_uva_available()` 带 `functools.cache`，晚设就没用了）。
"""

import json
import os
import sys
from collections import Counter
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
from minivllm.testing.tiny_models import (tiny_eagle3_dir, tiny_qwen3_config,  # noqa: E402
                                          tiny_qwen3_dir, tiny_structured_dir)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#: V2 的输入组装/采样/验证全是 Triton 内核 + UVA 常驻状态 → 没有 CUDA 就不跑（不是"通过"）。
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="V2 Model Runner 需要 CUDA（Triton 内核 + UVA 常驻状态）")

TINY = tiny_qwen3_dir("tiny_gqa")
HF = tiny_qwen3_config("tiny_gqa")
DRAFT = tiny_eagle3_dir("tiny_gqa")
DRAFT_HF = json.loads((Path(DRAFT) / "config.json").read_text())
# tiny 词表只有 11 个 id：prompt 里的 token 必须落在 [0, 11)，否则 embedding 直接 device assert。
VOCAB = int(HF["vocab_size"])


def draft_config(max_model_len: int = 64) -> ModelConfig:
    return ModelConfig(model=DRAFT, dtype="float32", max_model_len=max_model_len,
                       hf_config=DRAFT_HF)


def make_config(*, v2: bool, method: str = "eagle3", spec_k: int | None = 3,
                mode="none", budget: int = 64, blocks: int = 64, max_num_seqs: int = 3,
                max_model_len: int = 64, spec=None, structured: bool = False):
    """建配置。`spec_k=None` = 不开投机；`v2=True` 走 V2 分派（环境变量先设好）。

    `structured=True` 换成 68 关那份**带 JSON tokenizer 的 tiny 对**（`tiny_mqa` + 匹配的
    eagle3 draft）：`tiny_gqa` 的词表只有 11 个 id、没有 tokenizer，任何 JSON 语法都拼不出来
    （实测 xgrammar 会允许一个词表里不存在的 token → 调度器的严格校验当场报错）。
    """
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1" if v2 else "0"
    target_dir, target_hf, draft_dir_path = TINY, HF, DRAFT
    if structured:
        target_dir = tiny_structured_dir("tiny_mqa")
        target_hf = tiny_qwen3_config("tiny_mqa")
        draft_dir_path = tiny_eagle3_dir("tiny_mqa")
    if spec is None and spec_k is not None:
        if structured:
            draft_hf = json.loads((Path(draft_dir_path) / "config.json").read_text())
            spec = SpeculativeConfig(
                method=method, num_speculative_tokens=spec_k,
                draft_model_config=ModelConfig(model=draft_dir_path, dtype="float32",
                                               max_model_len=max_model_len, hf_config=draft_hf))
        else:
            spec = SpeculativeConfig(method=method, num_speculative_tokens=spec_k,
                                     draft_model_config=draft_config(max_model_len))
    return VllmConfig(
        model_config=ModelConfig(model=target_dir, dtype="float32",
                                 max_model_len=max_model_len, hf_config=target_hf,
                                 # 结构化输出的管理器从 ModelConfig.tokenizer_path 读 tokenizer
                                 tokenizer=target_dir if structured else None),
        cache_config=CacheConfig(block_size=4, num_gpu_blocks=blocks),
        scheduler_config=SchedulerConfig(max_num_seqs=max_num_seqs,
                                         max_num_batched_tokens=budget),
        device_config=DeviceConfig(device=DEVICE), speculative_config=spec,
        compilation_config=CompilationConfig(cudagraph_mode=mode),
    )


def make_engine(**kwargs):
    config = make_config(**kwargs)
    # 结构化输出要 tokenizer（grammar 编译要用它）；没有 tokenizer 的 tiny 模型目录返回 None。
    try:
        from minivllm.tokenizer_utils import cached_tokenizer_from_config

        tokenizer = cached_tokenizer_from_config(config.model_config)
    except Exception:  # noqa: BLE001 —— tiny_gqa 目录里没有 tokenizer 文件
        tokenizer = None
    engine = LLMEngine(config, UniProcExecutor(config, Worker(config)), tokenizer=tokenizer)
    core = engine.engine_core.engine_core
    return engine, core, core.model_executor.driver_worker.model_runner


def run_to_end(engine, limit=400):
    final = {}
    for _ in range(limit):
        if not engine.has_unfinished_requests():
            break
        for out in engine.step():
            final[out.request_id] = list(out.token_ids)
    return final


def greedy(requests, *, max_tokens=12, logprobs=None, **kwargs):
    """跑一个 tiny 引擎到结束，返回 `{req_id: token_ids}`。`requests` = [(id, prompt), ...]。"""
    engine, _core, _runner = make_engine(**kwargs)
    try:
        for req_id, prompt in requests:
            engine.add_request(req_id, list(prompt),
                               SamplingParams(max_tokens=max_tokens, temperature=0.0,
                                              eos_token_id=VOCAB + 7, logprobs=logprobs))
        return run_to_end(engine)
    finally:
        engine.shutdown()


class DraftRecorder:
    """记录 V2 runner 每轮的采样结果与草稿行数（观测，不改变行为）。

    `sample()` 的返回值是 `(SamplerOutput, num_sampled, num_rejected)`；这里只读不改，
    用来钉住"草稿真的进批了、num_rejected 由 GPU 算出来了"这两条管道事实。
    """

    def __init__(self, runner) -> None:
        self.runner = runner
        self.draft_rows: list[int] = []
        self.num_sampled: list[int] = []
        self.num_rejected: list[int] = []
        self.logits_rows: list[int] = []
        original = runner.sample

        def patched(hidden_states, input_batch, grammar_output):
            out = original(hidden_states, input_batch, grammar_output)
            self.draft_rows.append(int(input_batch.num_draft_tokens))
            self.logits_rows.append(int(input_batch.cu_num_logits[-1].item()))
            self.num_sampled += [int(x) for x in out[1].tolist()]
            self.num_rejected += [int(x) for x in out[2].tolist()]
            return out

        runner.sample = patched

    def histogram(self) -> dict[int, int]:
        return dict(Counter(self.num_sampled))
