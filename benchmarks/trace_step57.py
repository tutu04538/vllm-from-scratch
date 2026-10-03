"""57F 对照 A：**一条小场景的完整状态轨迹**（供 `docs/step57_architecture.md` 逐段对照）。

场景（一轮不漏地覆盖五种事件）：小池子 + ngram 投机，两条请求先后到达，池子不够时发生抢占，
被抢的请求再恢复，期间有草稿被接受、也有被拒绝。

每条记录打印：

    step / 每请求 status · num_tokens · num_computed_tokens（计划前→计划后→结果后）
    num_scheduled_tokens / scheduled_spec_decode_tokens
    逻辑块表 / free 块数 / ref_cnt 摘要
    Runner 侧的 req_ids · input_ids · positions · slot_mapping
    ModelRunnerOutput / 最终提交的 token

**只在调试路径上打**：所有字段都取自 CPU 侧的 Scheduler 与 InputBatch 镜像（块表 CPU 副本、
`PreparedInputs` 的 CPU 张量），不读 GPU、不做同步——生产热路径不开这个开关。
"""

import json
import sys

sys.path.insert(0, "/home/user/proj/vllm-from-scratch")

from minivllm import (CacheConfig, DeviceConfig, LLMEngine, ModelConfig, SamplingParams,
                    SchedulerConfig, SpeculativeConfig, UniProcExecutor, VllmConfig, Worker)

from minivllm.testing.tiny_models import tiny_qwen3_dir   # 测试模型现场生成（仓库不再放 fixtures）
TINY = tiny_qwen3_dir("tiny_gqa")
TINY_CONFIG = json.load(open(f"{TINY}/config.json"))
PROMPT = [1, 2, 3, 4, 1, 2, 3, 4]


def ref_summary(manager, request_id):
    blocks = manager.coordinator.get_blocks(request_id)
    return {"blocks": [block.block_id for block in blocks],
            "ref": [block.ref_cnt for block in blocks]}


def snapshot(scheduler, runner, step, submissions):
    """把一轮的状态抓成一行文本。"""
    rows = []
    for req_id, request in scheduler.requests.items():
        rows.append(f"{req_id}: {request.status} tokens={request.num_tokens} "
                    f"computed={request.num_computed_tokens} preempt={request.num_preemptions}")
    row_index = {req_id: row for req_id, row in
                 runner.input_batch.req_id_to_index.items()}
    lines = [
        f"step {step}",
        f"  Scheduler  running={[r.request_id for r in scheduler.running]} "
        f"waiting={scheduler.waiting.request_ids()}",
        f"             {rows if rows else '（没有活动请求）'}",
        f"  KV         空闲={scheduler.kv_cache_manager.num_free_blocks()} "
        f"缓存={scheduler.kv_cache_manager.num_cached_blocks()} "
        + " ".join(f"{req_id}:{ref_summary(scheduler.kv_cache_manager, req_id)}"
                   for req_id in row_index),
        f"  提交        {submissions}",
    ]
    if step in TRACE["packets"]:
        packet, inputs = TRACE["packets"][step]
        lines.insert(1, f"  计划        num_scheduled={dict(packet.num_scheduled_tokens)} "
                        f"spec={dict(packet.scheduled_spec_decode_tokens) or '{}'}")
        if inputs is not None:
            lines.insert(2, f"  Runner      req_ids={runner.input_batch.req_ids} "
                            f"input_ids={inputs.input_ids.tolist()} "
                            f"positions={inputs.positions.tolist()}")
            lines.insert(3, f"              slot_mapping={inputs.slot_mapping.tolist()} "
                            f"seq_lens={inputs.seq_lens.tolist()}")
    return "\n".join(lines)


def main() -> int:
    config = VllmConfig(
        model_config=ModelConfig(model=TINY, dtype="float32", max_model_len=64,
                                 hf_config=TINY_CONFIG),
        # 小池子（4 块 × 4 槽）：两条请求都跑长一点就会撞上抢占
        cache_config=CacheConfig(block_size=4, num_gpu_blocks=4),
        scheduler_config=SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=8),
        device_config=DeviceConfig(device="cpu"),
        speculative_config=SpeculativeConfig(method="ngram", num_speculative_tokens=3))
    engine = LLMEngine(config, UniProcExecutor(config, Worker(config)))
    scheduler = engine.engine_core.engine_core.scheduler
    runner = engine.engine_core.engine_core.model_executor.driver_worker.model_runner

    # 记录每一轮的"计划"与"Runner 输入"：包留一份快照，inputs 从 _prepare_inputs 里抄一份
    original_schedule, original_prepare = scheduler.schedule, runner._prepare_inputs
    TRACE["pending"] = None

    def traced_schedule():
        packet = original_schedule()
        TRACE["pending"] = packet
        return packet

    def traced_prepare(packet):
        inputs = original_prepare(packet)
        step = scheduler.num_steps
        TRACE["packets"][step] = (TRACE["pending"], inputs)
        return inputs

    scheduler.schedule, runner._prepare_inputs = traced_schedule, traced_prepare

    engine.add_request("A", PROMPT, SamplingParams(max_tokens=8, temperature=0.0,
                                                   eos_token_id=999), priority=0)
    engine.add_request("B", [5, 6, 7, 8, 5, 6, 7, 8], SamplingParams(
        max_tokens=8, temperature=0.0, eos_token_id=999), priority=5)   # 数值大 = 优先级低

    print("场景：两条 8-token prompt、池子 4 块、预算 8、ngram 投机 K=3、priority 策略\n")
    submissions = []
    while engine.has_unfinished_requests():
        step = scheduler.num_steps + 1
        outputs = engine.step()
        submissions = [(output.request_id, output.token_ids) for output in outputs]
        print(snapshot(scheduler, runner, step, submissions))
        print()
    engine.shutdown()
    print(scheduler.format_trace())
    return 0


TRACE = {"packets": {}, "pending": None}

if __name__ == "__main__":
    sys.exit(main())
