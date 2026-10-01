"""第五十七关入口：把真实模型放进协议后面，从一句话走到一句话。

    tokenizer 编码 → LLMEngine → InprocClient → EngineCore → UniProcExecutor → Worker
      → GPUModelRunner（_update_states → _prepare_inputs → 模型 → 采样）→ tokenizer 解码

与第 56 关的区别不在"能生成"，而在**每一层是谁**：调度器发的是纯数据快照，模型前向拿不到
Request，KV 由 Runner 按块表写入，采样在另一步做。这个文件只做两件事：转发公开 API、
跑一段短生成。

    python step57/step57.py                                  # 默认模型 + 默认问题
    python step57/step57.py --max-new-tokens 64 "问题一" "问题二"
    python step57/step57.py --model-dir /path/to/other --device cpu --trace
    python step57/step57.py --scheduler-trace "同样的问题" "同样的问题"    # 看前缀命中

`--scheduler-trace` 打印每一轮的调度决策（排了谁几个 token、命中多少、抢占了谁、块占用），
它是 57C 的交付物之一：**不运行模型也能看懂调度器在做什么**。
"""

import argparse
import json
import pathlib
import sys
import time

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

DEFAULT_MODEL_DIR = str(_ROOT / "models" / "Qwen3-1.7B")
DEFAULT_QUESTIONS = ["用一句话解释什么是 KV cache。", "用一个类比说明分页和连续内存的区别。"]


def encode(tokenizer, question):
    """走官方 chat template：角色标签与特殊 token 由模板加，不手工拼。"""
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def scheduler_of(engine):
    return engine.engine_core.engine_core.scheduler


def main(argv=None):
    parser = argparse.ArgumentParser(description="对齐 vLLM 架构的引擎：真实模型短生成")
    parser.add_argument("questions", nargs="*", default=None, help="要问的问题，可以给多条")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="模型目录（含 tokenizer）")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-num-seqs", type=int, default=2)
    parser.add_argument("--max-num-batched-tokens", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-kv-blocks", type=int, default=64)
    parser.add_argument("--no-prefix-caching", action="store_true",
                        help="关掉前缀缓存（57C）：块 hash、命中查询、LRU 逐出都不参与")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16",
                        help="运行精度。本机 Qwen3-1.7B 的检查点就是 bf16，按 bf16 加载省一半内存")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0 = 贪心（本关的最小采样器只支持贪心与温度随机，见 sample/sampler.py）")
    parser.add_argument("--trace", action="store_true", help="打印第一轮真正喂给模型的数字")
    parser.add_argument("--scheduler-trace", action="store_true",
                        help="打印每轮的调度决策（scheduler_trace）")
    args = parser.parse_args(argv)
    questions = args.questions or DEFAULT_QUESTIONS

    import torch
    from transformers import AutoTokenizer

    from step57 import (CacheConfig, DeviceConfig, LLMEngine, ModelConfig, SamplingParams,
                        SchedulerConfig, UniProcExecutor, VllmConfig, Worker)

    model_dir = pathlib.Path(args.model_dir)
    if not model_dir.is_dir():
        parser.error(f"模型目录不存在: {model_dir}（本关只读本地目录，不下载模型）")
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    hf_config = json.loads((model_dir / "config.json").read_text())
    config = VllmConfig(
        model_config=ModelConfig(model=str(model_dir), dtype=args.dtype,
                                 max_model_len=args.max_model_len, hf_config=hf_config),
        cache_config=CacheConfig(block_size=args.block_size, num_gpu_blocks=args.num_kv_blocks,
                                 enable_prefix_caching=not args.no_prefix_caching),
        scheduler_config=SchedulerConfig(max_num_seqs=args.max_num_seqs,
                                         max_num_batched_tokens=args.max_num_batched_tokens),
        device_config=DeviceConfig(device=device))

    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    # 没有任何 Runner 注入 → Worker 走真实路径：建 GPUModelRunner、读权重、按 KV 规格建物理缓存
    engine = LLMEngine(config, UniProcExecutor(config, Worker(config)))
    load_seconds = time.perf_counter() - started

    runner = engine.engine_core.engine_core.model_executor.driver_worker.model_runner
    model = runner.model
    num_params = sum(parameter.numel() for parameter in model.parameters())
    kv_bytes = sum(cache.numel() * cache.element_size() for cache in runner.kv_caches.values())
    print(f"模型: {model_dir}（{type(model).__name__}）")
    print(f"  {hf_config['num_hidden_layers']} 层, hidden={hf_config['hidden_size']}, "
          f"Q/KV heads={hf_config['num_attention_heads']}/{hf_config['num_key_value_heads']}"
          f"（GQA，head_dim={hf_config['head_dim']}）")
    print(f"  运行精度 {model.model.embed_tokens.weight.dtype}，设备 {device}，"
          f"参数 {num_params / 1e9:.2f} G，tied embedding={model.tie_word_embeddings}")
    print(f"  KV 缓存 {len(runner.kv_caches)} 层 × {args.num_kv_blocks} 块 × "
          f"{args.block_size} 槽 = {kv_bytes / 1e6:.0f} MB；加载 {load_seconds:.2f}s（含 tokenizer）")
    print(f"  前缀缓存：{'开' if config.cache_config.enable_prefix_caching else '关'}"
          f"（块 hash 链 + 引用计数 + LRU 逐出；两条相同 prompt 的请求会命中同一批块）")

    # --trace：只记第一轮。这是"协议 → 模型输入"这一段的真实数字，后面的轮次结构相同
    original_prepare = runner._prepare_inputs
    trace = {}

    def traced_prepare(scheduler_output):
        inputs = original_prepare(scheduler_output)
        trace.setdefault("packet", dict(scheduler_output.num_scheduled_tokens))
        trace.setdefault("inputs", inputs)
        return inputs

    if args.trace:
        runner._prepare_inputs = traced_prepare

    for index, question in enumerate(questions):
        prompt_token_ids = encode(tokenizer, question)
        engine.add_request(f"q{index}", prompt_token_ids,
                           SamplingParams(max_tokens=args.max_new_tokens,
                                          temperature=args.temperature,
                                          eos_token_id=hf_config.get("eos_token_id")))
        print(f"\n问: {question}（{len(prompt_token_ids)} 个 prompt token）")

    start = time.perf_counter()
    steps = 0
    finished: dict[str, str] = {}
    answers: dict[str, str] = {}
    while engine.has_unfinished_requests():
        for output in engine.step():
            answers[output.request_id] = tokenizer.decode(output.token_ids,
                                                          skip_special_tokens=True)
            if output.finished:
                # 结束原因是枚举，打印名字比打印数字可读（STOP=遇到 eos，LENGTH=到 max_tokens）
                finished[output.request_id] = output.finish_reason.name
        steps += 1
    seconds = time.perf_counter() - start
    runner._prepare_inputs = original_prepare

    if trace.get("inputs") is not None:
        inputs = trace["inputs"]
        print(f"\n第一轮喂给模型的数字（协议只给了 num_scheduled_tokens={trace['packet']}）：")
        print(f"  input_ids       = {inputs.input_ids.tolist()}")
        print(f"  positions       = {inputs.positions.tolist()}")
        print(f"  query_start_loc = {inputs.query_start_loc.tolist()}   "
              f"seq_lens={inputs.seq_lens.tolist()}")
        print(f"  slot_mapping    = {inputs.slot_mapping.tolist()}   "
              f"logits_indices={inputs.logits_indices.tolist()}")
        print(f"  块表镜像        = {runner.input_batch.block_table.cpu[0].tolist()}")
        print("  （输入都是执行侧按协议算出来的：Scheduler 只发了 num_scheduled_tokens 与块号）")

    for index, question in enumerate(questions):
        request_id = f"q{index}"
        print(f"\n答[{request_id}]: {answers.get(request_id, '')}")
        print(f"   结束原因 {finished.get(request_id)}")
    generated = sum(len(tokenizer.encode(text)) for text in answers.values())
    stats = runner.input_batch.block_table  # noqa: F841  （只是提醒：块表在执行侧）
    kv_manager = engine.engine_core.engine_core.kv_cache_manager
    print(f"\n{steps} 轮调度、{generated} 个 token、{seconds:.2f}s"
          f"（{generated / max(seconds, 1e-9):.1f} tok/s；本关是逐请求的 Torch attention，"
          f"没有做性能优化）")
    print(f"KV 池收尾：占用 {kv_manager.num_allocated_blocks} 块、空闲 {kv_manager.num_free_blocks()} 块、"
          f"缓存中 {kv_manager.num_cached_blocks()} 块、抢占 {scheduler_of(engine).num_preemptions} 次")
    if args.scheduler_trace:
        print()
        print(scheduler_of(engine).format_trace())
    engine.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
