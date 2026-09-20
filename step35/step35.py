"""第 31 关入口：从一句话走到一句话。

    tokenizer 编码 → 自己的 Engine → tokenizer 解码

实现按模块拆在同一个包里，本文件只做两件事：转发公开 API、跑文本演示。

    python step35/step35.py                                  # 默认模型 + 默认问题
    python step35/step35.py --max-new-tokens 64 "问题一" "问题二"
    python step35/step35.py --model-dir /path/to/other --backend torch
"""

import argparse
import pathlib
import sys
import time

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# 这个文件既可能被 `python step35/step35.py` 直接跑，也可能在 sys.path 指向
# step35/ 目录时被 `import step35` 命中。后一种情况下它挡住了同名包，让位给包。
_self = sys.modules.get("step35")
if _self is not None and not hasattr(_self, "__path__"):
    del sys.modules["step35"]
    import step35 as _package
    globals().update({k: v for k, v in vars(_package).items() if not k.startswith("__")})

DEFAULT_MODEL_DIR = "/home/user/proj/KuiperLLama/Qwen/Qwen3-0.6B"
DEFAULT_QUESTIONS = ["用一句话解释什么是 KV cache。", "用一个类比说明分页和连续内存的区别。"]


def encode(tokenizer, question):
    # 走官方 chat template：角色标签和特殊 token 由模板加，不手工拼
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def main(argv=None):
    parser = argparse.ArgumentParser(description="用自己的 Engine 生成真实文本")
    parser.add_argument("questions", nargs="*", default=None, help="要问的问题，可以给多条")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="模型目录（含 tokenizer）")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-num-seqs", type=int, default=3)
    parser.add_argument("--max-num-batched-tokens", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-kv-blocks", type=int, default=128)
    parser.add_argument("--backend", choices=("torch", "triton"), default="triton")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32",
                        help="运行精度；默认 float32，bfloat16 需要 CUDA")
    parser.add_argument("--norm-backend", choices=("torch", "triton"), default="torch",
                        help="RMSNorm 后端；默认 torch，triton 是融合 kernel（需要 CUDA）")
    parser.add_argument("--graph", action="store_true", help="打开 CUDA Graph（需要 CUDA + Triton）")
    parser.add_argument("--mode", choices=("greedy", "random"), default="greedy")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--presence-penalty", type=float, default=None)
    parser.add_argument("--frequency-penalty", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-prefix-caching", action="store_true")
    args = parser.parse_args(argv)
    questions = args.questions or DEFAULT_QUESTIONS

    import torch
    from transformers import AutoTokenizer

    from step35 import Engine

    runtime_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]

    model_dir = pathlib.Path(args.model_dir)
    if not model_dir.is_dir():
        parser.error(f"模型目录不存在: {model_dir}")

    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    prompts = [encode(tokenizer, question) for question in questions]

    engine = Engine.from_model_dir(
        model_dir,
        device=args.device,
        attention_backend=args.backend,
        use_cuda_graph=args.graph,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        block_size=args.block_size,
        num_kv_blocks=args.num_kv_blocks,
        enable_prefix_caching=not args.no_prefix_caching,
        dtype=runtime_dtype,
        norm_backend=args.norm_backend,
    )
    load_seconds = time.perf_counter() - started

    model = engine.model
    # 打印**实际**精度：参数、KV 池、输入缓冲各查一次，避免「配置写 BF16、其实还在跑 FP32」
    param = next(model.parameters())
    print(f"模型: {model_dir}")
    print(f"  运行精度 {model.dtype}：参数 {param.dtype}，KV 池 {engine.kv_cache_pool.k_cache.dtype}，"
          f"输入缓冲 {model.input_buffer.dtype}（整数索引）")
    print(f"  后端：attention={engine.attention_backend}，norm={model.norm_backend}"
          f"（两层的 norm1/norm2={model.layers[0].norm1.backend}，最终 norm={model.norm.backend}）")
    print(f"  {model.num_layers} 层, d_model={model.d_model}, Q/KV heads={model.num_q_heads}/{model.num_kv_heads}, "
          f"head_dim={model.head_dim}, use_qk_norm={model.use_qk_norm}")
    print(f"  词表 {model.vocab_size}, 停止 token {sorted(model.eos_token_ids)}, "
          f"加载 {load_seconds:.2f}s（含 tokenizer）")
    print(f"  权重 {sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9:.2f} GB, "
          f"KV 池 {engine.kv_cache_pool.k_cache.numel() * engine.kv_cache_pool.k_cache.element_size() * 2 / 1e6:.0f} MB")

    print(f"  采样后端：{engine.sampler.name}，模式 = {args.mode}")
    results = {}
    engine.scheduler.on_finished = lambda r: results.__setitem__(r["request_id"], list(r["output_ids"]))
    for i, prompt_ids in enumerate(prompts):
        request = {"request_id": f"q{i}", "prompt_ids": prompt_ids,
                   "max_new_tokens": args.max_new_tokens}
        if args.mode == "random":
            # 没显式给的随机参数就用一组能看出效果的默认值
            request.update(temperature=args.temperature if args.temperature is not None else 0.8,
                           top_k=args.top_k if args.top_k is not None else 20,
                           top_p=args.top_p if args.top_p is not None else 0.9)
        else:
            # greedy 模式：不执行 top-k / top-p 筛选，但**用户显式传进来的参数仍要合法**，
            # 不能因为「反正不用」就丢掉——那会把非法配置静默吞掉
            if args.temperature is not None and args.temperature != 0.0:
                parser.error(f"--mode greedy 与 --temperature {args.temperature} 冲突："
                             f"非 0 温度就是随机采样，请用 --mode random")
            for key, value in (("temperature", args.temperature), ("top_k", args.top_k),
                               ("top_p", args.top_p)):
                if value is not None:
                    request[key] = value
        for key, value in (("repetition_penalty", args.repetition_penalty),
                           ("presence_penalty", args.presence_penalty),
                           ("frequency_penalty", args.frequency_penalty),
                           ("seed", args.seed)):
            if value is not None:
                request[key] = value
        engine.add_request(request)

    start = time.perf_counter()
    steps = 0
    while engine.has_unfinished_requests():
        engine.step()
        steps += 1
    seconds = time.perf_counter() - start

    for i, question in enumerate(questions):
        output_ids = results[f"q{i}"]
        # 只 decode 新生成的部分，不把 prompt 也解出来
        answer = tokenizer.decode(output_ids, skip_special_tokens=True)
        print()
        print(f"问: {question}")
        print(f"答: {answer}")
        print(f"    {len(prompts[i])} prompt + {len(output_ids)} 生成 token，{steps} 步，{seconds:.2f}s")
    print()
    print(f"合计 {sum(len(v) for v in results.values())} 个 token / {seconds:.2f}s "
          f"= {sum(len(v) for v in results.values()) / seconds:.1f} tok/s"
          f"（含每步调度与采样，未单独计时）")


if __name__ == "__main__":
    main()
