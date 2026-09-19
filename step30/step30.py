"""第 30 关入口：演示「只给一个外部目录，就能加载并生成 token」。

实现按模块拆在同一个包里，本文件只做两件事：转发公开 API、跑一段演示。

    python step30/step30.py
"""

import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# 这个文件既可能被 `python step30/step30.py` 直接跑，也可能在 sys.path 指向
# step30/ 目录时被 `import step30` 命中。后一种情况下它挡住了同名包，让位给包。
_self = sys.modules.get("step30")
if _self is not None and not hasattr(_self, "__path__"):
    del sys.modules["step30"]
    import step30 as _package
    globals().update({k: v for k, v in vars(_package).items() if not k.startswith("__")})


def main():
    import torch

    from step30 import Engine, save_model

    project_root = _ROOT
    fixtures = project_root / "fixtures" / "step30_qwen3"

    def run(engine, requests):
        out = []
        engine.scheduler.on_finished = lambda r: out.append((r["request_id"], tuple(r["output_ids"])))
        for request in requests:
            engine.add_request(dict(request))
        rounds = 0
        while engine.has_unfinished_requests():
            engine.step()
            rounds += 1
        return out, rounds

    # 1. 外部 Qwen3 目录：只给目录和运行选项，尺寸、参数名都不由调用者提供
    print("=== 直接加载外部 Qwen3 目录 ===")
    external = Engine.from_model_dir(
        fixtures / "tiny_gqa",
        device="cuda" if torch.cuda.is_available() else None,
        attention_backend="torch",
        max_num_seqs=3,
        max_num_batched_tokens=8,
        num_kv_blocks=48,
        block_size=4,
    )
    model = external.model
    print(f"  tiny_gqa -> {model.num_layers} 层, d_model={model.d_model}, "
          f"Q/KV heads={model.num_q_heads}/{model.num_kv_heads}, head_dim={model.head_dim}, "
          f"use_qk_norm={model.use_qk_norm}")

    mini = Engine.from_model_dir(
        fixtures / "tiny_mqa",
        device="cuda" if torch.cuda.is_available() else None,
        attention_backend="torch",
        max_num_seqs=3,
        max_num_batched_tokens=8,
        num_kv_blocks=48,
        block_size=4,
    )
    print(f"  tiny_mqa -> {mini.model.num_layers} 层, d_model={mini.model.d_model}, "
          f"Q/KV heads={mini.model.num_q_heads}/{mini.model.num_kv_heads}, head_dim={mini.model.head_dim}")

    requests = [{"request_id": "A", "prompt_ids": [0, 1, 2, 3, 4], "max_new_tokens": 4},
                {"request_id": "B", "prompt_ids": [3], "max_new_tokens": 2},
                {"request_id": "C", "prompt_ids": [0, 1], "max_new_tokens": 2}]
    got, rounds = run(external, requests)
    print(f"  {rounds} 轮生成: {got}")

    # 2. 外部加载的模型能存回自己的格式，再加载得到相同结果
    print()
    print("=== 存成自己的格式再加载 ===")
    model_dir = pathlib.Path(__file__).parent / "my_external_model"
    print(f"  保存到 {save_model(model, model_dir)}")
    reloaded = Engine.from_model_dir(model_dir, device=model.device, attention_backend="torch",
                                     max_num_seqs=3, max_num_batched_tokens=8,
                                     num_kv_blocks=48, block_size=4)
    again, _ = run(reloaded, requests)
    print(f"  重新加载后输出一致 = {got == again}")

    # 3. 随机初始化 -> 保存 -> 只凭目录重建，旧路径仍然可用
    print()
    print("=== 自定义目录老路径 ===")
    torch.manual_seed(12345)
    random_dir = pathlib.Path(__file__).parent / "my_tiny_model"
    random_engine = Engine(device=model.device, attention_backend="torch", max_num_seqs=3,
                           max_num_batched_tokens=8, num_kv_blocks=48, block_size=4,
                           vocab_size=11, d_model=32, max_seq_len=64, num_q_heads=4,
                           num_kv_heads=2, num_layers=2, intermediate_size=48,
                           head_dim=16, use_qk_norm=True)
    save_model(random_engine.model, random_dir)
    rebuilt = Engine.from_model_dir(random_dir, device=model.device, attention_backend="torch",
                                    max_num_seqs=3, max_num_batched_tokens=8,
                                    num_kv_blocks=48, block_size=4)
    before, _ = run(random_engine, requests)
    after, _ = run(rebuilt, requests)
    print(f"  输出一致 = {before == after}")


if __name__ == "__main__":
    main()
