"""在混合请求的固定 KV 位置采集 QKV、norm 和 RoPE。"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.step111_diag_wrapper import main

if __name__ == '__main__': main('benchmarks.step111_kv_diag_hooks')
