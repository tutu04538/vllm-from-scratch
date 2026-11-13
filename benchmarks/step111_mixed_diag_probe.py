"""在混合请求的固定位置采集逐层输出。"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.step111_diag_wrapper import main

if __name__ == '__main__': main('benchmarks.step111_mixed_diag_hooks')
