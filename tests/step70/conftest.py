"""把本目录的辅助模块放进 import 路径（与其它 stepNN 一致：辅助模块名全局唯一）。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
