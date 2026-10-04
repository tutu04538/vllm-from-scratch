"""step62 的会话级夹具：把仓库根放进 sys.path（测试要按点号路径导入 `examples.*`）。"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
