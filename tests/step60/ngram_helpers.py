"""step60 测试的共享工具：引擎夹具 + 造"批行/历史"的工具。

引擎那部分直接复用 `tests/step59/helpers.py`（同一套 tiny 模型配置），这里只加 ngram 关需要的：
`make_rows()`（造 `TargetRows`）与 `fake_input_batch()`（造一份够用的批镜像替身）。
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_step59_helpers():
    """按路径加载 `tests/step59/helpers.py`（不能写 `import helpers`：会和本模块同名）。"""
    path = ROOT / "tests" / "step59" / "spec_helpers.py"
    spec = importlib.util.spec_from_file_location("step59_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


step59_helpers = _load_step59_helpers()
make_config = step59_helpers.make_config
make_engine = step59_helpers.make_engine
run_prompts = step59_helpers.run_prompts

from minivllm.spec_decode.utils import TargetRows  # noqa: E402


def make_rows(histories: list[list[int]], ready: bool = True,
              req_ids: list[str] | None = None) -> list[TargetRows]:
    """按历史长度造 `TargetRows`（`history_end` = 历史长度，`ready` 默认全 True）。"""
    req_ids = req_ids or [f"r{index}" for index in range(len(histories))]
    return [TargetRows(req_id=req_id, row=index, start=0, target_rows=1, num_rejected=0,
                       history_end=len(tokens), next_token_id=tokens[-1] if tokens else 0,
                       ready=ready)
            for index, (req_id, tokens) in enumerate(zip(req_ids, histories))]


class FakeInputBatch:
    """只带 ngram 提议者需要的字段：`token_ids_cpu` / `num_tokens_no_spec` / `vocab_size`。"""

    def __init__(self, histories: list[list[int]], max_model_len: int, vocab_size: int = 1000,
                 dtype=None, max_num_reqs: int | None = None):
        import torch

        self.max_num_reqs = max_num_reqs or max(len(histories), 1)
        self.vocab_size = vocab_size
        self.token_ids_cpu = torch.zeros((self.max_num_reqs, max_model_len),
                                         dtype=dtype or torch.int64)
        self.num_tokens_no_spec = torch.zeros(self.max_num_reqs, dtype=torch.int64)
        self.req_ids = []
        self.req_id_to_index = {}
        for index, tokens in enumerate(histories):
            self.set_row(index, tokens, req_id=f"r{index}")

    def set_row(self, index: int, tokens: list[int], req_id: str | None = None) -> None:
        assert index < self.max_num_reqs, f"行 {index} 超出 max_num_reqs={self.max_num_reqs}"
        self.token_ids_cpu[index].zero_()
        self.token_ids_cpu[index, :len(tokens)] = __import__("torch").tensor(tokens)
        self.num_tokens_no_spec[index] = len(tokens)
        if req_id is not None:
            while len(self.req_ids) <= index:
                self.req_ids.append(None)
            self.req_ids[index] = req_id
            self.req_id_to_index = {rid: i for i, rid in enumerate(self.req_ids)
                                    if rid is not None}

    @property
    def num_reqs(self) -> int:
        return sum(1 for req_id in self.req_ids if req_id is not None)

    def num_tokens(self, row: int) -> int:
        return int(self.num_tokens_no_spec[row])
