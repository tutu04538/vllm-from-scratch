"""The copy stream must wait for asynchronous initialization of mask buffers."""
from types import SimpleNamespace

import numpy as np
import torch

from minivllm.worker.gpu.structured_outputs import StructuredOutputsWorker


def test_copy_stream_waits_for_mask_initialization():
    for _ in range(5):
        torch.cuda.synchronize()
        cu = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device='cuda')
        logits = torch.full((3, 64), 3.5, device='cuda')
        bitmask = np.zeros((5, 2), dtype=np.int32)
        bitmask[0] = [6, 2]
        bitmask[2] = [1, -2147483648]
        batch = SimpleNamespace(req_ids=['r0', 'r1', 'r2'],
            cu_num_logits_np=np.array([0, 1, 2, 3], dtype=np.int32),
            num_draft_tokens_per_req=np.array([1, 0, 2], dtype=np.int32), cu_num_logits=cu)
        torch.cuda._sleep(100_000_000)
        worker = StructuredOutputsWorker(8, 64, torch.device('cuda'), 4, 1)
        worker.apply_grammar_bitmask(logits, batch, ['r0', 'r2'], bitmask)
        assert torch.isfinite(logits[0]).cpu().nonzero().flatten().tolist() == [1, 2, 33]
