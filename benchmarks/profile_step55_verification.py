"""定向观测现有拒绝验证的 GPU 标量回传，不是端到端性能基准。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from step55.speculative import verify_drafts_random


def main():
    assert torch.cuda.is_available()
    # 必定首拒绝：p[0]/q[0]=0.4，注入 GPU uniform=0.9。
    p = torch.tensor([0.2, 0.3, 0.5], device="cuda")
    q = torch.tensor([0.5, 0.3, 0.2], device="cuda")
    uniform = torch.tensor(0.9, device="cuda")
    generator = torch.Generator(device="cuda").manual_seed(42)

    def verify():
        return verify_drafts_random(
            [0, 0], [p, p, p], set(), 3,
            lambda: float(uniform),
            lambda probs: torch.multinomial(probs, 1, generator=generator),
            draft_probs=[q, q],
        )

    for _ in range(10):
        verify()
    torch.cuda.synchronize()
    records = []
    for batch in (1, 8, 16):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
            for _ in range(batch):
                result = verify()
                assert result.num_accepted == 0
        events = {event.key: event.count for event in prof.key_averages()}
        records.append({"requests": batch,
                        "aten::item": events.get("aten::item", 0),
                        "aten::_local_scalar_dense": events.get("aten::_local_scalar_dense", 0)})
    print(json.dumps(records, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
