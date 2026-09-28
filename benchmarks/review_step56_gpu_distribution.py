"""一般 q 的 GPU 拒绝验证：草稿确从 q 抽样，检查首 token 与条件两 token 联合分布。"""
from pathlib import Path
from types import SimpleNamespace
import json
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from step56.rejection import BatchedRejectionSampler, RejectionItem
from step56.sampling import SamplingParams, TorchSampler


def main():
    torch.set_num_threads(1)
    n = 20000
    p = torch.tensor([0.6, 0.3, 0.1], dtype=torch.float32, device='cuda')
    q = torch.tensor([0.1, 0.6, 0.3], dtype=torch.float32, device='cuda')
    conditional = torch.tensor([[0.2,0.3,0.5], [0.7,0.2,0.1], [0.1,0.7,0.2]],
                               dtype=torch.float32, device='cuda')
    # 草稿的随机源与验证流独立；这一份 CPU q 与传入 GPU 验证的 q 相同。
    proposals = torch.multinomial(q.cpu(), n, replacement=True,
                                  generator=torch.Generator().manual_seed(9107)).tolist()
    params = SamplingParams(temperature=0.8, seed=7)
    backend = BatchedRejectionSampler('triton', set(), TorchSampler(), device='cuda')
    states = [SimpleNamespace(sampling_params=params, rejection_seed=1000+i,
                              rejection_rng_counter=0, request_id=str(i)) for i in range(n)]
    batch = [RejectionItem(plan={'request':states[i]}, mode='distribution', draft_ids=[d],
                           remaining_outputs=2, row_probs=[p,conditional[d]], draft_probs=[q])
             for i,d in enumerate(proposals)]
    first = backend.materialize_results(backend.verify_batch(batch))
    assert all(r.error is None for r in first)
    pairs, pending, indices = [None]*n, [], []
    for i,r in enumerate(first):
        if len(r.committed_ids) == 2:
            pairs[i] = r.committed_ids
        else:
            assert len(r.committed_ids) == 1
            states[i].rejection_rng_counter += r.rng_consumed
            pending.append(RejectionItem(plan={'request':states[i]}, mode='distribution',
                                          draft_ids=[], remaining_outputs=1,
                                          row_probs=[conditional[r.committed_ids[0]]]))
            indices.append(i)
    second = backend.materialize_results(backend.verify_batch(pending))
    for i,r in zip(indices,second):
        assert r.error is None and len(r.committed_ids) == 1
        pairs[i] = first[i].committed_ids + r.committed_ids
    ids = torch.tensor(pairs)
    joint = torch.bincount(ids[:,0]*3+ids[:,1], minlength=9).reshape(3,3).double()/n
    expected = p.cpu().double()[:,None]*conditional.cpu().double()
    marginal = joint.sum(1)
    se_joint = (expected*(1-expected)/n).sqrt()
    p_cpu = p.cpu().double()
    se_first = (p_cpu*(1-p_cpu)/n).sqrt()
    checks = [('GPU 拒绝验证的首 token 分布回到 p', bool(((marginal-p_cpu).abs() < 6*se_first).all())),
              ('GPU 两 token 联合分布符合 p(y0)*p(y1|y0)', bool(((joint-expected).abs() < 6*se_joint).all()))]
    output=dict(samples=n, proposed_from_q=True, first_empirical=marginal.tolist(),
                joint_empirical=joint.tolist(), joint_expected=expected.tolist(),
                rejected=len(pending), max_joint_z=float(((joint-expected).abs()/se_joint).max()),
                checks=[dict(name=name,passed=ok) for name,ok in checks])
    destination=Path('/tmp/step56_recheck/gpu_distribution.json')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    for name,ok in checks:
        print('PASS' if ok else 'FAIL',name)
    print(json.dumps(output,ensure_ascii=False))
    return 0 if all(ok for _,ok in checks) else 1


if __name__=='__main__':
    sys.exit(main())
