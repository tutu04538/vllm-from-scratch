"""验收补测：接受 EOS 后必须立即停止验证，不消费后续接受随机数。"""
import json
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from step54.speculative import verify_drafts_random


def case(name, drafts, probs, expected_uniforms, expected_ids, expected_kept):
    calls = dict(uniform=0, token=0)
    generator = torch.Generator(device='cpu').manual_seed(7)
    reference = torch.Generator(device='cpu').manual_seed(7)
    # 始终给出 0.1 以确定接受，但真正消耗 generator，检验后续流位置。
    def draw_uniform():
        calls['uniform'] += 1
        torch.rand((), generator=generator)
        return 0.1
    def draw_token(p):
        calls['token'] += 1
        raise AssertionError('接受 EOS 后不应抽纠正/bonus token')
    result = verify_drafts_random(drafts, [torch.tensor(p) for p in probs],
                                  {2}, 8, draw_uniform, draw_token)
    for _ in range(expected_uniforms):
        torch.rand((), generator=reference)
    checks = dict(output=result.committed_ids == expected_ids,
                  num_accepted=result.num_accepted == len(expected_ids),
                  kept_inputs=result.kept_inputs == expected_kept,
                  uniform_calls=calls['uniform'] == expected_uniforms,
                  generator_state=torch.equal(generator.get_state(), reference.get_state()))
    return dict(name=name, checks=checks, actual_calls=calls,
                expected_uniform_calls=expected_uniforms,
                committed_ids=result.committed_ids, kept_inputs=result.kept_inputs,
                num_accepted=result.num_accepted)


if __name__ == '__main__':
    cases = [case('第一枚接受草稿为 EOS', [2, 1],
                  [[0.25, 0.25, 0.5], [0.3, 0.4, 0.3], [0.2, 0.3, 0.5]], 1, [2], 1),
             case('中间接受草稿为 EOS', [0, 2, 1],
                  [[0.5, 0.3, 0.2], [0.2, 0.3, 0.5], [0.3, 0.4, 0.3], [0.2, 0.3, 0.5]],
                  2, [0, 2], 2)]
    for result in cases:
        ok = all(result['checks'].values())
        print(('PASS' if ok else 'FAIL'), json.dumps(result, ensure_ascii=False))
    path = Path('/tmp/step54_acceptance/eos_rng.json')
    path.write_text(json.dumps(cases, indent=2, ensure_ascii=False) + '\n')
    sys.exit(0 if all(all(c['checks'].values()) for c in cases) else 1)
