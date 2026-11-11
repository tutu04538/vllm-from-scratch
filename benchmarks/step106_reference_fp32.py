"""在查看 mini 结果前，以真实权重对参考首行 FC 与映射 logits 做 FP32 重放。"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--draft', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    records = torch.load(args.reference, map_location='cpu', weights_only=True)
    fc = next(event for event in records if event['event'] == 'combine_hidden_states')
    logits = next(event for event in records if event['event'] == 'compute_logits')
    with safe_open(str(args.draft / 'model.safetensors'), framework='pt', device='cpu') as weights:
        fc_weight = weights.get_tensor('fc.weight').float()
        head_weight = weights.get_tensor('lm_head.weight').float()
        d2t = weights.get_tensor('d2t').long()
    fc_float = F.linear(fc['inputs'][0][:1].float(), fc_weight)
    head_float = F.linear(logits['inputs'][0][:1].float(), head_weight)
    mapped_ids = torch.arange(len(d2t)) + d2t
    references = {'combine_first_row': (fc['output'][:1].float(), fc_float),
                  'mapped_logits_first_row': (logits['output'][:1, mapped_ids].float(), head_float)}
    report = {}
    for name, (bf16, fp32) in references.items():
        delta = bf16 - fp32
        abs_delta = delta.abs().flatten()
        index = int(abs_delta.argmax())
        relative_l2 = float(delta.norm() / fp32.norm().clamp_min(1e-30))
        report[name] = {'shape': list(bf16.shape), 'max_abs': float(abs_delta[index]),
                        'max_index': index, 'reference_at_max': float(bf16.flatten()[index]),
                        'fp32_at_max': float(fp32.flatten()[index]), 'relative_l2': relative_l2,
                        'absolute_limit': max(1e-5, 2 * float(abs_delta[index])),
                        'relative_l2_limit': max(1e-5, 2 * relative_l2)}
        if name.startswith('mapped'):
            values, ids = torch.topk(bf16[0], 2)
            report[name]['top_ids_target'] = mapped_ids[ids].tolist()
            report[name]['top_values'] = values.tolist()
            report[name]['top_margin'] = float(values[0] - values[1])
    payload = {'scope': 'reference first observed row only; no model-forward/other-row guarantee',
               'reference_sha256': hashlib.sha256(args.reference.read_bytes()).hexdigest(),
               'draft': str(args.draft), 'results': report}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == '__main__':
    main()
