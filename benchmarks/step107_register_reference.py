"""Register repeat-only frozen TP reference before comparing project outputs."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open


def tensor_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--draft', type=Path, required=True)
    p.add_argument('--fixture', type=Path, required=True)
    p.add_argument('--contract', type=Path, required=True)
    a = p.parse_args()
    summary = json.loads((a.reference / 'summary.json').read_text())
    world = summary['tp']
    runs = [[torch.load(a.reference/f'run{run}'/f'rank{rank}.pt', weights_only=True)
             for rank in range(world)] for run in range(2)]
    labels = ('combine_hidden_states', 'draft_qkv', 'draft_attention', 'draft_layer', 'compute_logits')
    def event(records, label):
        return next(item for item in records if item['event'] == label)
    observed = {}
    for rank in range(world):
        observed[str(rank)] = {}
        for label in labels:
            x, y = [event(runs[run][rank], label)['output'] for run in range(2)]
            if isinstance(x, list):
                exact = all(torch.equal(m, n) for m, n in zip(x, y))
            else:
                exact = torch.equal(x, y)
            observed[str(rank)][label] = exact
            if not exact:
                raise ValueError(f'frozen reference repeats differ: rank {rank} {label}')
        first_layer = event(runs[0][rank], 'draft_layer')
        second_layer = event(runs[1][rank], 'draft_layer')
        exact_kv = torch.equal(first_layer['written_kv'], second_layer['written_kv'])
        observed[str(rank)]['draft_written_kv'] = exact_kv
        if not exact_kv:
            raise ValueError(f'frozen reference KV repeats differ: rank {rank}')
    records = runs[0][0]
    fc = event(records, 'combine_hidden_states')
    logits = event(records, 'compute_logits')
    raw = event(records, 'draft_raw_logits')['output']
    # Independently replay one frozen input row using FP32 accumulations. This is
    # diagnostic evidence; exact same-TP frozen repeats define the strict gate.
    fp32 = {}
    with safe_open(a.draft/'model.safetensors', framework='pt', device='cpu') as source:
        for name, inputs, reference in (
                ('fc.weight', fc['inputs'][0][:1], fc['output'][:1]),
                ('lm_head.weight', logits['inputs'][0][:1], raw[:1])):
            weight = source.get_tensor(name).to(torch.bfloat16).float()
            expected = torch.nn.functional.linear(inputs.float(), weight)
            delta = (reference.float()-expected).abs()
            i = int(delta.argmax())
            fp32[name] = {'max_abs': float(delta.max()),
                          'relative_l2': float(delta.norm()/expected.norm()),
                          'max_flat_index': i,
                          'bf16_value': float(reference.float().flatten()[i]),
                          'fp32_value': float(expected.flatten()[i])}
            del weight, expected, delta
    # Threshold is fixed only from reference repeats, before any project output is read.
    contract = {'model': summary['model'], 'tp': world, 'reference_repeats_exact': observed,
                'acceptance': {'kind': 'exact', 'reason': 'frozen repeated tensors are exactly equal'},
                'fp32_first_row': fp32,
                'draft_sha256': tensor_digest(a.draft/'model.safetensors'),
                'references': [tensor_digest(a.reference/f'run{run}'/f'rank{rank}.pt')
                               for run in range(2) for rank in range(world)]}
    a.contract.parent.mkdir(parents=True, exist_ok=True)
    a.contract.write_text(json.dumps(contract, ensure_ascii=False, indent=2)+'\n')
    layer = event(records, 'draft_layer')
    qkv = event(records, 'draft_qkv')
    attn = [event(runs[0][rank], 'draft_attention') for rank in range(world)]
    fixture = {'aux': fc['inputs'][0], 'combined': fc['output'],
               'logits_input': logits['inputs'][0], 'mapped_logits': logits['output'],
               'qkv_input': qkv['inputs'][0],
               'attention_input': [torch.cat([entry['inputs'][i] for entry in attn], dim=-1)
                                   for i in range(3)],
               'attention_output': torch.cat([entry['output'] for entry in attn], dim=-1),
               'attention_kv': torch.cat([event(runs[0][rank], 'draft_layer')['written_kv']
                                           for rank in range(world)], dim=2),
               'layer_inputs': dict(layer['kwargs']),
               'layer_output': layer['output'][0], 'layer_residual': layer['output'][1]}
    # The post-hook hidden argument has already been changed by fused residual RMSNorm;
    # the same round's captured combine output is the actual input before mutation.
    fixture['layer_inputs']['hidden_states'] = fc['output']
    a.fixture.parent.mkdir(parents=True, exist_ok=True)
    torch.save(fixture, a.fixture)
    print(json.dumps({'status': 'registered', 'tp': world, 'fixture': str(a.fixture)}))


if __name__ == '__main__':
    main()
