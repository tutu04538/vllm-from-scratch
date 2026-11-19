"""公开 AWQ 的生产加载和格式转换抽层 CPU 对照，不执行 GEMM。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'benchmarks'))
from step114_download_public import MODELS
from minivllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from minivllm.layers.quantization import resolve_quantization_config
from minivllm.model_loader.post_load import process_weights_after_loading

MANIFESTS = Path('/data/syh/models/converted/_quant_data/step114_public_manifests')


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def load_sample(root, index, name):
    output = {}
    for suffix in ('qweight', 'qzeros', 'scales'):
        key = f'{name}.{suffix}'
        with safe_open(root / index[key], framework='pt', device='cpu') as source:
            tensor = source.get_slice(key)
            output[suffix] = tensor[:256, :2] if suffix == 'qweight' else tensor[:2, :2] if suffix == 'qzeros' else tensor[:2, :16]
    return output


def source_model(repo):
    path = MANIFESTS / (repo.replace('/', '--') + '.json')
    manifest = json.loads(path.read_text())
    if manifest['repo'] != repo or manifest['revision'] != MODELS[repo] or manifest['status'] != 'download_and_integrity_passed_gpu_not_validated':
        raise ValueError(f'公开模型身份不符: {repo}')
    root = Path(manifest['snapshot'])
    for name in ('config.json', 'model.safetensors.index.json'):
        if sha(root / name) != manifest['files'][name]['sha256']:
            raise ValueError(f'源文件 hash 不符: {repo}/{name}')
    config = json.loads((root / 'config.json').read_text())
    index = json.loads((root / 'model.safetensors.index.json').read_text())['weight_map']
    layers = [0, config['num_hidden_layers'] // 2, config['num_hidden_layers'] - 1]
    expected_names = [f'model.layers.{i}.{part}.{suffix}' for i in layers for part in
                      ('self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
                       'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj')
                      for suffix in ('qweight', 'qzeros', 'scales')]
    shards = {index[name] for name in expected_names}
    for filename in shards:
        expected = manifest['files'][filename]['hub_lfs_sha256']
        if not expected or sha(root / filename) != expected:
            raise ValueError(f'LFS SHA256 不符: {repo}/{filename}')
    return root, config, index, layers, sha(path)


def make_layer(kind, cfg):
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        if kind == 'qkv':
            return QKVParallelLinear(256, 8, 2, 2, prefix='qkv_proj', quant_config=cfg)
        if kind == 'gate_up':
            return MergedColumnParallelLinear(256, [16, 16], prefix='gate_up_proj', quant_config=cfg)
        return RowParallelLinear(256, 16, prefix=kind, quant_config=cfg)
    finally:
        torch.set_default_dtype(old)


def run_check():
    cases, groups, reports = [], [], []
    for repo in ('Qwen/Qwen3-8B-AWQ', 'Qwen/Qwen3-14B-AWQ'):
        root, hf, index, layer_indices, manifest_sha = source_model(repo)
        cfg = resolve_quantization_config(None, hf['quantization_config'])
        cfg.packed_modules_mapping.update({'qkv_proj': ['q_proj', 'k_proj', 'v_proj'],
                                            'gate_up_proj': ['gate_proj', 'up_proj']})
        for i in layer_indices:
            definitions = [('qkv', [('q', 'self_attn.q_proj'), ('k', 'self_attn.k_proj'), ('v', 'self_attn.v_proj')]),
                           ('o_proj', [(None, 'self_attn.o_proj')]),
                           ('gate_up', [(0, 'mlp.gate_proj'), (1, 'mlp.up_proj')]),
                           ('down_proj', [(None, 'mlp.down_proj')])]
            for kind, parts in definitions:
                layer = make_layer(kind, cfg)
                samples = []
                for shard, part in parts:
                    name = f'model.layers.{i}.{part}'
                    tensors = load_sample(root, index, name)
                    for key, tensor in tensors.items():
                        getattr(layer, key).weight_loader(getattr(layer, key), tensor, shard)
                    cases.append({'name': f'{repo}/{name}', 'qweight': tensors['qweight'].tolist(),
                                  'qzeros': tensors['qzeros'].tolist()})
                    samples.append((name, tensors))
                model = torch.nn.Sequential(layer)
                process_weights_after_loading(model, None, 'cpu')
                if model._quant_postload_state != 'representation_ready':
                    raise ValueError('表示审计模型状态不正确')
                groups.append((repo, i, kind, layer, samples, manifest_sha))
    with tempfile.TemporaryDirectory(prefix='step119-') as dirname:
        inp, out = Path(dirname) / 'input.json', Path(dirname) / 'output.json'
        inp.write_text(json.dumps({'cases': cases}))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        subprocess.run([sys.executable, str(ROOT / 'tests/step118/step118_frozen_probe.py'),
                        '--input', str(inp), '--output', str(out)], env=env, timeout=180, check=True)
        upstream = json.loads(out.read_text())
    iterator = iter(upstream['cases'])
    for repo, i, kind, layer, samples, manifest_sha in groups:
        for j, (name, original) in enumerate(samples):
            oracle = next(iterator)
            if oracle['name'] != f'{repo}/{name}':
                raise ValueError('上游对照输入顺序不一致')
            w = layer.qweight[:, j * 16:(j + 1) * 16]
            z = layer.qzeros[j * 2:(j + 1) * 2, :]
            scales = layer.scales[:, j * 16:(j + 1) * 16]
            checks = {'qweight_exact': torch.equal(w, torch.tensor(oracle['qweight'], dtype=torch.int32)),
                      'qzeros_exact': torch.equal(z, torch.tensor(oracle['qzeros'], dtype=torch.int32)),
                      'scales_exact': torch.equal(scales, original['scales']),
                      'scales_dtype': scales.dtype == original['scales'].dtype,
                      'standard_layout': all(getattr(layer, p).layout_state == 'standard' for p in ('qweight', 'qzeros', 'scales'))}
            if not all(checks.values()):
                raise ValueError(f'{repo}/{name}: {checks}')
            reports.append({'repo': repo, 'layer': i, 'kind': kind, 'name': name,
                            'manifest_sha256': manifest_sha,
                            'raw_qweight_sha256': tensor_sha(original['qweight']),
                            'raw_qzeros_sha256': tensor_sha(original['qzeros']),
                            'raw_scales_sha256': tensor_sha(original['scales']),
                            'scale_dtype': str(scales.dtype), 'checks': checks})
    return {'status': 'passed', 'scope': 'real_checkpoint_cpu_production_loader_slices_tp1_only',
            'logical_slice': {'k': [0, 256], 'n': [0, 16], 'group_size': 128},
            'upstream_source_hashes': upstream['source_hashes'], 'projections': reports}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--compare', action='store_true')
    parser.add_argument('--trace', action='store_true')
    parser.add_argument('--case', choices=['all'], default='all')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not any((args.check, args.compare, args.trace)):
        parser.error('select --check, --compare or --trace')
    if args.trace:
        result, code = {'status': 'not_run', 'reason': '性能测量留待后续'}, 2
    else:
        result, code = run_check(), 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'output': str(args.output),
                      'projection_count': len(result.get('projections', []))}))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
