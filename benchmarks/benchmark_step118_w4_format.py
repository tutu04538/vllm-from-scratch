"""真实公开 checkpoint 的 W4 小块 CPU 对照；不执行性能测试。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests/step118'))
sys.path.insert(0, str(ROOT / 'benchmarks'))
import torch
from safetensors import safe_open
import step118_w4_reference as ref
from step114_download_public import MODELS

MANIFESTS = Path('/data/syh/models/converted/_quant_data/step114_public_manifests')


def sha_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def sample(manifest):
    item = json.loads(manifest.read_text())
    repo = item['repo']
    if repo not in MODELS or item['revision'] != MODELS[repo] or item['status'] != 'download_and_integrity_passed_gpu_not_validated':
        raise ValueError(f'公开模型身份无效: {repo}')
    root = Path(item['snapshot'])
    index_name = 'model.safetensors.index.json'
    if sha_file(root / index_name) != item['files'][index_name]['sha256']:
        raise ValueError(f'模型索引 hash 不匹配: {repo}')
    index = json.loads((root / index_name).read_text())['weight_map']
    names = {kind: f'model.layers.0.self_attn.q_proj.{kind}' for kind in ('qweight', 'qzeros', 'scales')}
    shards = {index[name] for name in names.values()}
    for shard in shards:
        expected = item['files'][shard]['hub_lfs_sha256']
        if expected is None or sha_file(root / shard) != expected:
            raise ValueError(f'模型分片 LFS hash 不匹配: {repo}/{shard}')
    awq = repo.endswith('-AWQ')
    tensors = {}
    for kind, name in names.items():
        with safe_open(root / index[name], framework='pt', device='cpu') as source:
            value = source.get_slice(name)
            if kind == 'qweight':
                tensors[kind] = value[:256, :2] if awq else value[:32, :16]
            elif kind == 'qzeros':
                tensors[kind] = value[:2, :2]
            else:
                tensors[kind] = value[:2, :16]
    return {'name': repo, 'format': 'awq' if awq else 'gptq',
            'qweight': tensors['qweight'].tolist(), 'qzeros': tensors['qzeros'].tolist(),
            'scales': tensors['scales'].float().tolist(), 'group_size': 128}, {
            'repo': repo, 'revision': item['revision'], 'manifest_sha256': sha_file(manifest),
            'index_sha256': item['files'][index_name]['sha256'],
            'source_shards_lfs_sha256': {x: item['files'][x]['hub_lfs_sha256'] for x in sorted(shards)},
            'tensor_names': names, 'qweight_shape': list(tensors['qweight'].shape),
            'qweight_stride': list(tensors['qweight'].stride()), 'qzeros_shape': list(tensors['qzeros'].shape),
            'qzeros_stride': list(tensors['qzeros'].stride()), 'scale_shape': list(tensors['scales'].shape),
            'scale_stride': list(tensors['scales'].stride()), 'scale_dtype': str(tensors['scales'].dtype),
            'logical_region': {'k': [0, 256], 'n': [0, 16], 'padding': 'none'}}


def run_real():
    paths = [MANIFESTS / (repo.replace('/', '--') + '.json') for repo in MODELS]
    cases, reports = [], []
    for path in paths:
        case, report = sample(path)
        cases.append(case); reports.append(report)
    with tempfile.TemporaryDirectory(prefix='step118-') as dirname:
        source, output = Path(dirname) / 'input.json', Path(dirname) / 'output.json'
        source.write_text(json.dumps({'cases': cases}))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        subprocess.run([sys.executable, str(ROOT / 'tests/step118/step118_frozen_probe.py'),
                        '--input', str(source), '--output', str(output)], env=env, check=True, timeout=120)
        reference = json.loads(output.read_text())
    for case, report, expected in zip(cases, reports, reference['cases'], strict=True):
        if case['name'] != expected['name']:
            raise ValueError('上游输出顺序不一致')
        w = torch.tensor(case['qweight'], dtype=torch.int32)
        z = torch.tensor(case['qzeros'], dtype=torch.int32)
        if case['format'] == 'awq':
            qw, qz, logical, zero = ref.awq_to_standard(w, z)
            checks = {'qweight_packed': torch.equal(qw, torch.tensor(expected['qweight'], dtype=torch.int32)),
                      'qzeros_packed': torch.equal(qz, torch.tensor(expected['qzeros'], dtype=torch.int32)),
                      'q': torch.equal(logical, torch.tensor(expected['q'], dtype=torch.int32)),
                      'zero': torch.equal(zero, torch.tensor(expected['z'], dtype=torch.int32))}
            scales = torch.tensor(case['scales'], dtype=torch.float32)
            fp32 = ref.dequantize(logical, scales, zero, group_size=128)
            checks['fp32_dequant'] = torch.equal(fp32, torch.tensor(expected['dequant'], dtype=torch.float32))
            report['standard_qweight_shape'] = list(qw.shape)
            report['standard_qzeros_shape'] = list(qz.shape)
        else:
            logical = ref.unpack_standard_rows(w)
            raw_zero = ref.unpack_standard_cols(z)
            checks = {'q': torch.equal(logical, torch.tensor(expected['q'], dtype=torch.int32)),
                      'zero_file_bits': torch.equal(raw_zero, torch.tensor(expected['z_file_raw'], dtype=torch.int32))}
            report['gptq_zero_runtime_semantics'] = 'not_interpreted_step127'
        if not all(checks.values()):
            raise ValueError(f"{case['name']} 对照失败: {checks}")
        report['checks'] = checks
    return {'status': 'passed', 'scope': 'cpu_public_checkpoint_subtensor_only',
            'vllm_source_hashes': reference['source_hashes'], 'models': reports}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--compare', action='store_true')
    parser.add_argument('--trace', action='store_true')
    parser.add_argument('--case', choices=['all', 'real'], default='all')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not any((args.check, args.compare, args.trace)):
        parser.error('select --check, --compare or --trace')
    if args.trace:
        result = {'status': 'not_run', 'reason': 'performance profiling deferred'}
        code = 2
    else:
        result = run_real()
        code = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'output': str(args.output)}))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
