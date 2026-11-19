"""CPU 身份与格式审计；不替代真实 vLLM 前向。"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open


def sha_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def check(path):
    path = Path(path)
    manifest = json.loads((path / 'conversion_manifest.json').read_text())
    if manifest['status'] != 'converted_not_validated_by_vllm':
        raise ValueError('unexpected conversion status')
    for name, meta in manifest['output_files'].items():
        file = path / name
        if not file.is_file() or file.stat().st_size != meta['bytes'] or sha_file(file) != meta['sha256']:
            raise ValueError(f'output hash mismatch: {name}')
    config = json.loads((path / 'config.json').read_text())
    qconfig = config['quantization_config']
    if qconfig['quant_method'] != 'fp8' or qconfig['activation_scheme'] != 'dynamic':
        raise ValueError('unexpected FP8 config')
    index = json.loads((path / 'model.safetensors.index.json').read_text())['weight_map']
    source = Path(manifest['source_path'])
    source_index = json.loads((source / 'model.safetensors.index.json').read_text())['weight_map']
    names = sorted(manifest['per_weight_scale'])
    if len(names) != manifest['scale_count'] or len(names) != config['num_hidden_layers'] * 7:
        raise ValueError('wrong number of FP8 projection weights')
    scale_names = {name.removesuffix('.weight') + '.weight_scale' for name in names}
    if set(index) != set(source_index) | scale_names:
        raise ValueError('output tensor list differs from source plus scales')
    for name in names:
        if index[name] != index[name.removesuffix('.weight') + '.weight_scale']:
            raise ValueError(f'weight/scale shard mismatch: {name}')
    shards = sorted(set(index.values()))
    for shard in shards:
        expected_keys = {name for name, current in index.items() if current == shard}
        with safe_open(str(path / shard), framework='pt', device='cpu') as converted:
            if set(converted.keys()) != expected_keys:
                raise ValueError(f'shard tensor/index mismatch: {shard}')
            for name in sorted(expected_keys & set(names)):
                scale_name = name.removesuffix('.weight') + '.weight_scale'
                scale = converted.get_tensor(scale_name)
                if (converted.get_slice(name).get_dtype() != 'F8_E4M3'
                        or scale.dtype != torch.float32 or scale.numel() != 1
                        or not torch.isfinite(scale).all() or scale.item() <= 0):
                    raise ValueError(f'invalid FP8 weight or scale: {name}')
                if scale.item() != manifest['per_weight_scale'][name]:
                    raise ValueError(f'scale differs from manifest: {name}')
    sampled = [f'model.layers.0.self_attn.q_proj.weight',
               f'model.layers.{config["num_hidden_layers"] - 1}.mlp.down_proj.weight']
    for name in sampled:
        with safe_open(str(source / source_index[name]), framework='pt', device='cpu') as original, \
             safe_open(str(path / index[name]), framework='pt', device='cpu') as converted:
            weight = original.get_tensor(name).float()
            q = converted.get_tensor(name)
            scale = converted.get_tensor(name.removesuffix('.weight') + '.weight_scale')
            maximum = weight.abs().max()
            expected_scale = (maximum / 448) if maximum != 0 else torch.tensor(1., dtype=torch.float32)
            torch.testing.assert_close(scale, expected_scale, rtol=0, atol=0)
            expected = (weight / expected_scale).clamp(-448, 448).to(torch.float8_e4m3fn)
            if not torch.equal(q.view(torch.uint8), expected.view(torch.uint8)):
                raise ValueError(f'FP8 bytes mismatch: {name}')
    for name in ('model.embed_tokens.weight', 'lm_head.weight'):
        with safe_open(str(source / source_index[name]), framework='pt', device='cpu') as original, \
             safe_open(str(path / index[name]), framework='pt', device='cpu') as converted:
            x = original.get_tensor(name)
            y = converted.get_tensor(name)
            if x.shape != y.shape or x.dtype != y.dtype or not torch.equal(x.flatten()[:1024], y.flatten()[:1024]):
                raise ValueError(f'non-quantized weight mismatch: {name}')
    return {'status': 'passed', 'kind': 'cpu_artifact_audit_only', 'path': str(path),
            'manifest_sha256': sha_file(path / 'conversion_manifest.json'),
            'source_revision': manifest['source_revision'], 'tensor_count': len(index),
            'scales': len(names), 'sampled_projection_weights': sampled,
            'real_vllm_load': 'not_run'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = check(args.artifact)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
