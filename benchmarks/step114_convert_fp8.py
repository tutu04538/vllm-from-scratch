"""CPU 离线逐张量 FP8 转换；输出必须经冻结 vLLM 验收后才能标 ready。"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HUB = Path('/data/syh/.cache/huggingface/hub')
SOURCES = {'8B': 'b968826d9c46dd6066d109eabc6255188de91218',
           '14B': '40c069824f4251a91eefaf281ebe4c544efd3e18'}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LFS_MANIFEST = PROJECT_ROOT / 'benchmarks/results/step090/source_lfs_verification_full.json'
SOURCE_CONTRACT = PROJECT_ROOT / 'learning_notes/14_vllm_from_scratch/量化完整需求/参考资料/模型与量化产物契约.json'
PROJ_SUFFIXES = tuple(f'.{name}.weight' for name in
                      ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'))


def sha_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def fp8_quantize(weight):
    """q=cast_e4m3fn(clamp(w/s)), s=max(abs(w))/448；全零张量的 s=1。"""
    if weight.dtype != torch.bfloat16 or weight.device.type != 'cpu' or weight.ndim != 2:
        raise ValueError('FP8 conversion requires a 2D CPU BF16 linear weight')
    f32 = weight.float()
    if not torch.isfinite(f32).all():
        raise ValueError('non-finite source weight')
    amax = float(f32.abs().max())
    scale = torch.tensor(amax / 448.0 if amax else 1.0, dtype=torch.float32)
    quantized = torch.clamp(f32 / scale, -448, 448).to(torch.float8_e4m3fn).contiguous()
    return quantized, scale


def is_linear_weight(name):
    return name.startswith('model.layers.') and name.endswith(PROJ_SUFFIXES)


def convert(size, out_root):
    revision = SOURCES[size]
    repo_name = f'Qwen3-{size}'
    source = HUB / f'models--Qwen--{repo_name}' / 'snapshots' / revision
    index_path = source / 'model.safetensors.index.json'
    config_path = source / 'config.json'
    if not index_path.is_file() or not config_path.is_file():
        raise FileNotFoundError(f'missing fixed source snapshot: {source}')
    source_identity = json.loads(LFS_MANIFEST.read_text())
    source_entry = source_identity['models'][size.lower()]
    if (source_identity.get('status') != 'passed' or not source_entry.get('passed')
            or source_entry['repo'] != 'Qwen/' + repo_name or source_entry['revision'] != revision):
        raise ValueError('source LFS verification manifest identity mismatch')
    source_contract = json.loads(SOURCE_CONTRACT.read_text())
    contract = next(row for row in source_contract['sources'] if row['repo'] == 'Qwen/' + repo_name)
    if contract['revision'] != revision:
        raise ValueError('source contract revision mismatch')
    for item in contract['metadata']:
        if sha_file(source / item['name']) != item['sha256']:
            raise ValueError(f'source metadata hash mismatch: {item["name"]}')
    source_index = json.loads(index_path.read_text())
    source_config = json.loads(config_path.read_text())
    if source_config.get('quantization_config') is not None:
        raise ValueError('source checkpoint is already quantized')
    mapping = source_index['weight_map']
    expected_count = source_config['num_hidden_layers'] * len(PROJ_SUFFIXES)
    names = [name for name in mapping if is_linear_weight(name)]
    if len(names) != expected_count:
        raise ValueError(f'expected {expected_count} logical projection weights, found {len(names)}')
    by_shard = defaultdict(set)
    for name, shard in mapping.items():
        if Path(shard).name != shard or not shard.endswith('.safetensors'):
            raise ValueError(f'invalid source shard name: {shard}')
        by_shard[shard].add(name)
    if set(by_shard) != set(source_entry['files']):
        raise ValueError('source shard list differs from verified LFS manifest')
    recipe = {'method': 'fp8', 'weight_dtype': 'float8_e4m3fn', 'activation_scheme': 'dynamic',
              'weight_scale': 'one float32 dequant multiplier per logical 2D projection',
              'conversion': 's=max(abs(w.float()))/448; zero s=1; q=cast_fp8(clamp(w/s,-448,448))',
              'source_dtype': 'bfloat16', 'nonlinear_layers': 'retain original bytes and dtype',
              'projection_suffixes': list(PROJ_SUFFIXES), 'script_sha256': sha_file(Path(__file__))}
    recipe_bytes = json.dumps(recipe, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
    recipe_hash = hashlib.sha256(recipe_bytes).hexdigest()
    parent = Path(out_root) / repo_name / revision
    destination = parent / ('fp8-' + recipe_hash[:16])
    if destination.exists():
        raise FileExistsError(f'refusing to overwrite {destination}')
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.fp8-preparing-', dir=parent) as dirname:
        tmp = Path(dirname)
        new_mapping = {}
        output_bytes = 0
        source_shards = {}
        scales = {}
        for shard in sorted(by_shard):
            file = source / shard
            if not file.is_file():
                raise FileNotFoundError(file)
            source_shards[shard] = {'sha256': sha_file(file), 'bytes': file.stat().st_size}
            if source_shards[shard]['sha256'] != source_entry['files'][shard]:
                raise ValueError(f'source LFS hash mismatch: {shard}')
            converted = {}
            with safe_open(str(file), framework='pt', device='cpu') as src:
                actual = set(src.keys())
                if actual != by_shard[shard]:
                    raise ValueError(f'{shard} index/tensor mismatch: {sorted(actual ^ by_shard[shard])[:10]}')
                for name in sorted(actual):
                    value = src.get_tensor(name)
                    if is_linear_weight(name):
                        q, scale = fp8_quantize(value)
                        scale_name = name.removesuffix('.weight') + '.weight_scale'
                        if scale_name in mapping or scale_name in converted:
                            raise ValueError(f'duplicate scale: {scale_name}')
                        converted[name], converted[scale_name] = q, scale
                        new_mapping[scale_name] = shard
                        scales[name] = float(scale)
                    else:
                        converted[name] = value.contiguous()
                    new_mapping[name] = shard
                output_bytes += sum(v.numel() * v.element_size() for v in converted.values())
                save_file(converted, str(tmp / shard), metadata={'format': 'pt'})
            print(f'{repo_name}: converted {shard}', flush=True)
            del converted
        if set(names) != set(scales):
            raise ValueError('not all logical projections received a scale')
        for path in source.iterdir():
            if path.is_file() and not path.name.endswith('.safetensors') and path.name not in (
                    'config.json', 'model.safetensors.index.json'):
                shutil.copy2(path, tmp / path.name)
        source_config['quantization_config'] = {'quant_method': 'fp8', 'activation_scheme': 'dynamic',
                                                'ignored_layers': [], 'weight_block_size': None}
        (tmp / 'config.json').write_text(json.dumps(source_config, ensure_ascii=False, indent=2) + '\n')
        (tmp / 'model.safetensors.index.json').write_text(json.dumps(
            {'metadata': {'total_size': output_bytes}, 'weight_map': new_mapping}, indent=2) + '\n')
        files = {path.name: {'sha256': sha_file(path), 'bytes': path.stat().st_size}
                 for path in sorted(tmp.iterdir()) if path.is_file()}
        manifest = {'status': 'converted_not_validated_by_vllm', 'source_repo': 'Qwen/' + repo_name,
                    'source_revision': revision, 'source_path': str(source),
                    'source_config_sha256': sha_file(config_path),
                    'source_index_sha256': sha_file(index_path), 'source_shards': source_shards,
                    'source_lfs_manifest_sha256': sha_file(LFS_MANIFEST),
                    'recipe': recipe, 'recipe_sha256': recipe_hash, 'torch_version': str(torch.__version__),
                    'scale_count': len(scales), 'per_weight_scale': scales,
                    'output_tensor_bytes': output_bytes, 'output_files': files}
        (tmp / 'conversion_manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
        os.rename(tmp, destination)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=sorted(SOURCES), required=True)
    parser.add_argument('--output-root', type=Path, default=Path('/data/syh/models/converted/Qwen'))
    args = parser.parse_args()
    destination = convert(args.model, args.output_root)
    print(json.dumps({'status': 'converted_not_validated_by_vllm', 'path': str(destination)}))


if __name__ == '__main__':
    main()
