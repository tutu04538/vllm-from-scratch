"""使用隔离 AutoAWQ 环境量化固定 Qwen3；必须由忙卡检查的启动器调用。"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import tempfile

import torch
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

from step114_convert_fp8 import SOURCES, sha_file
from step114_quantizer_common import prepare_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=sorted(SOURCES), required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, default=Path('/data/syh/models/converted/Qwen'))
    args = parser.parse_args()
    if importlib.metadata.version('autoawq') != '0.2.9' or importlib.metadata.version('transformers') != '4.51.3':
        raise ValueError('offline tool versions differ from the preregistered recipe')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('AutoAWQ 需要启动器指定恰好一个空闲 GPU')
    inputs = prepare_inputs(args.model, args.data)
    source, name, revision, samples = (inputs[k] for k in ('source', 'name', 'revision', 'samples'))
    recipe = {'method': 'autoawq', 'tool_version': '0.2.9', 'source_archive_sha256':
              'f3397687118c7d23c769de0de05ff1dca8950a2644acbe1b6eb1399c691fe7ee',
              'w_bit': 4, 'q_group_size': 128, 'zero_point': True, 'version': 'GEMM',
              'source_and_calibration_dtype': 'bfloat16', 'autoawq_apply_quant_weight_dtype': 'float16',
              'inference_activation_target': 'bfloat16', 'device_map': 'cpu',
              'duo_scaling': True, 'apply_clip': True,
              'max_calib_samples': 128, 'max_calib_seq_len': 512, 'n_parallel_calib_samples': 1,
              'data_manifest_sha256': inputs['data_manifest_sha256'],
              'calibration_sha256': inputs['calibration_sha256'],
              'derived_calibration_sha256': inputs['derived_calibration_sha256'],
              'calibration_derivation': 'first 512 tokens of each frozen training window', 'seed': 0,
              'script_sha256': sha_file(Path(__file__)),
              'common_script_sha256': sha_file(Path(__file__).with_name('step114_quantizer_common.py'))}
    digest = hashlib.sha256(json.dumps(recipe, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    parent = args.output_root / name / revision
    destination = parent / ('awq-' + digest[:16])
    if destination.exists():
        raise FileExistsError(destination)
    parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    random.seed(0)
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, trust_remote_code=False)
    model = AutoAWQForCausalLM.from_pretrained(
        str(source), torch_dtype=torch.bfloat16, trust_remote_code=False, device_map='cpu',
        low_cpu_mem_usage=True, use_cache=False, local_files_only=True)
    with tempfile.TemporaryDirectory(prefix='.awq-preparing-', dir=parent) as dirname:
        tmp = Path(dirname)
        model.quantize(tokenizer, quant_config={'zero_point': True, 'q_group_size': 128,
                       'w_bit': 4, 'version': 'GEMM'}, calib_data=[r['token_ids'] for r in samples],
                       duo_scaling=True, apply_clip=True, n_parallel_calib_samples=1,
                       max_calib_samples=128, max_calib_seq_len=512)
        model.save_quantized(str(tmp), safetensors=True, shard_size='5GB')
        tokenizer.save_pretrained(str(tmp))
        files = {p.name: {'sha256': sha_file(p), 'bytes': p.stat().st_size}
                 for p in sorted(tmp.iterdir()) if p.is_file()}
        manifest = {'status': 'quantized_not_validated_by_vllm', 'source_repo': 'Qwen/' + name,
                    'source_revision': revision, 'source_config_sha256': sha_file(source / 'config.json'),
                    'source_lfs_manifest_sha256': inputs['source_lfs_manifest_sha256'], 'recipe': recipe,
                    'recipe_sha256': digest, 'output_files': files,
                    'tool_versions': {key: importlib.metadata.version(key) for key in
                                      ('autoawq', 'torch', 'transformers', 'triton', 'accelerate', 'datasets')}}
        (tmp / 'conversion_manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
        os.rename(tmp, destination)
    print(json.dumps({'status': 'quantized_not_validated_by_vllm', 'path': str(destination)}))


if __name__ == '__main__':
    main()
