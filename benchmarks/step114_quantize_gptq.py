"""隔离 GPTQModel 环境中量化固定 Qwen3；必须由忙卡检查的启动器调用。"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import tempfile

import torch
from gptqmodel import GPTQModel, QuantizeConfig
from transformers import AutoTokenizer

from step114_convert_fp8 import SOURCES, sha_file
from step114_quantizer_common import prepare_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=sorted(SOURCES), required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, default=Path('/data/syh/models/converted/Qwen'))
    args = parser.parse_args()
    if importlib.metadata.version('gptqmodel') != '5.6.0' or importlib.metadata.version('transformers') != '4.57.6':
        raise ValueError('offline tool versions differ from the preregistered recipe')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('GPTQModel 需要启动器指定恰好一个空闲 GPU')
    inputs = prepare_inputs(args.model, args.data)
    source, name, revision, samples = (inputs[k] for k in ('source', 'name', 'revision', 'samples'))
    recipe = {'method': 'gptqmodel', 'tool_version': '5.6.0', 'source_archive_sha256':
              '44ed55a51fc63e699236323d4fb2a36831024ee317ef30846d5cd1d5d20d54ff',
              'bits': 4, 'group_size': 128, 'sym': True, 'desc_act': False, 'lm_head': False,
              'format': 'gptq', 'pack_dtype': 'int32', 'device': 'cuda:0',
              'source_and_calibration_dtype': 'bfloat16',
              'offload_to_disk': True, 'batch_size': 1, 'calibration_sort': None,
              'calibration_count': 128, 'calibration_tokens': 512,
              'data_manifest_sha256': inputs['data_manifest_sha256'],
              'calibration_sha256': inputs['calibration_sha256'],
              'derived_calibration_sha256': inputs['derived_calibration_sha256'],
              'calibration_derivation': 'first 512 tokens of each frozen training window', 'seed': 0,
              'script_sha256': sha_file(Path(__file__)),
              'common_script_sha256': sha_file(Path(__file__).with_name('step114_quantizer_common.py'))}
    digest = hashlib.sha256(json.dumps(recipe, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    parent = args.output_root / name / revision
    destination = parent / ('gptq-' + digest[:16])
    if destination.exists():
        raise FileExistsError(destination)
    parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    random.seed(0)
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, trust_remote_code=False)
    with tempfile.TemporaryDirectory(prefix='.gptq-preparing-', dir=parent) as dirname:
        tmp = Path(dirname)
        offload = tmp / 'offload'
        offload.mkdir()
        config = QuantizeConfig(bits=4, group_size=128, sym=True, desc_act=False,
                                lm_head=False, device='cuda:0', offload_to_disk=True,
                                offload_to_disk_path=str(offload))
        # GPTQModel 5.6 明确将未量化模型先加载到 CPU，逐层调度到 quantization device。
        model = GPTQModel.load(str(source), config, trust_remote_code=False, dtype=torch.bfloat16)
        data = [{'input_ids': row['token_ids'], 'attention_mask': [1] * len(row['token_ids'])}
                for row in samples]
        model.quantize(data, tokenizer=tokenizer, batch_size=1, calibration_sort=None)
        model.save(str(tmp), max_shard_size='5GB')
        tokenizer.save_pretrained(str(tmp))
        # 仅清理本次独立临时目录内的 offload 中间文件。
        import shutil
        if offload.exists():
            shutil.rmtree(offload)
        files = {p.name: {'sha256': sha_file(p), 'bytes': p.stat().st_size}
                 for p in sorted(tmp.iterdir()) if p.is_file()}
        manifest = {'status': 'quantized_not_validated_by_vllm', 'source_repo': 'Qwen/' + name,
                    'source_revision': revision, 'source_config_sha256': sha_file(source / 'config.json'),
                    'source_lfs_manifest_sha256': inputs['source_lfs_manifest_sha256'], 'recipe': recipe,
                    'recipe_sha256': digest, 'output_files': files,
                    'tool_versions': {key: importlib.metadata.version(key) for key in
                                      ('gptqmodel', 'torch', 'transformers', 'triton', 'accelerate', 'datasets')}}
        (tmp / 'conversion_manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
        os.rename(tmp, destination)
    print(json.dumps({'status': 'quantized_not_validated_by_vllm', 'path': str(destination)}))


if __name__ == '__main__':
    main()
