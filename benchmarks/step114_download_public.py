"""下载固定 revision 的公开量化模型并逐文件对账，不修改 Hub 快照。"""
import hashlib
import json
from pathlib import Path
import sys

from huggingface_hub import HfApi, snapshot_download

MODELS = {
    'Qwen/Qwen3-8B-AWQ': '4da05a8edb55c6046cce958586c33b61da07bb79',
    'Qwen/Qwen3-14B-AWQ': '31c69efc29464b6bb0aee1398b5a7b50a99340c3',
    'JunHowie/Qwen3-8B-GPTQ-Int4': 'f08c8ea5f60e1fe34ff4a45edb58e8ca958560c3',
    'JunHowie/Qwen3-14B-GPTQ-Int4': '5073a3e15216fb4facc39c953934910dbf0df51b',
}
OUT = Path('/data/syh/models/converted/_quant_data/step114_public_manifests')


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as src:
        for chunk in iter(lambda: src.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    name = sys.argv[1]
    revision = MODELS[name]
    info = HfApi().model_info(name, revision=revision, files_metadata=True)
    if info.sha != revision:
        raise ValueError('Hub revision mismatch')
    root = Path(snapshot_download(name, revision=revision, max_workers=4))
    files = {}
    for item in info.siblings:
        path = root / item.rfilename
        if not path.is_file() or path.stat().st_size != item.size:
            raise ValueError(f'missing file or size mismatch: {path}')
        digest = sha256(path)
        lfs = item.lfs.sha256 if item.lfs else None
        if lfs and digest != lfs:
            raise ValueError(f'LFS hash mismatch: {path}')
        files[item.rfilename] = {'bytes': item.size, 'sha256': digest, 'hub_lfs_sha256': lfs}
    config = json.loads((root / 'config.json').read_text())
    quant = config['quantization_config']
    method = 'awq' if name.endswith('AWQ') else 'gptq'
    if quant.get('quant_method') != method or quant.get('bits') != 4 or quant.get('group_size') != 128:
        raise ValueError('unexpected quantization configuration')
    if method == 'awq' and (quant.get('zero_point') is not True or quant.get('version', '').lower() != 'gemm'):
        raise ValueError('unexpected AWQ configuration')
    if method == 'gptq' and (quant.get('sym') is not True or quant.get('desc_act') is not False):
        raise ValueError('unexpected GPTQ configuration')
    report = {'status': 'download_and_integrity_passed_gpu_not_validated', 'repo': name,
              'revision': revision, 'snapshot': str(root), 'files': files,
              'quantization_config': quant, 'source_bf16_revision': 'unknown',
              'calibration_data_and_recipe': 'not independently established'}
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / (name.replace('/', '--') + '.json')
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'status': report['status'], 'manifest': str(target)}), flush=True)


if __name__ == '__main__':
    main()
