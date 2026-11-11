"""独立审计两份真实 draft 的身份、转换清单及词表映射。"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.check_step090_models import audit_converted, audit_eagle_mapping, file_hash, resolve_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    results = {}
    root = Path('/data/syh/models/converted/AngelSlim')
    for key, folder, revision in [('8b-draft', 'Qwen3-8B_eagle3', '9629dfce7a4a10564dd48d3e5485c3976095653c'),
                                  ('14b-draft', 'Qwen3-14B_eagle3', '3d13517724e81cb409ddf1d4650772ec52f1e18e')]:
        source, config = resolve_snapshot(key)
        converted = root/folder/revision
        manifest = json.loads((converted/'conversion_manifest.json').read_text())
        if file_hash(source/'pytorch_model.bin') != manifest['source_sha256']:
            raise ValueError(f'{key} 原始权重 hash 不符')
        if file_hash(converted/'model.safetensors') != manifest['converted_sha256']:
            raise ValueError(f'{key} 转换权重 hash 不符')
        if file_hash(converted/'config.json') != manifest['config_sha256']:
            raise ValueError(f'{key} 配置 hash 不符')
        audit_converted(converted/'model.safetensors', manifest['tensors'])
        mapping = audit_eagle_mapping(converted/'model.safetensors', config)
        results[key] = {'source': str(source), 'converted': str(converted), 'revision': revision,
                        'architecture': config['architectures'], 'hidden_size': config['hidden_size'],
                        'source_sha256': manifest['source_sha256'], 'converted_sha256': manifest['converted_sha256'],
                        'manifest_sha256': file_hash(converted/'conversion_manifest.json'),
                        'tensors': manifest['tensors'], 'mapping': mapping}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'status': 'passed', 'models': results}, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'status': 'passed', 'models': list(results)}))


if __name__ == '__main__':
    main()
