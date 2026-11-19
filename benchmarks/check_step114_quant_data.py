"""验证 step114 数据产物的文件身份、完整 token 和训练/评测分离。"""
import argparse
import hashlib
import json
from pathlib import Path
import struct


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def ids_hash(ids):
    return hashlib.sha256(struct.pack(f'<{len(ids)}I', *ids)).hexdigest()


def check(root):
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest['kind'] != 'data_preparation_only':
        raise ValueError('wrong artifact kind')
    for name, item in manifest['sources'].items():
        path = Path(item['path'])
        if not path.is_file() or sha(path) != item['sha256']:
            raise ValueError(f'source mismatch: {name}')
    expected_targets = {
        'Qwen3-8B': 'b968826d9c46dd6066d109eabc6255188de91218',
        'Qwen3-14B': '40c069824f4251a91eefaf281ebe4c544efd3e18',
    }
    if set(manifest['targets']) != set(expected_targets):
        raise ValueError('missing or unexpected target')
    report = {}
    for name, target in manifest['targets'].items():
        if target['source_revision'] != expected_targets[name]:
            raise ValueError(f'wrong source revision: {name}')
        for filename, expected in target['files'].items():
            if sha(root / name / filename) != expected:
                raise ValueError(f'artifact file mismatch: {name}/{filename}')
        train = json.loads((root / name / 'calibration.json').read_text())
        test = json.loads((root / name / 'heldout.json').read_text())
        gsm = json.loads((root / name / 'gsm8k.json').read_text())
        if len(train) != 128 or len(test) != target['heldout_windows'] or len(gsm) != 256:
            raise ValueError(f'unexpected sample counts: {name}')
        for split, samples in [('train', train), ('test', test)]:
            for row in samples:
                if len(row['token_ids']) != 2048 or ids_hash(row['token_ids']) != row['sha256_uint32_le']:
                    raise ValueError(f'invalid token window: {name}/{split}/{row["id"]}')
        a = {r['sha256_uint32_le'] for r in train}
        b = {r['sha256_uint32_le'] for r in test}
        if a & b or len(b) != len(test):
            raise ValueError(f'train/test overlap or duplicate evaluation window: {name}')
        if [row['id'] for row in gsm] != [f'main/test:{i}' for i in range(256)]:
            raise ValueError(f'GSM8K sample IDs mismatch: {name}')
        for row in gsm:
            if ids_hash(row['prompt_token_ids']) != row['prompt_sha256_uint32_le']:
                raise ValueError(f'GSM8K prompt mismatch: {row["id"]}')
        tokenizer = Path(target['tokenizer_directory'])
        for filename, expected in target['tokenizer_files'].items():
            if sha(tokenizer / filename) != expected:
                raise ValueError(f'tokenizer mismatch: {name}/{filename}')
        report[name] = {'calibration_windows': len(train), 'heldout_windows': len(test),
                        'gsm8k': len(gsm), 'train_test_intersection': 0}
    return {'status': 'passed', 'kind': 'data_integrity_only', 'manifest_sha256': sha(root / 'manifest.json'),
            'targets': report}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(check(args.data), indent=2))


if __name__ == '__main__':
    main()
