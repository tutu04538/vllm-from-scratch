"""AWQ/GPTQ 量化工具共享的源模型和校准样本身份检查。"""
import json
import hashlib
import struct

from step114_convert_fp8 import HUB, LFS_MANIFEST, SOURCES, SOURCE_CONTRACT, sha_file

DATA_MANIFEST_SHA256 = '3b2799bf8bc959e70616793fe4b038467ff3c81f77b338d6593f99c1ba6d1d0e'


def prepare_inputs(size, data):
    revision = SOURCES[size]
    name = f'Qwen3-{size}'
    source = HUB / f'models--Qwen--{name}' / 'snapshots' / revision
    lfs = json.loads(LFS_MANIFEST.read_text())
    entry = lfs['models'][size.lower()]
    if lfs['status'] != 'passed' or not entry['passed'] or entry['revision'] != revision or entry['repo'] != 'Qwen/' + name:
        raise ValueError('source LFS manifest identity mismatch')
    contract = json.loads(SOURCE_CONTRACT.read_text())
    item = next(row for row in contract['sources'] if row['repo'] == 'Qwen/' + name)
    if item['revision'] != revision:
        raise ValueError('source contract revision mismatch')
    for file in item['metadata']:
        if sha_file(source / file['name']) != file['sha256']:
            raise ValueError(f'source metadata mismatch: {file["name"]}')
    for file, expected in entry['files'].items():
        if sha_file(source / file) != expected:
            raise ValueError(f'source weight mismatch: {file}')
    manifest_path = data / 'manifest.json'
    if sha_file(manifest_path) != DATA_MANIFEST_SHA256:
        raise ValueError('calibration data manifest identity mismatch')
    data_manifest = json.loads(manifest_path.read_text())
    if data_manifest['targets'][name]['source_revision'] != revision:
        raise ValueError('calibration dataset does not belong to source revision')
    data_file = data / name / 'calibration.json'
    if sha_file(data_file) != data_manifest['targets'][name]['files']['calibration.json']:
        raise ValueError('calibration dataset hash mismatch')
    samples = json.loads(data_file.read_text())
    if len(samples) != 128 or any(len(row['token_ids']) != 2048 for row in samples):
        raise ValueError('calibration requires exactly 128 x 2048 tokens')
    samples, derived_sha256 = derive_calibration(samples)
    return {'source': source, 'name': name, 'revision': revision, 'samples': samples,
            'derived_calibration_sha256': derived_sha256,
            'data_manifest_sha256': sha_file(manifest_path), 'calibration_sha256': sha_file(data_file),
            'source_lfs_manifest_sha256': sha_file(LFS_MANIFEST)}


def derive_calibration(samples):
    """从冻结训练窗口确定性派生 128 条短校准序列。"""
    if len(samples) != 128 or any(len(row['token_ids']) != 2048 for row in samples):
        raise ValueError('source calibration requires exactly 128 x 2048 tokens')
    derived = [{'id': row['id'], 'token_ids': row['token_ids'][:512]} for row in samples]
    for row in derived:
        row['sha256_uint32_le'] = hashlib.sha256(struct.pack('<512I', *row['token_ids'])).hexdigest()
    payload = json.dumps(derived, sort_keys=True, separators=(',', ':')).encode()
    return derived, hashlib.sha256(payload).hexdigest()
