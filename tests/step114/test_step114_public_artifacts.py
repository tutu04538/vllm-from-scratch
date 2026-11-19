"""公开权重身份检查：模型、格式、tokenizer 和内容不一致均拒绝。"""
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'benchmarks'))
from benchmark_step114_vllm_load import check_public_artifact
from step114_download_public import MODELS


def test_public_artifact_identity_and_tampering(tmp_path, monkeypatch):
    import check_step114_public_tensors
    monkeypatch.setattr(check_step114_public_tensors, "check", lambda _: {"status": "test"})
    repo = 'Qwen/Qwen3-8B-AWQ'
    payload = b'tokenizer'
    (tmp_path / 'tokenizer.json').write_bytes(payload)
    sha = hashlib.sha256(payload).hexdigest()
    manifest = {'status': 'download_and_integrity_passed_gpu_not_validated', 'repo': repo,
                'revision': MODELS[repo], 'snapshot': str(tmp_path),
                'files': {'tokenizer.json': {'bytes': len(payload), 'sha256': sha, 'hub_lfs_sha256': sha}}}
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest))
    assert check_public_artifact(tmp_path, path, 'awq', sha, 'Qwen3-8B')['status'] == 'passed_file_integrity'
    with pytest.raises(ValueError, match='model or format'):
        check_public_artifact(tmp_path, path, 'gptq', sha, 'Qwen3-8B')
    with pytest.raises(ValueError, match='tokenizer'):
        check_public_artifact(tmp_path, path, 'awq', 'wrong', 'Qwen3-8B')
    (tmp_path / 'tokenizer.json').write_bytes(b'altered!!')
    with pytest.raises(ValueError, match='file mismatch'):
        check_public_artifact(tmp_path, path, 'awq', sha, 'Qwen3-8B')


def test_tensor_inventory_rejects_missing_quantization_parts(tmp_path):
    import torch
    from safetensors.torch import save_file
    from check_step114_public_tensors import check

    tensors = {'model.layer.qweight': torch.ones(1, dtype=torch.int32),
               'model.layer.scales': torch.ones(1)}
    save_file(tensors, str(tmp_path / 'model.safetensors'))
    (tmp_path / 'model.safetensors.index.json').write_text(json.dumps(
        {'weight_map': {key: 'model.safetensors' for key in tensors}}))
    with pytest.raises(ValueError, match='missing packed'):
        check(tmp_path)
    tensors['model.layer.qzeros'] = torch.ones(1, dtype=torch.int32)
    save_file(tensors, str(tmp_path / 'model.safetensors'))
    (tmp_path / 'model.safetensors.index.json').write_text(json.dumps(
        {'weight_map': {key: 'model.safetensors' for key in tensors}}))
    assert check(tmp_path)['quantized_projection_count'] == 1
