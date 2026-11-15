"""Audit catches installed-vLLM imports and hashes path/name as well as contents."""
import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location('step112_audit', Path(__file__).resolve().parents[2]/'benchmarks/check_step112_acceptance.py')
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def test_production_does_not_import_upstream():
    assert audit.upstream_imports() == []


def test_import_audit_detects_static_and_dynamic_but_excludes_testing(tmp_path):
    (tmp_path/'a.py').write_text('import vllm\nfrom vllm.v1 import worker\nx = __import__("vllm.model_executor")\n')
    (tmp_path/'testing').mkdir(); (tmp_path/'testing'/'reference.py').write_text('import vllm\n')
    assert [x['module'] for x in audit.upstream_imports(tmp_path)] == ['vllm', 'vllm.v1', 'vllm.model_executor']


def test_package_hash_includes_path_and_ignores_non_python(tmp_path):
    a = tmp_path/'a.py'; a.write_text('x = 1\n')
    first = audit.package_hash(tmp_path)
    (tmp_path/'data.txt').write_text('ignored')
    assert audit.package_hash(tmp_path) == first
    a.rename(tmp_path/'b.py')
    assert audit.package_hash(tmp_path) != first


def test_recorded_current_package_hash_matches():
    import json
    root = Path(__file__).resolve().parents[2]
    assert audit.package_hash() == json.loads((root/'docs/results.json').read_text())['current_package']['sha256']


def test_model_audit_detects_modified_or_missing_weights(tmp_path):
    import hashlib
    root = tmp_path/'revision'; root.mkdir()
    (root/'model.safetensors').write_bytes(b'weights')
    (root/'tokenizer.json').write_bytes(b'tokenizer')
    digest = lambda value: hashlib.sha256(value).hexdigest()
    identity = {'revision':'revision', 'local_path':str(root),
                'weights_sha256':{'model.safetensors':digest(b'weights')},
                'tokenizer_sha256':digest(b'tokenizer')}
    assert all(item['passed'] for item in audit.audit_model_identity(identity))
    (root/'model.safetensors').write_bytes(b'changed')
    assert not all(item['passed'] for item in audit.audit_model_identity(identity))
    (root/'tokenizer.json').unlink()
    assert audit.audit_model_identity(identity)[-1]['actual'] is None


def test_draft_manifest_must_match_weights_and_revision(tmp_path):
    import hashlib
    import json
    digest = lambda data: hashlib.sha256(data).hexdigest()
    source = tmp_path/'rev'; source.mkdir(); converted = tmp_path/'converted'; converted.mkdir()
    (source/'pytorch_model.bin').write_bytes(b'source')
    (converted/'model.safetensors').write_bytes(b'converted')
    (converted/'config.json').write_bytes(b'config')
    manifest = {'identity':{'repo':'test/model', 'revision':'rev'},
                'source_sha256':digest(b'source'), 'converted_sha256':digest(b'converted'),
                'config_sha256':digest(b'config')}
    path = converted/'conversion_manifest.json'; path.write_text(json.dumps(manifest))
    identity = {'repo':'test/model', 'revision':'rev', 'local_path':str(source), 'converted_path':str(converted),
                'source_sha256':digest(b'source'), 'converted_sha256':digest(b'converted'), 'conversion_manifest':str(path)}
    assert all(item['passed'] for item in audit.audit_model_identity(identity, draft=True))
    manifest['identity']['revision'] = 'wrong'; path.write_text(json.dumps(manifest))
    assert not all(item['passed'] for item in audit.audit_model_identity(identity, draft=True))


def test_gpu_occupancy_uses_selected_devices_and_restores_environment(monkeypatch):
    import os
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '3,2')
    seen = []
    def probe(count):
        seen.append((count, os.environ['CUDA_VISIBLE_DEVICES']))
        return {92, 17}
    monkeypatch.setattr(audit, 'compute_pids', probe)
    assert audit.selected_gpu_processes('0,1') == [17, 92]
    assert seen == [(2, '0,1')]
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '3,2'


def test_gpu_occupancy_restores_environment_on_error(monkeypatch):
    import os
    import pytest
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    def probe(_count):
        raise ValueError('cannot inspect GPU')
    monkeypatch.setattr(audit, 'compute_pids', probe)
    with pytest.raises(ValueError, match='cannot inspect GPU'):
        audit.selected_gpu_processes('0')
    assert 'CUDA_VISIBLE_DEVICES' not in os.environ


def test_regressions_must_match_current_package_and_all_groups():
    result = {'status': 'passed_regressions', 'package_sha256': 'source-a',
              'package_sha256_end': 'source-a',
              'runs': [{'name': name, 'exit_code': 0}
                       for name in ('pytest', 'legacy_final', 'parallel_final')]}
    assert audit.regressions_match_package(result, 'source-a')
    assert not audit.regressions_match_package(result, 'source-b')
    result['runs'][2]['name'] = 'legacy_final'
    assert not audit.regressions_match_package(result, 'source-a')


def test_regressions_use_all_gpus_for_parallel_even_with_single_gpu_environment(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    (tmp_path/'tests/step089').mkdir(parents=True)
    (tmp_path/'tests/step111').mkdir(parents=True)
    groups={name:(paths,devices) for name,paths,devices in audit.regression_groups(tmp_path)}
    assert groups['legacy_final'][1]=='0'
    assert groups['pytest'][1]==groups['parallel_final'][1]=='2,3,0,1'
    assert groups['parallel_final'][0]==['tests/step089','tests/step111']
