"""校准和 held-out 数据身份：CPU 小例与重叠负例。"""
import importlib.util
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / 'benchmarks/prepare_step114_quant_data.py'
spec = importlib.util.spec_from_file_location('step114_quant_data_prepare', SCRIPT)
data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data)


def test_windows_keep_non_overlapping_ids_and_hashes():
    rows = data.windows(list(range(13)), 'train', 3, width=4)
    assert [r['token_ids'] for r in rows] == [list(range(4)), list(range(4, 8)), list(range(8, 12))]
    assert [r['id'] for r in rows] == ['train:tokens:0:4', 'train:tokens:4:8', 'train:tokens:8:12']
    assert rows[0]['sha256_uint32_le'] == data.token_hash([0, 1, 2, 3])


def test_detects_content_overlap_despite_distinct_split_ids():
    train = data.windows([1, 2, 3, 4], 'train', 1, width=4)
    test = data.windows([1, 2, 3, 4], 'test', 1, width=4)
    with pytest.raises(ValueError, match='overlap'):
        data.check_no_leakage(train, test)


def test_duplicate_test_content_is_rejected():
    test = data.windows([1, 2, 1, 2], 'test', 2, width=2)
    with pytest.raises(ValueError, match='duplicate'):
        data.check_no_leakage([], test)


def test_data_audit_rejects_manifest_without_target_models(tmp_path):
    import json
    check_path = ROOT / 'benchmarks/check_step114_quant_data.py'
    audit_spec = importlib.util.spec_from_file_location('step114_quant_data_audit', check_path)
    audit = importlib.util.module_from_spec(audit_spec)
    audit_spec.loader.exec_module(audit)
    (tmp_path / 'manifest.json').write_text(json.dumps({
        'kind': 'data_preparation_only', 'sources': {}, 'targets': {},
    }))
    with pytest.raises(ValueError, match='missing or unexpected target'):
        audit.check(tmp_path)


@pytest.mark.parametrize('method', ['awq', 'gptq'])
def test_busy_quantizer_launcher_never_executes_quantizer(tmp_path, monkeypatch, method):
    import sys
    sys.path.insert(0, str(ROOT / 'benchmarks'))
    try:
        script = ROOT / 'benchmarks/step114_launch_quantizer.py'
        launch_spec = importlib.util.spec_from_file_location('step114_awq_launcher', script)
        module = importlib.util.module_from_spec(launch_spec)
        launch_spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(ROOT / 'benchmarks'))
    monkeypatch.setattr(module, 'gpu_occupancy', lambda device: {'status': 'busy'})
    def forbidden(*args, **kwargs):
        raise AssertionError('busy device must not start a process')
    monkeypatch.setattr(module.subprocess, 'run', forbidden)
    result = tmp_path / 'result.json'
    code = module.launch(device=0, model='8B', tool_python=tmp_path / 'python', data=tmp_path,
                         output_root=tmp_path, result_path=result, method=method)
    assert code == 2
    import json
    assert json.loads(result.read_text())['status'] == 'not_run'


def test_short_calibration_is_deterministic_and_preserves_source():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'benchmarks'))
    from step114_quantizer_common import derive_calibration

    samples = [{'id': str(i), 'token_ids': list(range(2048))} for i in range(128)]
    derived, digest = derive_calibration(samples)
    assert len(derived) == 128
    assert all(row['token_ids'] == list(range(512)) for row in derived)
    assert len(samples[0]['token_ids']) == 2048
    assert derive_calibration(samples)[1] == digest
    samples[0]['token_ids'][0] = 999
    assert derive_calibration(samples)[1] != digest
