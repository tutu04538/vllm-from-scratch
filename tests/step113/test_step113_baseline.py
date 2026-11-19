"""CPU 测试：身份校验检测损坏，缺 GPU 探针不能冒充通过。"""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / 'benchmarks/benchmark_step113_quant_capability.py'
spec = importlib.util.spec_from_file_location('step113_baseline_probe', PATH)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_manifest_detects_modified_missing_and_duplicate_files(tmp_path):
    source = tmp_path / 'source.py'
    source.write_bytes(b'original')
    row = {'path': 'source.py', 'bytes': 8, 'sha256': hashlib.sha256(b'original').hexdigest()}
    assert probe.check_manifest({'files': [row]}, tmp_path)['status'] == 'passed'
    source.write_bytes(b'changed!')
    assert probe.check_manifest({'files': [row]}, tmp_path)['status'] == 'failed'
    source.unlink()
    assert probe.check_manifest({'files': [row]}, tmp_path)['status'] == 'failed'
    assert probe.check_manifest({'files': [row, row]}, tmp_path)['status'] == 'failed'


def test_package_hash_uses_relative_paths(tmp_path):
    path = tmp_path / 'a.py'
    path.write_text('x = 1\n')
    before = probe.package_hash(tmp_path)
    (tmp_path / 'ignored.txt').write_text('ignored')
    assert probe.package_hash(tmp_path) == before
    path.rename(tmp_path / 'b.py')
    assert probe.package_hash(tmp_path)['sha256'] != before['sha256']


def test_unimplemented_gpu_probe_is_not_passed(tmp_path):
    out = tmp_path / 'probe.json'
    result = subprocess.run([sys.executable, str(PATH), '--trace', '--output', str(out)],
                            cwd=ROOT, capture_output=True, text=True, timeout=15)
    assert result.returncode != 0
    assert json.loads(out.read_text())['status'] == 'not_run'


def test_busy_gpu_does_not_start_reference_process(monkeypatch, tmp_path):
    monkeypatch.setattr(probe, 'gpu_occupancy', lambda index: {'status': 'busy', 'device': index})
    def must_not_run(*args, **kwargs):
        raise AssertionError('busy GPU must not start a subprocess')
    monkeypatch.setattr(probe.subprocess, 'run', must_not_run)
    result = probe.compare_gpu(0, tmp_path / 'result.json')
    assert result['status'] == 'not_run'
