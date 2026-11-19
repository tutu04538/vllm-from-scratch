"""CPU 生产导入隔离及失败子进程回收检查。"""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def test_project_import_does_not_import_or_link_installed_vllm():
    code = '''
import json, sys
import minivllm
import minivllm.models.qwen3
modules = sorted(name for name in sys.modules if name == "vllm" or name.startswith("vllm."))
with open("/proc/self/maps") as stream:
    linked = sorted({line.rsplit(None, 1)[-1] for line in stream
                     if "site-packages/vllm/" in line and ".so" in line})
print(json.dumps({"modules": modules, "linked": linked}))
'''
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    result = subprocess.run([sys.executable, '-c', code], cwd=ROOT, env=env,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout.splitlines()[-1])
    assert report['modules'] == [] and report['linked'] == []


def test_reference_process_failure_exits_and_is_reaped(tmp_path):
    code = 'raise RuntimeError("step113 intentional reference-process failure")'
    result = subprocess.run([sys.executable, '-c', code], cwd=ROOT,
                            capture_output=True, text=True, timeout=10)
    assert result.returncode != 0
    assert 'step113 intentional reference-process failure' in result.stderr
