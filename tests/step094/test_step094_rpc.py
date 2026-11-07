"""真实多 GPU RPC 的独立子进程测试。"""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
ROOT=Path(__file__).resolve().parents[2]
ENTRY=ROOT/'benchmarks/check_step094_acceptance.py'
spec=importlib.util.spec_from_file_location('step094_acceptance',ENTRY)
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


@pytest.mark.parametrize('case',['rpc','failure','engine'])
def test_gpu_rpc_scenarios(case,tmp_path):
    result=runner.execute(case,tmp_path/f'{case}.json',150)
    assert result['passed'],Path(result['log_path']).read_text()[-12000:]
    assert result['result']['status']=='passed'


def test_help_and_timeout_validation(tmp_path):
    p=subprocess.run([sys.executable,str(ENTRY),'--help'],capture_output=True,text=True,timeout=20)
    assert p.returncode==0 and '--case' in p.stdout
    p=subprocess.run([sys.executable,str(ENTRY),'--output',str(tmp_path/'x.json'),'--timeout','0'],capture_output=True,text=True,timeout=20)
    assert p.returncode!=0
