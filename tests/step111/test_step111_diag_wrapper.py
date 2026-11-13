"""逐层探针复用性能驱动，同时在门禁边界安装与导出观察。"""
import sys
from types import SimpleNamespace

from benchmarks import step111_diag_wrapper


def test_wrapper_installs_and_flushes_at_gate_boundaries(monkeypatch, tmp_path):
    import benchmarks
    calls=[]
    hooks=SimpleNamespace(install=object(),flush=object())
    def rpc(fn,*args): calls.append(('rpc',fn,args))
    def gate(rpc_arg,upstream,k): calls.append(('gate',rpc_arg,upstream,k))
    def flush(rpc_arg,directory,k):
        calls.append(('flush',rpc_arg,directory,k));return 'records'
    worker=SimpleNamespace(install_gate=gate,flush_gate=flush)
    def run():
        worker.install_gate(rpc,True,0)
        assert worker.flush_gate(rpc,tmp_path/'run',0)=='records'
    worker.main=run
    monkeypatch.setattr(benchmarks,'step111_parallel_worker',worker,raising=False)
    monkeypatch.setattr(step111_diag_wrapper.importlib,'import_module',lambda _:hooks)
    monkeypatch.setattr(sys,'argv',['probe','--diag-request','12','--diag-positions','850,1023',
                                    '--phase','gate'])
    step111_diag_wrapper.main('unused')
    assert calls==[('gate',rpc,True,0),('rpc',hooks.install,('12',[850,1023])),
                   ('rpc',hooks.flush,(str(tmp_path/'run')+'_layers',)),
                   ('flush',rpc,tmp_path/'run',0)]
    assert sys.argv==['probe','--phase','gate']


def test_trace_rejects_gate_for_another_package(tmp_path):
    import json
    import os
    import subprocess
    from pathlib import Path
    root=Path(__file__).resolve().parents[2]
    case={'model':'8b','tp':1,'batch':1,'lengths':[128],'k':0,'mode':'eager'}
    gate=tmp_path/'gate.json'
    gate.write_text(json.dumps({'cuda_visible_devices':'0','package':{'sha256':'wrong'},
        'results':[{'case':case,'status':'passed_performance','strict_equivalent':True}]}))
    command=[sys.executable,str(root/'benchmarks/trace_step111_parallel.py'),
             '--model','8b','--tp','1','--gate-report',str(gate),'--output',str(tmp_path/'out.json')]
    run=subprocess.run(command,cwd=root,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0'),
                       text=True,capture_output=True,timeout=30)
    assert run.returncode != 0 and 'gate report package differs' in run.stderr
    assert not (tmp_path/'out.json').exists()
