"""资源占用时，模型性能复测必须记录未运行并且不创建工作进程。"""
import importlib.util
import json
from pathlib import Path
import sys


def test_busy_gpu_never_starts_model_and_preserves_results(tmp_path, monkeypatch):
    root=Path(__file__).resolve().parents[2]
    spec=importlib.util.spec_from_file_location('step112_rope_resource_gate',
                                                root/'benchmarks/step112_rope_validation.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(module,'OUT',tmp_path/'runs')
    monkeypatch.setattr(module,'selected_gpu_processes',lambda selected:[1234])
    def unexpected(*_args,**_kwargs):
        raise AssertionError('资源占用时不能启动模型进程')
    monkeypatch.setattr(module.subprocess,'Popen',unexpected)
    result=tmp_path/'busy.json'
    monkeypatch.setattr(sys,'argv',['probe','--case','failed-cases','--output',str(result)])
    assert module.main() == 1
    rows=json.loads(result.read_text())
    assert len(rows)==3
    assert all(row['status']=='unrun' and row['exit_code']==125 for row in rows)
    assert all('--input-length' not in row['command'] and
               row['command'][row['command'].index('--group')+1]=='B' for row in rows)
    from benchmarks.step111_matrix import build_cases
    for row in rows:
        cmd=row['command']; model=cmd[cmd.index('--model')+1]; tp=int(cmd[cmd.index('--tp')+1])
        matches=[c for c in build_cases() if c['model']==model and c['tp']==tp and 'B' in c['groups']]
        assert len(matches)==1 and matches[0]['lengths']==[128]*8+[1024]*8
    assert not list((tmp_path/'runs').rglob('*.log'))


def test_resume_preserves_previous_summary_when_gpu_busy(tmp_path, monkeypatch):
    root=Path(__file__).resolve().parents[2]
    spec=importlib.util.spec_from_file_location('step112_rope_resume_gate',
                                                root/'benchmarks/step112_rope_validation.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(module,'OUT',tmp_path/'runs')
    monkeypatch.setattr(module,'selected_gpu_processes',lambda selected:[1234])
    summary=tmp_path/'run.json';summary.write_text('[{"prior": true}]\n')
    run_dir=tmp_path/'runs'/'run';run_dir.mkdir(parents=True)
    (run_dir/'full_matrix.json').write_text('{"results": []}\n')
    monkeypatch.setattr(sys,'argv',['probe','--case','matrix','--resume','--output',str(summary)])
    assert module.main()==1
    snapshots=list(run_dir.glob('previous_summary_*.json'))
    assert len(snapshots)==1 and json.loads(snapshots[0].read_text())==[{'prior':True}]
    assert json.loads(summary.read_text())[0]['status']=='unrun'
    assert json.loads((run_dir/'full_matrix.json').read_text())=={'results':[]}


def test_trace_registration_rejects_other_or_modified_gate(tmp_path):
    import hashlib
    root=Path(__file__).resolve().parents[2]
    spec=importlib.util.spec_from_file_location('step112_result_registration',
                                                root/'benchmarks/register_step111_results.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    gate=tmp_path/'gate.json';gate.write_text(json.dumps({'package':{'sha256':'package-a'}}))
    report=tmp_path/'trace.json'
    analysis=tmp_path/'analysis.json';analysis.write_text(json.dumps({'status':'passed'}))
    analyzer=hashlib.sha256((root/'benchmarks/step111_analyze_trace.py').read_bytes()).hexdigest()
    report.write_text(json.dumps({'status':'captured','gate_report':str(gate),
        'gate_report_sha256':hashlib.sha256(gate.read_bytes()).hexdigest(),
        'analysis':str(analysis),'harness_sha256':{'step111_analyze_trace.py':analyzer}}))
    assert module.validate_trace_source(report,'package-a')
    assert not module.validate_trace_source(report,'package-b')
    gate.write_text(json.dumps({'package':{'sha256':'package-a'},'changed':True}))
    assert not module.validate_trace_source(report,'package-a')


def test_matrix_registration_preserves_previous_source_evidence():
    root=Path(__file__).resolve().parents[2]
    spec=importlib.util.spec_from_file_location('step112_registration_history',
                                                root/'benchmarks/register_step111_results.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    previous={'history':[{'matrix_sha256':'old'}],
              'results':{'matrix_sha256':'current','package_sha256':'prior-package'}}
    assert module.retain_history(previous,'current')==previous['history']
    assert module.retain_history(previous,'new')==previous['history']+[previous['results']]
    assert previous['history']==[{'matrix_sha256':'old'}]


def test_registration_accepts_final_regressions_only_for_matching_package():
    root=Path(__file__).resolve().parents[2]
    spec=importlib.util.spec_from_file_location('step112_registration_regressions',
                                                root/'benchmarks/register_step111_results.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rows=[{'name':name,'exit_code':0,'summary':'1 passed',
           'command':['pytest','tests/step111'] if name=='parallel_final' else ['pytest']}
          for name in ('pytest','legacy_final','parallel_final')]
    record={'status':'passed_regressions','package_sha256':'a','package_sha256_end':'a','runs':rows}
    _,errors=module.normalize_regressions(record,'a')
    assert not errors
    _,errors=module.normalize_regressions(record,'b')
    assert 'regression source or status does not match matrix' in errors
    rows[-1]['command']=['pytest']
    _,errors=module.normalize_regressions(record,'a')
    assert 'parallel regressions did not include tests/step111' in errors


def test_legacy_regression_evidence_cannot_be_reused_for_changed_package():
    root=Path(__file__).resolve().parents[2]
    spec=importlib.util.spec_from_file_location('step112_legacy_regression_provenance',
                                                root/'benchmarks/register_step111_results.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rows=[{'group':name,'exit_code':0,'summary':'1 passed'}
          for name in ('step111','legacy','parallel')]
    assert not module.normalize_regressions(rows,'old','old')[1]
    assert 'legacy regressions do not match matrix source' in module.normalize_regressions(rows,'new','old')[1]
