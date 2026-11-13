"""The fairness matrix is deterministic and repeated workloads are merged."""
import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location('step111_parallel', Path(__file__).resolve().parents[2] / 'benchmarks/compare_step111_parallel.py')
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_complete_deduplicated_matrix():
    cases = module.build_cases()
    assert len(cases) == 50
    assert len({(c['model'], c['tp'], tuple(c['lengths']), c['k'], c['mode']) for c in cases}) == 50
    assert all(c['output_tokens'] == 128 for c in cases)
    assert all(c['tp'] != 1 for c in cases if c['model'] == '14b')
    assert sum('A' in c['groups'] for c in cases) == 10
    assert sum('B' in c['groups'] for c in cases) == 5
    assert sum('C' in c['groups'] for c in cases) == 20
    assert sum('D' in c['groups'] for c in cases) == 40
    for c in cases:
        if 'B' in c['groups']:
            assert c['batch'] == 16 and c['lengths'] == [128]*8 + [1024]*8


def test_summary_retains_all_repeats():
    rows = [{'wall_time_s': v, 'output_tokens_per_s': 128/v,
             'requests_per_s': 1/v, 'ttft_s': v/2, 'tpot_s': None,
             'inter_burst_latency_s': v/10} for v in (4, 2, 3)]
    result = module.summarize(rows)
    assert result['wall_time_s'] == {'values':[4,2,3], 'median':3, 'min':2, 'max':4}
    assert result['output_tokens_per_s']['values'] == [32,64,128/3]
    assert result['tpot_s']['median'] is None


def test_committed_token_throughput_does_not_count_drafts_or_ranks():
    from benchmarks.spec_decode_metrics import AcceptanceAggregate, RequestTiming, summarize_run
    timing = RequestTiming('r', submitted_at=0.0)
    timing.observe(1.0, 2)
    timing.observe(2.0, 3)
    timing.observe(2.0, 3)
    a = AcceptanceAggregate(num_drafts=2, num_draft_tokens=4, num_accepted_tokens=1)
    result = summarize_run(timings={'r': timing}, acceptance=a, wall_time_s=2.0, engine_steps=2)
    assert result['committed_output_tokens'] == 3
    assert result['output_tokens_per_s'] == 1.5
    assert result['tpot_s']['mean'] == 1.0
    assert result['inter_burst_latency_s']['mean'] == 1.0


def test_gpu_occupancy_checks_only_selected_visible_devices(monkeypatch):
    class Result:
        def __init__(self, stdout): self.stdout = stdout
    def run(command, **kwargs):
        if '--query-gpu=index,uuid' in command:
            return Result('0, GPU-a\n1, GPU-b\n2, GPU-c\n3, GPU-d\n')
        return Result('101, GPU-a\n102, GPU-c\n103, GPU-d\n')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '2,3,0,1')
    monkeypatch.setattr(module.subprocess, 'run', run)
    assert module.compute_pids(2) == {102, 103}
    assert module.compute_pids(1) == {102}


def test_gate_registers_reference_before_mini_and_detects_rank_error(tmp_path):
    import json
    import subprocess
    import sys
    import torch
    root = Path(__file__).resolve().parents[2]
    case = {'tp':2, 'batch':1}
    for name, implementation in [('ref','frozen'), ('repeat','frozen'), ('mini','mini')]:
        directory = tmp_path/name; directory.mkdir()
        (tmp_path/f'{name}.json').write_text(json.dumps({'case':case, 'implementation':implementation,
            'prompt_ids':[[1,2]], 'samples':[{'tokens':{'0':[3,4]}}]}))
        for rank in range(2):
            rows = [{'request_id':'0', 'position':step+1, 'sequence_length':step+2, 'sampled':3,
                     'hidden':torch.tensor([1.0,2.0]), 'logits':torch.tensor([1.0,2.0]),
                     'kv':torch.tensor([float(rank),1.0])} for step in range(4)]
            torch.save(rows, directory/f'run0_rank{rank}.pt')
    script = root/'benchmarks/step111_compare_gate.py'; contract = tmp_path/'contract.json'
    def compare(other, register=False):
        cmd = [sys.executable, str(script), '--reference', str(tmp_path/'ref'), '--other', str(tmp_path/other),
               '--contract', str(contract), '--output', str(tmp_path/f'{other}_report.json')]
        if register: cmd.append('--register')
        return subprocess.run(cmd, capture_output=True, text=True)
    assert compare('repeat', True).returncode == 0
    assert compare('mini').returncode == 0
    path = tmp_path/'mini'/'run0_rank1.pt'
    rows = torch.load(path, weights_only=True); rows[0]['sampled'] = 9; torch.save(rows, path)
    assert compare('mini').returncode != 0


def test_eagle_gate_detects_inconsistent_rank_state(tmp_path):
    import torch
    spec = importlib.util.spec_from_file_location('step111_rank_gate', Path(__file__).resolve().parents[2]/'benchmarks/step111_rank_gate.py')
    rank_gate = importlib.util.module_from_spec(spec); spec.loader.exec_module(rank_gate)
    base = [{'kind':'sample', 'req_ids':['0'], 'num_sampled':torch.tensor([1]),
             'sampled_ids':[torch.tensor([7])]}]
    torch.save(base, tmp_path/'state_rank0.pt'); torch.save(base, tmp_path/'state_rank1.pt')
    assert rank_gate.check(tmp_path, 2) == []
    bad = [{'kind':'sample', 'req_ids':['0'], 'num_sampled':torch.tensor([1]),
            'sampled_ids':[torch.tensor([8])]}]
    torch.save(bad, tmp_path/'state_rank1.pt')
    assert 'sampled_ids differs' in rank_gate.check(tmp_path, 2)[0]


def test_trace_collectives_count_input_bytes_without_replacing_output(monkeypatch):
    import torch
    from types import SimpleNamespace
    from benchmarks import step111_parallel_worker as worker
    from minivllm.distributed import parallel_state
    class Group:
        world_size = 2
        def all_reduce(self, tensor): return tensor + 1
        def all_gather(self, tensor, dim=-1): return torch.cat([tensor, tensor], dim=dim)
    group = Group(); obj = SimpleNamespace()
    monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: True)
    monkeypatch.setattr(torch.distributed, 'get_rank', lambda: 0)
    monkeypatch.setattr(torch.cuda, 'is_current_stream_capturing', lambda: False)
    monkeypatch.setattr(parallel_state, 'get_tp_group', lambda: group)
    worker.trace_install_collectives(obj, False)
    tensor = torch.tensor([1.,2.])
    assert torch.equal(group.all_reduce(tensor), torch.tensor([2.,3.]))
    assert group.all_gather(tensor).shape == (4,)
    observed = worker.trace_collectives(obj)
    assert observed['operations']['all_reduce']['input_bytes'] == 8
    assert observed['operations']['all_gather']['calls'] == 1
    assert 'graph replay is not counted' in observed['coverage']


def test_trace_aggregates_only_each_rank_steady_interval(tmp_path):
    import sqlite3
    import pytest
    spec = importlib.util.spec_from_file_location('step111_trace_analysis', Path(__file__).resolve().parents[2]/'benchmarks/step111_analyze_trace.py')
    trace = importlib.util.module_from_spec(spec); spec.loader.exec_module(trace)
    path = tmp_path/'trace.sqlite'; db = sqlite3.connect(path)
    db.execute('create table StringIds(id integer, value text)')
    db.execute('create table NVTX_EVENTS(start integer,end integer,text text,globalTid integer,textId integer)')
    db.execute('create table CUPTI_ACTIVITY_KIND_KERNEL(start integer,end integer,demangledName integer,globalPid integer,graphNodeId integer)')
    db.executemany('insert into StringIds values(?,?)', [(1,'ncclAllReduce'),(2,'ampere_sgemm')])
    db.execute('insert into NVTX_EVENTS values(10,100,?, ?, null)', ('step111_steady_rank_0', (10<<24)+123))
    db.execute('insert into NVTX_EVENTS values(10,100,?, ?, null)', ('step111_steady_rank_1', (11<<24)+456))
    db.executemany('insert into CUPTI_ACTIVITY_KIND_KERNEL values(?,?,?,?,?)',
                   [(20,30,1,10<<24,None),(40,60,2,10<<24,99),(101,110,1,10<<24,None),(20,25,1,11<<24,None)])
    db.commit(); db.close()
    result = trace.analyze(path, 2)
    assert result['ranks'][0]['categories']['collective'] == {'calls':1,'gpu_sum_ns':10,'graph_node_calls':0}
    assert result['ranks'][0]['categories']['gemm']['graph_node_calls'] == 1
    assert result['ranks'][1]['categories']['collective']['gpu_sum_ns'] == 5
    with pytest.raises(ValueError, match='missing rank'): trace.analyze(path, 4)


def test_upstream_rank_log_times_are_separate_and_capture_is_rounded():
    log = ('(Worker_TP1 pid=11) INFO Model loading took 14.53 GiB memory and 31.305286 seconds\n'
           '(Worker_TP1 pid=11) INFO Graph capturing finished in 1 secs, took 0.13 GiB\n')
    assert module.upstream_log_times(log) == {1:{'model_loading_seconds':31.305286, 'graph_capture_seconds_rounded':1}}
    mixed = '(Worker_TP0 pid=1) progress (Worker_TP2 pid=2) Graph capturing finished in 3 secs'
    assert module.upstream_log_times(mixed) == {2:{'graph_capture_seconds_rounded':3}}
    single = 'INFO Model loading took 16 GiB memory and 5.7 seconds\nINFO Graph capturing finished in 0 secs'
    assert module.upstream_log_times(single, 1) == {0:{'model_loading_seconds':5.7, 'graph_capture_seconds_rounded':0}}


def test_gate_observes_scheduler_snapshot_without_mutating_it():
    from types import SimpleNamespace
    from benchmarks.step111_parallel_worker import install_schedule_gate
    cached = SimpleNamespace(req_ids=['a-x'], num_computed_tokens=[2], num_output_tokens=[1],
                             new_block_ids=[([4],)], resumed_req_ids=set())
    packet = SimpleNamespace(scheduled_cached_reqs=cached, scheduled_new_reqs=[],
                             num_scheduled_tokens={'a-x':1}, scheduled_spec_decode_tokens={},
                             finished_req_ids=set())
    scheduler = SimpleNamespace(schedule=lambda: packet)
    engine = SimpleNamespace(engine_core=SimpleNamespace(engine_core=SimpleNamespace(scheduler=scheduler)))
    records = install_schedule_gate(engine)
    assert scheduler.schedule() is packet
    assert records[0]['scheduled_tokens'] == [('a',1)]
    assert cached.req_ids == ['a-x'] and cached.num_computed_tokens == [2]


def test_gpu_selection_rejects_missing_or_duplicate_devices(monkeypatch):
    import pytest
    class Result:
        def __init__(self, stdout): self.stdout = stdout
    monkeypatch.setattr(module.subprocess, 'run', lambda command, **kwargs: Result('0, GPU-a\n1, GPU-b\n'))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    with pytest.raises(ValueError, match='only 1'): module.compute_pids(2)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,0')
    with pytest.raises(ValueError, match='duplicate'): module.compute_pids(2)


def test_resolved_identity_rejects_silent_mode_or_capacity_changes():
    import pytest
    from enum import Enum
    from types import SimpleNamespace
    from benchmarks.step111_parallel_worker import check_identity
    class Mode(Enum): NONE=0
    cfg = SimpleNamespace(parallel_config=SimpleNamespace(tensor_parallel_size=2, pipeline_parallel_size=1,data_parallel_size=1),
        scheduler_config=SimpleNamespace(max_num_seqs=1,max_num_batched_tokens=128,async_scheduling=False),
        cache_config=SimpleNamespace(block_size=16,enable_prefix_caching=False),
        compilation_config=SimpleNamespace(cudagraph_mode=Mode.NONE,mode=Mode.NONE),
        model_config=SimpleNamespace(dtype='bfloat16',seed=0,max_model_len=2048),speculative_config=None,attention_backend='triton')
    case = {'tp':2,'batch':1,'mode':'eager','k':0}
    assert check_identity(cfg,case)['tp'] == 2
    cfg.scheduler_config.max_num_batched_tokens = 256
    with pytest.raises(ValueError,match='resolved configuration differs'): check_identity(cfg,case)


def test_performance_output_gate_reports_first_divergence():
    assert module.token_difference({'a':[1,2]}, {'a':[1,2]}) is None
    assert module.token_difference({'a':[1,2,3]}, {'a':[1,9,3]}) == {'request':'a','index':1,'left':2,'right':9}
    assert module.token_difference({'a':[1]}, {'a':[1,2]})['error'] == 'length differs'


def test_trace_refuses_unverified_case_before_gpu_use(tmp_path):
    import json
    import os
    import subprocess
    import sys
    gate = tmp_path/'gate.json'
    gate.write_text(json.dumps({'cuda_visible_devices':'0,1', 'results':[]}))
    script = Path(__file__).resolve().parents[2]/'benchmarks/trace_step111_parallel.py'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1')
    run = subprocess.run([sys.executable,str(script),'--model','8b','--tp','2','--gate-report',str(gate),
                          '--output',str(tmp_path/'out.json')], env=env, text=True, capture_output=True)
    assert run.returncode != 0 and 'matching case did not pass' in run.stderr
    assert not (tmp_path/'out.json').exists()


def test_rank_device_gate_detects_missing_or_duplicate_gpu():
    import pytest
    ranks = [{'rank':0,'device_uuid':'GPU-a'},{'rank':1,'device_uuid':'GPU-b'}]
    result = {'rank_before':ranks, 'rank_after':list(ranks)}
    assert module.rank_devices(result,2) == {0:'GPU-a',1:'GPU-b'}
    result['rank_after'] = [{'rank':0,'device_uuid':'GPU-a'},{'rank':1,'device_uuid':'GPU-a'}]
    with pytest.raises(ValueError): module.rank_devices(result,2)


def test_queue_time_uses_submission_to_first_schedule(monkeypatch):
    from benchmarks import step111_parallel_worker as worker
    events = []
    def run_once(*args):
        events.extend([{'at':12.0,'request_ids':['a-random']}, {'at':14.0,'request_ids':['a-random','b-random']}])
        return {'request_timing':{'a':{'submitted_at':10.0},'b':{'submitted_at':11.0}}}
    monkeypatch.setattr(worker,'run_once',run_once)
    sample = worker.measured_runs(None,[],128,False,1,events)[0]
    assert sample['queue_seconds_by_request'] == {'a':2.0,'b':3.0}
    assert sample['queue_seconds'] == 2.5


def test_schedule_comparison_preserves_one_to_one_physical_block_mapping():
    def row(blocks, finish):
        return {'scheduled_tokens':[['0',1]],'spec_tokens':[], 'finished_req_ids':finish,
                'new':[{'request_id':'0','prompt_token_ids':[7], 'num_computed_tokens':0,'block_ids':[blocks]}],
                'cached':{'req_ids':[],'num_computed_tokens':[],'num_output_tokens':[],
                          'resumed_req_ids':[],'new_block_ids':[]}}
    reference = [row([5,6],['warm0'])]
    other = [row([4,5],[]), {'scheduled_tokens':[], 'finished_req_ids':['0']}]
    result = module.compare_schedule(reference,other)
    assert result['passed'] and not result['raw_finished_notifications_equal']
    assert [(item['reference'],item['other']) for item in result['physical_block_mapping']] == [(5,4),(6,5)]
    assert not module.compare_schedule(reference,[row([4,4],[])])['passed']


def test_long_input_requires_registered_cross_tp_reference(tmp_path):
    import subprocess
    import sys
    script = Path(__file__).resolve().parents[2]/'benchmarks/compare_step111_parallel.py'
    run = subprocess.run([sys.executable,str(script),'--model','8b','--tp','1','--batch','1',
                          '--input-length','1024','--k','0','--mode','eager','--output',str(tmp_path/'out.json')],
                         text=True,capture_output=True)
    assert run.returncode != 0 and '--calibration' in run.stderr
    assert not (tmp_path/'out.json').exists()


def test_calibration_identifies_history_divergence_before_using_a_window():
    from benchmarks.register_step111_reference import first_differences
    import pytest
    a = {'r':[1,2,3,4,5]}
    assert first_differences(a, {'r':[1,2,3,4,6]}) == [{'request':'r','position':4,'reference':5,'other':6}]
    assert first_differences(a, {'r':[1,9,3,4,5]})[0]['position'] == 1
    with pytest.raises(ValueError, match='length'): first_differences(a, {'r':[1]})


def test_neartie_requires_reproduced_history_and_exact_bfloat16_tie(tmp_path):
    import json
    import pytest
    import torch
    from benchmarks.step111_neartie import verify
    ranks = [{'rank':0,'device_uuid':'GPU-a'}]
    ref_tokens={'0':[1,2,3,4,5]};mini_tokens={'0':[1,2,3,4,6]}
    state=[{'request_id':'0','position':i,'sequence_length':i+1,'sampled':i+1,
            'hidden':torch.tensor([1.]),'logits':torch.tensor([1.,2.]),'kv':torch.tensor([1.])} for i in range(4)]
    for name in ('reference','mini'):
        directory=tmp_path/name; directory.mkdir();torch.save(state,directory/'run0_rank0.pt')
    values={f'rank0/0/{i}/{field}':{'max_abs':0.0,'relative_l2':0.0,'threshold':{'max_abs':1e-5,'relative_l2':1e-5}}
            for i in range(4) for field in ('hidden','logits','kv')}
    report=tmp_path/'comparison.json';report.write_text(json.dumps({'values':values}))
    def result(tokens,name):
        return {'result':{'samples':[{'tokens':tokens}],'prompt_ids':[[7]],'rank_before':ranks},
                'command':['python','--output',str(tmp_path/f'{name}.json')]}
    gate={'case':{'tp':1,'batch':1},'gate':{'frozen_ref':result(ref_tokens,'reference'),
          'mini':result(mini_tokens,'mini'),'comparison':{'report':str(report)}}}
    def diag(tokens, top):
        return {'tokens':tokens,'prompt_ids':[[7]],'ranks':[{'rank':0,'device_uuid':'GPU-a',
                'events':[{'request':'0','step':4,'sampled':tokens['0'][4],'top_ids':[5,6],
                           'top_values':top,'dtype':'torch.bfloat16'}]}]}
    a,b=diag(ref_tokens,[3.0,3.0]),diag(mini_tokens,[3.0,2.75])
    assert verify(gate,a,b)['strict_equivalent'] is False
    gate['engines']={'frozen':{'result':{'samples':[{'tokens':ref_tokens}]}},
                     'mini':{'result':{'samples':[{'tokens':mini_tokens}]}}}
    assert verify(gate,a,b,'performance')['strict_equivalent'] is False
    a['ranks'][0]['events'][0]['top_values']=[3.0,2.75]
    with pytest.raises(ValueError,match='neither side'):verify(gate,a,b)


def test_trace_maps_graph_replay_to_captured_collective_payload(tmp_path):
    import sqlite3
    spec = importlib.util.spec_from_file_location('step111_graph_attribution', Path(__file__).resolve().parents[2]/'benchmarks/step111_analyze_trace.py')
    trace = importlib.util.module_from_spec(spec); spec.loader.exec_module(trace)
    path=tmp_path/'graph.sqlite';db=sqlite3.connect(path)
    db.execute('create table StringIds(id integer,value text)')
    db.execute('create table NVTX_EVENTS(start integer,end integer,text text,globalTid integer,textId integer)')
    db.execute('create table CUDA_GRAPH_EVENTS(start integer,globalTid integer,graphNodeId integer,originalGraphNodeId integer)')
    db.execute('create table CUPTI_ACTIVITY_KIND_KERNEL(start integer,end integer,demangledName integer,globalPid integer,graphNodeId integer)')
    tid=(10<<24)+123;pid=10<<24
    db.execute('insert into StringIds values(1,?)',('ncclAllReduce',))
    db.execute('insert into NVTX_EVENTS values(1,9,?,?,null)',('step111_capture_all_reduce_64',tid))
    db.execute('insert into NVTX_EVENTS values(10,100,?,?,null)',('step111_steady_rank_0',tid))
    db.execute('insert into CUDA_GRAPH_EVENTS values(5,?,?,null)',(tid,1))
    db.execute('insert into CUDA_GRAPH_EVENTS values(9,?,?,?)',(tid,2,1))
    db.executemany('insert into CUPTI_ACTIVITY_KIND_KERNEL values(?,?,?,?,?)',[(20,30,1,pid,2),(40,50,1,pid,2)])
    db.commit();db.close()
    worker={'trace_communication':[{'rank':0,'operations':{'all_reduce':{'calls':0,'input_bytes':0,'events':[]},
                                                              'all_gather':{'calls':0,'input_bytes':0,'events':[]}}}]}
    result=trace.analyze(path,1,worker)
    assert result['ranks'][0]['communication']['graph_replay']['all_reduce']['calls']==2
    assert result['ranks'][0]['communication']['graph_replay']['all_reduce']['input_bytes']==128


def test_resume_does_not_trust_a_status_without_artifacts(tmp_path):
    case={'tp':1,'mode':'eager'}
    entry={'case':case,'status':'passed_performance','artifact_hashes':{str(tmp_path/'missing'):'0'}}
    assert not module.reusable_case(entry,case,3)
    path=tmp_path/'file';path.write_bytes(b'one')
    digest=module.file_sha256(path);path.write_bytes(b'two')
    assert module.file_sha256(path)!=digest


def test_trace_classifies_cublas_gemv_as_matrix_work():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[2] / 'benchmarks/step111_analyze_trace.py'
    spec = importlib.util.spec_from_file_location('step111_trace_category', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.category('internal::gemvx::kernel<cublasGemvParamsEx>') == 'gemm'
    assert module.category('kernel_unified_attention') == 'attention'
    assert module.category('ncclDevKernel_AllReduce') == 'collective'
