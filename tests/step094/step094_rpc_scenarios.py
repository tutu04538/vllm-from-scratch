"""094 真实四 rank RPC、EngineCore 及失败传播探针。"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
os.environ['VLLM_USE_V2_MODEL_RUNNER']='1'
from minivllm.config import CacheConfig,DeviceConfig,ModelConfig,ParallelConfig,SchedulerConfig,VllmConfig
from minivllm.core.sched.output import CachedRequestData,NewRequestData,SchedulerOutput
from minivllm.engine import LLMEngine
from minivllm.executor import MultiprocExecutor
from minivllm.sampling_params import SamplingParams
from minivllm.testing.step094_probe import RPCProbeWorker,GatedRPCProbeWorker


class FailingExecuteProbe(RPCProbeWorker):
    def execute_model(self, scheduler_output, non_block=False):
        if self.rank == 1:
            raise RuntimeError('injected engine execution failure')
        return super().execute_model(scheduler_output, non_block=non_block)


def config(tp=2,pp=2):
    return VllmConfig(ModelConfig('unused',hf_config={'num_attention_heads':4,'num_key_value_heads':2,
                       'num_hidden_layers':4},max_model_len=64,enforce_eager=True),
                      cache_config=CacheConfig(num_gpu_blocks=16),
                      scheduler_config=SchedulerConfig(max_num_seqs=4,max_num_batched_tokens=16),
                      device_config=DeviceConfig('cuda'),
                      parallel_config=ParallelConfig(tensor_parallel_size=tp,pipeline_parallel_size=pp))


def snapshot():
    new=NewRequestData(req_id='A',prompt_token_ids=[1,2],sampling_params=SamplingParams(max_tokens=2),
                       block_ids=([1,2],),num_computed_tokens=0,prefill_token_ids=[1,2])
    cached=CachedRequestData(req_ids=['B'],resumed_req_ids={'B'},new_block_ids=[([3],)],
                             num_computed_tokens=[1],num_output_tokens=[0],all_token_ids={'B':[4,5]})
    return SchedulerOutput(scheduled_new_reqs=[new],scheduled_cached_reqs=cached,
                           num_scheduled_tokens={'A':2,'B':1},total_num_scheduled_tokens=3,
                           scheduled_spec_decode_tokens={},finished_req_ids={'finished'})


def do_rpc(outdir):
    gate=outdir/'gate';gate.mkdir(parents=True,exist_ok=True)
    for name in ('waiting','release'):
        (gate/name).unlink(missing_ok=True)
    os.environ['MINIVLLM_STEP094_GATE']=str(gate)
    cfg=config(); ex=MultiprocExecutor(cfg,_worker_cls=GatedRPCProbeWorker,_startup_timeout=90)
    results={}
    try:
        assert ex.output_rank == 2
        s0=snapshot()
        bad=snapshot();bad.scheduled_new_reqs=[object()]
        try:ex.execute_model(bad)
        except TypeError as exc:assert '快照' in str(exc)
        else:raise AssertionError('请求活对象必须在 RPC 前拒绝')
        f1=ex.execute_model(s0,non_block=True)
        f2=ex.sample_tokens(None,non_block=True)
        deadline=time.monotonic()+30
        while not (gate/'waiting').exists() and time.monotonic()<deadline:
            time.sleep(0.01)
        assert (gate/'waiting').exists()
        assert not f1.done() and not f2.done()
        (gate/'release').touch()
        result=f2.result()  # 故意先等第二个：第一个 Future 必须先读取其对应回复。
        assert f1.done() and f1.result() is None
        assert result.source_rank == 2 and result.req_ids == ['A','B']
        observed=ex.collective_rpc('snapshot',timeout=30)
        assert len(observed)==4 and all(asdict(x)==asdict(s0) for x in observed)
        assert ex.collective_rpc('rank_identity',unique_reply_rank=2,timeout=30)==2
        assert ex.collective_rpc(lambda worker, offset: worker.rank + offset,
                                 args=(10,), timeout=30)==[10,11,12,13]
        # 连续 RPC 与 queue 读取顺序，至少超过环形槽数。
        futures=[ex.collective_rpc('echo',args=(i,),non_block=True,unique_reply_rank=2,timeout=30) for i in range(12)]
        assert futures[-1].result()==11
        assert [f.result() for f in futures]==list(range(12))
        empty=ex.execute_model(SchedulerOutput.make_empty())
        assert empty.req_ids==[] and ex.take_draft_token_ids() is None
        # 主消息大于默认 24 MiB 槽，触发 ZMQ overflow。
        payload=b'X'*(24*1024*1024+4096)
        assert ex.collective_rpc('echo',args=(payload,),unique_reply_rank=2,timeout=60)==payload
        results={'status':'passed','world_size':4,'output_rank':ex.output_rank,
                 'future_pending':True,'reverse_wait_order':True,'snapshot_equal_all_ranks':True,
                 'sample_source_rank':result.source_rank,'sample_req_ids':result.req_ids,
                 'sampled_token_ids':result.sampled_token_ids,'overflow_bytes':len(payload),
                 'ranks':ex.ready_workers,'worker_pids':[h.process.pid for h in ex.workers]}
    finally:
        (gate/'release').touch()
        ex.shutdown()
        os.environ.pop('MINIVLLM_STEP094_GATE',None)
    assert all(not h.process.is_alive() for h in ex.workers)
    results['worker_exit_codes']=[h.process.exitcode for h in ex.workers]
    assert results['worker_exit_codes']==[0,0,0,0]
    return results


def do_failure():
    ex=MultiprocExecutor(config(),_worker_cls=RPCProbeWorker,_startup_timeout=90)
    try:
        f=ex.collective_rpc('fail_rank',args=(1,),unique_reply_rank=2,non_block=True,timeout=30)
        try: f.result()
        except RuntimeError as exc:
            assert 'rank 1' in str(exc) and 'injected RPC failure' in str(exc)
            error=str(exc)
        else: raise AssertionError('非输出 rank 的异常被丢弃')
        try: ex.collective_rpc('rank_identity',timeout=3)
        except RuntimeError as exc: assert '已失败' in str(exc)
        else: raise AssertionError('失败后仍可调度')
    finally:
        ex.shutdown()
    abrupt=MultiprocExecutor(config(),_worker_cls=RPCProbeWorker,_startup_timeout=90)
    names=list(abrupt._response_shm_names); directory=Path(abrupt._rendezvous.name)
    try:
        f=abrupt.collective_rpc('exit_rank',args=(1,),unique_reply_rank=2,non_block=True,timeout=30)
        try:f.result()
        except RuntimeError as exc:
            assert 'rank 1' in str(exc) and '退出' in str(exc)
            abrupt_error=str(exc)
        else:raise AssertionError('突然退出未传回')
    finally:
        abrupt.shutdown()
    assert not directory.exists() and all(not (Path('/dev/shm')/n).exists() for n in names)
    assert all(not h.process.is_alive() for h in abrupt.workers)
    return {'status':'passed','failed_rank':1,'output_rank':2,'error':error,
            'abrupt_exit_error':abrupt_error,'orphan_shm_cleaned':True,'ipc_directory_cleaned':True,
            'abrupt_worker_exit_codes':[h.process.exitcode for h in abrupt.workers]}


def do_engine():
    cfg=config(tp=2,pp=1);ex=MultiprocExecutor(cfg,_worker_cls=RPCProbeWorker,_startup_timeout=90)
    engine=LLMEngine(cfg,ex)
    try:
        engine.add_request('A',[1,2],SamplingParams(max_tokens=2,temperature=0.0))
        outputs=[]
        while engine.has_unfinished_requests(): outputs.extend(engine.step())
        assert outputs[-1].token_ids==[7,8]
        tokens=list(outputs[-1].token_ids)
    finally: engine.shutdown()
    ex2=MultiprocExecutor(cfg,_worker_cls=FailingExecuteProbe,_startup_timeout=90)
    failing=LLMEngine(cfg,ex2)
    try:
        failing.add_request('A',[1,2],SamplingParams(max_tokens=2,temperature=0.0))
        try:failing.step()
        except RuntimeError as exc:assert 'rank 1' in str(exc)
        else:raise AssertionError('非输出 rank 执行失败被遗漏')
        core=failing.engine_core.engine_core
        assert core.failure is not None
        def must_not_schedule():raise AssertionError('失败后不应再次调度')
        core.scheduler.schedule=must_not_schedule
        try:failing.step()
        except RuntimeError as exc:assert '失败' in str(exc)
        else:raise AssertionError('失败后继续执行')
    finally:failing.shutdown()
    return {'status':'passed','output_token_ids':tokens,'fixture':'FakeRunner，非真实模型',
            'nonoutput_failure_stops_before_next_schedule':True}


def main():
    p=argparse.ArgumentParser();p.add_argument('--case',choices=['rpc','failure','engine'],required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    value={'status':'failed','case':args.case}
    try:
        value.update({'rpc':lambda:do_rpc(args.output.parent),'failure':do_failure,'engine':do_engine}[args.case]())
    except BaseException as exc:
        import traceback
        value.update(error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
    args.output.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'case':args.case,'status':value['status'],'error':value.get('error')},ensure_ascii=False))
    return 0 if value['status']=='passed' else 1

if __name__=='__main__':sys.exit(main())
