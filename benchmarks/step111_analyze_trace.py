"""Summarize GPU kernel intervals inside each rank's steady NVTX range."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import sqlite3


def category(name):
    value = name.lower()
    if 'nccl' in value: return 'collective'
    if any(token in value for token in ('attention', 'attn', 'flash_fwd', 'flash_bwd')): return 'attention'
    if any(token in value for token in ('gemm', 'gemv', 'matmul', 'sgemm', 'hgemm', 's1688', 's16816')): return 'gemm'
    return 'other'


def analyze(path, tp, worker=None):
    db = sqlite3.connect(f'file:{Path(path).resolve()}?mode=ro', uri=True)
    try:
        ranges = db.execute("select start,end,coalesce(text, StringIds.value),globalTid from NVTX_EVENTS "
                            "left join StringIds on NVTX_EVENTS.textId=StringIds.id "
                            "where coalesce(text, StringIds.value) like 'step111_steady_rank_%'").fetchall()
        ranges_by_rank = {}
        for start,end,name,tid in ranges:
            match = re.fullmatch(r'step111_steady_rank_(\d+)', name or '')
            if match and end is not None:
                rank = int(match.group(1))
                if tid is None: raise ValueError(f'missing NVTX thread identity for rank {rank}')
                if rank in ranges_by_rank: raise ValueError(f'duplicate steady range for rank {rank}')
                ranges_by_rank[rank] = (start,end,tid - tid % (1 << 24))
        if set(ranges_by_rank) != set(range(tp)):
            raise ValueError(f'missing rank NVTX ranges: {sorted(ranges_by_rank)}')
        captures = defaultdict(set)
        graph_tables = {row[0] for row in db.execute("select name from sqlite_master where type='table'")}
        parents = {}
        if 'CUDA_GRAPH_EVENTS' in graph_tables:
            query = ("select n.globalTid,n.start,n.text,g.graphNodeId from NVTX_EVENTS n "
                     "join CUDA_GRAPH_EVENTS g on g.globalTid=n.globalTid and g.start between n.start and n.end "
                     "where n.text like 'step111_capture_%'")
            for tid,started,label,node in db.execute(query):
                match = re.fullmatch(r'step111_capture_(all_reduce|all_gather)_(\d+)', label or '')
                if match:
                    pid = tid - tid % (1 << 24)
                    captures[(pid,node)].add((started,match.group(1),int(match.group(2))))
            for node,original,tid in db.execute('select graphNodeId,originalGraphNodeId,globalTid from CUDA_GRAPH_EVENTS where originalGraphNodeId is not null'):
                key = (tid - tid % (1 << 24), node)
                if key in parents and parents[key] != original:
                    raise ValueError(f'graph node {key} has conflicting origins')
                parents[key] = original
        def source_call(pid,node):
            visited = set()
            while (pid,node) not in visited:
                visited.add((pid,node))
                options = captures.get((pid,node))
                if options:
                    if len(options) != 1: raise ValueError('ambiguous collective capture attribution')
                    return next(iter(options))
                if (pid,node) not in parents: return None
                node = parents[(pid,node)]
            raise ValueError('cycle in graph node origins')
        output = []
        for rank in range(tp):
            start,end,global_pid = ranges_by_rank[rank]
            kernels = db.execute('select k.start,k.end,s.value,k.graphNodeId from CUPTI_ACTIVITY_KIND_KERNEL k '
                                 'join StringIds s on k.demangledName=s.id '
                                 'where k.globalPid=? and k.start>=? and k.end<=?',
                                 (global_pid,start,end)).fetchall()
            if not kernels: raise ValueError(f'no steady GPU kernels for rank {rank}')
            totals = {}; names = {}; graph_comm = {}; source_nodes = defaultdict(set)
            for begin,finish,name,node in kernels:
                group = category(name); duration = finish-begin
                item = totals.setdefault(group, {'calls':0,'gpu_sum_ns':0,'graph_node_calls':0})
                item['calls'] += 1; item['gpu_sum_ns'] += duration; item['graph_node_calls'] += int(node is not None)
                item = names.setdefault(name, {'calls':0,'gpu_sum_ns':0})
                item['calls'] += 1; item['gpu_sum_ns'] += duration
                if worker is not None and group == 'collective' and node is not None:
                    source = source_call(global_pid,node)
                    if source is None: raise ValueError(f'rank {rank}: graph collective node {node} has no capture attribution')
                    capture_start, operation, size = source
                    source_nodes[(capture_start,operation,size)].add(node)
                    comm = graph_comm.setdefault(operation, {'calls':0,'input_bytes':0,'sizes':{}})
                    comm['calls'] += 1; comm['input_bytes'] += size
                    comm['sizes'][str(size)] = comm['sizes'].get(str(size),0)+1
            if any(len(nodes) != 1 for nodes in source_nodes.values()):
                raise ValueError(f'rank {rank}: collective capture mapped to multiple replay kernels')
            record = {'rank':rank, 'global_pid':global_pid, 'nvtx_range_ns':[start,end],
                      'categories':totals, 'kernels':names}
            if worker is not None:
                observed = {x['rank']:x for x in worker.get('trace_communication') or []}
                if rank not in observed: raise ValueError(f'rank {rank}: missing host communication events')
                host = observed[rank]['operations']
                if any(event['capturing'] for item in host.values() for event in item['events']):
                    raise ValueError(f'rank {rank}: graph capture occurred during steady trace')
                record['communication'] = {'host_issued':host,'graph_replay':graph_comm,
                    'byte_scope':'input tensor payload bytes, not inferred physical wire traffic',
                    'graph_mapping':'capture NVTX range -> CUDA graph node/origin -> replayed NCCL kernel; one kernel per observed capture call'}
            output.append(record)
        return {'status':'passed', 'ranks':output,
                'scope':'device kernel durations within each process NVTX interval; asynchronous durations are inclusive and cannot be summed into wall time',
                'classification':'name-based GEMM/GEMV, unrecognized kernels remain other'}
    finally:
        db.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sqlite', type=Path, required=True)
    p.add_argument('--tp', type=int, choices=(1,2,4), required=True)
    p.add_argument('--worker-json', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    try:
        worker = json.loads(a.worker_json.read_text()) if a.worker_json else None
        result = analyze(a.sqlite, a.tp, worker)
    except (ValueError, sqlite3.Error) as exc: result = {'status':'failed', 'error':str(exc)}
    a.output.write_text(json.dumps(result, indent=2)+'\n'); print(json.dumps({'status':result['status']}))
    return int(result['status'] != 'passed')

if __name__ == '__main__': raise SystemExit(main())
