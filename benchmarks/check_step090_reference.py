"""090 冻结 vLLM 参考探针：固定 token 历史，逐 rank 导出测试数据。"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


from functools import partial
from benchmarks.step090_reference_hooks import install, flush, local_fp32


def verify_oracle():
    import importlib.util
    import tarfile
    base=ROOT/'learning_notes/14_vllm_from_scratch/分布式并行完整需求/参考资料'
    manifest_path=base/'源码清单.json'
    manifest=json.loads(manifest_path.read_text())
    installed=Path(importlib.util.find_spec('vllm').origin).parent
    for row in manifest['files']:
        name=row['path']
        expected=row['sha256']
        for path in (base/'vllm_0_28_0'/name, installed/name):
            if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
                raise RuntimeError(f'冻结参考不匹配：{path}')
    archive=ROOT/'learning_notes/14_vllm_from_scratch/单卡执行优化需求/参考资料/vllm_execution_20261029.tar.gz'
    archive_count=0
    with tarfile.open(archive,'r:gz') as tar:
        for member in tar.getmembers():
            if not member.isfile() or not member.name.endswith('.py'):
                continue
            expected=hashlib.sha256(tar.extractfile(member).read()).hexdigest()
            if hashlib.sha256((installed/member.name).read_bytes()).hexdigest()!=expected:
                raise RuntimeError(f'历史冻结参考不匹配：{member.name}')
            archive_count+=1
    return {'source_manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            'checked_files':len(manifest['files']), 'historical_archive_files':archive_count,
            'historical_archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=('8b','14b'),required=True)
    p.add_argument('--tp',type=int,choices=(1,2,4),required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--repeats',type=int,default=3)
    a=p.parse_args()
    if a.model=='14b' and a.tp==1: p.error('14B BF16 单卡不允许')
    if a.repeats<2: p.error('必须至少重复两次')
    oracle=verify_oracle()
    from benchmarks.check_step090_models import resolve_snapshot
    from vllm import LLM, SamplingParams
    path,_=resolve_snapshot(a.model)
    a.output.mkdir(parents=True,exist_ok=True)
    prompt=json.loads((ROOT/'benchmarks/results/step090/forward_8b_retry.json').read_text())['forward']['prompt_token_ids']
    prompt=(prompt*((128+len(prompt)-1)//len(prompt)))[:128]
    result={'model':a.model,'tp':a.tp,'prompt_token_ids':prompt,'oracle':oracle,'status':'failed','iterations':[]}
    start=time.monotonic()
    try:
        llm=LLM(model=str(path),tokenizer=str(path),dtype='bfloat16',tensor_parallel_size=a.tp,
                enforce_eager=True,max_model_len=2048,max_num_seqs=1,max_num_batched_tokens=128,
                gpu_memory_utilization=.86,attention_config={'backend':'TRITON_ATTN'},
                enable_prefix_caching=False,disable_log_stats=True,seed=0)
        result['ranks']=llm.apply_model(install)
        for i in range(a.repeats):
            output=llm.generate([{'prompt_token_ids':prompt}],
                SamplingParams(temperature=0,max_tokens=1,logprobs=2,ignore_eos=True),use_tqdm=False)[0].outputs[0]
            records=llm.apply_model(partial(flush, directory=str(a.output.resolve()), iteration=i))
            result['iterations'].append({'token_ids':list(output.token_ids),'ranks':records})
        result['local_fp32']=llm.apply_model(partial(local_fp32, directory=str(a.output.resolve())))
        result['status']='passed'
    except Exception as exc:
        result['error']=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        result['elapsed_seconds']=time.monotonic()-start
        (a.output/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__':main()
