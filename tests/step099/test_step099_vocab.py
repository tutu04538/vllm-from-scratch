"""Real GPU/rank golden fixture and local API checks for 099."""
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
import torch
from minivllm.layers.linear import VocabParallelEmbedding, ParallelLMHead
from minivllm.layers.logits_processor import LogitsProcessor

ROOT=Path(__file__).resolve().parents[2]
WORKER=ROOT/'tests/step099/step099_rank_worker.py'

@pytest.mark.parametrize('world',[1,2,4])
def test_frozen_and_real_ranks(world,tmp_path):
    assert torch.cuda.device_count()>=world, '需要真实 GPU，不以 skip 代替验收'
    cmd=[sys.executable,'-m','torch.distributed.run','--standalone',f'--nproc-per-node={world}',str(WORKER),'--out',str(tmp_path)]
    proc=subprocess.run(cmd,cwd=ROOT,env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},capture_output=True,text=True,timeout=180)
    assert proc.returncode==0,proc.stdout+proc.stderr
    records=[json.loads((tmp_path/f'rank{i}.json').read_text()) for i in range(world)]
    assert all(r['complete'] and r['rank']==i for i,r in enumerate(records))
    assert len({r['pid'] for r in records})==world
    assert len({r['device_uuid'] for r in records})==world
    for r in records:
        assert [x['vocab'] for x in r['cases']]==[7,129,151936]
        assert all(len(c['source_sha256'])==64 and c['tied_same_parameter'] for c in r['cases'])
        assert all(c['storage_device']==f"cuda:{r['rank']}" for c in r['cases'])
    print(json.dumps(records,ensure_ascii=False))


def test_tp1_padding_weight_validation_and_no_pad_sampling():
    embedding=VocabParallelEmbedding(7,3);head=ParallelLMHead(7,3)
    assert embedding.weight.shape==(64,3)
    source=torch.arange(21,dtype=torch.float32).reshape(7,3)
    embedding.weight_loader(embedding.weight,source)
    assert torch.equal(embedding.weight[:7],source)
    assert torch.equal(embedding.weight[7:],torch.zeros_like(embedding.weight[7:]))
    with pytest.raises(ValueError,match='形状不匹配'):
        embedding.weight_loader(embedding.weight,source[:-1])
    with pytest.raises(ValueError,match='打包参数'):
        embedding.weight_loader(embedding.weight,source,shard_id=0)
    head.weight_loader(head.weight,-torch.ones_like(source))
    with torch.no_grad():head.weight[7:].fill_(100)
    logits=LogitsProcessor(7)(head,torch.ones(2,3))
    assert logits.shape==(2,7)
    assert torch.equal(logits,torch.full((2,7),-3.))
    assert (logits.argmax(dim=-1)<7).all()
    ids,_=LogitsProcessor(7).get_top_k_tokens(head,torch.ones(2,3),3)
    assert (ids<7).all()
    assert head.weight is not embedding.weight
    assert head.tie_weights(embedding).weight is embedding.weight
    with pytest.raises(ValueError,match='规格'):
        ParallelLMHead(8,3).tie_weights(embedding)
