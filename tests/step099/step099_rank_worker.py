"""Real NCCL rank: compare vocab shards and values with hash-matched frozen vLLM."""
import argparse
from contextlib import ExitStack
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import torch


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    rank=int(os.environ['RANK']); world=int(os.environ['WORLD_SIZE']); local=int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local);device=torch.device('cuda',local)
    from minivllm.distributed import init_distributed_environment,initialize_model_parallel,get_tp_group,destroy_model_parallel,destroy_distributed_environment
    from minivllm.layers.linear import VocabParallelEmbedding,ParallelLMHead
    from minivllm.layers.logits_processor import LogitsProcessor
    from vllm.model_executor.layers import vocab_parallel_embedding as ref
    from vllm.model_executor.layers import logits_processor as ref_logits
    from vllm.platforms import interface as ref_platform, cuda as ref_cuda
    frozen=ROOT/'learning_notes/14_vllm_from_scratch/分布式并行完整需求/参考资料/vllm_0_28_0/model_executor/layers/vocab_parallel_embedding.py'
    digest=hashlib.sha256(frozen.read_bytes()).hexdigest()
    assert hashlib.sha256(Path(ref.__file__).read_bytes()).hexdigest()==digest
    frozen_logits=frozen.parent/'logits_processor.py'
    assert hashlib.sha256(Path(ref_logits.__file__).read_bytes()).digest()==hashlib.sha256(frozen_logits.read_bytes()).digest()
    frozen_root=frozen.parents[2]
    for module,relative in ((ref_platform,'platforms/interface.py'),(ref_cuda,'platforms/cuda.py')):
        assert hashlib.sha256(Path(module.__file__).read_bytes()).digest()==hashlib.sha256((frozen_root/relative).read_bytes()).digest()
    init_distributed_environment(world,rank,'env://',local,timeout=timedelta(seconds=90))
    try:
        initialize_model_parallel(world,1,timeout=timedelta(seconds=90));group=get_tp_group();cases=[]
        with ExitStack() as stack:
            stack.enter_context(patch.object(ref,'get_tensor_model_parallel_rank',return_value=rank))
            stack.enter_context(patch.object(ref,'get_tensor_model_parallel_world_size',return_value=world))
            stack.enter_context(patch.object(ref,'tensor_model_parallel_all_reduce',side_effect=group.all_reduce))
            stack.enter_context(patch.object(ref_logits,'tensor_model_parallel_all_gather',side_effect=group.all_gather))
            stack.enter_context(patch.object(ref_logits,'get_current_vllm_config',return_value=SimpleNamespace(model_config=None)))
            for vocab in (7,129,151936):
                dim=4
                # Distinct, exactly representable rows expose incorrect shard or token order.
                source=(torch.arange(vocab,dtype=torch.float32).reshape(-1,1)-(vocab-1))*torch.tensor([[1.,2.,3.,4.]])*0.0001
                with torch.device(device):
                    ours=VocabParallelEmbedding(vocab,dim); oracle=ref.VocabParallelEmbedding(vocab,dim)
                    head=ParallelLMHead(vocab,dim); oracle_head=ref.ParallelLMHead(vocab,dim)
                for layer in (ours,oracle,head,oracle_head):layer.weight_loader(layer.weight,source)
                oi=ours.shard_indices; ri=oracle.shard_indices
                fields=('padded_org_vocab_start_index','padded_org_vocab_end_index','org_vocab_start_index','org_vocab_end_index','num_org_elements','num_org_elements_padded','num_org_vocab_padding')
                golden={f:getattr(ri,f) for f in fields}
                assert all(getattr(oi,f)==v for f,v in golden.items())
                assert ours.num_embeddings_padded==oracle.num_embeddings_padded
                assert tuple(ours.weight.shape)==tuple(oracle.weight.shape)
                assert torch.equal(ours.weight,oracle.weight)
                assert torch.equal(head.weight,oracle_head.weight)
                # Every shard's first and last valid ID plus every cross-shard boundary.
                boundaries={0,vocab-1}
                stride=ours.num_embeddings_per_partition
                for r in range(world):
                    for token in (r*stride-1,r*stride,(r+1)*stride-1,(r+1)*stride):
                        if 0<=token<vocab:boundaries.add(token)
                ids=torch.tensor(sorted(boundaries),dtype=torch.int64,device=device)
                got=ours(ids);expected=oracle(ids)
                assert torch.equal(got,expected)
                assert torch.equal(got,source.to(device)[ids])
                invalid=torch.tensor([vocab],device=device)
                if world>1 or vocab<ours.num_embeddings_padded:
                    assert torch.equal(ours(invalid),oracle(invalid))
                    assert torch.equal(ours(invalid),torch.zeros((1,dim),device=device))
                # Raw padded rows are poisoned to ensure they never reach top-k.
                with torch.no_grad():
                    head.weight[oi.num_org_elements:].fill_(1e5)
                    oracle_head.weight[oi.num_org_elements:].fill_(1e5)
                hidden=torch.tensor([[1.,2.,3.,4.],[4.,-2.,1.,3.]],device=device)
                processor=LogitsProcessor(vocab,scale=0.5,soft_cap=3.)
                logits=processor(head,hidden)
                upstream_processor=ref_logits.LogitsProcessor(vocab,scale=0.5,soft_cap=3.)
                assert upstream_processor.use_all_gather
                upstream_logits=upstream_processor(oracle_head,hidden)
                torch.testing.assert_close(logits,upstream_logits,rtol=1e-5,atol=1e-5)
                local_bias=torch.arange(head.num_embeddings_per_partition,device=device,dtype=torch.float32)/100
                biased=processor(head,hidden,embedding_bias=local_bias)
                upstream_biased=upstream_processor(oracle_head,hidden,embedding_bias=local_bias)
                torch.testing.assert_close(biased,upstream_biased,rtol=1e-5,atol=1e-5)
                expected=torch.nn.functional.linear(hidden,source.to(device))
                expected=torch.tanh(expected/3.)*3.*0.5
                torch.testing.assert_close(logits,expected,rtol=1e-5,atol=1e-5)
                assert logits.shape==(2,vocab)
                top_values,top_ids=logits.topk(2,dim=-1)
                assert torch.equal(top_ids,torch.tensor([[vocab-1,vocab-2]]*2,device=device))
                assert bool((top_values[:,0]>top_values[:,1]).all())
                assert head.weight is not ours.weight
                head.tie_weights(ours);assert head.weight is ours.weight
                tied=processor(head,hidden);torch.testing.assert_close(tied,expected,rtol=1e-5,atol=1e-5)
                delta=(logits-expected).abs()
                worst_flat=int(delta.argmax())
                worst_row,worst_token=divmod(worst_flat,vocab)
                cases.append({'vocab':vocab,'padded':ours.num_embeddings_padded,'local_rows':ours.num_embeddings_per_partition,'golden':golden,'source_sha256':digest,'logits_source_sha256':hashlib.sha256(frozen_logits.read_bytes()).hexdigest(),'storage_device':str(ours.weight.device),'tied_same_parameter':head.weight is ours.weight,'max_abs':float((logits-expected).abs().max().detach()),'frozen_max_abs':float((logits-upstream_logits).abs().max().detach()),'relative_l2':float(((logits-expected).norm()/expected.norm()).detach()),'max_location':[worst_row,worst_token],'max_actual':float(logits[worst_row,worst_token].detach()),'max_reference':float(expected[worst_row,worst_token].detach()),'top1':top_ids[:,0].tolist(),'top2':top_ids[:,1].tolist(),'top1_values':top_values[:,0].tolist(),'top2_values':top_values[:,1].tolist(),'margins':(top_values[:,0]-top_values[:,1]).tolist(),'ids':sorted(boundaries)})
        a.out.mkdir(parents=True,exist_ok=True);(a.out/f'rank{rank}.json').write_text(json.dumps({'rank':rank,'world':world,'pid':os.getpid(),'device_uuid':str(torch.cuda.get_device_properties(local).uuid),'complete':True,'cases':cases},indent=2)+'\n')
    finally:
        destroy_model_parallel();destroy_distributed_environment()

if __name__=='__main__':main()
