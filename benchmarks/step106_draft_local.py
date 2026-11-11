"""14B 真实草稿的局部参考或项目计算；不加载 14B target 权重。"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DRAFT = Path('/data/syh/models/converted/AngelSlim/Qwen3-14B_eagle3/3d13517724e81cb409ddf1d4650772ec52f1e18e')
TARGET = Path('/data/syh/.cache/huggingface/hub/models--Qwen--Qwen3-14B/snapshots/40c069824f4251a91eefaf281ebe4c544efd3e18')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--implementation', choices=['reference', 'mini'], required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    os.environ['VLLM_USE_V2_MODEL_RUNNER'] = '1'
    torch.cuda.set_device(0)
    config = json.loads((DRAFT / 'config.json').read_text())
    weights = load_file(str(DRAFT / 'model.safetensors'))
    torch.set_default_dtype(torch.bfloat16)
    if args.implementation == 'reference':
        from vllm.config import (DeviceConfig, ModelConfig, ParallelConfig, SpeculativeConfig,
                                 VllmConfig, set_current_vllm_config)
        from vllm.config.cache import CacheConfig
        from vllm.config.scheduler import SchedulerConfig
        from vllm.distributed import init_distributed_environment, initialize_model_parallel
        from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM
        from benchmarks.check_step090_reference import verify_oracle
        oracle = verify_oracle()
        target = ModelConfig(model=str(TARGET), dtype='bfloat16', max_model_len=64)
        draft = ModelConfig(model=str(DRAFT), dtype='bfloat16', max_model_len=64)
        parallel = ParallelConfig()
        speculative = SpeculativeConfig(model=str(DRAFT), method='eagle3', num_speculative_tokens=2,
                                        target_model_config=target, draft_model_config=draft,
                                        target_parallel_config=parallel)
        cfg = VllmConfig(model_config=target, speculative_config=speculative,
                         cache_config=CacheConfig(block_size=16, gpu_memory_utilization=.1),
                         parallel_config=parallel, device_config=DeviceConfig('cuda'),
                         scheduler_config=SchedulerConfig(max_num_seqs=1, max_num_batched_tokens=64,
                                                          max_model_len=64, is_encoder_decoder=False))
        with set_current_vllm_config(cfg):
            init_distributed_environment(world_size=1, rank=0, local_rank=0,
                                         distributed_init_method='tcp://127.0.0.1:29786', backend='gloo')
            initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
            model = Eagle3LlamaForCausalLM(vllm_config=cfg, prefix='')
            model.load_weights(iter(sorted(weights.items())))
    else:
        from minivllm.models.qwen3_eagle3 import Eagle3ForCausalLM
        oracle = None
        model = Eagle3ForCausalLM(config)
        model.load_weights(iter(sorted(weights.items())))
    model = model.to('cuda').eval()
    generator = torch.Generator(device='cpu').manual_seed(106)
    aux = torch.randn((1, 3 * config['hidden_size']), generator=generator, dtype=torch.float32).to('cuda', torch.bfloat16)
    with torch.inference_mode():
        combined = model.combine_hidden_states(aux)
        mapped = model.compute_logits(combined)
    args.output.mkdir(parents=True, exist_ok=True)
    path = (args.output / 'output.pt').resolve()
    torch.save({'aux': aux.cpu(), 'combined': combined.cpu(), 'mapped_logits': mapped.cpu()}, path)
    report = {'status': 'local_run_complete', 'implementation': args.implementation,
              'scope': '14B draft FC and mapped logits only; target/attention not executed',
              'oracle': oracle, 'trace': str(path), 'architecture': config['architectures'],
              'uuid': str(torch.cuda.get_device_properties(0).uuid)}
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
