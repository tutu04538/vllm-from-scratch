"""090: 固定真实模型、审计 checkpoint 并无损转换 EAGLE3。"""
from __future__ import annotations
import argparse
from collections import OrderedDict
from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from minivllm.model_loader.weight_utils import get_safetensors_files, iter_weights

MODELS = {
    '8b': ('Qwen/Qwen3-8B', 'b968826d9c46dd6066d109eabc6255188de91218', 36, 4096, 32),
    '14b': ('Qwen/Qwen3-14B', '40c069824f4251a91eefaf281ebe4c544efd3e18', 40, 5120, 40),
    '8b-draft': ('AngelSlim/Qwen3-8B_eagle3', '9629dfce7a4a10564dd48d3e5485c3976095653c', 1, 4096, 32),
    '14b-draft': ('AngelSlim/Qwen3-14B_eagle3', '3d13517724e81cb409ddf1d4650772ec52f1e18e', 1, 5120, 40),
}
CONFIG_HASHES = {
    '8b': 'f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30',
    '14b': 'e73c3664ca09b10a673fef0c22e8a6b456201d49bd4713c9691f775720e8857a',
    '8b-draft': '360aa6861f74205c9b07dede8325b4638464d077c895f438e4ee20f0b2512dd7',
    '14b-draft': '3b0e2906a1189d11f10284695c0f4d72c87405793c9b101274516e8059164152',
}
CONVERTED_ROOT = Path('/data/syh/models/converted')


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for data in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def tensor_record(tensor):
    if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided:
        raise TypeError('state dict 必须仅含 strided tensor')
    raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
    return {'shape': list(tensor.shape), 'dtype': str(tensor.dtype),
            'sha256': hashlib.sha256(memoryview(raw)).hexdigest(), 'bytes': raw.nbytes}


def check_config(key, config, revision):
    repo, expected_revision, layers, hidden, heads = MODELS[key]
    if revision != expected_revision:
        raise ValueError(f'{repo} revision 错误：{revision} != {expected_revision}')
    expected = {'num_hidden_layers': layers, 'hidden_size': hidden, 'num_attention_heads': heads,
                'num_key_value_heads': 8, 'head_dim': 128, 'vocab_size': 151936,
                'tie_word_embeddings': False}
    for name, value in expected.items():
        if config.get(name) != value:
            raise ValueError(f'{repo} 配置 {name} 错误：{config.get(name)!r} != {value!r}')
    if key.endswith('-draft'):
        if config.get('model_type') != 'llama' or config.get('architectures') != ['LlamaForCausalLMEagle3']:
            raise ValueError(f'{repo} 非冻结 Llama EAGLE3 架构')
        if config.get('draft_vocab_size') != 32000:
            raise ValueError(f'{repo} 草稿词表不匹配')
    elif config.get('model_type') != 'qwen3' or config.get('architectures') != ['Qwen3ForCausalLM']:
        raise ValueError(f'{repo} 非冻结 Qwen3 架构')
    return {'repo': repo, 'revision': revision, 'config': expected}


def validate_pair(target_key, draft_key, target_config, draft_config):
    if target_key not in ('8b', '14b') or draft_key != target_key + '-draft':
        raise ValueError('target 和 draft 必须按 8B/14B 对应配对')
    check_config(target_key, target_config, MODELS[target_key][1])
    check_config(draft_key, draft_config, MODELS[draft_key][1])
    if target_config['vocab_size'] != draft_config['vocab_size']:
        raise ValueError('target/draft 词表不匹配')


def load_source(path):
    value = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(value, (dict, OrderedDict)) or not value:
        raise ValueError('原始 checkpoint 必须是非空 state dict')
    for name, tensor in value.items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise TypeError(f'原始 checkpoint 包含非 tensor 项：{name!r}')
    return value


def convert_draft(source, dest, config_path, identity):
    source, dest, config_path = Path(source), Path(dest), Path(config_path)
    source_digest = file_hash(source)
    weights = load_source(source)
    original = {k: tensor_record(v) for k, v in sorted(weights.items())}
    # clone makes aliased storage serializable without changing names, shapes or values.
    tensors = {k: v.detach().cpu().contiguous().clone() for k, v in weights.items()}
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        manifest_path = dest / 'conversion_manifest.json'
        if not manifest_path.is_file() or not (dest / 'model.safetensors').is_file():
            raise FileExistsError(f'已有目录不能覆盖：{dest}')
        previous = json.loads(manifest_path.read_text())
        if previous.get('identity') != identity or previous.get('source_sha256') != source_digest or previous.get('tensors') != original:
            raise ValueError(f'已存在转换产物身份或张量不匹配：{dest}')
        if (not (dest / 'config.json').is_file()
                or file_hash(dest / 'config.json') != file_hash(config_path)
                or previous.get('config_sha256') != file_hash(config_path)
                or previous.get('converted_sha256') != file_hash(dest / 'model.safetensors')):
            raise ValueError(f'已存在转换产物文件或配置不匹配：{dest}')
        audit_converted(dest / 'model.safetensors', original)
        return previous
    with tempfile.TemporaryDirectory(prefix='.step090-', dir=dest.parent) as temp:
        stage = Path(temp) / 'model'
        stage.mkdir()
        save_file(tensors, stage / 'model.safetensors')
        audit_converted(stage / 'model.safetensors', original)
        shutil.copyfile(config_path, stage / 'config.json')
        if file_hash(source) != source_digest:
            raise RuntimeError('转换期间原始文件发生变化')
        manifest = {'identity': identity, 'source_path': str(source), 'source_sha256': source_digest,
                    'converted_sha256': file_hash(stage / 'model.safetensors'),
                    'config_sha256': file_hash(stage / 'config.json'), 'tensors': original}
        (stage / 'conversion_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
        os.rename(stage, dest)
    return manifest


def audit_converted(path, expected):
    with safe_open(path, framework='pt', device='cpu') as f:
        if set(f.keys()) != set(expected):
            raise ValueError('转换 tensor 名称缺失或多出')
        for name in sorted(expected):
            if tensor_record(f.get_tensor(name)) != expected[name]:
                raise ValueError(f'转换 tensor 不一致：{name}')


def audit_eagle_mapping(path, config):
    with safe_open(path, framework='pt', device='cpu') as f:
        if not {'d2t','t2d'}.issubset(f.keys()):
            raise ValueError('缩减词表草稿缺少 d2t/t2d')
        d2t=f.get_tensor('d2t'); t2d=f.get_tensor('t2d')
        if d2t.dtype != torch.int64 or t2d.dtype != torch.bool:
            raise ValueError('d2t/t2d dtype 必须分别为 int64/bool')
        if list(d2t.shape)!=[config['draft_vocab_size']] or list(t2d.shape)!=[config['vocab_size']]:
            raise ValueError('d2t/t2d shape 与词表不符')
        ids=torch.arange(d2t.numel(),dtype=torch.int64)+d2t
        if (ids<0).any() or (ids>=t2d.numel()).any() or ids.unique().numel()!=ids.numel():
            raise ValueError('d2t 映射越界或重复')
        expected=torch.zeros_like(t2d);expected[ids]=True
        if not torch.equal(t2d,expected):
            raise ValueError('t2d 与 d2t 映射集合不一致')
        return {'draft_size':d2t.numel(),'target_size':t2d.numel(),
                'min_target_id':int(ids.min()),'max_target_id':int(ids.max()),
                'target_members':int(t2d.sum())}


def resolve_snapshot(key):
    from huggingface_hub import snapshot_download
    repo, revision, *_ = MODELS[key]
    path = Path(snapshot_download(
        repo, revision=revision, local_files_only=True,
        allow_patterns=['*.json', '*.txt', '*.safetensors', 'pytorch_model.bin', 'LICENSE*'],
    ))
    if path.name != revision:
        raise ValueError(f'缓存快照 revision 不符：{path}')
    if file_hash(path / 'config.json') != CONFIG_HASHES[key]:
        raise ValueError(f'{repo} config hash 与冻结元数据不符')
    config = json.loads((path / 'config.json').read_text())
    check_config(key, config, path.name)
    return path, config


def audit_target(path):
    files, weight_map = get_safetensors_files(str(path))
    records = {}
    for name, tensor in iter_weights(str(path)):
        records[name] = tensor_record(tensor)
    if weight_map is not None:
        index = json.loads((path/'model.safetensors.index.json').read_text())
        advertised = index.get('metadata', {}).get('total_size')
        if advertised is not None and advertised != sum(x['bytes'] for x in records.values()):
            raise ValueError('索引 total_size 与实际张量字节数不符')
    return {'weight_count': len(records), 'tensor_bytes': sum(x['bytes'] for x in records.values()),
            'weight_files': {Path(f).name: file_hash(f) for f in files}, 'tensors': records,
            'indexed': weight_map is not None}


def extract_prompt_ids(encoded):
    """接受 tokenizer 直接返回 ID 列表或包含 input_ids 的 BatchEncoding。"""
    ids = encoded.get('input_ids') if isinstance(encoded, Mapping) else encoded
    if not isinstance(ids, (list, tuple)) or not ids or any(type(x) is not int for x in ids):
        raise ValueError('chat template 必须返回非空的一维整数 token IDs')
    return list(ids)


def forward_8b(path, config):
    from minivllm import CacheConfig, DeviceConfig, LLMEngine, ModelConfig, SamplingParams, SchedulerConfig, UniProcExecutor, VllmConfig, Worker
    from minivllm.config import CompilationConfig
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(str(path),local_files_only=True,trust_remote_code=False)
    prompt=tokenizer.apply_chat_template([{'role':'user','content':'请用一句话介绍你自己。'}],
                                         tokenize=True,add_generation_prompt=True,
                                         enable_thinking=False)
    prompt=extract_prompt_ids(prompt)
    os.environ['VLLM_USE_V2_MODEL_RUNNER'] = '1'
    cfg = VllmConfig(model_config=ModelConfig(model=str(path), dtype='bfloat16', max_model_len=2048, hf_config=config),
        cache_config=CacheConfig(block_size=16, num_gpu_blocks=128),
        scheduler_config=SchedulerConfig(max_num_seqs=1, max_num_batched_tokens=128),
        device_config=DeviceConfig(device='cuda'), compilation_config=CompilationConfig(cudagraph_mode='none'),
        attention_backend='triton_attn')
    torch.cuda.reset_peak_memory_stats()
    engine = LLMEngine(cfg, UniProcExecutor(cfg, Worker(cfg)))
    try:
        runner=engine.engine_core.engine_core.model_executor.driver_worker.model_runner
        loaded_parameters=[{'name':name,'shape':list(param.shape),'dtype':str(param.dtype),
                            'device':str(param.device)}
                           for name,param in runner.model.named_parameters()]
        if not loaded_parameters:
            raise RuntimeError('8B 模型没有已加载参数')
        kv_caches=[{'name':name,'shape':list(cache.shape),'dtype':str(cache.dtype),
                    'device':str(cache.device),'bytes':cache.numel()*cache.element_size()}
                   for name,cache in runner.kv_caches.items()]
        if not kv_caches:
            raise RuntimeError('8B 模型没有已绑定 KV 缓存')
        engine.add_request('step090', prompt, SamplingParams(max_tokens=1, temperature=0.0))
        outputs = []
        for _ in range(50):
            if not engine.has_unfinished_requests(): break
            outputs.extend(engine.step())
        if not outputs: raise RuntimeError('没有模型输出')
        return {'prompt_token_ids': prompt, 'output_token_ids': list(outputs[-1].token_ids),
                'tokenizer_sha256':file_hash(path/'tokenizer.json'),
                'model_parameters':loaded_parameters,
                'kv_caches':kv_caches,
                'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                'gpu_peak_reserved_bytes': torch.cuda.max_memory_reserved()}
    finally: engine.shutdown()


def load_draft_cpu(path, config):
    """核验草稿加载；未绑定 target 的共享 embedding 时不能前向。"""
    from minivllm import ModelConfig
    from minivllm.model_loader.loader import get_model
    model=get_model(ModelConfig(model=str(path),dtype='bfloat16',max_model_len=2048,
                                hf_config=config),'cpu')
    attention=model.model.layers[0].self_attn
    if attention.q_norm is not None or attention.k_norm is not None:
        raise ValueError('Llama EAGLE3 不能加入 Q/K norm')
    if not model.model.tie_embeddings or model.d2t is None or model.t2d is None:
        raise ValueError('草稿共享 embedding 或映射状态不符合真实权重')
    return {'architecture':type(model).__name__, 'qk_norm':False,
            'shared_embedding_required':True, 'embedding_bound_to_target':False,
            'd2t_shape':list(model.d2t.shape), 't2d_shape':list(model.t2d.shape),
            'parameters':[{'name':name,'shape':list(param.shape),'dtype':str(param.dtype)}
                          for name,param in model.named_parameters()],
            'note':'仅验证加载；checkpoint没有embedding，尚未绑定target，不可前向'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', required=True, choices=['download', 'audit', 'convert', 'load-draft', 'forward'])
    p.add_argument('--model', required=True, choices=list(MODELS))
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--timeout', type=int, default=1800)
    p.add_argument('--converted-root', type=Path, default=CONVERTED_ROOT)
    a=p.parse_args()
    if a.timeout <= 0: p.error('--timeout must be positive')
    def expired(signum, frame): raise TimeoutError(f'超过 {a.timeout}s')
    signal.signal(signal.SIGALRM, expired); signal.alarm(a.timeout)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    result={'mode':a.mode,'model':a.model,'status':'failed','elapsed_seconds':None}
    start=time.monotonic()
    try:
        repo, revision, *_=MODELS[a.model]
        if a.mode == 'download':
            from huggingface_hub import HfApi, snapshot_download
            from huggingface_hub.constants import HF_HUB_CACHE
            info=HfApi().model_info(repo,revision=revision,files_metadata=True,timeout=60)
            if info.sha != revision: raise ValueError('远端 revision 不符')
            selected = [x for x in info.siblings if x.rfilename.endswith(('.json', '.txt', '.safetensors'))
                        or x.rfilename == 'pytorch_model.bin' or x.rfilename.startswith('LICENSE')]
            if any(x.size is None for x in selected): raise ValueError('远端文件大小未知')
            needed=sum(x.size for x in selected)
            cache_root=Path(HF_HUB_CACHE)
            cache_root.mkdir(parents=True,exist_ok=True)
            free=shutil.disk_usage(cache_root).free
            print(json.dumps({'cache':str(cache_root),'download_bytes':needed,
                              'estimated_with_temporary_copy':2*needed,'available_bytes':free}),flush=True)
            if free < 2*needed: raise OSError('缓存所在磁盘余量不足')
            path=Path(snapshot_download(repo,revision=revision,allow_patterns=['*.json','*.txt','*.safetensors','pytorch_model.bin','LICENSE*'],max_workers=4))
        path, config=resolve_snapshot(a.model)
        result.update(repo=repo,revision=revision,local_path=str(path),config_sha256=file_hash(path/'config.json'))
        if a.mode == 'download':
            result['files'] = {}
            for sibling in selected:
                name=sibling.rfilename
                actual=file_hash(path/name)
                expected=sibling.lfs.sha256 if sibling.lfs else None
                if expected and actual != expected:
                    raise ValueError(f'{name} 不匹配 Hub LFS SHA256')
                result['files'][name]={'bytes':(path/name).stat().st_size,'sha256':actual,'hub_lfs_sha256':expected}
        elif a.mode == 'audit':
            if a.model.endswith('-draft'): raise ValueError('audit 只用于 target safetensors')
            result['audit']=audit_target(path)
        elif a.mode == 'convert':
            if not a.model.endswith('-draft'): raise ValueError('convert 只用于 EAGLE3')
            target_key=a.model.removesuffix('-draft'); _, target_config=resolve_snapshot(target_key)
            validate_pair(target_key,a.model,target_config,config)
            dest=a.converted_root / 'AngelSlim' / repo.split('/')[1] / revision
            result['converted_path']=str(dest)
            result['conversion']=convert_draft(path/'pytorch_model.bin',dest,path/'config.json',{'repo':repo,'revision':revision})
            result['mapping_audit']=audit_eagle_mapping(dest/'model.safetensors',config)
        elif a.mode == 'load-draft':
            if not a.model.endswith('-draft'): raise ValueError('load-draft 只用于 EAGLE3')
            target_key=a.model.removesuffix('-draft'); _, target_config=resolve_snapshot(target_key)
            validate_pair(target_key,a.model,target_config,config)
            dest=a.converted_root / 'AngelSlim' / repo.split('/')[1] / revision
            result['converted_path']=str(dest)
            result['mapping_audit']=audit_eagle_mapping(dest/'model.safetensors',config)
            result['draft_load']=load_draft_cpu(dest,config)
        elif a.mode == 'forward':
            if a.model != '8b': raise ValueError('本关只允许 8B 单卡前向')
            result['forward']=forward_8b(path,config)
        result['status']='passed'
    except Exception as exc:
        result['error']=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        result['elapsed_seconds']=time.monotonic()-start
        result['cpu_max_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        signal.alarm(0)

if __name__ == '__main__': main()
