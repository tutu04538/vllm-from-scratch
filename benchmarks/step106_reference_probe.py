"""106 真实 EAGLE3 冻结上游参考运行；不导入本项目模型。"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['VLLM_USE_V2_MODEL_RUNNER'] = '1'
os.environ['VLLM_ALLOW_INSECURE_SERIALIZATION'] = '1'

from benchmarks.check_step090_models import resolve_snapshot
from benchmarks.check_step090_reference import verify_oracle
from benchmarks.step103_workload import prompts


def install(worker, upstream=True, forced_history=None, forced_drafts=None, head_size=128,
            forced_first=None):
    import torch
    runner = worker.model_runner
    spec = runner.speculator
    draft = spec.model
    records = []
    runner._step106_records = records
    original_prepare = runner.prepare_attn
    def prepare(*args, **kwargs):
        tables, slots = original_prepare(*args, **kwargs)
        runner._step106_block_table = tables[0].detach().cpu().clone()
        return tables, slots
    runner.prepare_attn = prepare

    def snapshot(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()
        if isinstance(value, (tuple, list)):
            return [snapshot(item) for item in value]
        if isinstance(value, dict):
            return {key: snapshot(item) for key, item in value.items()}
        if isinstance(value, (int, float, bool, str)):
            return value
        return None

    def wrap(obj, name, label=None):
        original = getattr(obj, name)
        def measured(*args, **kwargs):
            result = original(*args, **kwargs)
            records.append({'event': label or name, 'inputs': snapshot(args), 'output': snapshot(result)})
            return result
        setattr(obj, name, measured)

    wrap(draft, 'combine_hidden_states')
    wrap(draft, 'compute_logits')
    wrap(draft.logits_processor, 'forward', 'draft_raw_logits')
    def layer_hook(module, args, kwargs, output):
        if upstream:
            from vllm.forward_context import get_forward_context
        else:
            from minivllm.forward_context import get_forward_context
        ctx = get_forward_context()
        attn = module.self_attn.attn
        mappings = ctx.slot_mapping or {}
        slots = mappings.get(attn.layer_name) if upstream else mappings.get('slot_mapping')
        if slots is not None and slots.ndim > 1:
            slots = slots[0]
        count = output[0].shape[0] if isinstance(output, tuple) else output.shape[0]
        slot_values = slots[:count] if slots is not None else None
        cache = attn.kv_cache
        kv = None
        if slot_values is not None:
            valid = slot_values[slot_values >= 0].long()
            block = torch.div(valid, 16, rounding_mode='floor')
            offset = valid % 16
            if upstream:
                kv = cache[block, :, offset, :].reshape(len(valid), -1, 2, head_size).permute(0, 2, 1, 3)
            else:
                kv = cache[:, block, offset, :, :].permute(1, 0, 2, 3)
        records.append({'event': 'draft_layer', 'inputs': snapshot(args),
                        'kwargs': snapshot(kwargs), 'output': snapshot(output),
                        'draft_input_ids': snapshot(spec.input_buffers.input_ids[:count]),
                        'draft_positions': snapshot(spec.input_buffers.positions[:count]),
                        'draft_slot_mapping': snapshot(slot_values), 'written_kv': snapshot(kv)})
    for layer in draft.model.layers:
        layer.register_forward_hook(layer_hook, with_kwargs=True)
        def save_kernel_event(name):
            def hook(module, args, output):
                value = output[0] if isinstance(output, tuple) else output
                records.append({'event': name, 'inputs': snapshot(args), 'output': snapshot(value)})
            return hook
        layer.self_attn.qkv_proj.register_forward_hook(save_kernel_event('draft_qkv'))
        layer.self_attn.attn.register_forward_hook(save_kernel_event('draft_attention'))
    original_propose = spec.propose
    forced_index = 0
    def propose(input_batch, attn_metadata, slot_mappings, last_hidden_states, aux_hidden_states,
                num_sampled, num_rejected, *args, **kwargs):
        nonlocal forced_index
        fields = ('input_ids', 'positions', 'query_start_loc', 'seq_lens', 'idx_mapping',
                  'num_tokens', 'num_tokens_after_padding', 'num_reqs')
        record = {'event': 'propose', 'batch': {name: snapshot(getattr(input_batch, name, None))
                                                for name in fields},
                  'slot_mappings': snapshot(slot_mappings), 'num_sampled': snapshot(num_sampled),
                  'num_rejected': snapshot(num_rejected)}
        records.append(record)
        result = original_propose(input_batch, attn_metadata, slot_mappings, last_hidden_states,
                                  aux_hidden_states, num_sampled, num_rejected, *args, **kwargs)
        record['output'] = snapshot(result)
        # 每请求的有效上下文，不包含本轮被拒的末尾行。按逻辑位置读取各自 KV；
        # 两端允许物理块不同，因此不能直接比较整个 cache 或 block 编号。
        if int(input_batch.num_reqs) != 1:
            raise RuntimeError('固定历史探针目前只支持 B1')
        length = int(input_batch.seq_lens[0]) - int(num_rejected[0])
        if length < 0:
            raise RuntimeError('有效上下文长度为负')
        table = runner._step106_block_table[0].to(input_batch.positions.device).long()
        logical = torch.arange(length, device=input_batch.positions.device)
        blocks = table[torch.div(logical, 16, rounding_mode='floor')]
        offsets = logical % 16
        def read_kv(cache):
            if upstream:
                return cache[blocks, :, offsets, :].reshape(length, -1, 2, head_size).permute(0, 2, 1, 3)
            return cache[:, blocks, offsets, :, :].permute(1, 0, 2, 3)
        record['valid_context_length'] = length
        record['target_valid_kv'] = snapshot(read_kv(runner.model.model.layers[0].self_attn.attn.kv_cache))
        record['draft_valid_kv'] = snapshot(read_kv(draft.model.layers[0].self_attn.attn.kv_cache))
        if forced_drafts is not None:
            if forced_index >= len(forced_drafts):
                raise RuntimeError('固定草稿历史不足')
            forced = torch.as_tensor(forced_drafts[forced_index], dtype=result.dtype, device=result.device)
            if forced.shape != result.shape:
                raise RuntimeError('固定草稿形状不匹配')
            result.copy_(forced)
            record['forced_output'] = snapshot(result)
            forced_index += 1
        elif forced_first is not None and forced_index == 0:
            forced = torch.as_tensor(forced_first, dtype=result.dtype, device=result.device)
            if forced.shape != result.shape:
                raise RuntimeError('首轮固定草稿形状不匹配')
            result.copy_(forced)
            record['forced_output'] = snapshot(result)
            forced_index += 1
        return result
    spec.propose = propose
    if forced_history is not None:
        original_sample = runner.sample
        cursor = 0
        def sample(*args, **kwargs):
            nonlocal cursor
            output = original_sample(*args, **kwargs)
            sampled, counts, _ = output
            if counts.numel() != 1:
                raise RuntimeError('固定历史探针目前只支持 B1')
            count = int(counts[0])
            actual = snapshot(sampled.sampled_token_ids[0, :count])
            values = forced_history[cursor:cursor + count]
            if len(values) != count:
                # 结束轮采样可能超过 max_tokens，Scheduler 只提交历史中剩余的有效部分。
                # 多余项保留原采样结果，不将它们冒充已提交历史。
                values = values + actual[len(values):].tolist()
            sampled.sampled_token_ids[0, :count] = torch.as_tensor(
                values, dtype=sampled.sampled_token_ids.dtype, device=sampled.sampled_token_ids.device)
            records.append({'event': 'target_sample', 'output': actual,
                            'forced_output': snapshot(sampled.sampled_token_ids[0, :count])})
            cursor += count
            return output
        runner.sample = sample
    target = runner.model
    def storages(model):
        unique = {}
        for parameter in model.parameters():
            storage = parameter.untyped_storage()
            unique[(str(parameter.device), storage.data_ptr())] = storage.nbytes()
        return unique
    target_storage = storages(target)
    draft_storage = storages(draft)
    shared = target_storage.keys() & draft_storage.keys()
    architectures = (draft.config.get('architectures') if isinstance(draft.config, dict)
                     else draft.config.architectures)
    return {'draft_class': type(draft).__name__, 'draft_architectures': architectures,
            'embedding_shared': draft.model.embed_tokens is target.model.embed_tokens,
            'lm_head_shared': draft.lm_head is target.lm_head,
            'target_unique_parameter_storage_bytes': sum(target_storage.values()),
            'draft_unique_parameter_storage_bytes': sum(draft_storage.values()),
            'shared_parameter_storage_bytes': sum(draft_storage[key] for key in shared),
            'draft_incremental_parameter_storage_bytes': sum(draft_storage.values())
                - sum(draft_storage[key] for key in shared),
            'gpu_allocated_bytes': torch.cuda.memory_allocated(worker.device),
            'gpu_reserved_bytes': torch.cuda.memory_reserved(worker.device),
            'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated(worker.device),
            'device_uuid': str(torch.cuda.get_device_properties(worker.device).uuid)}


def flush(worker, path):
    import torch
    runner = worker.model_runner
    torch.save(runner._step106_records, path)
    events = len(runner._step106_records)
    runner._step106_records.clear()
    return events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--k', type=int, choices=[1, 2, 4], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--forced-from', type=Path)
    args = parser.parse_args()
    if args.repeats < 2 and args.forced_from is None:
        parser.error('参考测量至少重复两次')
    args.output.mkdir(parents=True, exist_ok=True)
    oracle = verify_oracle()
    target, _ = resolve_snapshot('8b')
    draft = Path('/data/syh/models/converted/AngelSlim/Qwen3-8B_eagle3/9629dfce7a4a10564dd48d3e5485c3976095653c')
    forced_history = forced_drafts = None
    if args.forced_from:
        import torch
        prior = json.loads((args.forced_from / 'summary.json').read_text())
        if prior['k'] != args.k:
            raise ValueError('固定历史的 K 不匹配')
        forced_history = prior['runs'][0]['tokens']
        events = torch.load(prior['runs'][0]['trace'], map_location='cpu', weights_only=True)
        forced_drafts = [event['output'].tolist() for event in events if event['event'] == 'propose']
        if args.repeats != 1:
            parser.error('固定历史探针独立进程每次运行一次')
    from vllm import LLM, SamplingParams
    start = time.monotonic()
    llm = LLM(model=str(target), skip_tokenizer_init=True, dtype='bfloat16',
              tensor_parallel_size=1, enforce_eager=True, max_model_len=2048,
              max_num_seqs=1, max_num_batched_tokens=128,
              kv_cache_memory_bytes=192 * 2 * 16 * 8 * 128 * 2 * 36,
              gpu_memory_utilization=.86, attention_config={'backend': 'TRITON_ATTN'},
              enable_prefix_caching=False, disable_log_stats=True, seed=0,
              speculative_config={'method': 'eagle3', 'model': str(draft),
                                  'num_speculative_tokens': args.k})
    identity = llm.collective_rpc(install, args=(True, forced_history, forced_drafts))
    prompt = prompts('8b', 1)[0]
    runs = []
    for index in range(args.repeats):
        output = llm.generate([{'prompt_token_ids': prompt}],
                              SamplingParams(temperature=0, max_tokens=32, ignore_eos=True),
                              use_tqdm=False)
        path = (args.output / f'run{index}.pt').resolve()
        counts = llm.collective_rpc(flush, args=(str(path),))
        runs.append({'tokens': list(output[0].outputs[0].token_ids),
                     'trace': str(path), 'events_per_rank': counts})
    payload = {'status': 'reference_runs_complete', 'oracle': oracle, 'k': args.k,
               'identity': identity, 'prompt_token_ids': prompt, 'runs': runs,
               'elapsed_seconds': time.monotonic() - start}
    (args.output / 'summary.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': payload['status'], 'k': args.k, 'runs': len(runs)}))


if __name__ == '__main__':
    main()
