"""Test-only layer observations for the 111 mixed-input numerical failure."""
from pathlib import Path


def install(worker, request_id, positions):
    import torch
    runner = worker.model_runner
    model = runner.model
    model._step111_diag = {}
    model._step111_row = None
    original = runner.prepare_attn

    def prepare(batch):
        result = original(batch)
        model._step111_row = None
        for index, req in enumerate(batch.req_ids):
            if req != request_id and not req.startswith(request_id + '-'):
                continue
            start, end = map(int, batch.query_start_loc_np[index:index + 2])
            for offset, position in enumerate(batch.positions[start:end].detach().cpu().tolist()):
                if position in positions:
                    model._step111_row = (start + offset, position)
        return result
    runner.prepare_attn = prepare

    def hook(name):
        def capture(_module, _args, output):
            if model._step111_row is None or model._step111_current_layer != int(name.split(':')[0]):
                return
            row, pos = model._step111_row
            outputs = output if isinstance(output, tuple) else (output,)
            values = [x.detach()[row].cpu().clone() if isinstance(x, torch.Tensor)
                      and x.ndim >= 2 and row < x.shape[0] else None for x in outputs]
            model._step111_diag.setdefault(name, {})[pos] = values
        return capture

    model._step111_current_layer = None
    def rope_pre(module, args):
        if model._step111_row is None or model._step111_current_layer != 0:
            return
        row, logical_position = model._step111_row
        actual_position = int(args[0][row].item())
        model._step111_diag.setdefault('0:rope_position', {})[logical_position] = [
            torch.tensor([actual_position])]
        model._step111_diag.setdefault('0:rope_cache', {})[logical_position] = [
            module.cos_sin_cache[actual_position].detach().cpu().clone()]
    model.model.layers[0].self_attn.rotary_emb.register_forward_pre_hook(rope_pre)

    def layer_pre(index):
        def record(_module, _args):
            model._step111_current_layer = index
        return record
    for index, layer in enumerate(model.model.layers):
        layer.register_forward_pre_hook(layer_pre(index))
        for name, module in [('input_norm', layer.input_layernorm),
                             ('qkv', layer.self_attn.qkv_proj),
                             ('q_norm', layer.self_attn.q_norm),
                             ('k_norm', layer.self_attn.k_norm),
                             ('rope', layer.self_attn.rotary_emb),
                             ('attn', layer.self_attn.attn),
                             ('o_proj', layer.self_attn.o_proj),
                             ('post_norm', layer.post_attention_layernorm),
                             ('gate_up', layer.mlp.gate_up_proj),
                             ('silu', layer.mlp.act_fn),
                             ('down_proj', layer.mlp.down_proj), ('layer', layer)]:
            module.register_forward_hook(hook(f'{index}:{name}'))
    return True


def flush(worker, directory):
    import torch
    import torch.distributed as dist
    rank = dist.get_rank() if dist.is_initialized() else 0
    path = Path(directory) / f'rank{rank}.pt'
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(worker.model_runner.model._step111_diag, path)
    return {'rank': rank, 'path': str(path)}
