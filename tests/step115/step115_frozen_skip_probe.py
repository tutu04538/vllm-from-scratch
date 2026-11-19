"""独立进程的冻结 vLLM 融合层排除规则参考。"""
import json
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped

mapping = {'qkv_proj': ['q_proj', 'k_proj', 'v_proj'],
           'gate_up_proj': ['gate_proj', 'up_proj']}
cases = [
    ('model.layers.0.self_attn.qkv_proj', [], 'substring'),
    ('model.layers.0.self_attn.qkv_proj', ['q_proj', 'k_proj', 'v_proj'], 'substring'),
    ('model.layers.0.self_attn.qkv_proj', ['q_proj'], 'substring'),
    ('model.layers.0.self_attn.qkv_proj', ['model.layers.0.self_attn.qkv_proj'], 'exact'),
    ('model.layers.0.self_attn.qkv_proj', ['model.layers.0.self_attn.q_proj'], 'exact'),
    ('model.layers.0.mlp.gate_up_proj', ['gate_proj', 'up_proj'], 'substring'),
    ('model.layers.0.mlp.gate_up_proj', ['gate_proj'], 'substring'),
    ('model.layers.0.mlp.gate_up_proj', ['model.layers.0.mlp.gate_up_proj'], 'exact'),
    ('model.layers.0.self_attn.o_proj', ['o_proj'], 'suffix'),
]
results = []
for prefix, ignored, mode in cases:
    try:
        value = is_layer_skipped(prefix, ignored, mapping, match_mode=mode)
        results.append({'status': 'ok', 'value': value})
    except ValueError as exc:
        results.append({'status': 'error', 'value': str(exc)})
print(json.dumps({'cases': cases, 'results': results}))
