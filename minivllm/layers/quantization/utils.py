"""冻结 vLLM 融合层排除规则的本项目子集。"""


def is_layer_skipped(prefix, ignored_layers, fused_mapping, *, match_mode='exact'):
    if match_mode == 'exact':
        match = lambda value: value in ignored_layers
    elif match_mode == 'substring':
        match = lambda value: any(name in value for name in ignored_layers)
    elif match_mode == 'suffix':
        match = lambda value: any(value == name or value.endswith('.' + name) for name in ignored_layers)
    else:
        raise ValueError(f'Unsupported layer skip match mode: {match_mode}')
    name = prefix.split('.')[-1]
    if name in fused_mapping and match(prefix):
        return True
    if name in fused_mapping:
        decisions = [match(prefix.replace(name, child)) for child in fused_mapping[name]]
        if not decisions:
            raise ValueError(f'融合层 {prefix} 的逻辑分片不能为空')
        if any(value != decisions[0] for value in decisions[1:]):
            raise ValueError(f'Detected some but not all shards of {prefix} are quantized. '
                             'All shards of fused layers to have the same precision.')
        return decisions[0]
    if 'experts' in prefix and match_mode == 'exact':
        return any(prefix in item for item in ignored_layers if 'experts' in item)
    return match(prefix)
