"""Test-only rank-local audit of AWQ methods after model load."""
def audit(worker):
    from minivllm.layers.quantization.awq import AutoAWQMarlinLinearMethod
    model = worker.model_runner.model
    layers = []
    for name, module in model.named_modules():
        method = getattr(module, 'quant_method', None)
        if not isinstance(method, AutoAWQMarlinLinearMethod):
            continue
        assert module._method_weights_ready and method.supports_execution
        assert method.post_load_layout == 'kernel'
        params = {key: getattr(module, key) for key in ('qweight', 'qzeros', 'scales')}
        assert all(param.layout_state == 'kernel' for param in params.values())
        layers.append({'name': name,
                       'parameter_bytes': sum(t.numel() * t.element_size() for t in params.values()),
                       'workspace_bytes': method.workspace.numel() * method.workspace.element_size(),
                       'logical_nk': list(module.marlin_logical_nk),
                       'padded_nk': list(module.marlin_padded_nk)})
    assert layers and model._quant_postload_state == 'ready'
    return {'rank': worker.rank or 0, 'state': model._quant_postload_state,
            'layers': layers, 'quant_parameter_bytes': sum(x['parameter_bytes'] for x in layers),
            'workspace_bytes': sum(x['workspace_bytes'] for x in layers)}
