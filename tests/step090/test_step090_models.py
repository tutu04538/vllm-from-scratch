from __future__ import annotations
import json
from pathlib import Path
import pytest
import torch
from safetensors.torch import save_file
from benchmarks.check_step090_models import MODELS, audit_converted, check_config, convert_draft, tensor_record, validate_pair
from minivllm.model_loader.weight_utils import iter_weights


def _config(key):
    _, _, layers, hidden, heads = MODELS[key]
    return {'num_hidden_layers':layers, 'hidden_size':hidden, 'num_attention_heads':heads,
            'num_key_value_heads':8, 'head_dim':128, 'vocab_size':151936,
            'tie_word_embeddings':False, 'model_type':'llama' if key.endswith('-draft') else 'qwen3',
            'architectures':['LlamaForCausalLMEagle3' if key.endswith('-draft') else 'Qwen3ForCausalLM'],
            **({'draft_vocab_size':32000} if key.endswith('-draft') else {})}


def test_conversion_is_lossless_for_alias_and_mapping_dtypes(tmp_path):
    data = torch.tensor([1., 2.], dtype=torch.bfloat16)
    weights = {'a':data, 'alias':data, 'd2t':torch.tensor([3, -1],dtype=torch.int64),
               't2d':torch.tensor([True,False],dtype=torch.bool)}
    src=tmp_path/'source.bin'; torch.save(weights,src)
    cfg=tmp_path/'config.json'; cfg.write_text('{}')
    dest=tmp_path/'converted'; identity={'repo':'test','revision':'fixed'}
    original=src.read_bytes()
    first=convert_draft(src,dest,cfg,identity)
    assert src.read_bytes()==original
    assert first['tensors']=={k:tensor_record(v) for k,v in sorted(weights.items())}
    assert first['tensors']['d2t']['dtype']=='torch.int64'
    assert first['tensors']['t2d']['dtype']=='torch.bool'
    assert convert_draft(src,dest,cfg,identity)==first
    with pytest.raises(ValueError,match='身份或张量不匹配'):
        convert_draft(src,dest,cfg,{'repo':'test','revision':'other'})
    audit_converted(dest/'model.safetensors', first['tensors'])


def test_bad_state_and_unknown_existing_path_are_rejected(tmp_path):
    src=tmp_path/'source.bin';torch.save({'a':torch.ones(2),'not_tensor':'no'},src)
    cfg=tmp_path/'config.json';cfg.write_text('{}')
    with pytest.raises(TypeError, match='非 tensor'):
        convert_draft(src,tmp_path/'out',cfg,{})
    torch.save({'a':torch.ones(2)},src)
    dest=tmp_path/'out';dest.mkdir();(dest/'user_file').write_text('keep')
    with pytest.raises(FileExistsError): convert_draft(src,dest,cfg,{})
    assert (dest/'user_file').read_text()=='keep'


def test_config_revision_pair_and_vocab_validation():
    for key in MODELS: check_config(key,_config(key),MODELS[key][1])
    with pytest.raises(ValueError,match='revision'):
        check_config('8b',_config('8b'),'main')
    with pytest.raises(ValueError,match='对应配对'):
        validate_pair('8b','14b-draft',_config('8b'),_config('14b-draft'))
    broken=_config('8b-draft');broken['vocab_size']=10
    with pytest.raises(ValueError,match='vocab_size|词表'):
        validate_pair('8b','8b-draft',_config('8b'),broken)


def test_weight_iterator_groups_layers_across_shards(tmp_path):
    parts={'one.safetensors':{'model.layers.0.a':torch.tensor([1]),'model.layers.1.a':torch.tensor([3])},
           'two.safetensors':{'model.layers.0.b':torch.tensor([2]),'model.layers.2.a':torch.tensor([4])}}
    index={}
    for filename,tensors in parts.items():
        save_file(tensors,tmp_path/filename)
        index.update({name:filename for name in tensors})
    (tmp_path/'model.safetensors.index.json').write_text(json.dumps({'weight_map':index}))
    got=list(iter_weights(str(tmp_path)))
    assert [n for n,_ in got]==sorted(index)
    assert [int(v.item()) for _,v in got]==[1,2,3,4]


def test_checkpoint_rejects_duplicate_missing_and_wrong_shard(tmp_path):
    save_file({'layer.a':torch.ones(1)},tmp_path/'a.safetensors')
    save_file({'layer.a':torch.ones(1)},tmp_path/'b.safetensors')
    index=tmp_path/'model.safetensors.index.json'
    index.write_text(json.dumps({'weight_map':{'layer.a':'a.safetensors','other':'b.safetensors'}}))
    with pytest.raises(ValueError,match='出现了两次'): list(iter_weights(str(tmp_path)))
    save_file({'other':torch.ones(1)},tmp_path/'b.safetensors')
    index.write_text(json.dumps({'weight_map':{'layer.a':'b.safetensors','other':'a.safetensors'}}))
    with pytest.raises(ValueError,match='索引说'): list(iter_weights(str(tmp_path)))
    index.write_text(json.dumps({'weight_map':{'missing':'a.safetensors','other':'b.safetensors'}}))
    with pytest.raises(ValueError): list(iter_weights(str(tmp_path)))
    index.write_text('{"weight_map":{"a":"a.safetensors","a":"b.safetensors"}}')
    with pytest.raises(ValueError,match='JSON 键重复'): list(iter_weights(str(tmp_path)))


def test_frozen_real_configs_and_split_index_grouping():
    base = Path(__file__).resolve().parents[2] / 'learning_notes/14_vllm_from_scratch/分布式并行完整需求/参考资料/模型元数据'
    names={'8b':'Qwen--Qwen3-8B','14b':'Qwen--Qwen3-14B',
           '8b-draft':'AngelSlim--Qwen3-8B_eagle3','14b-draft':'AngelSlim--Qwen3-14B_eagle3'}
    from minivllm.model_loader.weight_utils import _check_grouping
    for key, name in names.items():
        config=json.loads((base/name/'config.json').read_text())
        check_config(key,config,MODELS[key][1])
        if not key.endswith('-draft'):
            weight_map=json.loads((base/name/'model.safetensors.index.json').read_text())['weight_map']
            old_order=[k for shard in sorted(set(weight_map.values())) for k in sorted(weight_map) if weight_map[k]==shard]
            with pytest.raises(ValueError,match='不是按模块分组'):
                _check_grouping(old_order)
            _check_grouping(sorted(weight_map))


def test_frozen_loader_sources_match_installed_vllm():
    import ast
    import hashlib
    import importlib.util
    root=Path(__file__).resolve().parents[2]
    base=root/'learning_notes/14_vllm_from_scratch/分布式并行完整需求/参考资料'
    manifest=json.loads((base/'源码清单.json').read_text())
    entries={x['path']:x for x in manifest['files']}
    installed=Path(importlib.util.find_spec('vllm').origin).parent
    anchors={'model_executor/model_loader/default_loader.py':('DefaultModelLoader','load_weights'),
             'model_executor/models/llama_eagle3.py':('Eagle3LlamaForCausalLM','load_weights')}
    for name in ['model_executor/model_loader/weight_utils.py',*anchors,
                 'model_executor/models/qwen3.py','config/speculative.py']:
        data=(base/'vllm_0_28_0'/name).read_bytes()
        expected=entries[name]['sha256']
        assert hashlib.sha256(data).hexdigest()==expected
        assert hashlib.sha256((installed/name).read_bytes()).hexdigest()==expected
        if name in anchors:
            cls, method=anchors[name]
            tree=ast.parse(data)
            node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls)
            assert any(isinstance(n,ast.FunctionDef) and n.name==method for n in node.body)


def test_converted_tiny_eagle_checkpoint_loads_through_project_loader(tmp_path):
    from safetensors.torch import load_file
    from minivllm.config import ModelConfig
    from minivllm.model_loader.loader import get_model
    from minivllm.testing.tiny_models import tiny_eagle3_dir
    tiny=Path(tiny_eagle3_dir('tiny_gqa'))
    state=load_file(tiny/'model.safetensors',device='cpu')
    source=tmp_path/'draft.bin'
    torch.save(state,source)
    destination=tmp_path/'converted'
    convert_draft(source,destination,tiny/'config.json',{'repo':'tiny','revision':'test'})
    config=json.loads((tiny/'config.json').read_text())
    reference=get_model(ModelConfig(model=str(tiny),dtype='float32',hf_config=config),'cpu')
    converted=get_model(ModelConfig(model=str(destination),dtype='float32',hf_config=config),'cpu')
    left=dict(reference.named_parameters()); right=dict(converted.named_parameters())
    assert set(left)==set(right)
    # No embedding in this draft: ownership is transferred from target later.
    assert not any("embed_tokens" in name for name in state)
    assert reference.model.tie_embeddings and converted.model.tie_embeddings
    differences={name for name in left if not torch.equal(left[name],right[name])}
    assert differences <= {"model.embed_tokens.weight"}


def test_interleaved_shards_load_tiny_qwen_without_weight_changes(tmp_path):
    from safetensors.torch import load_file
    from minivllm.config import ModelConfig
    from minivllm.model_loader.loader import get_model
    from minivllm.testing.tiny_models import tiny_qwen3_dir, tiny_qwen3_config
    tiny=Path(tiny_qwen3_dir('tiny_gqa'))
    state=load_file(tiny/'model.safetensors',device='cpu')
    names=sorted(state)
    shards={'one.safetensors':{},'two.safetensors':{}}
    for i,name in enumerate(names): shards['one.safetensors' if i%2==0 else 'two.safetensors'][name]=state[name]
    for name,tensors in shards.items(): save_file(tensors,tmp_path/name)
    index={key:name for name,tensors in shards.items() for key in tensors}
    (tmp_path/'model.safetensors.index.json').write_text(json.dumps({'weight_map':index}))
    cfg=tiny_qwen3_config('tiny_gqa')
    expected=get_model(ModelConfig(model=str(tiny),dtype='float32',hf_config=cfg),'cpu')
    actual=get_model(ModelConfig(model=str(tmp_path),dtype='float32',hf_config=cfg),'cpu')
    left=dict(expected.named_parameters());right=dict(actual.named_parameters())
    assert set(left)==set(right)
    assert all(torch.equal(left[key],right[key]) for key in left)


def test_resolve_snapshot_accepts_selected_cache_files(tmp_path, monkeypatch):
    import hashlib
    import huggingface_hub
    import benchmarks.check_step090_models as probe
    revision=MODELS['8b'][1]
    path=tmp_path/revision
    path.mkdir()
    config=json.dumps(_config('8b')).encode()
    (path/'config.json').write_bytes(config)
    monkeypatch.setitem(probe.CONFIG_HASHES,'8b',hashlib.sha256(config).hexdigest())
    def local_snapshot(repo, revision, local_files_only, allow_patterns):
        assert repo == MODELS['8b'][0] and local_files_only is True
        assert '*.safetensors' in allow_patterns and 'README.md' not in allow_patterns
        return str(path)
    monkeypatch.setattr(huggingface_hub,'snapshot_download',local_snapshot)
    result, parsed=probe.resolve_snapshot('8b')
    assert result == path and parsed == _config('8b')


def test_chat_template_token_ids_not_mapping_keys():
    from benchmarks.check_step090_models import extract_prompt_ids
    assert extract_prompt_ids({'input_ids':[1,2], 'attention_mask':[1,1]}) == [1,2]
    assert extract_prompt_ids([3,4]) == [3,4]
    for invalid in ({'attention_mask':[1]}, ['input_ids'], [[1,2]], []):
        with pytest.raises(ValueError, match='token IDs'):
            extract_prompt_ids(invalid)


def test_reference_error_statistics_are_explicit_and_reject_nonfinite():
    from benchmarks.compare_step090_reference import metrics
    observed=torch.tensor([1.,4.]);baseline=torch.tensor([1.,2.])
    result=metrics(observed,baseline)
    assert result['max_abs']==2.0 and result['max_error_flat_index']==1
    assert result['max_error_index']==[1] and result['shape']==[2]
    assert result['observed_value']==4.0 and result['reference_value']==2.0
    assert result['relative_l2']==pytest.approx(2/(5**0.5))
    with pytest.raises(ValueError,match='非有限'):
        metrics(torch.tensor([float('nan')]),torch.tensor([0.]))


def test_eagle_mapping_offsets_membership_and_failure_cases(tmp_path):
    from benchmarks.check_step090_models import audit_eagle_mapping
    path=tmp_path/'mapping.safetensors'
    cfg={'draft_vocab_size':3,'vocab_size':6}
    def write(d2t,t2d):
        save_file({'d2t':d2t,'t2d':t2d},path)
    d=torch.tensor([0,1,3],dtype=torch.int64)
    valid=torch.tensor([True,False,True,False,False,True])
    write(d,valid)
    assert audit_eagle_mapping(path,cfg)['target_members']==3
    write(d,~valid)
    with pytest.raises(ValueError,match='集合不一致'):
        audit_eagle_mapping(path,cfg)
    write(torch.tensor([0,0,-1],dtype=torch.int64),valid)
    with pytest.raises(ValueError,match='越界或重复'):
        audit_eagle_mapping(path,cfg)
