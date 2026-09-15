"""第十五关 Engine 完整 Engine 生命周期对照；CPU 同权重，源文件不变，调试 print 临时关闭。"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import copy
import hashlib
import importlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.benchmark import Timer

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
RECORDS = Path("/home/user/proj/vllm-omni/learning_notes/14_vllm_from_scratch/验收记录")
FIXTURES = {"step14": RECORDS / "step14_engine_contract_20260915T111003.407774Z.json",
            "step15": RECORDS / "step15_engine_contract_20260915T132042.147602Z.json"}
REQUESTS = [
    {"request_id":"A", "prompt_ids":[0], "max_new_tokens":3},
    {"request_id":"B", "prompt_ids":[1,2,3], "max_new_tokens":2},
    {"request_id":"C", "prompt_ids":[0,1,0], "max_new_tokens":4},
]
TOKENS = {"A":[2,2,0], "B":[3,0], "C":[0,2,2,2]}
ORDER = {1:"ABC", 2:"BAC", 3:"BAC"}
STEPS = {1:9, 2:6, 3:4}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parameter_hash(model):
    h = hashlib.sha256()
    for name, t in model.state_dict().items():
        h.update(name.encode())
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", choices=["step14", "step15"], default="step15")
    parser.add_argument("--capacity", type=int, choices=[1,2,3], default=2)
    args = parser.parse_args()
    module = importlib.import_module(args.module)
    source = Path(module.__file__)
    source_hash = sha(source)
    accepted = json.loads(FIXTURES[args.module].read_text())
    assert accepted["status"] == "passed" and accepted["source_sha256"] == source_hash, "先对当前源码完成验收"
    case_name = "random_capacity_" + str(args.capacity)
    fixture = next(c for c in accepted["cases"] if c["name"] == case_name)
    wanted_calls = [{"kind":c["kind"], "ids":c["ids"]} for c in fixture["calls"]]
    wanted_samples = [c["tokens"] for c in fixture["samples"]]
    expected = [{"request_id":rid, "output_ids":TOKENS[rid]} for rid in ORDER[args.capacity]]
    torch.set_num_threads(1)
    torch.manual_seed(0)
    rng = torch.get_rng_state().clone()
    weight_hash = parameter_hash(module.TinyCausalLM(vocab_size=5,d_model=8,max_seq_len=32))

    def run_static():
        torch.set_rng_state(rng)
        engine = module.Engine(max_num_seqs=args.capacity,vocab_size=5,d_model=8,max_seq_len=32)
        for request in REQUESTS: engine.add_request(request)
        results = []
        while engine.has_unfinished_requests(): results.extend(engine.step())
        return results

    def check():
        calls, samples, steps, models = [], [], [], []
        inputs_before = copy.deepcopy(REQUESTS)
        original_init = module.TinyCausalLM.__init__
        forward_name = "forward_prefill"
        original_forward = getattr(module.TinyCausalLM, forward_name)
        original_sample, original_step = module.Sampler.sample, module.Engine.step
        def init(model,*a,**kw):
            original_init(model,*a,**kw)
            assert parameter_hash(model) == weight_hash
            models.append(model)
        def observe(kind,fn,model,ids,*a,**kw):
            assert len(calls) < len(wanted_calls), "多余模型调用"
            assert torch.is_inference_mode_enabled() and not model.training and not torch.is_grad_enabled()
            assert ids.dtype == torch.long and ids.device.type == "cpu"
            calls.append({"kind":kind,"ids":ids.tolist()})
            result = fn(model,ids,*a,**kw)
            logits = result[0] if isinstance(result,tuple) else result
            assert not logits.requires_grad
            return result
        def forward(model,ids,*a,**kw):
            return observe("prefill",original_forward,model,ids,*a,**kw)
        def sample(sampler,logits):
            assert torch.is_inference_mode_enabled()
            ids = original_sample(sampler,logits)
            samples.append(ids.tolist())
            return ids
        def step(engine):
            assert len(steps) < STEPS[args.capacity], "额外 step 或无法结束"
            a,b = len(calls),len(samples)
            result = original_step(engine)
            steps.append([len(calls)-a,len(samples)-b])
            return result
        from contextlib import ExitStack
        with ExitStack() as stack:
            for obj,name,fn in ((module.TinyCausalLM,"__init__",init),(module.TinyCausalLM,forward_name,forward),
                                (module.Sampler,"sample",sample),(module.Engine,"step",step)):
                stack.enter_context(patch.object(obj,name,fn))
            if args.module in ("step14", "step15"):
                original_decode = module.TinyCausalLM.forward_decode
                stack.enter_context(patch.object(module.TinyCausalLM,"forward_decode",
                    lambda model,ids,*a,**kw:observe("decode",original_decode,model,ids,*a,**kw)))
            results = run_static()
        assert results == expected and calls == wanted_calls and samples == wanted_samples
        assert len(models) == 1 and parameter_hash(models[0]) == weight_hash
        assert REQUESTS == inputs_before and len(steps) == STEPS[args.capacity]
        assert all(pair[1] == 1 for pair in steps)
        return {"outputs":results,"model_calls":calls,"sampled_tokens":samples,"calls_per_step":steps,
                "processed_input_positions":sum(len(row) for c in calls for row in c["ids"]),
                "initialization_count":1,"parameter_sha256":weight_hash,"stop_reasons":{"A":"budget","B":"budget","C":"budget"}}

    started = datetime.now(timezone.utc).isoformat()
    # 临时遮蔽模块内的调试 print，阻止 Tensor 格式化；不修改学生源码，不全局替换 print。
    # 空函数调用本身仍计时；patch 的设置和恢复不计时。两版使用相同测试环境。
    with patch.object(module,"print",lambda *a,**kw:None,create=True):
        before = check()
        m = Timer(stmt="run_static()",globals={"run_static":run_static},num_threads=1,
                  label=f"{args.module} Engine lifecycle",description=f"capacity={args.capacity}, CPU, 9 tokens"
                  ).blocked_autorange(min_run_time=1.0)
        after = check()
        assert before == after
    assert sha(source) == source_hash
    report = {
        "status":"measured","started_at_utc":started,"pid":os.getpid(),"module":args.module,
        "source":str(source),"source_sha256":source_hash,"benchmark_sha256":sha(Path(__file__)),
        "fixture":str(FIXTURES[args.module]),"fixture_sha256":sha(FIXTURES[args.module]),
        "python":sys.version,"python_executable":sys.executable,"torch":torch.__version__,"platform":platform.platform(),
        "device":"cpu","dtype":"float32","num_threads":1,"max_num_seqs":args.capacity,"seed":0,
        "parameter_sha256":weight_hash,"rng_state_sha256":hashlib.sha256(rng.numpy().tobytes()).hexdigest(),
        "model_config":{"vocab_size":5,"d_model":8,"max_seq_len":32},"requests":REQUESTS,"expected_new_tokens":9,
        "boundary":"restore fixed CPU RNG state + Engine/model initialization/eval + submit + query/step loop (input preparation, model calls, inference contexts, sampling, state/cache writeback and cleanup) + collect results; excludes imports, test fixtures, correctness hooks/checks; module-local diagnostic print temporarily replaced by noop without source edits, noop call cost included",
        "debug_print_disabled":True,"min_run_time_s":1.0,"before_check":before,"after_check":after,
        "median_us":m.median*1e6,"iqr_us":m.iqr*1e6,"has_warnings":m.has_warnings,
        "number_per_run":m.number_per_run,"raw_times_s":m.raw_times,
    }
    directory = PROJECT / "benchmarks/results"
    directory.mkdir(exist_ok=True,parents=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    path = directory / f"step15_engine_compare_{args.module}_cap{args.capacity}_{stamp}_{os.getpid()}.json"
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(m)
    print("Saved:",path)


if __name__ == "__main__": main()
