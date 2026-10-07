"""58：固定输入工作区与端到端（需求 §8.C）。"""

import pytest
import torch

from helpers import WorkspaceTrace, greedy_outputs, make_engine, run_to_end
from minivllm import SamplingParams


def test_workspace_is_allocated_once_and_stable(tiny_dir, hf_config):
    engine, _core, runner = make_engine(tiny_dir=tiny_dir, hf_config=hf_config, spec_k=3,
                                        budget=16)
    proposer = runner.proposer
    assert proposer.max_num_tokens == 16 and proposer.max_num_reqs == 2
    assert proposer.input_ids.shape == (16,)
    assert proposer.query_start_loc.shape == (3,)
    assert proposer.seq_lens.shape == (2,)
    assert proposer.block_table.shape == (2, 16)
    addresses = {name: getattr(proposer, name).data_ptr()
                 for name in ("input_ids", "positions", "slot_mapping",
                              "is_rejected_token_mask", "query_start_loc", "seq_lens",
                              "block_table")}
    engine.add_request("A", [1, 2, 3, 4], SamplingParams(max_tokens=4, temperature=0.0,
                                                         eos_token_id=999))
    engine.add_request("B", [5, 6], SamplingParams(max_tokens=3, temperature=0.0,
                                                   eos_token_id=999))
    run_to_end(engine)
    engine.add_request("C", [7, 8, 9], SamplingParams(max_tokens=3, temperature=0.0,
                                                      eos_token_id=999))
    run_to_end(engine)
    assert all(getattr(proposer, name).data_ptr() == address
               for name, address in addresses.items())
    engine.shutdown()


def test_only_valid_slices_are_read(tiny_dir, hf_config):
    """把工作区尾部灌成垃圾，输出必须不变（有效长度由切片表达，不是"内容恰好是 0"）。"""
    reference = greedy_outputs(spec_k=3, tiny_dir=tiny_dir, hf_config=hf_config,
                               prompts=(("r", [1, 2, 3, 4, 5, 6]),), max_tokens=6)
    engine, _core, runner = make_engine(tiny_dir=tiny_dir, hf_config=hf_config, spec_k=3,
                                        budget=16)
    proposer = runner.proposer
    original = proposer._forward

    def poisoning(num_tokens, num_reqs):
        hidden = original(num_tokens, num_reqs)
        proposer.input_ids_cpu[num_tokens:] = 999
        proposer.positions_cpu[num_tokens:] = 999
        proposer.slot_mapping_cpu[num_tokens:] = -1
        proposer.query_start_loc_cpu[num_reqs + 1:] = 999
        proposer.seq_lens_cpu[num_reqs:] = 999
        proposer.block_table_cpu[num_reqs:] = 999
        return hidden

    proposer._forward = poisoning
    engine.add_request("r", [1, 2, 3, 4, 5, 6], SamplingParams(max_tokens=6, temperature=0.0,
                                                               eos_token_id=999))
    outputs = run_to_end(engine)
    proposer._forward = original
    assert outputs == reference
    engine.shutdown()


@pytest.mark.parametrize("k", [1, 3])
@pytest.mark.parametrize("prefix", [False, True])
def test_greedy_speculative_matches_non_speculative(tiny_dir, hf_config, k, prefix):
    reference = greedy_outputs(spec_k=None, tiny_dir=tiny_dir, hf_config=hf_config,
                               prefix=prefix)
    speculative = greedy_outputs(spec_k=k, tiny_dir=tiny_dir, hf_config=hf_config,
                                 prefix=prefix)
    assert speculative == reference
    assert len(speculative["r"]) == 6


def test_two_requests_with_different_lengths(tiny_dir, hf_config):
    prompts = (("A", [1, 2, 3, 4]), ("B", [5, 6, 7, 8, 9]))
    reference = greedy_outputs(spec_k=None, tiny_dir=tiny_dir, hf_config=hf_config,
                               prompts=prompts, max_tokens=5)
    speculative = greedy_outputs(spec_k=3, tiny_dir=tiny_dir, hf_config=hf_config,
                                 prompts=prompts, max_tokens=5)
    assert speculative == reference


def test_abort_mid_flight(tiny_dir, hf_config):
    engine, _core, _runner = make_engine(tiny_dir=tiny_dir, hf_config=hf_config, spec_k=3,
                                         budget=16)
    engine.add_request("keep", [1, 2, 3], SamplingParams(max_tokens=3, temperature=0.0,
                                                         eos_token_id=999))
    engine.add_request("drop", [4, 5, 6], SamplingParams(max_tokens=8, temperature=0.0,
                                                         eos_token_id=999))
    engine.step()
    engine.abort_request(["drop"])
    outputs = run_to_end(engine)
    assert "drop" not in outputs
    assert len(outputs["keep"]) == 3
    engine.shutdown()


def test_request_id_reuse_after_cleanup(tiny_dir, hf_config):
    engine, _core, _runner = make_engine(tiny_dir=tiny_dir, hf_config=hf_config, spec_k=1,
                                         prefix=True)
    engine.add_request("reuse", [1, 2, 3, 4, 5, 6, 7, 8], SamplingParams(max_tokens=1,
                                                                        temperature=0.0,
                                                                        eos_token_id=999))
    run_to_end(engine)
    engine.add_request("reuse", [8, 7, 6, 5], SamplingParams(max_tokens=2, temperature=0.0,
                                                             eos_token_id=999))
    assert len(run_to_end(engine)["reuse"]) == 2
    engine.shutdown()


def test_failure_state_blocks_next_step(tiny_dir, hf_config):
    engine, _core, runner = make_engine(tiny_dir=tiny_dir, hf_config=hf_config, spec_k=1,
                                        budget=16)
    engine.add_request("r", [1, 2, 3, 4, 5, 6], SamplingParams(max_tokens=4, temperature=0.0,
                                                               eos_token_id=999))
    original = runner.proposer._forward

    def injected(num_tokens, num_reqs):
        raise RuntimeError("injected-draft-forward-failure")

    runner.proposer._forward = injected
    # 70 关：异步调度下"提议"发生在**结果结账**那一刻（比同步晚一轮），所以失败可能在第二次
    # step 才暴露。断言不变的是语义：**一旦失败就停摆**（这点由下面两处钉住：失败必须出现、
    # 之后每一轮都拒绝）。允许"晚一轮暴露"不是放宽断言，而是异步的定义（结果本来就不当轮交付）。
    with pytest.raises(RuntimeError):
        for _ in range(3):
            engine.step()
    runner.proposer._forward = original
    assert runner.failure is not None
    with pytest.raises(RuntimeError):
        engine.step()
    engine.shutdown()


def test_context_limit_matches_non_speculative(tiny_dir, hf_config):
    reference = greedy_outputs(spec_k=None, tiny_dir=tiny_dir, hf_config=hf_config,
                               prompts=(("r", [1, 2, 3, 4, 5, 6, 7, 8]),), max_tokens=2,
                               max_model_len=10)
    speculative = greedy_outputs(spec_k=3, tiny_dir=tiny_dir, hf_config=hf_config,
                                 prompts=(("r", [1, 2, 3, 4, 5, 6, 7, 8]),), max_tokens=2,
                                 max_model_len=10)
    assert speculative == reference and len(speculative["r"]) == 2


def test_seeded_runs_are_reproducible(tiny_dir, hf_config):
    def seeded():
        engine, _core, _runner = make_engine(tiny_dir=tiny_dir, hf_config=hf_config, spec_k=3,
                                             budget=16)
        engine.add_request("r", [1, 2, 3, 4, 5, 6], SamplingParams(max_tokens=6, temperature=0.8,
                                                                   seed=1234, eos_token_id=999))
        outputs = run_to_end(engine)
        engine.shutdown()
        return outputs

    assert seeded() == seeded()


def test_workspace_capacity_limits_batch(tiny_dir, hf_config):
    """工作区（= max_num_batched_tokens）是硬上限：一次排进来的行数不会超过它。"""
    engine, core, runner = make_engine(tiny_dir=tiny_dir, hf_config=hf_config, spec_k=3,
                                       budget=8, max_num_seqs=4)
    for index in range(4):
        engine.add_request(f"r{index}", [1, 2, 3], SamplingParams(max_tokens=4, temperature=0.0,
                                                                 eos_token_id=999))
    engine.step()
    scheduled = core.scheduler.last_token_budget is not None
    plan = sum(core.scheduler.trace[-1]["scheduled"].values()) if core.scheduler.trace else 0
    assert scheduled and plan <= 8
    runner.proposer  # 提议者仍然可用
    run_to_end(engine)
    engine.shutdown()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA")
def test_cuda_end_to_end_matches_non_speculative(tiny_dir, hf_config):
    reference = greedy_outputs(spec_k=None, tiny_dir=tiny_dir, hf_config=hf_config,
                               device="cuda")
    speculative = greedy_outputs(spec_k=3, tiny_dir=tiny_dir, hf_config=hf_config,
                                 device="cuda")
    assert speculative == reference and len(speculative["r"]) == 6


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA")
def test_matches_upstream_expansion_kernel(tiny_dir, hf_config):
    """与上游 `copy_and_expand_eagle_inputs_kernel`（shift_input_ids=False）逐值差分。"""
    from vllm.v1.spec_decode.utils import copy_and_expand_eagle_inputs_kernel

    from minivllm.spec_decode.utils import DraftInputRows, expand_draft_inputs

    device = "cuda"
    target_ids = torch.tensor([5, 6, 7, 8, 9, 10], dtype=torch.int32, device=device)
    target_positions = torch.tensor([6, 7, 8, 8, 9, 10], dtype=torch.int32, device=device)
    next_ids = torch.tensor([99, 111], dtype=torch.int32, device=device)
    query_start_loc = torch.tensor([0, 3, 6], dtype=torch.int32, device=device)
    query_end_loc = torch.tensor([1, 5], dtype=torch.int32, device=device)
    rows = 8
    out_ids = torch.zeros(rows, dtype=torch.int32, device=device)
    out_positions = torch.zeros(rows, dtype=torch.int32, device=device)
    out_rejected = torch.zeros(rows, dtype=torch.bool, device=device)
    out_masked = torch.zeros(rows, dtype=torch.bool, device=device)
    out_new = torch.zeros(2, dtype=torch.int32, device=device)
    out_hidden = torch.zeros(6, dtype=torch.int32, device=device)
    copy_and_expand_eagle_inputs_kernel[(2, 1)](
        target_ids, target_positions, next_ids, out_ids, out_positions, out_rejected,
        out_masked, out_new, out_hidden, query_start_loc, query_end_loc, 0, 0, 6, 1, False,
        BLOCK_SIZE_TOKENS=4)
    ids, positions, rejected, sample = expand_draft_inputs([
        DraftInputRows(valid_token_ids=[5, 6], start=6, next_token_id=99, num_rejected=1),
        DraftInputRows(valid_token_ids=[8, 9, 10], start=8, next_token_id=111,
                       num_rejected=0)])
    assert out_ids.tolist() == ids
    assert out_positions.tolist() == positions
    assert out_rejected.tolist() == [bool(x) for x in rejected]
    assert out_new.tolist() == sample
    assert not bool(out_masked.any())
