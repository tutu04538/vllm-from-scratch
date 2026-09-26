"""采样执行层：把本轮的 picked logits 变成 token，再提交进请求状态。

一轮的顺序固定：**行映射**（纯）→ **三条采样路径** → **验证与 KV 回滚** → **提交**。
这一层不认识模型前向、不认识请求队列，也不认识 `Engine` 这个类型：它需要的东西
（采样后端、KV 池、停止 token）在构造时给它，每轮变的部分（logits、本轮计划、
输出回调）按参数传进来。

依赖方向是单向的：`sampling` / `speculative` → `sample_runtime` → `engine`。

为什么是**组合**而不是继承：vLLM 就是这么分的——`vllm/v1/sample/sampler.py` 的
`Sampler` 与 `rejection_sampler.py` 的 `RejectionSampler` 都是独立的类，被
`v1/worker/gpu_model_runner.py` **持有为属性**（`:594 self.sampler = Sampler(...)`、
`:705 self.rejection_sampler = RejectionSampler(self.sampler, self.speculative_config,
self.device)`），构造参数正好是「它需要的稳定依赖 + 配置」。
"""

from types import SimpleNamespace

import torch

from .sampling import apply_penalties, row_distribution
from .speculative import verify_drafts, verify_drafts_random


class SampleRuntime:
    """本轮采样的执行者：行映射 → 三条路径 → 验证与回滚 → 提交。

    它需要的东西分两类：**稳定的**（采样后端、KV 池、停止 token）在构造时给它；
    **每轮变的**（logits、本轮计划、输出回调）按参数传。输出回调归 Engine 所有
    （`engine.on_token` 是公开参数，用户可能在任何时候设置它），所以走 `run()` 的参数。
    """

    def __init__(self, sampler, kv_cache_pool, eos_token_ids):
        self.sampler = sampler
        self.kv_cache_pool = kv_cache_pool
        self.eos_token_ids = eos_token_ids

    # -------- 1) 行映射（纯） --------

    @staticmethod
    def plan_sample_rows(scheduled_items):
        # 把「哪些请求就绪、各取哪几行」整理一次，产出两样东西：
        #   rows   —— 传给模型的**原始输入行号**（模型只认行号，不需要认识请求对象）
        #   picked —— 要采样的那些 item，采样结果按同一顺序写回去
        #
        # 这里有两个坐标系，不能混用：
        #   * 原始输入行号：本轮全部输入 token 拼成一维后的下标（中间 prefill 也占位置）
        #   * 筛选后偏移  ：模型返回的 logits 只包含选中的行，行内下标从 0 开始
        # 所以每个 picked 项都记下自己的 `num_sample_rows` 与 `sample_offset`，
        # 后者在 `run()` 里直接用来切 Python 列表。
        #
        # 普通项只取「片段末行」；投机项取整段输入的行——前 K 行验证草稿，最后一行
        # 给 bonus。中间 prefill（`can_sample` 为假）不取行，但它的 token 照样占
        # 原始输入位置，所以 `offset` 对**每个** item 都要累加。
        rows = []
        picked = []
        offset = 0
        for item in scheduled_items:
            offset += item["num_scheduled_tokens"]
            if not item["can_sample"]:
                continue
            item["sample_offset"] = len(rows)          # 在筛选后 logits 里的起点
            if item["draft_ids"]:
                item["num_sample_rows"] = item["num_scheduled_tokens"]
                rows.extend(range(offset - item["num_scheduled_tokens"], offset))
            else:
                item["num_sample_rows"] = 1
                rows.append(offset - 1)
            picked.append(item)
        return rows, picked

    # -------- 2) 三条采样路径的分派 --------

    def run(self, logits, picked, on_token=None):
        # logits 已经是「需要采样的那几行」，行序与 picked 一致，不再按原始行号二次索引
        if not picked:
            return None
        expected = sum(item["num_sample_rows"] for item in picked)
        if logits.shape[0] != expected:
            raise RuntimeError(f"模型返回 {logits.shape[0]} 行 logits，但本轮需要 {expected} 行")

        # 1) 没有草稿的项：走采样后端（随机采样、惩罚项、按行独立的历史都在那条路上）。
        #    整批一次 select_batch + 一次 .tolist()，先把 token 算好，**提交留到下面按
        #    picked 顺序做**——在这里就提交的话，事件顺序会变成「先普通后投机」。
        plain = [item for item in picked if not item["draft_ids"]]
        plain_tokens = {}
        if plain:
            for item, token in zip(plain, self._sampler_tokens(logits, plain)):
                plain_tokens[id(item)] = token

        # 2) 贪心且无惩罚的投机项：保留第五十三关的整批 argmax 快路径（一次回传）。
        #    带惩罚项的贪心**不能**走这条：每一行看到的生成历史不同，argmax 必须在
        #    逐行施加惩罚之后再取。
        any_fast = any(item["draft_ids"] and self._is_greedy_without_penalty(item["request"])
                       for item in picked)
        greedy = torch.argmax(logits, dim=-1).tolist() if any_fast else None

        # 3) 严格按 picked 顺序提交：同一请求内按 token 顺序，跨请求按本轮采样顺序
        for item in picked:
            seq = item["request"]
            start, nrows = item["sample_offset"], item["num_sample_rows"]
            if not item["draft_ids"]:
                self._commit_tokens(seq, [plain_tokens[id(item)]], on_token)
            elif greedy is not None and self._is_greedy_without_penalty(seq):
                self._commit_drafts(item, greedy[start:start + nrows], on_token)
            else:
                self._commit_drafts_random(item, logits[start:start + nrows], on_token)
        return None

    @staticmethod
    def _is_greedy_without_penalty(seq):
        """能不能走「整批 argmax」快路径：贪心，而且没有任何惩罚项。

        有惩罚项时每行看到的生成历史都不同（需求 §3），必须先逐行施加惩罚再取
        argmax——那时 one-hot 才是**该行**的目标分布。
        """
        params = seq.sampling_params
        return params.is_greedy and not params.has_penalty

    def _sampler_tokens(self, logits, items):
        """按每项自己的采样参数与惩罚项抽一枚 token，返回与 items 同序的 Python 列表。

        整批一次 `select_batch`、一次 `.tolist()`，不逐请求 `.item()`。
        **这里必须把 logits 转成 FP32**，和投机那条路不同：`apply_penalties()` 算
        presence / frequency 惩罚时惩罚量是 FP32，回写要 `index_put` 进 logits 那一行，
        dtype 不匹配会直接抛（BF16 模型 + 这两种惩罚，不转 FP32 就跑不起来）。

        **不复制**（不带 `copy=True`）：Graph 的输出缓冲确实会被下次 replay 覆盖，
        但 logits 只在本轮 `step()` 内被消费，`apply_penalties()` 也只读输入。
        **如果以后把采样挪进异步调度、要跨步持有 logits，这里就得改回来。**
        """
        seqs = [item["request"] for item in items]
        rows = [apply_penalties(logits[item["sample_offset"]].to(torch.float32),
                                seq.sampling_params, seq.sampling_state)
                for item, seq in zip(items, seqs)]
        tokens = self.sampler.select_batch(
            rows, [s_.sampling_params for s_ in seqs], [s_.sampling_state for s_ in seqs])
        # token id 整批回传，不逐请求 .item()
        return torch.stack(tokens).tolist()

    # -------- 3) 提交：token 进入请求状态的唯一入口 --------

    def _commit_tokens(self, seq, token_ids, on_token):
        """把 token 提交进请求状态：已提交历史、惩罚计数、回调**同进同退**。

        这里是唯一的提交点，普通路径与投机路径共用。只有真正被接受的 token 才
        会走到这儿——草稿在验证通过之前一个都不进来。

        回调顺序沿用 `picked` —— 也就是本轮采样的顺序，不等于 `running` 列表的顺序。
        """
        for token_id in token_ids:
            # 通知点就在这儿：token 已经是 Python int、马上要提交进请求状态，
            # 但请求还没被判停、没被回收。
            index = len(seq.output_ids)      # 提交前的长度就是这次的序号（只数生成 token）
            seq.append_output_ids(token_id)   # 唯一写入点：同步更新已提交历史与输出
            # 只有真正提交的输出 token 才进惩罚计数；M=0、中间 prefill 块都不经过这里
            seq.sampling_state.note_output_token(token_id)
            if on_token is not None:
                # 每次新建一个独立字典，只放 CPU 上的 ID/整数：调用方存起来或改它
                # 都不会碰到引擎状态。不传 SequenceConfig，也不传内部列表。
                on_token({"request_id": seq.request_id, "token_id": token_id,
                          "output_index": index})

    # -------- 4) 验证与回滚 --------

    def _commit_drafts(self, item, greedy_ids, on_token):
        """投机项（贪心快路径）：用目标模型一次 forward 的 K+1 行验证草稿，回滚 KV，再提交。

        顺序不能换：**回滚必须在 `scheduler.post_step()` 之前**。被拒绝草稿的 KV
        已经随本轮输入写进了物理块，先把 `cache.length` 退回去，post_step 里的
        发布与判停读到的才是真实进度。

        `greedy_ids` 是**调用方整批算好、切好**的 K+1 枚目标模型贪心结果（见 `run()`）。
        这里不再自己 argmax：批量下每个请求各做一次 argmax 会多付一次设备同步，
        而且会把「一次 forward 一次回传」拆散。
        """
        seq = item["request"]
        remaining_outputs = seq.max_new_tokens - len(seq.output_ids)
        result = verify_drafts(item["draft_ids"], greedy_ids, self.eos_token_ids,
                               remaining_outputs)

        # 1) 先回滚：本轮输入 [x, d0..] 里只有 x 和「被接受且还要当下一轮输入」的草稿要留
        self.kv_cache_pool.truncate(seq, item["start_cache_length"] + result.kept_inputs)
        # 2) 再提交：逐枚走和普通路径同一个入口，事件序号自然连续
        self._commit_tokens(seq, result.committed_ids, on_token)

    def _commit_drafts_random(self, item, rows, on_token):
        """投机项（随机采样，或带惩罚项的贪心）：逐行构造目标分布做拒绝采样。

        **每一行的惩罚历史不同**（需求 §3）：行 j 预测的是「真实历史 + 前 j 枚草稿」
        之后那个 token。所以这里用的是一份**临时计数**——从真实计数复制一份，接受
        一枚草稿就往里加一枚；真实 `sampling_state` 与 `all_token_ids` 在
        `_commit_tokens()` 之前一个字都不动。

        行的分布按「全都接受」构造：拒绝点之后的行根本不会被读到（`verify_drafts_random`
        首次拒绝就停），而拒绝点之前的行历史恰好就是「前 j 枚都被接受」。
        """
        seq = item["request"]
        params, state = seq.sampling_params, seq.sampling_state
        draft_ids = item["draft_ids"]

        # 只复制计数，不复制整段 token 历史；与真实状态不共享底层字典
        temp = SimpleNamespace(prompt_token_ids=state.prompt_token_ids,
                               generated_counts=dict(state.generated_counts))
        row_probs = []
        for index in range(len(draft_ids) + 1):
            row_probs.append(row_distribution(rows[index].to(torch.float32), params, temp))
            if index < len(draft_ids):
                token = draft_ids[index]
                temp.generated_counts[token] = temp.generated_counts.get(token, 0) + 1

        if params.is_greedy:
            # 贪心请求没有 generator（第五十二关起只给随机采样建），这里也确实不需要：
            # 目标分布是 one-hot，`p[d]` 非 0 即 1，接受与否由 `verify_drafts_random()`
            # 直接判定、一次 uniform 都不抽；纠正与 bonus 就是该行分布的 argmax。
            def draw_uniform():
                raise AssertionError("贪心路径不该抽接受随机数")
            draw_token = lambda probs: int(torch.argmax(probs))
        else:
            def draw_uniform():
                return float(torch.rand((), generator=state.generator,
                                        device=state.generator.device))
            draw_token = lambda probs: torch.multinomial(probs, num_samples=1,
                                                         generator=state.generator)

        remaining_outputs = seq.max_new_tokens - len(seq.output_ids)
        result = verify_drafts_random(draft_ids, row_probs, self.eos_token_ids,
                                      remaining_outputs, draw_uniform, draw_token)

        # 先回滚（被拒草稿的 KV 已经随本轮输入写进物理块），再走唯一提交入口
        self.kv_cache_pool.truncate(seq, item["start_cache_length"] + result.kept_inputs)
        self._commit_tokens(seq, result.committed_ids, on_token)
