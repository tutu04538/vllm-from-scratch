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

import math
from types import SimpleNamespace

import torch

from .rejection import BatchedRejectionSampler, is_greedy_without_penalty
from .speculative import verify_drafts, verify_drafts_random


def is_speculative_item(item):
    """这一项本轮是不是**按投机计划排出的 decode 采样行**。

    判据是**调度时的计划**（`item["speculative"]`，由 scheduler 在 decode 分支上打），
    **不看本轮实际跑出几枚草稿**。「有几枚草稿」是运行结果，「是不是投机行」是配置：

    - ngram：草稿在计划阶段就用纯函数算好了，而 `num_reserved_drafts` **恒为 0**——只看
      预留名额会把它漏掉；
    - draft_model：target 预算只够 pending token、draft 池不够、补算吃光预算…… 都会让
      实际草稿数为 0，而 `num_reserved_drafts` 也可能是 0（预算阶段就被缩零了）。
      这些行**仍然是投机请求的 decode 步**，随机流必须留在 counter 那套上；按「有没有
      草稿」去判，它们就会切回请求自己的 `torch.Generator`——同一条请求中途用上两套随机
      机制，输出还会随调度（池子紧不紧）漂移。

    非投机行（prefill 与中间重算的 chunk）是 False：末块虽然也采样，但它的 `kept_inputs`
    语义不同（要保留**整块**输入而不是 1 枚），不能套投机行的回滚公式。
    """
    return bool(item["speculative"])


class SampleRuntime:
    """本轮采样的执行者：行映射 → 三条路径 → 验证与回滚 → 提交。

    它需要的东西分两类：**稳定的**（采样后端、KV 池、停止 token）在构造时给它；
    **每轮变的**（logits、本轮计划、输出回调）按参数传。输出回调归 Engine 所有
    （`engine.on_token` 是公开参数，用户可能在任何时候设置它），所以走 `run()` 的参数。
    """

    def __init__(self, sampler, kv_cache_pool, eos_token_ids, draft_proposer=None,
                 rejection_backend="torch", device=None):
        self.sampler = sampler
        self.kv_cache_pool = kv_cache_pool
        self.eos_token_ids = eos_token_ids
        # 第五十五关：draft model 那一层（None 表示不投机或 ngram）。
        # 采样层只用一个入口 `align()`——验证之后两套 KV 的回滚/对齐在同一个地方做。
        self.draft_proposer = draft_proposer
        # 累计接受了多少枚草稿（跨请求、跨轮）。投机的收益全在接受率上，所以这个数
        # 要能被外面看见——验收报告指出过「只看提议数 > 0 不是接受链路的证据」。
        # 与提议侧 `DraftModelProposer.num_proposed_tokens` 配套看。
        self.num_accepted_drafts = 0
        # 第五十六关：拒绝验证的批量执行层。**验证**（产生结论）搬进它，
        # **回滚 / 对齐 / 提交 / 收尾**仍然留在这里，且严格按 picked 顺序。
        # 设备在这里**显式**传进去（不从 sampler 猜）：triton 后端要把元数据建在
        # 与 logits 同一个设备上，猜错的表现是内核直接报
        # 「Pointer argument cannot be accessed from Triton (cpu tensor?)」。
        self.rejection_sampler = BatchedRejectionSampler(
            rejection_backend, eos_token_ids, sampler, device)

    def _align_draft(self, seq):
        """验证之后把 draft 的 KV 夹回 target 的真实计算边界。

        与 target 的 `truncate` 是**同一时刻**的两件事：都在提交 token 之前、
        `post_step()` 之前，所以 post_step 读到的两套进度都是真实进度。
        """
        if self.draft_proposer is not None:
            self.draft_proposer.align(seq)

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

        # 路由：每一项**要么**走普通采样后端、**要么**走批量验证，互斥且完备。判据只算一次
        # 且只有一份（`_needs_verification`）——两个列表各写一个表达式的话，条件一旦不
        # 一致就会出现「既采样又验证」的项：triton 后端的 K=0 回退项曾经就是（普通采样
        # 那份算完被丢掉，但请求自己的 `torch.Generator` 白白前进了一格，下次它走普通
        # 路径时抽到的随机数就跟着漂了）。
        verify = {id(item): self._needs_verification(item) for item in picked}

        # 1) 不走验证的项：交给采样后端（随机采样、惩罚项、按行独立的历史都在那条路上）。
        #    整批一次 select_batch + 一次 .tolist()，先把 token 算好，**提交留到下面按
        #    picked 顺序做**——在这里就提交的话，事件顺序会变成「先普通后投机」。
        plain = [item for item in picked if not verify[id(item)]]
        plain_tokens = {}
        if plain:
            for item, token in zip(plain, self._sampler_tokens(logits, plain)):
                plain_tokens[id(item)] = token

        # 2) 贪心且无惩罚的投机项：保留第五十三关的整批 argmax 快路径（一次回传）。
        #    带惩罚项的贪心**不能**走这条：每一行看到的生成历史不同，argmax 必须在
        #    逐行施加惩罚之后再取。
        #    **只有真的会走这条快路径的后端才算它**：triton 后端在 `prepare_batch()` 里
        #    一律走一般路径（它要那几行分布），这份 CPU 结果没人用——算了就是白回传一次
        #    （`argmax(...).tolist()` 是一次 D2H），验收方在真实 tiny 双模型上数到过。
        any_fast = (self.rejection_sampler.uses_greedy_fast_path
                    and any(item["draft_ids"] and is_greedy_without_penalty(item["request"])
                            for item in picked))
        greedy = torch.argmax(logits, dim=-1).tolist() if any_fast else None

        # 3) 投机项：交给批量验证层**先把结论算好**（triton 后端在这里做一次 GPU 批量
        #    计算 + 一次集中回传；torch 后端就是原来的逐请求参考路径）。提交留到下面，
        #    与普通项的 token 一样，按 picked 顺序走。
        #    `verify_batch` 里不碰任何请求状态：回滚、对齐、提交都在提交循环里。
        spec = [item for item in picked if verify[id(item)]]
        results = {}
        if spec:
            outcome = self.rejection_sampler.verify_batch(
                self.rejection_sampler.prepare_batch(logits, spec, greedy))
            item_results = self.rejection_sampler.materialize_results(outcome)
            # **整批先检查再提交**：任何一项非法都不允许「已经提交了半批」。
            errors = [f"{item['request'].request_id}: {r.error}"
                      for item, r in zip(spec, item_results) if r.error]
            if errors:
                raise ValueError("拒绝验证发现非法输入：" + "；".join(errors))
            results = {id(item): r for item, r in zip(spec, item_results)}

        # 4) 严格按 picked 顺序提交：同一请求内按 token 顺序，跨请求按本轮采样顺序
        for item in picked:
            seq = item["request"]
            if not verify[id(item)]:
                self._commit_tokens(seq, [plain_tokens[id(item)]], on_token)
                continue
            result = results[id(item)]
            self._commit_verified(item, result, on_token)

        # 5) 本轮收尾：把**预留了却没算到**的整块还给池子
        #    （有草稿的两条路径在验证时就还过了，这里对它们是空操作）
        for item in picked:
            self._release_unused_blocks(item["request"])
        return None

    def _release_unused_blocks(self, seq):
        """把本轮**预留了但没算到**的整块还给池子。

        计划阶段按 `1 + num_reserved_drafts` 占块——draft_model 在草稿提出来之前就得把
        target 的块先占住，而实际可能一枚草稿都没提出来（补算吃光预算、draft 池运行时
        不够）。那时 target 只算了 `[x]` 一个位置，多占的整块如果不还，会一直挂在这条
        请求名下挤占别人的容量：**不是永久泄漏**（请求结束时引用照样归零），而是运行中
        白占。

        **有草稿的两条路径不需要它**：验证之后已经 `truncate` 到真正保留的长度，多预留的
        部分当场就还了。这里对它们是空操作——块表本来就等于 `ceil(cache.length / block_size)`。
        所以这一步可以无脑对每个 item 调：它同时把「块表长度 == 已算到的整块数」这条
        不变量**执行**了一遍，而不只是写在文档里。

        不动有效 KV 内容、不重新抽样、也不会重复释放（长度没变时 `truncate` 一个块都不动）。
        """
        pool = self.kv_cache_pool
        if seq.cache is None or not seq.cache.block_table:
            return
        used_blocks = math.ceil(seq.cache.length / pool.block_size)
        if len(seq.cache.block_table) > used_blocks:
            pool.truncate(seq, seq.cache.length)

    def _sampler_tokens(self, logits, items):
        """按每项自己的采样参数与惩罚项抽一枚 token，返回与 items 同序的 Python 列表。

        整批一次 `select_batch`、一次 `.tolist()`，不逐请求 `.item()`。
        交给采样器的还是**原始**行（惩罚由 `TorchSampler` 内部做），但
        **这里必须把 logits 转成 FP32**，和投机那条路不同：`apply_penalties()` 算
        presence / frequency 惩罚时惩罚量是 FP32，回写要 `index_put` 进 logits 那一行，
        dtype 不匹配会直接抛（BF16 模型 + 这两种惩罚，不转 FP32 就跑不起来）。

        **不复制**（不带 `copy=True`）：Graph 的输出缓冲确实会被下次 replay 覆盖，
        但 logits 只在本轮 `step()` 内被消费，`apply_penalties()` 也只读输入。
        **如果以后把采样挪进异步调度、要跨步持有 logits，这里就得改回来。**
        """
        seqs = [item["request"] for item in items]
        rows = [logits[item["sample_offset"]].to(torch.float32) for item in items]
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

    def _needs_verification(self, item):
        """这一项要不要走**批量拒绝验证**（= 用不用 counter RNG 那条流）。

        判据只看**配置**，不看这一轮草稿跑出来几枚；两个后端规则不同、理由也不同：

        - triton 后端：**本轮所有投机采样行**都走，包括实际一枚草稿都没有的那些
          （预算缩零、池子不够、ngram 无匹配）。
          它必须用同一套 counter RNG 采样——中途切回 target 的 `torch.Generator`，这条
          请求的随机流就换了一条，「同 seed 逐 token 复现」也就不成立了。
        - torch 后端：只有真有草稿的项才走（与第五十五关完全一致）。这个后端里「走不走
          验证」不影响随机流——两条路抽随机数用的都是 `sampling_state.generator`，
          所以 K=0 的项退回普通采样没有任何副作用。
        """
        if self.rejection_sampler.uses_counter_rng:
            return is_speculative_item(item)
        return bool(item["draft_ids"])

    def _commit_verified(self, item, result, on_token):
        """把一条验证结论落到实处：回滚两套 KV → 记账 → 经唯一入口提交。

        顺序不能换：**回滚必须在 `scheduler.post_step()` 之前**。被拒绝草稿的 KV 已经
        随本轮输入写进了各自的物理块，先把 `cache.length` 退回去，post_step 里的发布与
        判停读到的才是真实进度；draft 那边在同一时刻夹回同一条边界。
        """
        seq = item["request"]
        self.kv_cache_pool.truncate(seq, item["start_cache_length"] + result.kept_inputs)
        self._align_draft(seq)
        self.num_accepted_drafts += result.num_accepted
        seq.rejection_rng_counter += result.rng_consumed
        self._commit_tokens(seq, result.committed_ids, on_token)

