"""draft model 的提议层：补算真实历史 → 分轮批量提议 → 对齐 → 释放。

第五十五关的新东西。draft 是**第二个模型**，有自己的 KV 池和自己的随机流，但它
**不拥有**任何请求状态：真实历史（`all_token_ids` / 输出 / 惩罚计数 / target 的
随机流）仍然只有一份，在 `SequenceConfig` 上；这里只多记一份 `seq.draft_cache`
和一条独立的 `seq.draft_generator`。

不创建第二套 `Engine.step()`，也不为两个模型引入通用 pipeline：draft 就是拿
`TinyCausalLM._forward_append()` 跑自己的池子，读的是**请求自己的**已提交历史。

三条不变量（测试都盯着）：

1. **两套 KV 绝不互换**：draft 的物理块只从 draft 池分配，写它的只有这一个类；
   两个池子各按自己模型的层数/头数/精度创建。
2. **draft 不领先于真实历史**：它最多覆盖到 target 的真实计算边界
   （`seq.cache.length`）。越过的部分只能是被拒或还没验证的草稿，验证一结束就
   由 `align()` 夹掉；下一轮提议之前用 `catch_up()` 从已提交历史统一补齐。
3. **提议真的从 q 抽**：每枚草稿的 q 是**抽出它时用的那个分布**（含温度、过滤、
   逐行临时惩罚历史），原样交给验证层算 `min(1, p/q)` 与 `max(p-q, 0)`。不能
   argmax 提议却拿 softmax 的 q 去验证——那样接受概率是假的。

`q` 按请求逐枚存小张量（词表大小），本关不做压缩，也不做 GPU rejection kernel。
"""

from dataclasses import dataclass
from types import SimpleNamespace

import torch

from .request import CacheConfig


@dataclass
class DraftProposal:
    """一次提议的结果：K 枚候选 + 每枚的**实际**提议分布 q。

    提议前的 draft KV 进度**不在这里**：那时它按约定恒等于 `seq.cache.length`
    （提议的硬前置是「draft 已追到 target 的真实计算边界」），另存一份只会多出
    一个可能与 `seq.draft_cache.length` 对不上的真相。
    """
    draft_ids: list
    draft_probs: list


# draft 的种子派生：一次线性同余混合，确定、无状态、与 target 的种子不同但由它唯一决定
DRAFT_SEED_MULTIPLIER = 6364136223846793005
DRAFT_SEED_INCREMENT = 1442695040888963407
DRAFT_SEED_MODULUS = 2 ** 63


def derive_draft_seed(seed):
    """由请求的 `seed` **稳定派生** draft 的种子。

    不能用 Python 的 `hash()`：同一进程里它也会被 PYTHONHASHSEED 打乱，拿它派生
    种子会让「同 seed、同工作序列必须能复现」这条约定失效。`seed=None`（调用方没
    指定）时返回 None，由 `make_draft_generator()` 从全局随机源取一个——
    那时本来就不承诺复现，和 target 侧的处理一致。
    """
    if seed is None:
        return None
    return (seed * DRAFT_SEED_MULTIPLIER + DRAFT_SEED_INCREMENT) % DRAFT_SEED_MODULUS


def make_draft_generator(params, device):
    """建 draft 的随机流，返回 `(generator, seed)`；贪心请求返回 `(None, None)`。

    贪心不需要随机数（`TorchSampler.draw()` 走 argmax），也就不建 generator——
    与 target 侧 `SamplingState` 的约定一致。

    **必须在请求创建时建一次**：抢占、重算、被缩草稿都不重置它，否则同一个请求的
    随机流会在中途换一条，同 seed 复现就无从谈起。
    """
    if params.is_greedy:
        return None, None
    seed = derive_draft_seed(params.seed)
    if seed is None:
        seed = int(torch.randint(0, DRAFT_SEED_MODULUS, (1,)).item())
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator, seed


class DraftModelProposer:
    """draft 模型的执行者：它不认识 Engine，也不认识 Scheduler。

    构造时给它**稳定的**依赖（draft 模型、draft 池、采样后端、停止 token、预算）；
    每轮变的东西（本轮的 item 列表、请求状态）按参数传进来。
    """

    def __init__(self, model, kv_cache_pool, sampler, eos_token_ids, max_num_batched_tokens):
        self.model = model
        self.kv_cache_pool = kv_cache_pool
        self.sampler = sampler
        self.eos_token_ids = set(eos_token_ids)
        # draft 自己的计算预算，与 target 的 max_num_batched_tokens **分开计数**：
        # 小模型的前向也是计算，不能偷偷记成 0。补算与提议都从这里扣。
        self.max_num_batched_tokens = max_num_batched_tokens
        # 累计统计（测试与文档用）：draft 前向次数、补算 token 数、提议位置数
        self.num_forwards = 0
        self.num_catchup_tokens = 0
        self.num_proposed_tokens = 0

    # -------- 前向 --------

    def _run(self, input_ids, num_scheduled_tokens, caches, sample_rows=None):
        """跑一次 draft 前向。`caches` 与 `num_scheduled_tokens` 逐项对应。

        补算是「一条请求、一段连续位置」（`sample_rows=[]`，不要 logits）；
        提议是「多条请求、各一个位置」（要 logits，行号就是批内下标）。
        """
        tokens = torch.tensor(list(input_ids), dtype=torch.long, device=self.model.device)
        self.num_forwards += 1
        return self.model._forward_append(tokens, list(num_scheduled_tokens), list(caches),
                                          self.kv_cache_pool, sample_rows=sample_rows)

    # -------- 1) 补算：draft 追到 target 的真实计算边界 --------

    def backlog(self, seq):
        """draft 还差多少个 token 才追上 target 已算完的前缀。"""
        return max(seq.cache.length - seq.draft_cache.length, 0)

    def fits(self, seq, k):
        """只读地问：draft 池容不容得下「补算缺口 + k 枚草稿」。给计划阶段用。

        补算也是 draft 的计算，同样占 draft 池的块，所以**必须一起算进去**——
        否则会出现「计划说放得下 k 枚，实际连历史都补不完」。
        """
        return self.kv_cache_pool.can_grow_for(seq.draft_cache, self.backlog(seq) + k)

    def catch_up(self, seq, budget):
        """把已提交历史补进 draft 的 KV，返回 `(是否追平, 用掉多少预算)`。

        补的永远是**已提交**历史的一段 `[draft.length, target.length)`：draft 落后
        只会发生在（a）刚准入或命中前缀，（b）被抢占后重算，（c）上一轮没排上、
        草稿池不够或走了 fallback。三种情形走同一条路径，不维护第二份可分叉的
        真实 token 列表。

        chunk 大小受 `budget` 限制，追不平就返回 False：**这一轮不提议**，请求照常
        走普通 target 路径（草稿是可选加速）。绝不为了草稿去抢占 target 的其他请求。
        """
        used = 0
        while True:
            remaining = self.backlog(seq)
            if remaining <= 0:
                return True, used
            chunk = min(remaining, budget - used)
            if chunk <= 0:
                return False, used
            # 池子不够：一个字节都不改（池子自己是计划/提交分离的），本轮不提议
            if not self.kv_cache_pool.ensure_blocks_for(seq.draft_cache, chunk):
                return False, used
            start = seq.draft_cache.length
            self._run(seq.all_token_ids[start:start + chunk], [chunk], [seq.draft_cache],
                      sample_rows=[])
            used += chunk
            self.num_catchup_tokens += chunk

    # -------- 2) 提议：逐位置、跨请求合成一个 batch --------

    def run_round(self, items):
        """一轮的全部 draft 工作，把草稿写回 item。**在 target forward 之前调用**。

        `item["max_draft_k"]` 是调度器给的预留上限（已按两个池子的容量缩过）。
        实际草稿数可能更少：draft 提出终止 token、池子中途不够、预算用尽。
        """
        budget = self.max_num_batched_tokens
        ready = []
        for item in items:
            if item["max_draft_k"] <= 0:
                continue
            done, used = self.catch_up(item["request"], budget)
            budget -= used
            if done:
                ready.append(item)
        if ready:
            self._propose_batch(ready, budget)
        for item in items:
            if item["max_draft_k"] > 0:
                self._fill_item(item)

    def _propose_batch(self, items, budget):
        """逐位置提议：第 j 步把「还想要第 j 枚」的请求合成一个 batch。

        不套「请求循环 × K 次单请求前向」——那是 K 倍的小 batch，白扔掉 batch 维度。
        不同请求的 K 可以不同（终止 token、池子不够、预算见底），batch 因此是
        ragged 的：某条请求退出，其余照常继续。
        """
        drafts = {id(item): ([], []) for item in items}      # 每请求：草稿 id 与 q
        stopped = set()                                      # 本轮不再提议的请求
        max_steps = max(item["max_draft_k"] for item in items)
        for step in range(max_steps):
            active = [it for it in items
                      if id(it) not in stopped and it["max_draft_k"] > step]
            if not active or budget <= 0:
                break
            batch = active[:budget]
            # 每条请求这一步都要写一个新位置：补不出块的（以及预算没轮上的）本轮到此为止
            batch = [it for it in batch
                     if self.kv_cache_pool.ensure_blocks_for(it["request"].draft_cache, 1)]
            for it in active:
                if it not in batch:
                    stopped.add(id(it))
            if not batch:
                break

            input_ids, caches = [], []
            for it in batch:
                seq = it["request"]
                # 第 0 枚喂 pending x（真实历史最后一枚，还没进过任何模型），
                # 之后每枚喂上一枚草稿——与 target 那 K+1 行的排布逐个对齐。
                input_ids.append(seq.all_token_ids[seq.cache.length] if step == 0
                                 else drafts[id(it)][0][step - 1])
                caches.append(seq.draft_cache)
            logits = self._run(input_ids, [1] * len(batch), caches,
                               sample_rows=list(range(len(batch))))
            budget -= len(batch)
            self.num_proposed_tokens += len(batch)

            for row, it in enumerate(batch):
                seq = it["request"]
                ids, qs = drafts[id(it)]
                # 提议也必须真的从 q 抽：贪心走 argmax（q 是 one-hot），
                # 随机走这条请求自己的 draft 随机流
                q = self.sampler.distribution(logits[row].to(torch.float32),
                                              seq.sampling_params, self._history(seq, ids))
                ids.append(int(self.sampler.draw(q, seq.sampling_params, seq.draft_generator)))
                qs.append(q)
                if ids[-1] in self.eos_token_ids:
                    # 终止 token 之后再提没有意义：它要么被接受（验证在那里就结束）、
                    # 要么被拒（首个拒绝就是它），后面的草稿永远读不到
                    stopped.add(id(it))

        for it in items:
            ids, qs = drafts[id(it)]
            it["_proposal"] = DraftProposal(ids, qs)

    @staticmethod
    def _history(seq, draft_ids):
        """第 i 行使用的惩罚历史：真实已生成 + **前 i 枚**草稿（需求 §3）。

        与 target 侧 `_commit_drafts_random()` 是同一条规则、同一种构造：只复制计数，
        不复制整段 token 历史，也不与真实状态共享底层字典——真实计数只在唯一提交
        入口更新，提议、重算都不重复计数。
        """
        state = seq.sampling_state
        temp = SimpleNamespace(prompt_token_ids=state.prompt_token_ids,
                               generated_counts=dict(state.generated_counts))
        for token in draft_ids:
            temp.generated_counts[token] = temp.generated_counts.get(token, 0) + 1
        return temp

    @staticmethod
    def _fill_item(item):
        """把草稿写进本轮计划：输入行、计数、以及给验证层用的 q。

        计划里预留的 `max_draft_k` 与实际草稿数可能不一致。实际更少时**不在这里
        还块**：多预留的 target 块由验证之后的回滚（`truncate` 到保留长度）自然还回
        池子，draft 那边由 `align()` 夹回边界——两个池子各有一条归还路径，不重复还。
        """
        proposal = item.pop("_proposal", None) or DraftProposal([], [])
        item["draft_ids"] = list(proposal.draft_ids)
        item["draft_probs"] = list(proposal.draft_probs)
        item["input_ids"] = item["input_ids"][:1] + list(proposal.draft_ids)
        item["num_scheduled_tokens"] = 1 + len(proposal.draft_ids)

    # -------- 3) 验证之后：对齐 --------

    def align(self, seq):
        """把 draft 的 KV 夹回 target 的真实计算边界。

        target 侧刚刚回滚到「本轮真正保留的长度」，draft 这边多出来的一定是被拒或
        未验证的草稿（被接受的那些位置恰好就是 target 保留的那些），所以只有两条：

        - draft 比 target **长**：夹到 `seq.cache.length`，丢掉草稿尾巴；
        - draft 比 target **短**：什么都不做，下一次提议前统一由 `catch_up()` 补齐。

        时机与 target 的回滚一样有硬约束：**必须在 `Scheduler.post_step()` 之前**。
        草稿的 KV 是随本轮输入写进物理块的，先夹掉才不会被误当成已提交前缀。
        """
        if seq.draft_cache.length > seq.cache.length:
            self.kv_cache_pool.truncate_cache(seq.draft_cache, seq.cache.length)

    # -------- 4) 释放 --------

    def release(self, seq):
        """释放一条请求的 draft 资源（完成、失败、被抢占）。

        只动 draft 那一套：物理块还回 draft 池、进度归零。**真实历史、两条随机流、
        惩罚计数一律不动**——抢占不是完成，也不是失败。
        """
        if seq.draft_cache.block_table:
            self.kv_cache_pool.release_cache(seq.draft_cache)
        seq.draft_cache = CacheConfig()
