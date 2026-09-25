"""调度：把等待中的请求放进 running，并为本轮排出一份 prefill/decode 合并的计划。

这一层只管「本轮谁跑、跑几个 token」，不碰模型，也不碰 attention 元数据。
"""

from .cache import CacheConfig, InfeasibleRequest, KVCachePool, SequenceConfig
from .model import DEFAULT_EOS_TOKEN_IDS
from .sampling import SamplingParams, SamplingState
from .speculative import propose_ngram

# 请求字典里属于采样参数的键；其余键必须显式认识，避免把写错的参数名静默忽略
SAMPLING_KEYS = ("temperature", "top_k", "top_p", "repetition_penalty",
                 "presence_penalty", "frequency_penalty", "seed")
REQUEST_KEYS = ("request_id", "prompt_ids", "max_new_tokens", "priority") + SAMPLING_KEYS


class Scheduler:

    def __init__(self, max_num_seqs=1, max_num_batched_tokens=4, block_size=4, enable_prefix_caching=True, on_finished=None, kv_cache_pool: KVCachePool=None, eos_token_ids=None, vocab_size=None, preemption_mode=None, scheduling_policy="fcfs", speculative_mode=None, num_speculative_tokens=2, prompt_lookup_n=2, max_seq_len=None):

        if max_num_batched_tokens <= 0:
            # 一步都排不出 token 的配置没有意义，构造时就明确拒绝，
            # 不要留到运行时变成「第一次 step 静默返回、第二次才报零进展」
            raise ValueError(f"max_num_batched_tokens 必须为正，收到 {max_num_batched_tokens}")
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.running : list[SequenceConfig] = []
        self.waiting : list[SequenceConfig] = []
        self.step_done = []
        # 本轮有没有明确结束/拒绝过请求，给零进展守卫用
        self._failed_this_step = 0
        self.enable_prefix_caching = enable_prefix_caching
        # None = 承诺式（第四十三关行为）；"recompute" = 允许超卖 + 尾部犠牲者抢占
        self.preemption_mode = preemption_mode
        # "fcfs"（默认，忽略优先级）或 "priority"（按 (priority, arrival_order) 排序）
        self.scheduling_policy = scheduling_policy
        # 到达序：内部递增整数，抢占与恢复都不改它，只由 add_request 发一次
        self._arrival_counter = 0
        self.num_preemptions = 0          # 全局抢占总次数
        self.num_blocked_admissions = 0   # 因阻塞者未结束而跳过准入的次数
        self.num_priority_preemptions = 0  # 其中「名额被高优先级顶掉」的次数
        self.block_size = block_size
        self.on_finished = on_finished
        self.num_scheduled_tokens = []
        self.scheduled_items = []  # 本轮计划：prefill 与 decode 合成一份
        self.kv_cache_pool = kv_cache_pool
        # 停止规则来自模型配置；没给就沿用旧默认值 {4}
        self.eos_token_ids = set(eos_token_ids) if eos_token_ids else set(DEFAULT_EOS_TOKEN_IDS)
        self.vocab_size = vocab_size
        # 投机解码：None = 关（逐步行为与第五十一关完全相同）；"ngram" = 用 n-gram
        # 从已提交历史猜草稿。组合是否合法在 Engine 构造时校验，这里只管用。
        self.speculative_mode = speculative_mode
        self.num_speculative_tokens = num_speculative_tokens
        self.prompt_lookup_n = prompt_lookup_n
        # 剩余上下文长度的上限来源。草稿会让本轮多算几个位置，必须提前卡住，
        # 不能等到模型准备输入时才报「超出 max_seq_len」。
        self.max_seq_len = max_seq_len

    def add_request(self, request):
        # 先把参数校验完再建请求对象：非法参数在入队、分配 KV 之前就报出来
        unknown = [k for k in request if k not in REQUEST_KEYS]
        if unknown:
            raise ValueError(f"请求里有不认识的字段 {sorted(unknown)}；"
                             f"本实现只接受 {list(REQUEST_KEYS)}")
        priority = request.get("priority", 0)
        if isinstance(priority, bool) or not isinstance(priority, int):
            # bool 是 int 的子类，但 true/false 不是优先级，明确拒绝
            raise ValueError(f"priority 必须是整数（bool 不算），收到 {priority!r}")
        params = SamplingParams(vocab_size=self.vocab_size,
                                **{k: request[k] for k in SAMPLING_KEYS if k in request})
        if self.speculative_mode is not None and (not params.is_greedy or params.has_penalty):
            # 验证用的是「目标模型的贪心结果」，有惩罚项或随机采样时这个结论不成立，
            # 会静默地错。明确拒绝，不退化成普通解码。
            raise ValueError(f"投机解码只支持贪心且无惩罚项：请求 {request['request_id']!r} "
                             f"的参数是 {params}")
        seq = SequenceConfig(request["request_id"], request["prompt_ids"], request["max_new_tokens"],
                             self.block_size, priority=priority,
                             arrival_order=self._arrival_counter)
        self._arrival_counter += 1
        seq.sampling_params = params
        seq.sampling_state = SamplingState(params, seq.prompt_ids, self.kv_cache_pool.device)
        self._enqueue(seq)

    def _ordered_insert(self, lst, seq):
        """把请求放进列表并保持调度顺序。

        priority 模式按 (priority, arrival_order) 有序插入；fcfs 模式直接追加
        （列表本身就是到达序）。这样 `lst[0]` 在两个模式下都是「下一个该跑的」。
        """
        if self.scheduling_policy != "priority":
            lst.append(seq)
            return
        key = seq.sort_key
        for i, other in enumerate(lst):
            if other.sort_key > key:
                lst.insert(i, seq)
                return
        lst.append(seq)

    def _enqueue(self, seq):
        """新请求进入 waiting。fcfs 追加（到达序），priority 按排序键插入。"""
        self._ordered_insert(self.waiting, seq)

    def has_unfinished_requests(self):
        return len(self.running) + len(self.waiting) > 0

    def schedule(self):
        """一轮调度的四个阶段，顺序不能换。

        每个阶段各自负责一件事，方便指着某一阶段回答「它读什么、能改什么、
        失败时留下什么」。顶层保持能一眼看出时序。
        """
        self._begin_step()
        self._finish_zero_budget_waiting()
        self._admit_waiting()
        planned_items = self._plan_tokens()          # 只决定谁跑几个 token
        self.scheduled_items = self._reserve_blocks(planned_items)   # 再补物理块
        self._check_progress()
        return self.scheduled_items

    def _begin_step(self):
        # 上一轮的计划与完成记录都在这里清掉
        self.scheduled_items = []
        self.step_done = []
        self._failed_this_step = 0

    def _finish_zero_budget_waiting(self):
        # 零输出预算的请求不占运行名额、也不进 KV，直接完成
        for seq in self.waiting:
            if seq.max_new_tokens == 0:
                if self.on_finished:
                    self.on_finished({"request_id": seq.request_id, "output_ids": []})
                self.step_done.append({"request_id": seq.request_id, "output_ids": []})
        self.waiting = [seq for seq in self.waiting if seq.max_new_tokens > 0]

    def _admit_waiting(self):
        """把 waiting 队首放进 running。

        FCFS：队首拿不到就整队等。唯一能越过队首的情况是队首**不可能完成**——
        那时把它明确结束掉再试下一个，否则它会永远堵在这里（第 36 关记的就是这个）。
        """
        while self.waiting:
            next_seq = self.waiting[0]
            if len(self.running) >= self.max_num_seqs:
                # 名额满。fcfs（以及 priority 但没有严格更低优先级的候选）直接整队等；
                # priority 模式允许队首顶掉一条严格更低优先级的 running 请求。
                # 这发生在**本轮计划与补块之前**，所以被顶掉的请求还没拿到本轮的
                # token 预算和物理块，不需要回滚任何东西。
                if not self._preempt_for_slot(next_seq):
                    break
            if self._resume_blocked(next_seq):
                # 刚被抢占的请求别急着恢复：阻塞者还没结束，现在重建 KV 大概率
                # 又被同一条请求抢走，白白重算一次。
                # 这里直接 break：被阻塞的队首同样不能让更晚到达的请求越过它，
                # 即使 running 还有空名额。等待本身不算计算进展，零进展守卫照旧。
                self.num_blocked_admissions += 1
                break
            try:
                admitted = self.kv_cache_pool.allocate_block(next_seq)
            except InfeasibleRequest as exc:
                self.waiting.remove(next_seq)
                self._fail(next_seq, str(exc))
                continue
            if admitted:
                self.waiting.remove(next_seq)
                self._ordered_insert(self.running, next_seq)
                next_seq.resume_blocker = None   # 关系已经用上了，不再保留引用
            else:
                break  # 暂时不够，等 running 里的请求让出来

    def _plan_tokens(self):
        """本轮 token budget 的分配，返回**草稿计划**。

        只决定「谁跑几个 token」，不碰物理块、不改 cache.length、不写输出。
        真正会 forward 的是 `_reserve_blocks()` 挑出来的子集。

        分配按调度顺序，priority 模式下**按优先级分层**：高优先级组先拿额度，
        同级内部保持「ready（只差一个 token）的各留 1 个、其余给 prefill」。
        这样即使 max_num_batched_tokens=1，高优先级 prompt 也不会被低优先级的
        decode 长期占住唯一额度。
        fcfs 模式下只有一个组 = 整个 running，与第四十六关逐字节等价。
        """
        planned = []
        token_budget_remaining = self.max_num_batched_tokens
        for group in self._budget_groups():
            ready_req = [req for req in group if req.is_ready_for_next_token]
            if self.scheduling_policy != "priority":
                assert len(ready_req) <= token_budget_remaining, \
                    "Decode token budget exceeds max_num_batched_tokens"
            elif len(ready_req) > token_budget_remaining:
                # 额度不够这么多 ready 请求：按组内顺序（到达序）保留靠前的，
                # 其余的这轮不算，额度留给更高优先级用
                ready_req = ready_req[:token_budget_remaining]
            ready_ids = {id(req) for req in ready_req}
            prefill_token_budget = token_budget_remaining - len(ready_req)

            for seq in group:
                # 本轮要算的永远是 all_token_ids 的一段：抢占后自然就从 cache.length
                # 处重放 prompt + 旧 output，不需要为「重算」另写一条分支。
                if id(seq) in ready_ids:
                    # 真实历史只差最后一个 token，正好是投机的时机：
                    # 输入 = [x] + 草稿，num_real 记的是「真实历史那几个 token」。
                    draft_ids = self._plan_drafts(seq, prefill_token_budget)
                    num_real = 1
                    prefill_token_budget -= len(draft_ids)
                    num_scheduled_tokens = num_real + len(draft_ids)
                else:
                    num_uncomputed = seq.num_uncomputed_tokens
                    if num_uncomputed == 0 or prefill_token_budget == 0:
                        continue
                    num_scheduled_tokens = min(num_uncomputed, prefill_token_budget)
                    prefill_token_budget -= num_scheduled_tokens
                    num_real = num_scheduled_tokens
                    draft_ids = []

                start = seq.cache.length
                input_ids = seq.all_token_ids[start:start + num_real]
                if draft_ids:
                    # 草稿只是**临时计划**：它绝不进 seq 的历史，只作为本轮输入
                    input_ids = input_ids + list(draft_ids)
                planned.append({
                    "request": seq,
                    "input_ids": input_ids,
                    "num_scheduled_tokens": num_scheduled_tokens,
                    "draft_ids": list(draft_ids),
                    # 回滚与重算统计都要「本轮从哪儿开始算」；用 end - num_scheduled_tokens
                    # 反推在投机下是错的——那个 end 是回滚后的长度。
                    "start_cache_length": start,
                    # 只有本轮算到当前 all_token_ids 末尾，才允许采样；
                    # 中间的重算 chunk 不采样，也不重放 on_token。
                    # 投机项多出来的部分是草稿，不算进「真实历史覆盖到了末尾」。
                    "can_sample": num_real == seq.num_uncomputed_tokens
                })
            token_budget_remaining = prefill_token_budget
        return planned

    def _plan_drafts(self, seq, spare_budget):
        """给一条 ready 的请求算草稿。**只读**：不改请求状态，也不碰 KV。

        K 由四个上限共同决定，任何一个不满足就缩短，缩到 0 就是普通的 1-token 路径：

        - 配置上限 `num_speculative_tokens`；
        - 剩余 token 预算（草稿同样占 `max_num_batched_tokens`）；
        - 剩余输出上限：全部接受还会多一个 bonus，所以 `K <= R - 1`；
        - 剩余上下文长度：本轮要算 `[cache.length, cache.length + 1 + K)`。

        容量（物理块）不在这里判——那是 `_reserve_blocks()` 的事，它能真的补块。
        """
        if self.speculative_mode != "ngram":
            return []
        if not seq.output_ids or seq.num_uncomputed_tokens != 1:
            # prefill、重算途中都不投机：历史还没到「只差最后一个 token」。
            # 也要求至少生成过 1 个 token，否则没有可比较的续写行为。
            return []
        remaining_outputs = seq.max_new_tokens - len(seq.output_ids)
        k = min(self.num_speculative_tokens, spare_budget, remaining_outputs - 1)
        if self.max_seq_len is not None:
            k = min(k, self.max_seq_len - seq.cache.length - 1)
        if k <= 0:
            return []
        return propose_ngram(seq.all_token_ids, self.prompt_lookup_n, k)

    def _reserve_blocks(self, planned_items):
        """按计划顺序补物理块，返回**真的会 forward** 的那批 item。

        容量不足时（只有超卖模式会走到）从排序键更靠后、且本轮尚未安排的请求里选犠牲者。
        已经补过块的不动：本轮的 token 预算和刚分配的块都已经记账。

        草稿计划里被牺牲的请求直接跳过——它的 token 预算留空、不转给别人，
        也不能进 Engine 或算进重算量。
        """
        running_snapshot = list(self.running)
        committed_items = []
        for item in planned_items:
            seq = item["request"]
            if seq not in self.running:
                continue        # 本轮被选成犠牲者，计划作废
            if item["draft_ids"]:
                # 草稿是**可选**的加速：容量不够就逐枚缩短，最终退回普通 1-token 路径。
                # 缩短只改这一轮进模型几个 token，不动任何已提交状态。
                # 只读地问（can_grow 不摘链），真正补块仍然只有 ensure_blocks 一处。
                while item["draft_ids"] and not self.kv_cache_pool.can_grow(
                        seq, item["num_scheduled_tokens"]):
                    self._shrink_draft(item)
            if self.kv_cache_pool.ensure_blocks(seq, item["num_scheduled_tokens"]):
                committed_items.append(item)
                continue
            if self._make_room(seq, item["num_scheduled_tokens"], running_snapshot):
                committed_items.append(item)
            # 否则：当前请求自己就是最后可选犠牲者，本轮先不排它，
            # 让已经安排在它前面的工作照常跑完。
        return committed_items

    @staticmethod
    def _shrink_draft(item):
        """把计划里的草稿砍掉最后一枚。

        只改这一轮的临时计划（输入、计数）。`_plan_tokens()` 已经把 token 预算
        按原长度发下去了，砍短之后多出来的额度**不回收**——本关投机只允许
        `max_num_seqs=1`，没有别的请求在等这份额度，留给同一个请求反而更简单。
        """
        item["draft_ids"] = item["draft_ids"][:-1]
        item["input_ids"] = item["input_ids"][:-1]
        item["num_scheduled_tokens"] -= 1

    def _check_progress(self):
        # 不变量：还有未完成的请求时，这一步必须至少真正算一个 token、
        # 或明确结束/拒绝一条。**接纳本身不算进展**——只把请求从 waiting 挪到
        # running、一个 token 也没算，那是空转，不是前进。
        if (self.has_unfinished_requests() and not self.scheduled_items
                and not self._failed_this_step):
            raise RuntimeError(
                f"调度没有任何进展：running={len(self.running)} waiting={len(self.waiting)}，"
                f"本轮没有排出任何 token，也没有明确结束或拒绝任何请求。"
                f"Pool: {len(self.kv_cache_pool._free_block_indices())} 空闲 / "
                f"{self.kv_cache_pool.promised_blocks} 已承诺 / "
                f"{self.kv_cache_pool.num_kv_blocks} 总块")

    def _budget_groups(self):
        """本轮 token budget 的分配单位，按「先高优先级」的顺序返回。

        fcfs：一个组 = 整个 running（保持第四十六关的算法）。
        priority：按优先级数值升序分层，组内保持到达序 —— running 已经按排序键
        维护，所以同优先级的请求本来就是连续的一段。
        """
        if self.scheduling_policy != "priority":
            return [list(self.running)]
        groups, cur = [], []
        for seq in self.running:
            if cur and seq.priority != cur[-1].priority:
                groups.append(cur)
                cur = []
            cur.append(seq)
        if cur:
            groups.append(cur)
        return groups

    def _preempt_for_slot(self, seq):
        """名额满时，让优先级请求顶掉一条**严格更低优先级**的 running 请求。

        相同优先级不得互相顶替，否则同级请求会互相插队（需求 §2.1）。
        fcfs 模式永远返回 False，调度顺序与第四十六关完全一致。

        选中的是排序键最大的那条：优先级数值最大、同级里到达最晚的。
        """
        if self.scheduling_policy != "priority":
            return False
        candidates = [v for v in self.running if v.priority > seq.priority]
        if not candidates:
            return False
        self._preempt(max(candidates, key=lambda v: v.sort_key), blocker=seq,
                      priority_preemption=True)
        return True

    def _victims_after(self, seq, running):
        """可以当犠牲者的请求：本轮 forward 还没跑、且排在 seq 之后，从尾部往前。

        priority 模式按排序键取严格更靠后的——可以是优先级更低，**也可以是同级
        但到达更晚**；后者必不可少，否则两条同级请求把池子占满时谁也不动不了。
        更靠前的排序键一律不碰。
        fcfs 模式就是 running 尾部的未安排请求（第四十四关语义，逐字节一致）。

        顺序不变量：补块循环按 `scheduled_items` 的顺序依次调用本函数，而
        `scheduled_items` 是按 `running` 的顺序排出来的（priority 下 `_budget_groups()`
        按排序键升序）。本函数只返回排序键严格更靠后的请求，也就是**还没轮到**的那些，
        所以候选一定都是「本轮尚未补过块」的，不会回滚已记账的计划。
        """
        if self.scheduling_policy != "priority":
            out = []
            for victim in reversed(running):
                if victim is seq:
                    break
                out.append(victim)
            return out
        later = [v for v in running if v.sort_key > seq.sort_key]
        later.sort(key=lambda v: v.sort_key, reverse=True)
        return later

    def _make_room(self, seq, num_tokens, running):
        """释放排序键更靠后的、本轮尚未安排的请求，直到当前请求能补到块。

        返回 False 表示已经走到当前请求自己——按 FCFS 不再往后找犠牲者。

        候选里不会出现「本轮已经补过块」的请求：补块循环按 `scheduled_items` 的顺序
        走，而那个顺序就是 `running` 的顺序，`_victims_after()` 又只返回排序键严格
        更靠后的——两者方向刚好相反。详见 `_victims_after()` 的说明。
        """
        for victim in self._victims_after(seq, running):
            if victim not in self.running:
                continue        # 本轮已经被抢占过了
            self._preempt(victim, blocker=seq)
            if self.kv_cache_pool.ensure_blocks(seq, num_tokens):
                return True
        return False

    def _resume_blocked(self, seq):
        """队首的请求是否应该继续等：它的阻塞者还没结束。

        阻塞者结束或明确失败后关系立即失效，并顺手清掉引用，不留残余。

        只对 recompute 模式有意义：`resume_blocker` 只在 `_preempt()` 里被赋值，
        而 `_preempt()` 只能从 `_make_room()` 到达，后者只在超卖模式（`ensure_blocks()`
        返回 False）下被调用。承诺式路径不会经过这里，行为与第四十四关一致。
        """
        blocker = seq.resume_blocker
        if blocker is None:
            return False
        # 「仍未完成」= 还在队列里，不限于正在 running：被抢占的阻塞者依然未完成。
        # `waiting` 那一半当前走不到（阻塞者抢占时插 waiting 队首，必然排在它所阻塞的
        # 请求前面），保留是有意的兜底——真被走到且队列推不动时零进展守卫会报错。
        # 判断依据与实测见 docs/step45_blocker_aware_resume.md §2.4。
        if blocker in self.running or blocker in self.waiting:
            return True
        seq.resume_blocker = None
        return False

    def _preempt(self, seq, blocker=None, priority_preemption=False):
        """重算式抢占：丢掉物理 KV，保留历史，放回 waiting。

        blocker 是「当前是谁要块」——就是迫使 seq 让路的请求。
        priority_preemption 为真表示这次是「名额被更高优先级顶掉」，
        单独计进 num_priority_preemptions，便于与普通容量压力区分。

        抢占不是完成，也不是失败，所以这里不碰 on_finished / on_token，
        不清空 output_ids，不重置惩罚计数和随机数发生器。
        """
        # 记下抢占前已经算到哪儿：恢复时本轮 [start, end) 与它的重叠就是重算量
        seq.high_water = max(seq.high_water, seq.cache.length)
        self.running.remove(seq)
        self.kv_cache_pool.deallocate_block(seq)
        # 物理 KV 进度归零；prompt_ids / output_ids / SamplingState 原样留着，
        # 下一轮从 all_token_ids[cache.length] 继续重放
        seq.cache = CacheConfig()
        # 物理块已经还回去了，hash 链跟着作废；下次准入会按当时的历史重建
        seq.block_hashes = []
        seq.num_preemptions += 1
        self.num_preemptions += 1
        if priority_preemption:
            self.num_priority_preemptions += 1
        # 记住是谁迫使它让路：在阻塞者结束之前不要急着恢复它。
        # 若它后来又被另一条更早的请求抢占，这里会覆盖成那次真正让它让路的请求。
        seq.resume_blocker = blocker
        # 回到 waiting。
        # fcfs：按**全局 FCFS** 插到队首——本轮从 running 尾部依次取 C、B，
        #   每条都插队首，合起来正好是 [B, C] + 原有的 waiting。
        # priority：按 (priority, arrival_order) 插回原位；arrival_order 没变，
        #   所以它就是「按优先级和原到达序」重新排队（需求 §3）。
        # 同一次 schedule() 不会再准入它们——准入在选犠牲者之前就做完了。
        if self.scheduling_policy == "priority":
            self._ordered_insert(self.waiting, seq)
        else:
            self.waiting.insert(0, seq)

    def _fail(self, seq, message):
        # 明确结束一条不可能完成的请求，并把它的 KV 还回池子
        if seq.cache is not None and seq.cache.block_table:
            self.kv_cache_pool.deallocate_block(seq)
        seq.cache = None
        self._failed_this_step += 1
        record = {"request_id": seq.request_id, "output_ids": list(seq.output_ids),
                  "error": message}
        if self.on_finished:
            self.on_finished(record)
        self.step_done.append(record)

    def post_step(self):
        """一步执行完之后的三件事，顺序同样不能换。"""
        self._publish_computed_blocks()      # 必须在 forward 之后
        self._update_recompute_metrics()     # 只统计真的跑过的计划
        self._finish_completed_requests()    # 回调、释放引用、移出 running

    def _publish_computed_blocks(self):
        """把已经算完的完整块登记为可复用。

        时机是**模型 forward 之后、完成释放之前**：cache.length 已经是本轮真正
        写进 KV 的进度，而采样 append 发生在这之前，所以刚采样、还没进模型的 token
        绝不会被登记。每步都做（而不是只在抢占/完成前），被抢占时它的进度才早就在
        缓存里，恢复时才有东西可复用。
        **和 preemption_mode 无关**：什么时候更新 prefix cache、和「准入时要不要按最坏
        情况承诺未来块」是两件不相干的事。两种模式都在这里每步登记。
        （第四十六关曾把它限制在 recompute 下，理由是「旧模式行为不变」——那是一条
        范围约束，不是正确性约束；第五十一关按这个解耦改掉，行为差异见文档。）
        """
        if not self.kv_cache_pool.enable_prefix_caching:
            return
        for seq in self.running:
            self.kv_cache_pool.publish_computed_blocks(seq)

    def _update_recompute_metrics(self):
        """按**实际执行**的计划记重算量。

        被补块阶段丢掉的 item 根本没跑，cache.length 没动过，按计划记会把同一个
        区间重复计入。跑到这里的 item 都真的进过模型，所以 [start, end) 就是本轮
        真正算过的区间，与旧高水位 high_water 的重叠就是「重算」的部分。

        起点用计划里记下的 `start_cache_length`，**不是** `end - num_scheduled_tokens`：
        投机解码拒绝草稿之后 `cache.length` 被回滚过，那个 end 是**最终保留**的长度，
        而 `num_scheduled_tokens` 仍是本轮真实进模型的 token 数（K+1），两者相减
        会得到一个比真实起点靠后的位置。`high_water` 同样按最终保留的长度推进。
        """
        for item in self.scheduled_items:
            seq = item["request"]
            start = item["start_cache_length"]
            end = seq.cache.length
            seq.recomputed_tokens += max(0, min(end, seq.high_water) - start)
            seq.high_water = max(seq.high_water, end)

    def _finish_completed_requests(self):
        """判停、回调、释放 KV、把完成的请求移出 running。"""
        for seq in self.running:
            if not seq.output_ids:
                continue  # 还没产生过输出（prompt 还在算，或重算还没追上）——没有可判停的 token

            if seq.output_ids[-1] in self.eos_token_ids or len(seq.output_ids) >= seq.max_new_tokens:
                # 这里不再单独登记完整块：post_step 开头的 _publish_computed_blocks()
                # 已经覆盖了本轮所有 running 请求（含此刻即将完成的这条），
                # 而且 publish 本身是幂等的。少一次调用、少一处重复判断。
                # 对外接口仍然是普通 list：显式转一次，不让只读视图泄漏出去
                output_ids = list(seq.output_ids)
                if self.on_finished:
                    self.on_finished({"request_id": seq.request_id, "output_ids": output_ids})
                self.step_done.append({"request_id": seq.request_id, "output_ids": output_ids})
                self.kv_cache_pool.deallocate_block(seq)
                seq.cache = None  # Reset past_kv for completed sequences

        self.running = [seq for seq in self.running
                        if len(seq.output_ids) < seq.max_new_tokens
                        and (not seq.output_ids or seq.output_ids[-1] not in self.eos_token_ids)]
