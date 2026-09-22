"""调度：把等待中的请求放进 running，并为本轮排出一份 prefill/decode 合并的计划。

这一层只管「本轮谁跑、跑几个 token」，不碰模型，也不碰 attention 元数据。
"""

from .cache import CacheConfig, InfeasibleRequest, KVCachePool, SequenceConfig
from .model import DEFAULT_EOS_TOKEN_IDS
from .sampling import SamplingParams, SamplingState

# 请求字典里属于采样参数的键；其余键必须显式认识，避免把写错的参数名静默忽略
SAMPLING_KEYS = ("temperature", "top_k", "top_p", "repetition_penalty",
                 "presence_penalty", "frequency_penalty", "seed")
REQUEST_KEYS = ("request_id", "prompt_ids", "max_new_tokens") + SAMPLING_KEYS


class Scheduler:

    def __init__(self, max_num_seqs=1, max_num_batched_tokens=4, block_size=4, enable_prefix_caching=True, on_finished=None, kv_cache_pool: KVCachePool=None, eos_token_ids=None, vocab_size=None, preemption_mode=None):

        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.running : list[SequenceConfig] = []
        self.waiting : list[SequenceConfig] = []
        self.step_done = []
        # 本轮有没有接纳 / 明确结束过请求，给零进展守卫用
        self._admitted_this_step = 0
        self._failed_this_step = 0
        self.enable_prefix_caching = enable_prefix_caching
        # None = 承诺式（第四十三关行为）；"recompute" = 允许超卖 + 尾部犠牲者抢占
        self.preemption_mode = preemption_mode
        self.num_preemptions = 0          # 全局抢占总次数
        self._preempted_this_step = 0     # 本轮已抢占几条，给 waiting 定位用
        self._allocated_this_step = set()  # 本轮已经补过块的请求：不可再被抢占
        self.block_size = block_size
        self.on_finished = on_finished
        self.num_scheduled_tokens = []
        self.scheduled_items = []  # 本轮计划：prefill 与 decode 合成一份
        self.kv_cache_pool = kv_cache_pool
        # 停止规则来自模型配置；没给就沿用旧默认值 {4}
        self.eos_token_ids = set(eos_token_ids) if eos_token_ids else set(DEFAULT_EOS_TOKEN_IDS)
        self.vocab_size = vocab_size

    def add_request(self, request):
        # 先把参数校验完再建请求对象：非法参数在入队、分配 KV 之前就报出来
        unknown = [k for k in request if k not in REQUEST_KEYS]
        if unknown:
            raise ValueError(f"请求里有不认识的字段 {sorted(unknown)}；"
                             f"本实现只接受 {list(REQUEST_KEYS)}")
        params = SamplingParams(vocab_size=self.vocab_size,
                                **{k: request[k] for k in SAMPLING_KEYS if k in request})
        seq = SequenceConfig(request["request_id"], request["prompt_ids"], request["max_new_tokens"],
                             self.block_size)
        seq.sampling_params = params
        seq.sampling_state = SamplingState(params, seq.prompt_ids, self.kv_cache_pool.device)
        self.waiting.append(seq)

    def has_unfinished_requests(self):
        return len(self.running) + len(self.waiting) > 0

    def schedule(self):
        # Fill running with waiting sequences if there's space
        self.scheduled_items = []
        self.step_done = []
        self._admitted_this_step = 0
        self._failed_this_step = 0
        self._preempted_this_step = 0
        self._allocated_this_step = set()

        for seq in self.waiting:
            if seq.max_new_tokens == 0:
                # Zero budget requests are processed immediately
                if self.on_finished:
                    self.on_finished({"request_id": seq.request_id, "output_ids": []})
                self.step_done.append({"request_id": seq.request_id, "output_ids": []})

        self.waiting = [seq for seq in self.waiting if seq.max_new_tokens > 0]
        # 接纳。FCFS：队首拿不到就整队等（下面的 break）。
        # 唯一能越过队首的情况是队首**不可能完成**——那时把它明确结束掉再试下一个，
        # 否则它会永远堵在这里（第 36 关记的就是这个）。
        while len(self.running) < self.max_num_seqs and self.waiting:
            next_seq = self.waiting[0]
            try:
                admitted = self.kv_cache_pool.allocate_block(next_seq)
            except InfeasibleRequest as exc:
                self.waiting.pop(0)
                self._fail(next_seq, str(exc))
                continue
            if admitted:
                self.waiting.pop(0)
                self.running.append(next_seq)
                self._admitted_this_step += 1
            else:
                break  # 暂时不够，等 running 里的请求让出来

        # 预留：历史只差最后一个 token 的请求，各自先占住 1 个 token 额度，
        # 保证它们不会被正在重算/预填的请求挤掉。无抢占时这与「decode 优先级」
        # 完全等价（那时 prefill_len == 0 就是 is_ready_for_next_token）。
        ready_req = [req for req in self.running if req.is_ready_for_next_token]
        decode_token_budget = len(ready_req)
        assert decode_token_budget <= self.max_num_batched_tokens, "Decode token budget exceeds max_num_batched_tokens"
        prefill_token_budget = self.max_num_batched_tokens - decode_token_budget

        for seq in self.running:
            # 本轮要算的永远是 all_token_ids 的一段：抢占后自然就从 cache.length
            # 处重放 prompt + 旧 output，不需要为「重算」另写一条分支。
            if seq.is_ready_for_next_token:
                num_scheduled_tokens = 1
            else:
                num_uncomputed = seq.num_uncomputed_tokens
                if num_uncomputed == 0 or prefill_token_budget == 0:
                    continue
                num_scheduled_tokens = min(num_uncomputed, prefill_token_budget)
                prefill_token_budget -= num_scheduled_tokens

            start = seq.cache.length
            end = start + num_scheduled_tokens

            self.scheduled_items.append({
                "request": seq,
                "input_ids": seq.all_token_ids[start:end],
                "num_scheduled_tokens": num_scheduled_tokens,
                # 只有本轮算到当前 all_token_ids 末尾，才允许采样一个新 token；
                # 中间的重算 chunk 不采样，也不重放 on_token
                "can_sample": num_scheduled_tokens == seq.num_uncomputed_tokens
            })

        # 计划定下来之后再按需补块：补多少取决于本轮真的排到多少 token，
        # 而不是「这条请求最多可能生成多少」。这一步必须在 forward 之前。
        #
        # 容量不足时（只有超卖模式会走到）从 running 尾部、且**本轮尚未安排**的请求里
        # 选犠牲者。已安排的不动：本轮的 token 预算、scheduled_items 和刚分配的块
        # 都已经记账，回滚它们不属于第一阶段。
        running = list(self.running)
        for item in list(self.scheduled_items):
            seq = item["request"]
            if seq not in self.running:
                # 刚被本轮选成犠牲者：它的计划作废（它的 token 预算留空，不转给别人）
                self.scheduled_items.remove(item)
                continue
            if self.kv_cache_pool.ensure_blocks(seq, item["num_scheduled_tokens"]):
                self._allocated_this_step.add(id(seq))
                continue
            if not self._make_room(seq, item["num_scheduled_tokens"], running):
                # 当前请求自己就是最后可选犠牲者：本轮先不排它，
                # 让已经安排在它前面的工作照常跑完。
                self.scheduled_items.remove(item)

        # 不变量：还有未完成的请求时，这一步必须至少做成一件事——
        # 排出 token、接纳新请求、或明确结束一条。三件都没有就是死锁，宁可报出来也不要静默空转。
        if (self.has_unfinished_requests() and not self.scheduled_items
                and not self._admitted_this_step and not self._failed_this_step):
            raise RuntimeError(
                f"调度没有任何进展：running={len(self.running)} waiting={len(self.waiting)}，"
                f"本轮没有排出任何 token、没有接纳、也没有结束任何请求。"
                f"Pool: {len(self.kv_cache_pool._free_block_indices())} 空闲 / "
                f"{self.kv_cache_pool.promised_blocks} 已承诺 / "
                f"{self.kv_cache_pool.num_kv_blocks} 总块")
        return self.scheduled_items

    def _make_room(self, seq, num_tokens, running):
        """从 running 尾部释放尚未安排的请求，直到当前请求能补到块。

        返回 False 表示已经走到当前请求自己——按 FCFS 不再往后找犠牲者。
        """
        for victim in reversed(running):
            if victim is seq:
                return False
            if victim not in self.running:
                continue        # 本轮已经被抢占过了
            if id(victim) in self._allocated_this_step:
                # 排在它前面的都补过块了，再往前找就会回滚已记账的计划
                return False
            self._preempt(victim)
            if self.kv_cache_pool.ensure_blocks(seq, num_tokens):
                return True
        return False

    def _preempt(self, seq):
        """重算式抢占：丢掉物理 KV，保留历史，放回 waiting。

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
        seq.num_preemptions += 1
        self.num_preemptions += 1
        # 被抢占者按原有的 FCFS 先后回到 waiting：本轮先从尾部抢最后一条，
        # 所以每条新的都插在本轮已抢的前面。同一次 schedule() 不会再准入它们
        # ——准入在选犠牲者之前就已经做完了。
        self.waiting.insert(len(self.waiting) - self._preempted_this_step, seq)
        self._preempted_this_step += 1

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

        # 重算量只在这里记：计划可能在补块阶段被丢掉（没跑就没有重算），
        # 那时 cache.length 根本没动，按计划记会把同一个区间重复计入。
        # 跑到这里的 item 都真的进过模型，所以 end - start 就是本轮真正算的 token 数，
        # 与旧高水位 high_water 的重叠就是「重算」的部分。
        for item in self.scheduled_items:
            seq = item["request"]
            end = seq.cache.length
            start = end - item["num_scheduled_tokens"]
            seq.recomputed_tokens += max(0, min(end, seq.high_water) - start)
            seq.high_water = max(seq.high_water, end)

        for seq in self.running:

            if not seq.output_ids:
                continue  # 还没产生过输出（prompt 还在算，或重算还没追上）——没有可判停的 token

            if seq.output_ids[-1] in self.eos_token_ids or len(seq.output_ids) >= seq.max_new_tokens:
                if self.enable_prefix_caching:
                    # 先把可复用的完整 prompt 块登记为缓存，再释放本请求的活动引用
                    self.kv_cache_pool.publish_completed_prompt_blocks(seq)
                if self.on_finished:
                    self.on_finished({"request_id": seq.request_id, "output_ids": seq.output_ids})
                self.step_done.append({"request_id": seq.request_id, "output_ids": seq.output_ids})
                self.kv_cache_pool.deallocate_block(seq)
                seq.cache = None  # Reset past_kv for completed sequences

        self.running = [seq for seq in self.running
                        if len(seq.output_ids) < seq.max_new_tokens
                        and (not seq.output_ids or seq.output_ids[-1] not in self.eos_token_ids)]
