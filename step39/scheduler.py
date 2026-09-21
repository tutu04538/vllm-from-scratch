"""调度：把等待中的请求放进 running，并为本轮排出一份 prefill/decode 合并的计划。

这一层只管「本轮谁跑、跑几个 token」，不碰模型，也不碰 attention 元数据。
"""

from .cache import KVCachePool, SequenceConfig
from .model import DEFAULT_EOS_TOKEN_IDS
from .sampling import SamplingParams, SamplingState

# 请求字典里属于采样参数的键；其余键必须显式认识，避免把写错的参数名静默忽略
SAMPLING_KEYS = ("temperature", "top_k", "top_p", "repetition_penalty",
                 "presence_penalty", "frequency_penalty", "seed")
REQUEST_KEYS = ("request_id", "prompt_ids", "max_new_tokens") + SAMPLING_KEYS


class Scheduler:

    def __init__(self, max_num_seqs=1, max_num_batched_tokens=4, block_size=4, enable_prefix_caching=True, on_finished=None, kv_cache_pool: KVCachePool=None, eos_token_ids=None, vocab_size=None):

        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.running : list[SequenceConfig] = []
        self.waiting : list[SequenceConfig] = []
        self.step_done = []
        self.enable_prefix_caching = enable_prefix_caching
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

        for seq in self.waiting:
            if seq.max_new_tokens == 0:
                # Zero budget requests are processed immediately
                if self.on_finished:
                    self.on_finished({"request_id": seq.request_id, "output_ids": []})
                self.step_done.append({"request_id": seq.request_id, "output_ids": []})

        self.waiting = [seq for seq in self.waiting if seq.max_new_tokens > 0]
        # Limit the number of batched tokens
        while len(self.running) < self.max_num_seqs and self.waiting:
            next_seq = self.waiting[0]
            if self.kv_cache_pool.allocate_block(next_seq):
                self.waiting.pop(0)
                self.running.append(next_seq)
            else:
                break  # No more blocks available, cannot schedule more sequences

        decode_req = [req for req in self.running if req.prefill_len == 0]
        decode_token_budget = len(decode_req)
        assert decode_token_budget <= self.max_num_batched_tokens, "Decode token budget exceeds max_num_batched_tokens"
        prefill_token_budget = self.max_num_batched_tokens - decode_token_budget

        for seq in self.running:
            if seq.prefill_len > 0:
                if prefill_token_budget == 0:
                    continue

                num_scheduled_tokens = min(seq.prefill_len, prefill_token_budget)
                prefill_token_budget -= num_scheduled_tokens
                self.scheduled_items.append({
                    "request": seq,
                    "input_ids": seq.prompt_ids[seq.cache.length:seq.cache.length + num_scheduled_tokens],
                    "num_scheduled_tokens": num_scheduled_tokens,
                    "can_sample": num_scheduled_tokens == seq.prefill_len
                })
            else:
                self.scheduled_items.append({
                    "request": seq,
                    "input_ids": [seq.output_ids[-1]],
                    "num_scheduled_tokens": 1,
                    "can_sample": True
                })

        return self.scheduled_items

    def post_step(self):

        for seq in self.running:

            if seq.prefill_len > 0:
                continue  # Prefill not finished yet

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
