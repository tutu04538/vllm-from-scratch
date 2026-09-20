"""Beam search：不只选一个 token，而是保留若干条候选续写。

和普通采样是两套东西：普通采样每轮只产出一个 token，beam 每轮要
**把所有父分支的子候选放在一起比较**，保留累计分数最高的若干条。
所以它不是「随机生成几遍再选」，也不是「每个父分支各选一个」。

三个必须做对的地方：

1. 剪枝是全局的 —— A 的两个子候选（-1.7、-1.8）要好过 B 的最佳子候选（-2.1），
   于是保留的全是 A 的孩子。这正是需要「同一父 KV 分叉」的原因。
2. KV 不能串分支 —— 兄弟分支必须各有自己的物理块，否则一个孩子写进去的新 KV
   会把另一个孩子的尾巴改掉。首版用完整深复制，不做 copy-on-write。
3. 刚选出的新 token 还没有 KV —— 它的 KV 要等**下一轮** forward 才算，
   不能提前把 cache.length 加一。
"""

import math

import torch

from .sampling import SamplingParams, SamplingState, apply_penalties


class BeamCandidate:
    """一条候选续写。tokens 是已生成的部分（不含 prompt），score 是累计 log 概率。"""

    __slots__ = ("tokens", "score", "seq", "counts", "done", "hit_eos", "idx")

    def __init__(self, tokens, score, seq, counts, done=False, hit_eos=False):
        self.tokens = tokens          # list[int]
        self.score = score            # float，累计 log probability
        self.seq = seq                # SequenceConfig，持有自己的 KV
        self.counts = counts          # dict，这条分支自己的生成计数（惩罚用）
        self.done = done              # 不再扩展（选中 EOS 或到达预算）
        self.hit_eos = hit_eos

    def key(self):
        # 并列时按完整输出 token 序列的字典序升序
        return tuple(self.tokens)


def _final_score(candidate, length_penalty):
    length = max(1, len(candidate.tokens))
    return candidate.score / (length ** length_penalty)


def _rank(candidates, length_penalty):
    # 先按最终分数降序，再按输出序列字典序升序
    return sorted(candidates, key=lambda c: (-_final_score(c, length_penalty), c.key()))


def _branch_state(params, prompt_ids, counts, device):
    state = SamplingState(params, prompt_ids, device)
    state.generated_counts = dict(counts)
    return state


def _top_tokens(logprobs, k):
    """按「分数降序、同分取小 token id」取前 k 个下标。"""
    vocab = logprobs.shape[0]
    k = min(k, vocab)
    cutoff = torch.topk(logprobs, k, largest=True, sorted=True).values[-1]
    # 边界并列可能多于 k 个；nonzero 给出的是升序下标，稳定排序保持这个次序，
    # 于是同分时小的 token id 排在前面
    tied = (logprobs >= cutoff).nonzero().flatten()
    order = torch.argsort(logprobs[tied], descending=True, stable=True)
    return tied[order[:k]]


class BeamSearch:
    """教学版 beam search。约定见 docs/step35_*；不声称与其他库的终止策略一致。"""

    def __init__(self, engine):
        self.engine = engine

    # ---------- 参数与前置检查 ----------

    def _prepare(self, beam_width, length_penalty, repetition_penalty, presence_penalty,
                 frequency_penalty):
        engine = self.engine
        _reject_bool_int(beam_width, "beam_width")
        if not 1 <= beam_width <= 4:
            raise ValueError(f"beam_width 必须在 [1, 4] 内，收到 {beam_width}")
        if isinstance(length_penalty, bool) or not isinstance(length_penalty, (int, float)):
            raise ValueError(f"length_penalty 必须是数值，收到 {length_penalty!r}")
        length_penalty = float(length_penalty)
        if math.isnan(length_penalty) or math.isinf(length_penalty) or not 0.0 <= length_penalty <= 2.0:
            raise ValueError(f"length_penalty 必须在 [0, 2] 内，收到 {length_penalty}")

        # beam 走的是惩罚 + 完整 log_softmax，不接抽样参数；
        # 这里复用 SamplingParams 只为拿到同一套惩罚校验
        params = SamplingParams(repetition_penalty=repetition_penalty,
                                presence_penalty=presence_penalty,
                                frequency_penalty=frequency_penalty)

        if engine.has_unfinished_requests():
            raise RuntimeError("beam_search 要求 Engine 空闲：先把已提交的请求跑完再调用")
        if engine.enable_prefix_caching:
            # 明确拒绝，不偷偷改全局开关（那会影响之后的普通请求）
            raise RuntimeError("beam_search 首版要求关闭 prefix cache："
                               "请用 enable_prefix_caching=False 建 Engine")
        if beam_width > engine.scheduler.max_num_seqs:
            raise ValueError(f"beam_width={beam_width} 超过 max_num_seqs={engine.scheduler.max_num_seqs}；"
                             f"一次 forward 里最多只能打包这么多个请求")
        if beam_width > engine.scheduler.max_num_batched_tokens:
            raise ValueError(f"beam_width={beam_width} 超过 max_num_batched_tokens="
                             f"{engine.scheduler.max_num_batched_tokens}")
        return params, length_penalty

    def _reserve_budget(self, prompt_len, max_new_tokens, beam_width):
        """按「父 + 子」的最坏复制峰值保守检查容量，不做整段预留。"""
        engine = self.engine
        pool = engine.kv_cache_pool
        per_seq = math.ceil((prompt_len + max_new_tokens - 1) / pool.block_size)
        need = 2 * beam_width * per_seq
        free = len(pool._free_block_indices())
        if free < need:
            raise RuntimeError(f"beam 的保守块预算 {need} 超过当前空闲块 {free}"
                               f"（2 × beam_width({beam_width}) × ceil(({prompt_len}+{max_new_tokens}-1)"
                               f"/{pool.block_size}) = {per_seq}）；请调大 num_kv_blocks 或减小 beam_width")
        return need

    # ---------- 主流程 ----------

    def search(self, prompt_ids, max_new_tokens, beam_width=1, length_penalty=0.0,
               repetition_penalty=1.0, presence_penalty=0.0, frequency_penalty=0.0):
        engine = self.engine
        pool = engine.kv_cache_pool
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
            raise ValueError(f"max_new_tokens 必须是整数，收到 {max_new_tokens!r}")
        if max_new_tokens < 0:
            raise ValueError(f"max_new_tokens 不能为负，收到 {max_new_tokens}")
        params, length_penalty = self._prepare(beam_width, length_penalty, repetition_penalty,
                                               presence_penalty, frequency_penalty)
        prompt_ids = list(prompt_ids)
        if not prompt_ids:
            raise ValueError("prompt_ids 不能为空")

        if max_new_tokens == 0:
            # 预算为 0：返回一条空续写，分数 0，不运行模型、不占块
            return [{"output_ids": [], "sum_logprob": 0.0, "score": 0.0}]

        self._reserve_budget(len(prompt_ids), max_new_tokens, beam_width)

        held = []
        try:
            with torch.inference_mode():
                return self._run(prompt_ids, max_new_tokens, beam_width, length_penalty, params, held)
        finally:
            # 无论成功还是中途报错，块都要还回去：不能占着块空转
            for seq in held:
                if seq.cache is not None and seq.cache.block_table:
                    pool.deallocate_block(seq)
                seq.cache = None
            held.clear()

    def _run(self, prompt_ids, max_new_tokens, beam_width, length_penalty, params, held):
        engine = self.engine
        device = engine.device

        root = self._make_seq(prompt_ids, max_new_tokens, held)
        root_cand = BeamCandidate([], 0.0, root, {})
        root_cand.idx = 0
        # 根请求只做一次 prefill；之后每条分支各自前进一个 token
        logprobs = self._prefill(root, prompt_ids, device)
        live = [root_cand]
        finished = []

        for step in range(max_new_tokens):
            # 1) 每个未结束分支按**自己**的历史施加惩罚，再算完整 log_softmax
            kids = []
            for cand in live:
                row = logprobs[cand.idx].to(torch.float32, copy=True)
                state = _branch_state(params, prompt_ids, cand.counts, device)
                row = apply_penalties(row, params, state)
                lp = torch.log_softmax(row, dim=-1)
                for tok in _top_tokens(lp, beam_width):
                    token = int(tok)
                    counts = dict(cand.counts)
                    counts[token] = counts.get(token, 0) + 1
                    # 孩子的 KV 此刻就等于父亲的 KV：新 token 的 KV 下一轮才算
                    kids.append(BeamCandidate(cand.tokens + [token],
                                              cand.score + float(lp[token]),
                                              cand.seq, counts,
                                              done=(token in engine.model.eos_token_ids),
                                              hit_eos=(token in engine.model.eos_token_ids)))

            # 2) 全局剪枝：所有父分支的子候选 + 已结束候选一起比，留 beam_width 条。
            #    不是「每个父分支各留一个」——A 的两个孩子可能都好过 B 的最佳孩子
            pool_all = kids + finished
            pool_all.sort(key=lambda c: (-c.score, c.key()))
            keep = pool_all[:beam_width]
            finished = [c for c in keep if c.done]
            live = [c for c in keep if not c.done]

            # 3) 没有活分支、或已经是最后一轮：不用再准备 KV
            if not live or step == max_new_tokens - 1:
                break
            live = self._reparent(live, prompt_ids, max_new_tokens, held)
            logprobs = self._step(live, device)

        result = finished + live
        return [{"output_ids": list(c.tokens),
                 "sum_logprob": c.score,
                 "score": _final_score(c, length_penalty)}
                for c in _rank(result, length_penalty)]

    # ---------- 下面都是和图外无关的搬运 ----------

    def _make_seq(self, prompt_ids, max_new_tokens, held):
        seq = _new_seq(self.engine, prompt_ids, max_new_tokens)
        held.append(seq)
        return seq

    def _prefill(self, seq, prompt_ids, device):
        """根请求的 prefill，按 max_num_batched_tokens 分块。

        中间块不需要任何 logits（sample_rows=[] 就是 M=0 那一档），只有最后一块
        要最后一行——这正是第三十四关那个「只为需要采样的行算 logits」的用法。
        """
        pool = self.engine.kv_cache_pool
        budget = self.engine.scheduler.max_num_batched_tokens
        out = None
        for start in range(0, len(prompt_ids), budget):
            chunk = prompt_ids[start:start + budget]
            last = start + len(chunk) >= len(prompt_ids)
            out = self.engine.model._forward_append(
                torch.tensor(chunk, device=device), [len(chunk)], [seq.cache], pool,
                sample_rows=[len(chunk) - 1] if last else [])
        return out.float()

    def _step(self, live, device):
        """所有未结束分支各前进一个 token，返回 [len(live), vocab] 的 FP32 logits。

        每个分支输入的是它自己刚选出的那个 token——它的 KV 上一轮还没算，
        正好在这一轮补上。返回的 logits 才是「下一个 token」的分数。
        """
        pool = self.engine.kv_cache_pool
        tokens = [cand.tokens[-1] for cand in live]
        logits = self.engine.model._forward_append(
            torch.tensor(tokens, device=device), [1] * len(live),
            [cand.seq.cache for cand in live], pool, sample_rows=list(range(len(live))))
        for i, cand in enumerate(live):
            cand.idx = i
        return logits.float()

    def _reparent(self, keep, prompt_ids, max_new_tokens, held):
        """给留下的分支各准备一条独立 KV：深复制父分支的块，再释放上一轮的全部块。

        兄弟分支必须各有自己的物理块。只复制 Python 的 block_table 列表、
        却让两条分支写同一个物理尾块，是这一步最容易犯的错。
        """
        pool = self.engine.kv_cache_pool
        stale = list(held)
        for cand in keep:
            new_seq = self._make_seq(prompt_ids, max_new_tokens, held)
            _copy_kv(pool, cand.seq.cache, new_seq.cache)
            cand.seq = new_seq
        # 复制完才释放：上一轮的父块、被淘汰分支的块全在这里
        for seq in stale:
            pool.deallocate_block(seq)
            held.remove(seq)
            seq.cache = None
        return keep


def _reject_bool_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} 必须是整数，收到 {value!r}")


def _new_seq(engine, prompt_ids, max_new_tokens):
    from .cache import SequenceConfig
    seq = SequenceConfig("beam", list(prompt_ids), max_new_tokens, engine.kv_cache_pool.block_size)
    if not engine.kv_cache_pool.allocate_block(seq):
        raise RuntimeError("KV 块不足，无法为 beam 分支分配块")
    return seq


def _copy_kv(pool, src_cache, dst_cache):
    """把父分支已算好的全部层 K/V 与有效长度复制到子分支自己的块里。"""
    src_blocks = src_cache.block_table or []
    dst_blocks = dst_cache.block_table or []
    length = src_cache.length
    full = length // pool.block_size
    rest = length % pool.block_size
    for i in range(full):
        for cache in (pool.k_cache, pool.v_cache):
            cache[:, dst_blocks[i]] = cache[:, src_blocks[i]]
    if rest:
        for cache in (pool.k_cache, pool.v_cache):
            cache[:, dst_blocks[full], :rest] = cache[:, src_blocks[full], :rest]
    dst_cache.length = length
