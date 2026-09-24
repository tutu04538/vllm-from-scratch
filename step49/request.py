"""请求自身的状态：它已知什么、已经算到哪儿、块表长什么样。

这一层只描述「一个请求自己的状态」，不知道物理 KV 池在哪，也不认识调度器——
依赖方向是 request -> cache -> scheduler -> engine，这里不能反向 import。

`cache.length` 是「已经写入 KV 的 token 数」的唯一真相；不要再造一个平行的
「已计算长度」计数，否则两者迟早互相矛盾。
"""

from dataclasses import dataclass


@dataclass
class CacheConfig:
    block_table: list[int] = None # List of block indices in the KV cache
    length: int = 0


class SequenceConfig:
    def __init__(self, request_id, prompt_ids, max_new_tokens, block_size,
                 sampling_params=None, sampling_state=None, priority=0, arrival_order=0):
        self.request_id = request_id
        # 调度用的稳定排序键的两半。arrival_order 是调度器发的内部递增整数，
        # 抢占、恢复都不改它；不使用外部 request_id，因为引擎没有建立「ID 唯一」的约束。
        self.priority = priority
        self.arrival_order = arrival_order
        self.prompt_ids = prompt_ids
        self.max_new_tokens = max_new_tokens
        self.output_ids = []
        self.cache = CacheConfig()
        self.block_size = block_size  # Size of each block in the KV cache
        self.block_hashes = []  # 本请求已确定的前缀块 hash 链，命中时从缓存里的前缀接上
        # 准入时按最坏情况承诺、还没分配出去的块数；随 ensure_blocks 递减、随释放归还
        self.promised_blocks = 0
        # 采样参数与状态跟着请求走，不跟着 batch 行号走
        self.sampling_params = sampling_params
        self.sampling_state = sampling_state
        # 抢占：本请求被抢占的次数；真正重新进入模型的历史 token 数；曾计算到的最大位置
        self.num_preemptions = 0
        self.recomputed_tokens = 0
        self.high_water = 0
        # 准入时真正从 prefix cache 借到的完整块换来的 token 数（累加）。
        # 命中失败/被撤销的块不计入——只有最终成功借到才算。
        self.reused_tokens = 0
        # 是谁迫使本请求让出 KV（阻塞者）。保存**请求对象引用**而不是 request_id
        # 字符串——本引擎没有建立「ID 唯一」的强约束，比字符串可能认错人。
        # 只在 recompute 模式被赋值；阻塞者结束后会被清掉，不残留引用。
        self.resume_blocker = None

    @property
    def sort_key(self):
        # 数字越小越优先；同级按到达先后。只按这个键比较，不比较对象本身。
        return (self.priority, self.arrival_order)

    @property
    def all_token_ids(self):
        """模型应当「看过」的完整历史：prompt + 已生成的 token。

        最后一个已生成的 token 通常还没进模型——它就是下一轮的输入——
        所以 all_token_ids 总是比 KV 里已有的内容多一个。
        """
        return self.prompt_ids + self.output_ids

    @property
    def num_uncomputed_tokens(self):
        """还没进入模型、没写进 KV 的 token 数。

        「已计算长度」的唯一真相是 cache.length，这里只算差值，不再另存一份
        可能和它互相矛盾的计数。
        """
        return max(len(self.prompt_ids) + len(self.output_ids) - self.cache.length, 0)

    @property
    def is_ready_for_next_token(self):
        """历史只差最后一个 token 没算：本轮再算 1 个就能采样。

        被抢占过的请求重算到这一步时同样成立，所以重算结束后的第一次采样
        走的是和 decode 完全一样的记账，不需要另开一条路径。
        """
        return (self.num_uncomputed_tokens == 1
                and self.cache.length >= len(self.prompt_ids))

    @property
    def prefill_len(self):
        # 旧名字，等价于 num_uncomputed_tokens；无抢占时 clamp 前后的值相同
        return self.num_uncomputed_tokens
