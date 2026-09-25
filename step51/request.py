"""请求自身的状态：它已知什么、已经算到哪儿、块表长什么样。

这一层只描述「一个请求自己的状态」，不知道物理 KV 池在哪，也不认识调度器——
依赖方向是 request -> cache -> scheduler -> engine，这里不能反向 import。

`cache.length` 是「已经写入 KV 的 token 数」的唯一真相；不要再造一个平行的
「已计算长度」计数，否则两者迟早互相矛盾。
"""

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass
class CacheConfig:
    block_table: list[int] = None # List of block indices in the KV cache
    length: int = 0


class ReadOnlyTokenList(Sequence):
    """token 列表的**只读视图**：不复制底层 list，也不提供 append / extend。

    为什么不用「每次返回 list(...) 或 tuple(...)」来做只读：那会把整段历史再复制一遍，
    正是本关要消掉的开销。这里只包一层引用，切片/索引/len 都直接落到原 list 上。

    对外的可读接口尽量与普通 list 一致（索引、切片、len、in、迭代、与 list 的 +
    和 ==），所以既有代码里 `rec["output_ids"]`、`prompt_ids + output_ids` 这类写法
    照旧可用。改内容只有一条路：`SequenceConfig.append_output_ids()`。
    """

    __slots__ = ("_backing",)

    def __init__(self, backing: list):
        self._backing = backing

    def __getitem__(self, index):
        return self._backing[index]

    def __len__(self):
        return len(self._backing)

    def __iter__(self):
        return iter(self._backing)

    def __contains__(self, item):
        return item in self._backing

    def __eq__(self, other):
        if isinstance(other, ReadOnlyTokenList):
            other = other._backing
        return self._backing == other

    __hash__ = None                     # 底层可变，不该可哈希

    def __add__(self, other):
        return self._backing + list(other)

    def __radd__(self, other):
        return list(other) + self._backing

    def __repr__(self):
        return repr(self._backing)


class SequenceConfig:
    def __init__(self, request_id, prompt_ids, max_new_tokens, block_size,
                 sampling_params=None, sampling_state=None, priority=0, arrival_order=0):
        self.request_id = request_id
        # 调度用的稳定排序键的两半。arrival_order 是调度器发的内部递增整数，
        # 抢占、恢复都不改它；不使用外部 request_id，因为引擎没有建立「ID 唯一」的约束。
        self.priority = priority
        self.arrival_order = arrival_order
        # 已提交历史 = prompt + 已生成的 token，**增量维护**：初始化时复制一次 prompt，
        # 之后每生成一个 token 只追加，不再在读取时执行 prompt_ids + output_ids。
        # 两份列表不是冗余：_all_token_ids 按逻辑位置切模型输入、算 prefix hash；
        # _output_ids 直接回答「生成了多少 / 最后是什么 / 返回给用户什么」。
        # 两者只由 append_output_ids() 一处同步写入，外部拿到的是只读视图。
        self.prompt_ids = list(prompt_ids)
        self.max_new_tokens = max_new_tokens
        self._output_ids = []
        self._all_token_ids = list(self.prompt_ids)
        # 视图只构造一次，之后每次读属性都返回同一个包装（不复制、不新建）
        self._all_token_ids_view = ReadOnlyTokenList(self._all_token_ids)
        self._output_ids_view = ReadOnlyTokenList(self._output_ids)
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

    # 两个只读属性：外部拿不到私有列表，也不能用赋值把视图换掉
    # （那样 append_output_ids 改的私有列表就和外界看到的分家了）。
    @property
    def all_token_ids(self):
        """已提交的完整历史 prompt + output 的只读视图。"""
        return self._all_token_ids_view

    @property
    def output_ids(self):
        """已生成 token 的只读视图。"""
        return self._output_ids_view

    def append_output_ids(self, token_ids):
        """**唯一**写入点：把新生成的 token 同步追加进已提交历史与输出列表。

        `token_ids` 可以是单个 int 或一列 int。
        注意：追加进 _all_token_ids **不等于**增加 cache.length——刚采样的 token
        还没进模型，不允许发布它所在的 KV 块（见 cache.publish_computed_blocks）。
        将来投机解码里未被接受的草稿 token 也不能提前走这里。
        """
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        self._output_ids.extend(token_ids)
        self._all_token_ids.extend(token_ids)

    @property
    def all_token_ids_len(self):
        """完整历史的长度。O(1)，不需要构造整段历史。"""
        return len(self._all_token_ids)

    @property
    def num_uncomputed_tokens(self):
        """还没进入模型、没写进 KV 的 token 数。

        「已计算长度」的唯一真相是 cache.length，这里只算差值，不再另存一份
        可能和它互相矛盾的计数。
        """
        return max(len(self._all_token_ids) - self.cache.length, 0)

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
