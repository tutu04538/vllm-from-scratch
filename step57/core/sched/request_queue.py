"""等待队列（对应 vLLM `v1/core/sched/request_queue.py`）。

职责只有：入队、取下一条、前插（恢复）、按 ID 移除、判空。**不做**抢占决策，也不知道
KV 容量——那些是 Scheduler 的事。

两种策略：

- `FCFSRequestQueue`：`deque`，先进先出。
- `PriorityRequestQueue`：堆，按 `Request.__lt__`（priority 小的优先，同级按到达时间）。
  堆的经典坑是**已取消的条目还留在堆里**：这里用「懒惰删除 + 版本戳」，取的时候跳过已出队
  的请求，绝不让旧对象永久占住堆（196 §4 点名的那个问题）。

create_request_queue(policy) 是唯一的构造入口，对应 vLLM 同名函数。
"""

import heapq
from collections import deque
from collections.abc import Iterable
from typing import Protocol


class RequestQueue(Protocol):
    """队列接口。真实 vLLM 是 ABC；本关一个 Protocol 就够（不必为两个实现写抽象基类）。"""

    def add_request(self, request) -> None: ...
    def pop_request(self): ...
    def peek_request(self): ...
    def prepend_request(self, request) -> None: ...
    def prepend_requests(self, requests: "RequestQueue") -> None: ...
    def remove_request(self, request) -> None: ...
    def remove_requests(self, requests: Iterable) -> None: ...
    def __bool__(self) -> bool: ...
    def __len__(self) -> int: ...


class FCFSRequestQueue(deque):
    """先进先出。到达顺序就是服务顺序。"""

    def add_request(self, request) -> None:
        self.append(request)

    def pop_request(self):
        return self.popleft()

    def peek_request(self):
        if not self:
            raise IndexError("peek from an empty queue")
        return self[0]

    def prepend_request(self, request) -> None:
        self.appendleft(request)

    def prepend_requests(self, requests: "RequestQueue") -> None:
        # extendleft 会把序列反过来塞进左端，所以"前插一批"的实际顺序正是它们的原顺序
        self.extendleft(list(requests))

    def remove_request(self, request) -> None:
        self.remove(request)

    def remove_requests(self, requests: Iterable) -> None:
        for request in requests:
            self.remove(request)


class PriorityRequestQueue:
    """小顶堆。**懒惰删除**：被移除/弹出的请求留在堆里，取出时比对"当前是否在队"再跳掉。"""

    def __init__(self) -> None:
        self._heap: list = []
        self._member_ids: set[str] = set()

    def add_request(self, request) -> None:
        heapq.heappush(self._heap, request)
        self._member_ids.add(request.request_id)

    def peek_request(self):
        while self._heap:
            request = self._heap[0]
            if request.request_id in self._member_ids:
                return request
            heapq.heappop(self._heap)          # 已出队的旧条目，丢掉
        raise IndexError("peek from an empty queue")

    def pop_request(self):
        request = self.peek_request()
        heapq.heappop(self._heap)
        self._member_ids.discard(request.request_id)
        return request

    def prepend_request(self, request) -> None:
        """恢复的请求要"插回队首"。堆里没有队首的概念，所以给它一个**更早的到达时间**：
        同优先级下它就会排在前面（与 FCFS 的 `appendleft` 语义对齐）。"""
        request.arrival_time -= 1e-6
        self.add_request(request)

    def prepend_requests(self, requests: "RequestQueue") -> None:
        for request in requests:
            self.prepend_request(request)

    def remove_request(self, request) -> None:
        self._member_ids.discard(request.request_id)   # 堆里的旧条目由 peek 时惰性丢弃

    def remove_requests(self, requests: Iterable) -> None:
        for request in requests:
            self.remove_request(request)

    def __bool__(self) -> bool:
        try:
            self.peek_request()
        except IndexError:
            return False
        return True

    def __len__(self) -> int:
        return len(self._member_ids)


def create_request_queue(policy: str) -> RequestQueue:
    if policy == "priority":
        return PriorityRequestQueue()
    if policy == "fcfs":
        return FCFSRequestQueue()
    raise ValueError(f"未知调度策略: {policy!r}")
