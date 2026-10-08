"""动态投机长度（Dynamic SD）的配置校验与查找表（对应 vLLM
`v1/spec_decode/dynamic/utils.py`，L7-L148 逐行对齐）。

**为什么需要它**：低并发时"一次猜 4 枚"很划算，高并发时"猜 1 枚、甚至不猜"更快——因为投机
唯一的收益是"一次权重读取喂多条 token"，并发越高、这一步本来就被摊薄了（需求 071 §1）。
于是配置里给一张**闭区间表**：

    [(1, 2, 4), (5, 8, 1)]      读作：批大小 1~2 猜 4 枚；5~8 猜 1 枚

两个函数分工：

    validate_and_normalize_dynamic_sd_schedule()   校验 + 排序（配置期一次）
    build_dynamic_sd_schedule_lookup()             展开成"批大小 → K"的稠密表（索引 0 不用）

**它不是自适应策略**：不看接受率、不做在线搜索、也不改历史候选的解释——K 只由这张表决定，
而"本轮 K"与"本轮验证的旧候选"是两件事（见 `Scheduler.schedule()` 里的两轮时序）。

本仓库的差异只有一条、且是照抄上游的口径：`num_speculative_tokens_per_batch_size` 允许
`tuple` 或 `list`（上游 `isinstance(entry, list | tuple)`），表里的 K 一律 `int()` 转换
（"3" 这样的字符串会被转换成 3，而不是报错——这是上游的行为，本仓库不改）。
"""

DynamicSDSchedule = list[tuple[int, int, int]]


def validate_and_normalize_dynamic_sd_schedule(
    num_speculative_tokens_per_batch_size: object,
) -> DynamicSDSchedule:
    """校验并归一化"批大小 → 投机长度"的区间表（上游同名函数，L7-L74）。

    规则逐条照抄：
      - 必须是**非空 list**（None / 其它类型都报错）；
      - 每一项是 3 元组/列表 `(range_start, range_end, K)`，逐个 `int()` 转换；
      - 区间端点必须为正、起点 ≤ 终点、K ≥ 0；
      - 按起点排序后，区间**不许重叠**（`start <= previous_end` 就报错：区间是闭区间，
        上一段到 8、下一段从 8 开始就算重叠）；
      - 首段必须从 1 开始（否则 1 到首段起点之间的批大小没有定义）。

    空隙与尾部**不在这里**处理：那是 `build_dynamic_sd_schedule_lookup()` 的事（沿用前段 K）。
    """
    if num_speculative_tokens_per_batch_size is None:
        raise ValueError(
            "num_speculative_tokens_per_batch_size is required for "
            "dynamic speculative decoding."
        )
    if not isinstance(num_speculative_tokens_per_batch_size, list):
        raise ValueError(
            "num_speculative_tokens_per_batch_size must be a non-empty list of "
            "(range_start, range_end, num_speculative_tokens) entries."
        )
    if not num_speculative_tokens_per_batch_size:
        raise ValueError("num_speculative_tokens_per_batch_size must not be empty.")

    parsed_schedule: DynamicSDSchedule = []
    for entry in num_speculative_tokens_per_batch_size:
        if not isinstance(entry, list | tuple) or len(entry) != 3:
            raise ValueError(
                "Each num_speculative_tokens_per_batch_size entry must be a "
                "3-item sequence: (range_start, range_end, num_speculative_tokens)."
            )

        range_start, range_end, num_speculative_tokens = (
            int(entry[0]),
            int(entry[1]),
            int(entry[2]),
        )

        if range_start <= 0 or range_end <= 0:
            raise ValueError(
                f"Batch-size range ({range_start}, {range_end}) must be positive."
            )
        if range_start > range_end:
            raise ValueError(
                "Batch-size range start must be <= end for "
                f"({range_start}, {range_end}, {num_speculative_tokens})."
            )
        if num_speculative_tokens < 0:
            raise ValueError(
                "num_speculative_tokens_per_batch_size values must be >= 0."
            )

        parsed_schedule.append((range_start, range_end, num_speculative_tokens))

    parsed_schedule.sort(key=lambda entry: entry[0])

    previous_end = 0
    for range_start, range_end, _ in parsed_schedule:
        if range_start <= previous_end:
            raise ValueError("Batch-size ranges must be non-overlapping and sorted.")
        previous_end = range_end

    first_range_start = parsed_schedule[0][0]
    if first_range_start != 1:
        raise ValueError(
            "The first batch-size range must start at 1 so every runtime "
            "batch size has a defined schedule."
        )

    return parsed_schedule


def build_dynamic_sd_schedule_lookup(
    num_speculative_tokens_per_batch_size: object,
    vllm_max_batch_size: int,
    vllm_num_speculative_tokens: int,
) -> list[int]:
    """把区间表展开成稠密查找表 `dense_schedule[batch_size] = K`（上游同名函数，L77-L148）。

    "稠密"的意思是：调度器每轮只做一次**数组下标访问**，不去搜区间。

        dense_schedule[0]  **故意不用**（批大小从 1 开始，直接下标查）
        首段之前的空隙     —— 不可能有：首段必须从 1 开始
        段与段之间的空隙   —— 沿用**前一段**的 K（例：`[(1,16,3),(32,128,2)]` 的 17~31 是 3）
        最后一段之后到上界 —— 沿用**最后一段**的 K（例：B>8 延续 K=1）
        每一项都按**全局最大值** `vllm_num_speculative_tokens` 裁剪：表里写了 5、而配置的
        最大 K 是 4 时，实际用 4（工作区/图都是按最大 K 建的，表不能超出这个容量）

    `vllm_max_batch_size` 取 `scheduler_config.max_num_seqs`（上游同款）：批大小不可能超过
    并发上限，表比它长的部分不展开。

    ⚠️ 用的是"**实际被调度位置**"的个数（`len(num_scheduled_tokens)`），不是 waiting+running
    的总数——需求 071 §3.2 点名了这一条：没被排上的请求这一轮不产生草稿，也不该影响 K。
    """
    if vllm_max_batch_size <= 0:
        raise ValueError("vllm_max_batch_size must be > 0.")
    if vllm_num_speculative_tokens <= 0:
        raise ValueError("vllm_num_speculative_tokens must be > 0.")

    parsed_schedule = validate_and_normalize_dynamic_sd_schedule(
        num_speculative_tokens_per_batch_size
    )

    # 索引 0 故意留空：合法的批大小可以直接用 dense_schedule[batch_size] 查
    dense_schedule = [0] * (vllm_max_batch_size + 1)
    next_batch_size = 1
    last_num_speculative_tokens: int | None = None

    for range_start, range_end, num_speculative_tokens in parsed_schedule:
        if range_start > next_batch_size and last_num_speculative_tokens is not None:
            # 段间空隙：沿用前一段的 K。例：[(1,16,3),(32,128,2)] 把 17~31 填成 3。
            for batch_size in range(
                next_batch_size,
                min(range_start, vllm_max_batch_size + 1),
            ):
                dense_schedule[batch_size] = min(
                    vllm_num_speculative_tokens,
                    last_num_speculative_tokens,
                )

        # 本段（闭区间）填自己的 K
        for batch_size in range(
            max(range_start, next_batch_size),
            min(range_end, vllm_max_batch_size) + 1,
        ):
            dense_schedule[batch_size] = min(
                vllm_num_speculative_tokens,
                num_speculative_tokens,
            )

        next_batch_size = max(next_batch_size, range_end + 1)
        last_num_speculative_tokens = num_speculative_tokens

        if next_batch_size > vllm_max_batch_size:
            break

    if last_num_speculative_tokens is None:
        raise ValueError(
            "num_speculative_tokens_per_batch_size must contain at least "
            "one valid batch-size range."
        )

    # 最后一段之后到上界：沿用最后一段的 K
    for batch_size in range(next_batch_size, vllm_max_batch_size + 1):
        dense_schedule[batch_size] = min(
            vllm_num_speculative_tokens,
            last_num_speculative_tokens,
        )

    return dense_schedule
