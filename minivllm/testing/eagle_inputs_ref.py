"""**只给测试用**的 EAGLE 第一遍输入参考实现（63 关）。

生产路径**不** import 本模块：默认 EAGLE 通路现在在
`minivllm/spec_decode/eagle.py::EagleProposer.set_inputs_first_pass` 里**逐行照抄**上游
`llm_base_proposer.py` 的不扩容分支（整体左移 + 打补丁 + positions/特征逐行原样）。
这里留两份参考实现，用途各不相同：

- `eagle_first_pass_input_ids()`：上游那四行的**纯函数版**——需求 063 §3 的两请求例子、
  以及"补丁下标错一位会把下一条请求的 token 留给自己"的反证，都靠它当可对照的期望值来源；
- `expand_eagle_inputs_shifted()`：上游内核 `copy_and_expand_eagle_inputs_kernel`
  （`shift_input_ids=True`）的等价展开，对应**扩容分支**（`needs_extra_input_slots=True`，
  并行提议 / DFlash / PARD 那类）。默认 EAGLE 走不到它，但 72 关（P-EAGLE 并行提议）接入时
  要用，而且它现在就有与真内核的逐值差分（`tests/step63/test_eagle_inputs.py`）。

两个函数都**不**属于生产路径——放这里是为了让 `minivllm/spec_decode/` 只留被调用的代码。

"""

from ..spec_decode.utils import PADDING_TOKEN_ID, DraftInputRows



def eagle_first_pass_input_ids(target_token_ids: list[int], next_token_ids: list[int],
                               query_start_loc: list[int]) -> tuple[list[int], list[int]]:
    """上游 `SpecDecodeBaseProposer.set_inputs_first_pass()` 的**默认 EAGLE 通路**（需求 063 §3）。

    上游注释里的三步（`llm_base_proposer.py:846-872`）逐条照抄：

        # 1) 整体左移一格
        E.g., [a1, b1, b2, c1, c2, c3] -> [b1, b2, c1, c2, c3, c3]
        # 2) 每条请求的**最后一格**换成这条请求的新 token
        E.g., [b1, b2, c1, c2, c3, c3] -> [a2, b2, b3, c2, c3, c4]

    为什么必须"整体左移 + 按请求打补丁"，而不是逐请求自己左移：两者只在**每请求最后一格**不同，
    而那一格正是要被 `next_token_ids` 覆盖的位置（`token_indices_to_sample = query_start_loc[1:] - 1`）。
    换句话说"最后一格留着谁"完全由这个下标决定——算成 `query_start_loc[1:]`（下一条请求的第一格）
    就会把 A 的新 token 写到 B 头上，而 A 的最后一格留着 B 的 token（需求原文点名的坑）。

    参数与返回：
        `target_token_ids`  本轮 target 真正喂进去的行（扁平、按请求连续；含被拒草稿那些行）
        `next_token_ids`    每条请求本轮新采出的 token（按请求序）
        `query_start_loc`   `[0, n_A, n_A+n_B, ...]` 长度 = 请求数 + 1
        返回 `(input_ids, token_indices_to_sample)`——**特征（hidden states）与 positions 都不动**，
        所以不在这里返回：配对关系是"第 i 行用第 i 个位置/特征去预测第 i+1 个 token"。
    """
    num_tokens = len(target_token_ids)
    if len(query_start_loc) < 2:
        raise ValueError(f"query_start_loc 至少要有两个元素（起点 + 一条请求的末尾），"
                         f"收到 {query_start_loc!r}")
    if query_start_loc[-1] != num_tokens:
        raise ValueError(f"query_start_loc 的末位 {query_start_loc[-1]} 必须等于 token 行数 "
                         f"{num_tokens}（它描述的就是这轮 target 的物理行）")

    input_ids = list(target_token_ids)
    if num_tokens > 1:
        # 第 1 步：整体左移。最后一格暂时是"下一条请求的第一个 token"（或脏值），马上会被覆盖
        input_ids[:num_tokens - 1] = target_token_ids[1:]

    # 第 2 步：每请求最后一格 = 这条请求的新 token
    token_indices_to_sample = [query_start_loc[i + 1] - 1
                               for i in range(len(query_start_loc) - 1)]
    if len(next_token_ids) != len(token_indices_to_sample):
        raise ValueError(f"next_token_ids 有 {len(next_token_ids)} 个，但 query_start_loc 描述"
                         f" {len(token_indices_to_sample)} 条请求：两者必须一条一个")
    for index, token in zip(token_indices_to_sample, next_token_ids):
        input_ids[index] = token
    return input_ids, token_indices_to_sample


def expand_eagle_inputs_shifted(rows: list[DraftInputRows]) -> tuple[list[int], list[int],
                                                                    list[int], list[int]]:
    """上游 `copy_and_expand_eagle_inputs_kernel`（`shift_input_ids=True`）的等价展开。

    与 `expand_draft_inputs`（`False` 分支）的差别只有三处，对应内核里的同一个 if
    （`v1/spec_decode/utils.py:345-365`）：

        shift_input_ids=True:  num_valid_tokens = query_end - query_start        # 比 False 少 1
                               input_offset     = 1                              # 跳过第一个 token
                               output_start     = query_start + i*(slots - 1)    # 每请求少占 1 行

    即：**被拷进工作区的有效 token 少一个**（第一个 token 被跳掉），少的这一格由"扩容行"
    （`next_token_id`）补回来，所以每请求的总行数仍是 `有效 + 1 + 被拒`。

    **positions 不跟着移**（内核注释："Positions are NOT shifted"）：`positions[j] = start_pos + j`，
    也就是与 target 逐行相同；扩容行的位置就是这条请求最后一行的位置。hidden states 同理——
    内核里 `out_hidden_state_mapping[query_start + j] = output_start + j`（把该请求**包括被拒行**
    在内的所有 target 行按行号平移到工作区），所以扩容行拿到的是**这条请求最后一行的特征**，
    正好构成 EAGLE 的 `(h_last, next_token) → 下一个 token` 配对。

    用途：EAGLE 的**扩容分支**（并行提议 / DFlash / PARD 那类需要额外槽位的组合会走内核；
    默认 EAGLE 通路不需要额外槽位，走 `eagle_first_pass_input_ids`）。本关把它实现出来是为了
    能和上游内核逐值差分——58 关已经差分过 `False` 分支。
    """
    input_ids: list[int] = []
    positions: list[int] = []
    is_rejected: list[int] = []
    token_indices_to_sample: list[int] = []
    for row in rows:
        base = len(input_ids)
        # shift=True：跳掉第一个 token（它由上一行的"下一格"提供）
        shifted = row.valid_token_ids[1:]
        num_valid = len(shifted)
        input_ids.extend(shifted)
        # positions 不跟着移：第 j 行的位置仍是 start + j
        positions.extend(range(row.start, row.start + num_valid))
        is_rejected.extend([0] * num_valid)
        input_ids.append(row.next_token_id)
        # 扩容行的位置 = 这条请求最后一行的位置（= start + 原有效行数 - 1）
        positions.append(row.start + num_valid)
        is_rejected.append(0)
        token_indices_to_sample.append(base + num_valid)
        input_ids.extend([PADDING_TOKEN_ID] * row.num_rejected)
        positions.extend([0] * row.num_rejected)
        is_rejected.extend([1] * row.num_rejected)
    return input_ids, positions, is_rejected, token_indices_to_sample


