"""示例：**自定义提议者插件**（需求 62 §3.2）。

用途只有一个：**验证接口**——证明"候选来源换成用户自己的类"以后，Scheduler、KV 分配、
拒绝采样验证、停止判定全都不需要改。它**不是性能算法**（重复猜同一个 token 的接受率很低），
不要拿它当加速方案。

怎么被接入：
    SpeculativeConfig(method="custom_class", model="examples.custom_proposer.RepeatLastTokenProposer",
                      num_speculative_tokens=4)
  → config 期推断 `method="custom_class"`（或显式给）
  → Runner 构造期 `create_custom_proposer()`：import 这个模块、取出这个类、**用 VllmConfig 构造**
  → 每轮采样之后调用 `propose(sampled_token_ids, num_tokens_no_spec, token_ids_cpu,
                              slot_mappings=None)`，返回值 `list[list[int]]` 直接当草稿

插件拿到什么、拿不到什么（这是本关的安全边界）：
    拿到：`VllmConfig`（只读配置）＋ 调用时传入的三个 CPU 缓冲。
    拿不到：Scheduler 的 `Request`、`KVCacheManager`、`InputBatch` 本体、任何"改状态"的入口。
    所以插件无法绕过状态所有权（`Request` 只能由 Scheduler 改），也无法偷偷改 KV 或输出。

命令行用法（`minivllm/demo.py` 会把仓库根塞进 `sys.path`，所以点号路径写得出来）：
    python minivllm/demo.py --spec-method custom_class \\
        --spec-model examples.custom_proposer.RepeatLastTokenProposer --spec-k 4 "问题"
"""


class RepeatLastTokenProposer:
    """确定性提议者：把每一行"最近一个已提交的 token"重复 K 次。

    为什么拿它做示例：
      - **完全确定**（不含随机数、不依赖模型）：同一个输入永远给同一个候选，便于对照；
      - **候选基本都会被拒**：这样正好验证"错误候选不改变 greedy 最终答案"；
      - **长度可变**：贴着 `max_model_len` 的行会给更短的候选，甚至空列表；
      - **中间 prefill 块**（本轮没采样）必须跳过——`sampled_token_ids[row]` 是空列表时返回空。
    """

    def __init__(self, vllm_config):
        """只收 `VllmConfig`（上游约定：构造函数必须接受它）。"""
        spec_config = vllm_config.speculative_config
        assert spec_config is not None, "没有投机配置就不会构造自定义提议者"
        self.num_speculative_tokens = spec_config.num_speculative_tokens
        self.max_model_len = vllm_config.model_config.max_model_len

    def propose(self, sampled_token_ids, num_tokens_no_spec, token_ids_cpu, slot_mappings=None):
        """按上游 `custom_class` 分支的签名实现（**不要**加 `num_speculative_tokens` 参数）。

        `sampled_token_ids[row]` 是本轮第 row 行采到的 token（空 = 这行本轮没采样），它决定
        **要遍历几行**（也决定返回几行，必须一行不少）。

        `num_tokens_no_spec` / `token_ids_cpu` 是**定长缓冲**（长度分别是 `max_num_reqs` 与
        `max_num_reqs × max_model_len`），只有前 `len(sampled_token_ids)` 行是这一轮的；
        千万别 `for row in range(len(num_tokens_no_spec))`——那会读到上一轮留下的脏行。
        每行的历史是 `token_ids_cpu[row, :num_tokens_no_spec[row]]`。
        `slot_mappings` 本仓库暂时是 None（69 关 CUDA Graph 时才有）。
        """
        drafts: list[list[int]] = []
        for row, sampled in enumerate(sampled_token_ids):
            if not sampled:
                drafts.append([])          # 中间 prefill 块：没有已确认的输出可作依据
                continue
            num_tokens = int(num_tokens_no_spec[row])
            # 位置预算：猜出来的 token 下一轮要放进上下文，不能越过 max_model_len
            budget = min(self.num_speculative_tokens, self.max_model_len - num_tokens - 1)
            drafts.append([int(sampled[-1])] * max(budget, 0))
        return drafts
