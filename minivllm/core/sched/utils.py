"""停止判定（对应 vLLM `v1/core/sched/utils.py::check_stop`）。

**逐 token 判定**，判完立刻改状态——调用方（`Scheduler._update_request_with_output`）拿到
True 就截断本轮剩余候选。顺序与 vLLM 一致，因为顺序本身有语义：

1. 先生成数不够 `min_tokens` → 不检查任何停止条件（强制生成够数）；
2. 最后一个是 eos → FINISHED_STOPPED（`ignore_eos` 时跳过这一条）；
3. 最后一个是 stop token → FINISHED_STOPPED（额外记 `stop_reason`）；
4. 上下文到顶或生成数够 `max_tokens` → FINISHED_LENGTH_CAPPED。

vLLM 里还有 repetition detection 与字符串 stop 的匹配，本关不做（字符串匹配属于 tokenizer/
解码层，57B 以后再说）。
"""

from ...request import RequestStatus


def check_stop(request, max_model_len: int) -> bool:
    """返回 True 表示这条请求到此结束（并已经把 `request.status` 设好）。"""
    sampling_params = request.sampling_params

    # 1) min_tokens 没满足：什么都不查
    if request.num_output_tokens < sampling_params.min_tokens:
        return False

    last_token_id = request.output_token_ids[-1]

    # 2) eos
    if (not sampling_params.ignore_eos
            and sampling_params.eos_token_id is not None
            and last_token_id == sampling_params.eos_token_id):
        request.status = RequestStatus.FINISHED_STOPPED
        return True

    # 3) 显式 stop token
    if last_token_id in (sampling_params.stop_token_ids or ()):
        request.status = RequestStatus.FINISHED_STOPPED
        request.stop_reason = last_token_id
        return True

    # 4) 长度：上下文到顶，或生成数够 max_tokens
    if (request.num_tokens >= max_model_len
            or request.num_output_tokens >= request.max_tokens):
        request.status = RequestStatus.FINISHED_LENGTH_CAPPED
        return True

    return False
