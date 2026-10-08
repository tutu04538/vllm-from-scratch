"""动态投机解码（对应 vLLM `v1/spec_decode/dynamic/`）。

这一层只有**配置表**：把"批大小 → 本轮猜几枚"的闭区间三元组表，展开成按批大小直接索引的
稠密查找表。**它不含任何自适应策略**——不按最近的接受率调 K、不做在线搜索，K 完全由用户
给的表决定（需求 071 §1 明确要求）。
"""

from .utils import (DynamicSDSchedule, build_dynamic_sd_schedule_lookup,
                    validate_and_normalize_dynamic_sd_schedule)

__all__ = ["DynamicSDSchedule", "build_dynamic_sd_schedule_lookup",
           "validate_and_normalize_dynamic_sd_schedule"]
