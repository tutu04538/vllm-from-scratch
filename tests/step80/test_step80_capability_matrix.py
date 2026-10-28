"""80H：能力矩阵/收口记录的一致性测试（文档/测试收口，不做模型测试）。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / "docs/step80_model_matrix.md"
ALIGNMENT = ROOT / "docs/step80_alignment.md"
RESULTS = ROOT / "docs/results.json"
README = ROOT / "docs/README.md"

COLUMNS = [
    "模型/配置", "实际工厂分支", "模块结构是否对齐", "输入/KV协议",
    "tiny差分", "真实checkpoint", "后端/group前置", "下一任务",
]
REQUIRED_ROWS = [
    "Qwen3 MTP tiny",
    "Gemma4 assistant tiny",
    "Gemma4 target + assistant 真实",
    "Step3p5 MTP V1",
    "Step3p5 MTP V2",
    "DeepSeek MTP 稀疏索引契约",
    "DeepSeek V3.2/V4.1 真实",
]


def _matrix_rows() -> tuple[list[str], list[list[str]]]:
    lines = MATRIX.read_text().splitlines()
    start = next(i for i, line in enumerate(lines)
                 if line.strip().startswith("| 模型/配置 |"))
    header = [cell.strip() for cell in lines[start].strip().strip("|").split("|")]
    rows = []
    for line in lines[start + 2:]:
        if not line.strip().startswith("|"):
            break
        rows.append([cell.strip() for cell in line.strip().strip("|").split("|")])
    return header, rows


def test_matrix_has_fixed_columns_and_required_rows():
    header, rows = _matrix_rows()
    assert header == COLUMNS
    assert len(rows) == len(REQUIRED_ROWS)
    for row, label in zip(rows, REQUIRED_ROWS):
        assert row[0].startswith(label), (label, row[0])
        assert len(row) == len(COLUMNS), row
        assert all(cell for cell in row), row
        assert row[7] not in ("无", "-"), f"{label} 缺下一任务"


def test_matrix_does_not_turn_tiny_or_contract_into_real_support():
    _, rows = _matrix_rows()
    by_label = {label: next(row for row in rows if row[0].startswith(label))
                for label in REQUIRED_ROWS}
    # 真实权重的三种状态必须分开写，不能都写成“已支持”。
    assert "未下载" in by_label["Qwen3 MTP tiny"][5]
    step3p5 = by_label["Step3p5 MTP V1"][5]
    assert "已下载" in step3p5 and "196B target 未下载" in step3p5
    deepseek_contract = by_label["DeepSeek MTP 稀疏索引契约"][5]
    assert "未接" in deepseek_contract
    deepseek_real = by_label["DeepSeek V3.2/V4.1 真实"]
    assert deepseek_real[2].startswith("未实现")
    assert "未下载" in deepseek_real[5]
    # Step3p5 V2 不能把“真实 MTP 子集”写成“真实 target 已跑”。
    assert "未跑" in by_label["Step3p5 MTP V2"][5]


def test_results_cover_all_subrequirements_and_package_hash():
    data = json.loads(RESULTS.read_text())
    subreqs = data["step80"]["subrequirements"]
    expected = [f"80{c}" for c in "ABCDEFGH"]
    assert set(expected) <= set(subreqs)
    entry = subreqs["80H"]
    assert "完成" in entry["status"]
    assert entry["results"]["commands"]
    package = entry["results"]["package"]
    root = ROOT / "minivllm"
    names = sorted(root.rglob("*.py"), key=lambda p: str(p.relative_to(root)))
    digest = hashlib.sha256("".join(
        f"{p.relative_to(root)}\0{hashlib.sha256(p.read_bytes()).hexdigest()}\n"
        for p in names).encode()).hexdigest()
    assert package["sha256"] == digest
    assert package["python_files"] == len(names)


def test_alignment_and_index_are_current():
    text = ALIGNMENT.read_text()
    assert "80F–80H 未开始" not in text
    assert "## 16. 80H" in text
    assert "step80_model_matrix.md" in text
    readme = README.read_text()
    assert "step80_model_matrix.md" in readme
