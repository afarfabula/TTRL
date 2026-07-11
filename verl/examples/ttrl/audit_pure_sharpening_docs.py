#!/usr/bin/env python3
"""Documentation evidence audit for the active pure-sharpening goal."""

from __future__ import annotations

from pathlib import Path


ROOT = Path("/opt/tiger/TTRL")
HANDOFF = ROOT / "TTRL_SPS_HANDOFF.md"
CN_SUMMARY = ROOT / "TTRL_SPS_CN_SUMMARY.md"


def require(name: str, text: str, needle: str) -> None:
    if needle not in text:
        raise AssertionError(f"{name}: missing required snippet: {needle}")


def count_any(text: str, needles: list[str]) -> int:
    return sum(1 for needle in needles if needle in text)


def main() -> None:
    handoff = HANDOFF.read_text()
    cn = CN_SUMMARY.read_text()

    handoff_required = [
        "### 2026-07-07 literature refresh for the next iteration",
        "### Active goal completion audit snapshot: 2026-07-07",
        "Read and summarize at least 10 related papers/code directions",
        "v68 design: count-neutral process-preference sharpening",
        "v69 design: pairwise process preference on count-neutral targets",
        "Completion gate now requires no-majority-target audit",
        "val-core/MATH-TTT/acc/mean@4 >= 0.75",
        "MISSING_REQUIREMENT=no complete v68/v69 pure-sharpening run reaches target",
    ]
    for needle in handoff_required:
        require(str(HANDOFF), handoff, needle)

    cn_required = [
        "## 新 goal：纯锐化分布训练到 75%",
        "这轮快速读了 10 类相关工作/代码启发",
        "## v68/v69 no-majority-target 静态审计",
        "## completion gate 新增 no-majority-target 硬检查",
        "goal 未完成",
        "健康 worker 上完整跑 v69",
    ]
    for needle in cn_required:
        require(str(CN_SUMMARY), cn, needle)

    paper_markers = [
        "arXiv:2504.16084",
        "arXiv:2505.21444",
        "arXiv:2505.16022",
        "arXiv:2505.22660",
        "arXiv:2506.06395",
        "arXiv:2505.22617",
        "arXiv:2508.11016",
        "arXiv:2508.11356",
        "arXiv:2510.17472",
        "arXiv:2510.17923",
        "arXiv:2511.01191",
        "arXiv:2512.15146",
        "arXiv:2512.04359",
    ]
    paper_count = count_any(handoff, paper_markers)
    if paper_count < 10:
        raise AssertionError(f"literature evidence has only {paper_count} paper markers")

    for prior in ["v48", "v58", "v59", "v60", "v61", "v62", "v63", "v67"]:
        require("prior-observation evidence", handoff + cn, prior)

    print(f"PURE_SHARPENING_DOC_AUDIT_OK papers={paper_count}")


if __name__ == "__main__":
    main()
