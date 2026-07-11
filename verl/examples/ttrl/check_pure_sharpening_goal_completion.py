#!/usr/bin/env python3
"""Completion audit for the active pure-sharpening 50-step goal."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path("/opt/tiger/TTRL/verl")
SUMMARY = ROOT / "pure_sharpening_ablation_summary.tsv"
SUMMARY_HELPER = ROOT / "examples/ttrl/summarize_pure_sharpening_ablation.py"
VERIFY = ROOT / "examples/ttrl/verify_pure_sharpening_strict_configs.py"
NO_MAJORITY_AUDIT = ROOT / "examples/ttrl/audit_v68_v69_no_majority_target.py"
DOC_AUDIT = ROOT / "examples/ttrl/audit_pure_sharpening_docs.py"
POST_AUDIT = ROOT / "examples/ttrl/audit_pure_sharpening_result.py"
PYTHON = Path("/opt/tiger/modelchef/.venv/bin/python3")
TARGET = 0.75
PURE_CANDIDATES = {
    "v68_count_neutral",
    "v69_pairwise_count_neutral",
    "v75_support_gated",
    "v76_support_mixture",
    "v77_support_calibrated",
    "v78_support_residual",
    "v79_split_support",
    "v80_capacity_braked",
    "v81_basin_contrast",
}


def run(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return proc.returncode, proc.stdout.strip()


def refresh_summary() -> tuple[int, str]:
    code, output = run([str(PYTHON), str(SUMMARY_HELPER)])
    if code == 0:
        SUMMARY.write_text(output + "\n")
    return code, output


def read_rows() -> list[dict[str, str]]:
    if not SUMMARY.is_file() or SUMMARY.stat().st_size == 0:
        raise AssertionError(f"missing summary TSV: {SUMMARY}")
    with SUMMARY.open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def parse_float(value: str) -> float | None:
    if value in {"", "NA"}:
        return None
    return float(value)


def checklist_item(name: str, status: bool, evidence: str) -> None:
    print(f"CHECKLIST {name}={'PASS' if status else 'FAIL'} evidence={evidence}")


def main() -> None:
    print("OBJECTIVE=Qwen2.5-Math-7B strict 50-step pure sharpening mean@4 >= 0.75")
    print(f"TARGET={TARGET}")
    checklist_item(
        "objective_restatement",
        True,
        "Qwen2.5-Math-7B; 50 steps; val n=4; no validation selection; mean@4>=0.75",
    )

    code, output = refresh_summary()
    print(f"SUMMARY_REFRESH_EXIT={code}")
    print(f"SUMMARY_PATH={SUMMARY}")
    checklist_item("summary_refresh", code == 0 and SUMMARY.is_file(), str(SUMMARY))
    if code != 0:
        print(output)
        print("RESULT=FAIL")
        raise SystemExit(1)

    code, output = run([str(PYTHON), str(VERIFY)])
    print(f"STRICT_CONFIG_AUDIT_EXIT={code}")
    print(f"STRICT_CONFIG_AUDIT_OUTPUT={output}")
    checklist_item("strict_config_audit", code == 0, output)
    if code != 0:
        print("RESULT=FAIL")
        raise SystemExit(1)

    code, output = run([str(PYTHON), str(NO_MAJORITY_AUDIT)])
    print(f"NO_MAJORITY_TARGET_AUDIT_EXIT={code}")
    print(f"NO_MAJORITY_TARGET_AUDIT_OUTPUT={output}")
    checklist_item(
        "no_majority_target_audit",
        code == 0 and "PURE_SHARPENING_NO_MAJORITY_TARGET_AUDIT_OK" in output,
        output,
    )
    if code != 0 or "PURE_SHARPENING_NO_MAJORITY_TARGET_AUDIT_OK" not in output:
        print("MISSING_REQUIREMENT=no-majority-target audit did not pass")
        print("RESULT=FAIL")
        raise SystemExit(1)

    code, output = run([str(PYTHON), str(DOC_AUDIT)])
    print(f"DOC_AUDIT_EXIT={code}")
    print(f"DOC_AUDIT_OUTPUT={output}")
    checklist_item(
        "doc_audit",
        code == 0 and "PURE_SHARPENING_DOC_AUDIT_OK" in output,
        output,
    )
    if code != 0 or "PURE_SHARPENING_DOC_AUDIT_OK" not in output:
        print("MISSING_REQUIREMENT=documentation evidence audit did not pass")
        print("RESULT=FAIL")
        raise SystemExit(1)

    rows = read_rows()
    checklist_item("ablation_rows_present", len(rows) > 0, f"rows={len(rows)}")
    completed_candidates: list[tuple[str, float]] = []
    candidate_labels_seen = set()
    for row in rows:
        label = row["label"]
        if label not in PURE_CANDIDATES:
            continue
        candidate_labels_seen.add(label)
        value = parse_float(row["strict_mean4"])
        status = row["artifact_status"]
        print(f"CANDIDATE={label} ARTIFACT_STATUS={status} STRICT_MEAN@4={value if value is not None else 'NA'}")
        if status == "COMPLETE_METRICS" and value is not None:
            completed_candidates.append((label, value))
    checklist_item(
        "pure_candidates_listed",
        candidate_labels_seen == PURE_CANDIDATES,
        f"seen={','.join(sorted(candidate_labels_seen))}",
    )

    best_label = "NA"
    best_value: float | None = None
    if completed_candidates:
        best_label, best_value = max(completed_candidates, key=lambda item: item[1])
    print(f"BEST_PURE_CANDIDATE={best_label}")
    print(f"BEST_PURE_MEAN@4={best_value if best_value is not None else 'NA'}")
    checklist_item(
        "target_metric_reached",
        best_value is not None and best_value >= TARGET,
        f"best={best_value if best_value is not None else 'NA'} target={TARGET}",
    )

    if best_value is None or best_value < TARGET:
        print("MISSING_REQUIREMENT=no complete pure-sharpening run reaches target")
        print("RESULT=FAIL")
        raise SystemExit(2)

    # A passing TSV row is not enough; require the stricter post-run audit too.
    version_by_label = {
        "v68_count_neutral": "v68",
        "v69_pairwise_count_neutral": "v69",
        "v75_support_gated": "v75",
        "v76_support_mixture": "v76",
        "v77_support_calibrated": "v77",
        "v78_support_residual": "v78",
        "v79_split_support": "v79",
        "v80_capacity_braked": "v80",
        "v81_basin_contrast": "v81",
    }
    version = version_by_label[best_label]
    code, output = run([str(PYTHON), str(POST_AUDIT), version])
    print(f"POST_RUN_AUDIT_VERSION={version}")
    print(f"POST_RUN_AUDIT_EXIT={code}")
    print(output)
    checklist_item("post_run_audit_pass", code == 0 and "RESULT=PASS" in output, f"version={version}")
    if code != 0 or "RESULT=PASS" not in output:
        print("MISSING_REQUIREMENT=post-run audit did not pass")
        print("RESULT=FAIL")
        raise SystemExit(3)

    print("RESULT=PASS")


if __name__ == "__main__":
    main()
