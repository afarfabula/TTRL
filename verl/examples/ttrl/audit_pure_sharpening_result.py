#!/usr/bin/env python3
"""Post-run audit for strict pure-distribution sharpening experiments."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


ROOT = Path("/opt/tiger/TTRL/verl")
EXPS = {
    "v68": "sps_count_neutral_process_sharpened_prob_qwen25_math_7b_50step_v68_strict_n4",
    "v69": "sps_pairwise_process_count_neutral_sharpened_prob_qwen25_math_7b_50step_v69_strict_n4",
    "v67": "sps_band_limited_process_sharpened_prob_qwen25_math_7b_50step_v67_strict_n4",
    "v75": "sps_support_gated_process_sharpened_prob_qwen25_math_7b_50step_v75_strict_n4",
    "v76": "sps_support_mixture_process_sharpened_prob_qwen25_math_7b_50step_v76_strict_n4",
    "v77": "sps_support_calibrated_process_sharpened_prob_qwen25_math_7b_50step_v77_strict_n4",
    "v78": "sps_support_residual_process_sharpened_prob_qwen25_math_7b_50step_v78_strict_n4",
    "v79": "sps_split_support_process_sharpened_prob_qwen25_math_7b_50step_v79_strict_n4",
    "v80": "sps_capacity_braked_process_sharpened_prob_qwen25_math_7b_50step_v80_strict_n4",
    "v81": "sps_basin_contrast_process_sharpened_prob_qwen25_math_7b_50step_v81_strict_n4",
}
TARGET = 0.75
DIAGNOSTIC_METRICS = [
    ("TARGET_CONFIDENCE", "train/sps/direct_target_confidence"),
    ("TARGET_ENTROPY", "train/sps/direct_target_entropy"),
    ("TARGET_EFFECTIVE_K", "train/sps/direct_target_effective_K"),
    ("UNIQUE_ANSWER_COUNT", "train/sps/direct_unique_answer_count"),
    ("TARGET_MAJORITY_MASS", "train/sps/direct_majority_target_mass"),
    ("TARGET_MAJORITY_AGREEMENT", "train/sps/direct_majority_agreement"),
    ("BASE_TOP_CONFIDENCE", "train/sps/direct_base_top_confidence"),
    ("BASE_AGREEMENT", "train/sps/direct_base_agreement"),
    ("LOW_BUDGET_PARSEABLE_RATE", "train/sps/direct_low_budget_parseable_rate"),
    ("LOW_BUDGET_CLIP_RATE", "train/sps/direct_low_budget_clip_rate"),
    ("LOW_BUDGET_TOP_MASS", "train/sps/direct_low_budget_top_mass"),
    ("PROCESS_CONSISTENT_RATE", "train/sps/direct_process_consistent_rate"),
    ("PROCESS_TOP_SUPPORT", "train/sps/direct_process_top_support"),
    ("COUNT_NEUTRAL", "train/sps/direct_count_neutral_aggregation"),
    ("PAIRWISE_TOP_PREF", "train/sps/direct_pairwise_process_top_preference"),
    ("PAIRWISE_PREF_STD", "train/sps/direct_pairwise_process_preference_std"),
    ("SUPPORT_GATE_STRENGTH", "train/sps/direct_support_gate_strength_mean"),
    ("SUPPORT_GATE_TOP_CONFIDENCE", "train/sps/direct_support_gate_top_confidence"),
    ("SUPPORT_GATE_TARGET_OVERLAP", "train/sps/direct_support_gate_target_overlap"),
    ("SUPPORT_GATE_TARGET_AGREEMENT", "train/sps/direct_support_gate_target_agreement"),
    ("SUPPORT_GATE_LOW_BUDGET_ENTROPY", "train/sps/direct_support_gate_low_budget_entropy"),
    ("SUPPORT_GATE_PROCESS_ENTROPY", "train/sps/direct_support_gate_process_entropy"),
    ("SUPPORT_MIXTURE_ALPHA", "train/sps/direct_support_mixture_alpha"),
    ("SUPPORT_MIXTURE_PRE_CONFIDENCE", "train/sps/direct_support_mixture_pre_confidence"),
    ("SUPPORT_MIXTURE_PRE_EFFECTIVE_K", "train/sps/direct_support_mixture_pre_effective_K"),
    ("SUPPORT_MIXTURE_PRE_OVERLAP", "train/sps/direct_support_mixture_pre_overlap"),
    ("SUPPORT_CAP_ALPHA", "train/sps/direct_support_confidence_cap_alpha"),
    ("SUPPORT_CAP_VALUE", "train/sps/direct_support_confidence_cap_value"),
    ("SUPPORT_CAP_PRE_CONFIDENCE", "train/sps/direct_support_confidence_cap_pre_confidence"),
    ("SUPPORT_RESIDUAL_STRENGTH", "train/sps/direct_support_residual_strength_mean"),
    ("SUPPORT_RESIDUAL_MEAN", "train/sps/direct_support_residual_mean"),
    ("SUPPORT_RESIDUAL_STD", "train/sps/direct_support_residual_std"),
    ("SUPPORT_RESIDUAL_TOP_VALUE", "train/sps/direct_support_residual_top_value"),
    ("SPLIT_SUPPORT_STRENGTH", "train/sps/direct_split_support_strength_mean"),
    ("SPLIT_SUPPORT_TOP_CONFIDENCE", "train/sps/direct_split_support_top_confidence"),
    ("SPLIT_SUPPORT_TARGET_OVERLAP", "train/sps/direct_split_support_target_overlap"),
    ("SPLIT_SUPPORT_TARGET_AGREEMENT", "train/sps/direct_split_support_target_agreement"),
    ("SPLIT_SUPPORT_VIEW_OVERLAP", "train/sps/direct_split_support_view_overlap"),
    ("CAPACITY_BRAKE_ALPHA", "train/sps/direct_capacity_brake_alpha"),
    ("CAPACITY_BRAKE_PRE_CONFIDENCE", "train/sps/direct_capacity_brake_pre_confidence"),
    ("CAPACITY_BRAKE_PRE_EFFECTIVE_K", "train/sps/direct_capacity_brake_pre_effective_K"),
    ("CAPACITY_BRAKE_PRE_OVERLAP", "train/sps/direct_capacity_brake_pre_overlap"),
    ("CAPACITY_BRAKE_POST_EFFECTIVE_K", "train/sps/direct_capacity_brake_post_effective_K"),
    ("BASIN_CONTRAST_ALPHA", "train/sps/direct_basin_contrast_alpha"),
    ("BASIN_CONTRAST_TARGET_MARGIN", "train/sps/direct_basin_contrast_target_margin"),
    ("BASIN_CONTRAST_SUPPORT_MARGIN", "train/sps/direct_basin_contrast_support_margin"),
    ("BASIN_CONTRAST_BASE_MARGIN", "train/sps/direct_basin_contrast_base_margin"),
    ("BASIN_CONTRAST_LOW_BUDGET_MARGIN", "train/sps/direct_basin_contrast_low_budget_margin"),
    ("BASIN_CONTRAST_PROCESS_MARGIN", "train/sps/direct_basin_contrast_process_margin"),
    ("BASIN_CONTRAST_POST_EFFECTIVE_K", "train/sps/direct_basin_contrast_post_effective_K"),
    ("SPS_PICK_ACCURACY", "train/sps_pick_accuracy"),
    ("SPS_CORRECT_WEIGHT_MASS", "train/sps_correct_weight_mass"),
    ("PASS_AT_32", "train/pass@32"),
    ("VAL_RESPONSE_CLIP_MEAN4", "val-aux/MATH-TTT/response_clip/mean@4"),
    ("VAL_FORMAT_SCORE_MEAN4", "val-aux/MATH-TTT/format_score/mean@4"),
    ("VAL_RESPONSE_AVG_LOGPROB_MEAN4", "val-aux/MATH-TTT/response_avg_logprob/mean@4"),
    ("TRAIN_RESPONSE_CLIP_RATIO", "response_length/clip_ratio"),
    ("PERF_TIME_PER_STEP", "perf/time_per_step"),
    ("PERF_THROUGHPUT", "perf/throughput"),
]


def read(path: Path) -> str:
    return path.read_text(errors="ignore")


def find_last_float(text: str, metric: str) -> float | None:
    float_re = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"

    # Prefer the final validation metrics dict because step-line logging rounds
    # values to three decimals.
    dict_pattern = re.compile(
        rf"['\"]{re.escape(metric)}['\"]\s*:\s*({float_re})"
    )
    dict_matches = dict_pattern.findall(text)
    if dict_matches:
        return float(dict_matches[-1])

    step_pattern = re.compile(re.escape(metric) + rf"[:=]({float_re})")
    step_matches = step_pattern.findall(text)
    return float(step_matches[-1]) if step_matches else None


def require_file(path: Path) -> str:
    if not path.is_file() or path.stat().st_size == 0:
        raise AssertionError(f"missing or empty artifact: {path}")
    return read(path)


def effective_cli_value(text: str, key: str) -> str | None:
    pattern = re.compile(rf"(?:^|\s)(?:[+~])?{re.escape(key)}=('[^']*'|\"[^\"]*\"|[^\s]+)")
    matches = pattern.findall(text)
    if not matches:
        return None
    value = matches[-1]
    if (value.startswith("'") and value.endswith("'")) or (value.startswith('"') and value.endswith('"')):
        value = value[1:-1]
    return value


def require_effective_config(text: str, key: str, expected: str) -> None:
    actual = effective_cli_value(text, key)
    if actual != expected:
        raise AssertionError(f"effective config mismatch: {key} expected={expected!r} actual={actual!r}")


def print_diagnostics(text: str) -> None:
    for label, metric in DIAGNOSTIC_METRICS:
        value = find_last_float(text, metric)
        print(f"{label}={value if value is not None else 'NA'}")


def audit_exp(label: str, exp: str) -> None:
    log = ROOT / f"{exp}.log"
    metrics = ROOT / f"{exp}_metrics.txt"
    proc = ROOT / f"{exp}_proc_health.txt"
    throughput = ROOT / f"{exp}_throughput_summary.txt"
    ray_log = ROOT / f"{exp}_ray_taskrunner.log"

    log_text = require_file(log)
    metrics_text = require_file(metrics)
    proc_text = require_file(proc)
    throughput_text = require_file(throughput)
    require_file(ray_log)

    combined_metrics = "\n".join([log_text, metrics_text])
    mean4 = find_last_float(combined_metrics, "val-core/MATH-TTT/acc/mean@4")
    if mean4 is None:
        raise AssertionError("strict mean@4 metric not found")

    best4 = find_last_float(combined_metrics, "val-core/MATH-TTT/acc/best@4/mean")
    maj4 = find_last_float(combined_metrics, "val-core/MATH-TTT/acc/maj@4/mean")

    expected_mode = {
        "v67": "band_limited_process_sharpened_prob",
        "v68": "count_neutral_process_sharpened_prob",
        "v69": "pairwise_process_count_neutral_sharpened_prob",
        "v75": "support_gated_process_sharpened_prob",
        "v76": "support_mixture_process_sharpened_prob",
        "v77": "support_calibrated_process_sharpened_prob",
        "v78": "support_residual_process_sharpened_prob",
        "v79": "split_support_process_sharpened_prob",
        "v80": "capacity_braked_process_sharpened_prob",
        "v81": "basin_contrast_process_sharpened_prob",
    }[label]
    for key, expected in [
        ("actor_rollout_ref.rollout.val_kwargs.n", "4"),
        ("trainer.validation_answer_selection_enable", "False"),
        ("trainer.total_training_steps", "50"),
        ("trainer.test_freq", "50"),
        ("trainer.val_before_train", "False"),
        ("ttrl.sps_majority_reward_coef", "0.0"),
        ("ttrl.sps_format_reward_coef", "0.0"),
        ("ttrl.sps_reward_mode", expected_mode),
    ]:
        require_effective_config(log_text, key, expected)

    if "PROC_SELF_OK_FINAL" not in proc_text and "PROC_SELF_OK_AFTER" not in proc_text:
        raise AssertionError("final proc self health is not OK")
    if "PROC_MEMINFO_OK_FINAL" not in proc_text and "PROC_MEMINFO_OK_AFTER" not in proc_text:
        raise AssertionError("final proc meminfo health is not OK")
    if "PROC_COUNT_FINAL 0" in proc_text or "PROC_COUNT_AFTER 0" in proc_text:
        raise AssertionError("final proc count is zero")

    if "step_rows=" not in throughput_text or "NO_STEP_ROWS" in throughput_text:
        raise AssertionError("throughput summary has no parsed step rows")

    status = "PASS" if mean4 >= TARGET else "FAIL"
    print(f"EXP={exp}")
    print(f"STRICT_MEAN@4={mean4}")
    print(f"BEST@4={best4 if best4 is not None else 'NA'}")
    print(f"MAJ@4={maj4 if maj4 is not None else 'NA'}")
    print_diagnostics(combined_metrics)
    print(f"TARGET={TARGET}")
    print(f"RESULT={status}")
    if status != "PASS":
        raise SystemExit(2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("version", choices=sorted(EXPS))
    args = parser.parse_args()
    audit_exp(args.version, EXPS[args.version])


if __name__ == "__main__":
    main()
