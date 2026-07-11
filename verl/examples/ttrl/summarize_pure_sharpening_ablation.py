#!/usr/bin/env python3
"""Summarize strict low-budget pure-sharpening ablations as TSV."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path("/opt/tiger/TTRL/verl")
RUNS = [
    (
        "majvote_baseline",
        "Majority pseudo-label baseline",
        "majvote_qwen25_math_7b_50step_strict_n4",
    ),
    (
        "v48_best_sps",
        "Best prior SPS process-consistency capacity",
        "sps_efficient_ttrl_qwen25_math_7b_50step_v48_process_consistency_strict_n4",
    ),
    (
        "v62_process_quality",
        "Process-quality SPS negative result",
        "sps_efficient_ttrl_qwen25_math_7b_50step_v62_process_quality_strict_n4",
    ),
    (
        "v63_direct_prob",
        "Pure direct sharpened probability",
        "sps_direct_sharpened_prob_qwen25_math_7b_50step_v63_strict_n4",
    ),
    (
        "v67_band_limited",
        "Band-limited process sharpening",
        "sps_band_limited_process_sharpened_prob_qwen25_math_7b_50step_v67_strict_n4",
    ),
    (
        "v68_count_neutral",
        "Count-neutral process sharpening",
        "sps_count_neutral_process_sharpened_prob_qwen25_math_7b_50step_v68_strict_n4",
    ),
    (
        "v69_pairwise_count_neutral",
        "Pairwise process preference on count-neutral target",
        "sps_pairwise_process_count_neutral_sharpened_prob_qwen25_math_7b_50step_v69_strict_n4",
    ),
    (
        "v75_support_gated",
        "Support-gated process sharpening",
        "sps_support_gated_process_sharpened_prob_qwen25_math_7b_50step_v75_strict_n4",
    ),
    (
        "v76_support_mixture",
        "Support-mixture process sharpening",
        "sps_support_mixture_process_sharpened_prob_qwen25_math_7b_50step_v76_strict_n4",
    ),
    (
        "v77_support_calibrated",
        "Support-calibrated confidence-cap sharpening",
        "sps_support_calibrated_process_sharpened_prob_qwen25_math_7b_50step_v77_strict_n4",
    ),
    (
        "v78_support_residual",
        "Support-residual process sharpening",
        "sps_support_residual_process_sharpened_prob_qwen25_math_7b_50step_v78_strict_n4",
    ),
    (
        "v79_split_support",
        "Split-view support process sharpening",
        "sps_split_support_process_sharpened_prob_qwen25_math_7b_50step_v79_strict_n4",
    ),
    (
        "v80_capacity_braked",
        "Stability-overconfidence capacity brake",
        "sps_capacity_braked_process_sharpened_prob_qwen25_math_7b_50step_v80_strict_n4",
    ),
    (
        "v81_basin_contrast",
        "Basin-contrast margin calibration",
        "sps_basin_contrast_process_sharpened_prob_qwen25_math_7b_50step_v81_strict_n4",
    ),
]
METRICS = [
    ("strict_mean4", "val-core/MATH-TTT/acc/mean@4"),
    ("best4", "val-core/MATH-TTT/acc/best@4/mean"),
    ("maj4_diag", "val-core/MATH-TTT/acc/maj@4/mean"),
    ("target_conf", "train/sps/direct_target_confidence"),
    ("target_entropy", "train/sps/direct_target_entropy"),
    ("target_effective_K", "train/sps/direct_target_effective_K"),
    ("unique_answers", "train/sps/direct_unique_answer_count"),
    ("target_majority_mass_diag", "train/sps/direct_majority_target_mass"),
    ("base_agreement", "train/sps/direct_base_agreement"),
    ("low_budget_parseable", "train/sps/direct_low_budget_parseable_rate"),
    ("low_budget_clip", "train/sps/direct_low_budget_clip_rate"),
    ("low_budget_top_mass", "train/sps/direct_low_budget_top_mass"),
    ("process_consistent", "train/sps/direct_process_consistent_rate"),
    ("process_top_support", "train/sps/direct_process_top_support"),
    ("count_neutral", "train/sps/direct_count_neutral_aggregation"),
    ("pairwise_top_pref", "train/sps/direct_pairwise_process_top_preference"),
    ("pairwise_pref_std", "train/sps/direct_pairwise_process_preference_std"),
    ("support_gate_strength", "train/sps/direct_support_gate_strength_mean"),
    ("support_gate_top_conf", "train/sps/direct_support_gate_top_confidence"),
    ("support_target_overlap", "train/sps/direct_support_gate_target_overlap"),
    ("support_target_agreement", "train/sps/direct_support_gate_target_agreement"),
    ("support_low_budget_entropy", "train/sps/direct_support_gate_low_budget_entropy"),
    ("support_process_entropy", "train/sps/direct_support_gate_process_entropy"),
    ("support_mixture_alpha", "train/sps/direct_support_mixture_alpha"),
    ("support_mixture_pre_conf", "train/sps/direct_support_mixture_pre_confidence"),
    ("support_mixture_pre_effective_K", "train/sps/direct_support_mixture_pre_effective_K"),
    ("support_mixture_pre_overlap", "train/sps/direct_support_mixture_pre_overlap"),
    ("support_cap_alpha", "train/sps/direct_support_confidence_cap_alpha"),
    ("support_cap_value", "train/sps/direct_support_confidence_cap_value"),
    ("support_cap_pre_conf", "train/sps/direct_support_confidence_cap_pre_confidence"),
    ("support_residual_strength", "train/sps/direct_support_residual_strength_mean"),
    ("support_residual_mean", "train/sps/direct_support_residual_mean"),
    ("support_residual_std", "train/sps/direct_support_residual_std"),
    ("support_residual_top_value", "train/sps/direct_support_residual_top_value"),
    ("split_support_strength", "train/sps/direct_split_support_strength_mean"),
    ("split_support_top_conf", "train/sps/direct_split_support_top_confidence"),
    ("split_support_target_overlap", "train/sps/direct_split_support_target_overlap"),
    ("split_support_target_agreement", "train/sps/direct_split_support_target_agreement"),
    ("split_support_view_overlap", "train/sps/direct_split_support_view_overlap"),
    ("capacity_brake_alpha", "train/sps/direct_capacity_brake_alpha"),
    ("capacity_brake_pre_conf", "train/sps/direct_capacity_brake_pre_confidence"),
    ("capacity_brake_pre_effective_K", "train/sps/direct_capacity_brake_pre_effective_K"),
    ("capacity_brake_pre_overlap", "train/sps/direct_capacity_brake_pre_overlap"),
    ("capacity_brake_post_effective_K", "train/sps/direct_capacity_brake_post_effective_K"),
    ("basin_contrast_alpha", "train/sps/direct_basin_contrast_alpha"),
    ("basin_contrast_target_margin", "train/sps/direct_basin_contrast_target_margin"),
    ("basin_contrast_support_margin", "train/sps/direct_basin_contrast_support_margin"),
    ("basin_contrast_base_margin", "train/sps/direct_basin_contrast_base_margin"),
    ("basin_contrast_low_budget_margin", "train/sps/direct_basin_contrast_low_budget_margin"),
    ("basin_contrast_process_margin", "train/sps/direct_basin_contrast_process_margin"),
    ("basin_contrast_post_effective_K", "train/sps/direct_basin_contrast_post_effective_K"),
    ("sps_pick_acc", "train/sps_pick_accuracy"),
    ("sps_correct_weight_mass", "train/sps_correct_weight_mass"),
    ("pass_at_32", "train/pass@32"),
    ("val_clip_mean4", "val-aux/MATH-TTT/response_clip/mean@4"),
    ("val_format_mean4", "val-aux/MATH-TTT/format_score/mean@4"),
    ("val_avg_logprob_mean4", "val-aux/MATH-TTT/response_avg_logprob/mean@4"),
    ("train_clip_ratio", "response_length/clip_ratio"),
    ("step_time_s", "perf/time_per_step"),
    ("throughput", "perf/throughput"),
]


def read(path: Path) -> str:
    return path.read_text(errors="ignore")


def find_last_float(text: str, metric: str) -> float | None:
    float_re = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    dict_pattern = re.compile(rf"['\"]{re.escape(metric)}['\"]\s*:\s*({float_re})")
    dict_matches = dict_pattern.findall(text)
    if dict_matches:
        return float(dict_matches[-1])
    step_pattern = re.compile(re.escape(metric) + rf"[:=]({float_re})")
    step_matches = step_pattern.findall(text)
    return float(step_matches[-1]) if step_matches else None


def fmt(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.12g}"


def main() -> None:
    header = ["label", "method", "artifact_status"] + [name for name, _ in METRICS]
    print("\t".join(header))
    for label, method, exp in RUNS:
        metrics_path = ROOT / f"{exp}_metrics.txt"
        if not metrics_path.is_file() or metrics_path.stat().st_size == 0:
            row = [label, method, "MISSING"] + ["NA"] * len(METRICS)
            print("\t".join(row))
            continue

        text = read(metrics_path)
        values = [fmt(find_last_float(text, metric)) for _, metric in METRICS]
        status = "COMPLETE_METRICS" if values[0] != "NA" else "NO_STRICT_METRIC"
        print("\t".join([label, method, status] + values))


if __name__ == "__main__":
    main()
