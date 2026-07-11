#!/usr/bin/env python3
"""Static no-majority-target audit for current pure sharpening paths.

This checker is intentionally narrow.  The repo still contains older TTRL/SPS
majority-label modes for reproducibility, so a global grep for "majority" is
not useful.  Instead, this audit verifies that the two current pure-sharpening
    runners enter the direct-distribution path, and that the pure direct path
does not call the old majority-label target builders.
"""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path("/opt/tiger/TTRL")
RUNNERS = {
    "v68": ROOT
    / "verl/examples/ttrl/worker_run_sps_count_neutral_process_sharpened_prob_qwen25_math_7b_50step_v68_strict_n4.sh",
    "v69": ROOT
    / "verl/examples/ttrl/worker_run_sps_pairwise_process_count_neutral_sharpened_prob_qwen25_math_7b_50step_v69_strict_n4.sh",
    "v75": ROOT
    / "verl/examples/ttrl/worker_run_sps_support_gated_process_sharpened_prob_qwen25_math_7b_50step_v75_strict_n4.sh",
    "v76": ROOT
    / "verl/examples/ttrl/worker_run_sps_support_mixture_process_sharpened_prob_qwen25_math_7b_50step_v76_strict_n4.sh",
    "v77": ROOT
    / "verl/examples/ttrl/worker_run_sps_support_calibrated_process_sharpened_prob_qwen25_math_7b_50step_v77_strict_n4.sh",
    "v78": ROOT
    / "verl/examples/ttrl/worker_run_sps_support_residual_process_sharpened_prob_qwen25_math_7b_50step_v78_strict_n4.sh",
    "v79": ROOT
    / "verl/examples/ttrl/worker_run_sps_split_support_process_sharpened_prob_qwen25_math_7b_50step_v79_strict_n4.sh",
    "v80": ROOT
    / "verl/examples/ttrl/worker_run_sps_capacity_braked_process_sharpened_prob_qwen25_math_7b_50step_v80_strict_n4.sh",
    "v81": ROOT
    / "verl/examples/ttrl/worker_run_sps_basin_contrast_process_sharpened_prob_qwen25_math_7b_50step_v81_strict_n4.sh",
}
RAY_TRAINER = ROOT / "verl/verl/trainer/ppo/ray_trainer.py"
TTRL_UTILS = ROOT / "verl/verl/trainer/ppo/ttrl_utils.py"


def require(name: str, text: str, needle: str) -> None:
    if needle not in text:
        raise AssertionError(f"{name}: missing required snippet: {needle}")


def reject(name: str, text: str, needle: str) -> None:
    if needle in text:
        raise AssertionError(f"{name}: forbidden snippet present: {needle}")


def source_segment(text: str, start: str, end: str, name: str) -> str:
    try:
        start_idx = text.index(start)
        end_idx = text.index(end, start_idx)
    except ValueError as exc:
        raise AssertionError(f"{name}: could not isolate source segment") from exc
    return text[start_idx:end_idx]


def check_runner(label: str, mode: str) -> None:
    text = RUNNERS[label].read_text()
    required = [
        "SRC_MODEL=/opt/tiger/qwen2.5_math_7b",
        f"ttrl.sps_reward_mode={mode}",
        "ttrl.sps_majority_reward_coef=0.0",
        "ttrl.sps_format_reward_coef=0.0",
        "actor_rollout_ref.rollout.val_kwargs.n=4",
        "trainer.validation_answer_selection_enable=False",
        "trainer.total_training_steps=50",
        "trainer.test_freq=50",
    ]
    for needle in required:
        require(f"{label} runner", text, needle)
    forbidden = [
        "trainer.validation_answer_selection_enable=True",
        "actor_rollout_ref.rollout.val_kwargs.n=32",
        "ttrl.sps_majority_reward_coef=1",
        "ttrl.sps_reward_mode=answer_weighted_vote",
        "ttrl.sps_reward_mode=answer_weighted_gate",
        "ttrl.sps_reward_mode=answer_conf_filter",
        "ttrl.sps_reward_mode=answer_conf_weight",
        "ttrl.sps_reward_mode=answer_rule_conf_weight",
    ]
    for needle in forbidden:
        reject(f"{label} runner", text, needle)


def check_ray_direct_path() -> None:
    text = RAY_TRAINER.read_text()
    direct_segment = source_segment(
        text,
        'if sps_mode in (\n                                "direct_sharpened_prob"',
        'elif sps_mode in (\n                                "answer_weighted_vote"',
        "ray_trainer direct sharpening branch",
    )
    for needle in [
        '"count_neutral_process_sharpened_prob"',
        '"pairwise_process_count_neutral_sharpened_prob"',
        '"support_gated_process_sharpened_prob"',
        '"support_mixture_process_sharpened_prob"',
        '"support_calibrated_process_sharpened_prob"',
        '"support_residual_process_sharpened_prob"',
        '"split_support_process_sharpened_prob"',
        '"capacity_braked_process_sharpened_prob"',
        '"basin_contrast_process_sharpened_prob"',
        "apply_direct_sharpened_ttrl_reward",
        "select_top_k_per_prompt",
        "count_neutral_aggregation=(",
        '"sps_direct_count_neutral_aggregation", False',
        '"sps_direct_pairwise_process_preference_strength", 0.0',
        '"sps_direct_pairwise_process_preference_temperature", 1.0',
        '"sps_direct_support_gate_strength", 0.0',
        '"sps_direct_support_gate_temperature", 1.0',
        '"sps_direct_support_gate_floor", 0.05',
        '"sps_direct_support_gate_base_weight", 1.0',
        '"sps_direct_support_gate_low_budget_weight", 1.0',
        '"sps_direct_support_gate_process_weight", 1.0',
        '"sps_direct_support_mixture_strength", 0.0',
        '"sps_direct_support_mixture_confidence_threshold", 0.85',
        '"sps_direct_support_mixture_effective_k_threshold", 2.0',
        '"sps_direct_support_mixture_overlap_threshold", 0.85',
        '"sps_direct_support_confidence_cap_strength", 0.0',
        '"sps_direct_support_confidence_cap_margin", 0.20',
        '"sps_direct_support_confidence_cap_min", 0.55',
        '"sps_direct_support_residual_strength", 0.0',
        '"sps_direct_split_support_strength", 0.0',
        '"sps_direct_split_support_floor", 0.05',
        '"sps_direct_capacity_brake_strength", 0.0',
        '"sps_direct_capacity_brake_min_effective_k", 0.0',
        '"sps_direct_capacity_brake_confidence_threshold", 1.0',
        '"sps_direct_capacity_brake_overlap_threshold", 1.0',
        '"sps_direct_basin_contrast_strength", 0.0',
        '"sps_direct_basin_contrast_margin", 0.0',
        "20.0",
        "19.0",
        "18.0",
        "17.0",
        "16.0",
        "15.0",
        "14.0",
        "13.0",
        "12.0",
    ]:
        require("ray_trainer direct sharpening branch", direct_segment, needle)
    for needle in [
        "apply_ttrl_gt",
        "apply_sps_weighted_ttrl_gt",
        "select_majority_first_per_prompt",
        "select_sharpened_cluster_per_prompt",
        "use_majority_fallback",
        "majority_gt_list",
        "sps_raw_majority_gt_list",
    ]:
        reject("ray_trainer direct sharpening branch", direct_segment, needle)


def check_direct_function_ast() -> None:
    text = TTRL_UTILS.read_text()
    module = ast.parse(text)
    func = next(
        (
            node
            for node in module.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "apply_direct_sharpened_ttrl_reward"
        ),
        None,
    )
    if func is None:
        raise AssertionError("ttrl_utils: apply_direct_sharpened_ttrl_reward not found")

    direct_text = ast.get_source_segment(text, func)
    if direct_text is None:
        raise AssertionError("ttrl_utils: could not read direct function source")

    for needle in [
        "count_neutral_aggregation=False",
        "pairwise_process_preference_strength=0.0",
        "support_gate_strength=0.0",
        "support_gate_temperature=1.0",
        "support_gate_floor=0.05",
        "support_mixture_strength=0.0",
        "support_mixture_confidence_threshold=0.85",
        "support_mixture_effective_k_threshold=2.0",
        "support_mixture_overlap_threshold=0.85",
        "support_confidence_cap_strength=0.0",
        "support_confidence_cap_margin=0.20",
        "support_confidence_cap_min=0.55",
        "support_residual_strength=0.0",
        "split_support_strength=0.0",
        "split_support_floor=0.05",
        "capacity_brake_strength=0.0",
        "capacity_brake_min_effective_k=0.0",
        "capacity_brake_confidence_threshold=1.0",
        "capacity_brake_overlap_threshold=1.0",
        "basin_contrast_strength=0.0",
        "basin_contrast_margin=0.0",
        "Majority vote is only logged as a diagnostic.",
        "answer_logits = torch.stack",
        "support_probs = torch.full_like",
        "target_probs = torch.softmax",
        "target_prob_by_answer =",
        "majority_target_mass = target_prob_by_answer.get(majority_gt, 0.0)",
        '"sps/direct_majority_target_mass"',
        '"sps/direct_majority_agreement"',
        '"sps/direct_support_mixture_alpha"',
        '"sps/direct_support_confidence_cap_alpha"',
        '"sps/direct_support_residual_strength_mean"',
        '"sps/direct_split_support_strength_mean"',
        '"sps/direct_capacity_brake_alpha"',
        '"sps/direct_basin_contrast_alpha"',
    ]:
        require("apply_direct_sharpened_ttrl_reward", direct_text, needle)

    for needle in [
        "apply_ttrl_gt",
        "apply_sps_weighted_ttrl_gt",
        "select_majority_first_per_prompt",
        "reward_model",
        "ground_truth",
        "sps_raw_majority_gt_list",
    ]:
        reject("apply_direct_sharpened_ttrl_reward", direct_text, needle)

    target_idx = direct_text.index("target_prob_by_answer =")
    majority_idx = direct_text.index("majority_gt, majority_count = counter.most_common(1)[0]")
    if majority_idx < target_idx:
        raise AssertionError(
            "apply_direct_sharpened_ttrl_reward: majority vote appears before target distribution construction"
        )

    before_target = direct_text[:target_idx]
    for needle in ["majority_gt", "majority_count"]:
        reject("apply_direct_sharpened_ttrl_reward before target construction", before_target, needle)

    after_majority = direct_text[majority_idx:]
    for needle in ["answer_logits =", "target_probs ="]:
        reject("apply_direct_sharpened_ttrl_reward after majority diagnostic", after_majority, needle)

    for line_no, line in enumerate(direct_text.splitlines(), start=func.lineno):
        if "prompt_rewards" in line and "majority" in line:
            raise AssertionError(
                "apply_direct_sharpened_ttrl_reward: majority term touches prompt reward "
                f"on source line {line_no}: {line.strip()}"
            )


def main() -> None:
    check_runner("v68", "count_neutral_process_sharpened_prob")
    check_runner("v69", "pairwise_process_count_neutral_sharpened_prob")
    check_runner("v75", "support_gated_process_sharpened_prob")
    check_runner("v76", "support_mixture_process_sharpened_prob")
    check_runner("v77", "support_calibrated_process_sharpened_prob")
    check_runner("v78", "support_residual_process_sharpened_prob")
    check_runner("v79", "split_support_process_sharpened_prob")
    check_runner("v80", "capacity_braked_process_sharpened_prob")
    check_runner("v81", "basin_contrast_process_sharpened_prob")
    check_ray_direct_path()
    check_direct_function_ast()
    print("PURE_SHARPENING_NO_MAJORITY_TARGET_AUDIT_OK")


if __name__ == "__main__":
    main()
