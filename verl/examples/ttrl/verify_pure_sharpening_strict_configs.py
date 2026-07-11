#!/usr/bin/env python3
"""Static guardrails for strict pure-distribution sharpening runners.

This checker is intentionally conservative.  It does not prove algorithmic
success; it only prevents obvious configuration leaks before launching an
expensive worker run.
"""

from __future__ import annotations

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
CONFIG = ROOT / "verl/verl/trainer/config/ppo_trainer_ttrl.yaml"
V81_SMOKE = ROOT / "verl/examples/ttrl/smoke_v81_basin_contrast_cpu.py"


def require(name: str, text: str, needle: str) -> None:
    if needle not in text:
        raise AssertionError(f"{name}: missing required snippet: {needle}")


def reject(name: str, text: str, needle: str) -> None:
    if needle in text:
        raise AssertionError(f"{name}: forbidden snippet present: {needle}")


def check_runner(label: str, path: Path, mode: str, extra: list[str]) -> None:
    text = path.read_text()
    required = [
        "actor_rollout_ref.model.path=\"$LOCAL_MODEL\"",
        "SRC_MODEL=/opt/tiger/qwen2.5_math_7b",
        f"ttrl.sps_reward_mode={mode}",
        "ttrl.sps_majority_reward_coef=0.0",
        "ttrl.sps_format_reward_coef=0.0",
        "actor_rollout_ref.rollout.val_kwargs.n=4",
        "trainer.validation_answer_selection_enable=False",
        "trainer.total_training_steps=50",
        "trainer.test_freq=50",
        "trainer.val_before_train=False",
        "+ray_init.no_runtime_env=True",
        "+ray_init.include_dashboard=False",
        "+ray_init.node_ip_address=127.0.0.1",
        "CACHE_ROOT=/tmp/ttrl_cache/${EXP}",
        'export HF_HOME="$CACHE_ROOT/hf_home"',
        'export TRANSFORMERS_CACHE="$CACHE_ROOT/transformers"',
        'export TORCH_HOME="$CACHE_ROOT/torch"',
        'export VLLM_CACHE_ROOT="$CACHE_ROOT/vllm"',
        'export TRITON_CACHE_DIR="$CACHE_ROOT/triton"',
        'export TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/torchinductor"',
        'export XDG_CACHE_HOME="$CACHE_ROOT/xdg"',
        "PROC_COUNT_BEFORE",
        "GPU_COUNT_BEFORE_TRAIN",
        "check_cuda_compat_preflight.sh",
        "ttrl_record_cuda_preflight",
    ]
    for needle in required + extra:
        require(label, text, needle)

    forbidden = [
        "trainer.validation_answer_selection_enable=True",
        "actor_rollout_ref.rollout.val_kwargs.n=32",
        "trainer.total_training_steps=20",
        "ttrl.sps_majority_reward_coef=1",
        "ttrl.sps_reward_mode=answer_weighted_vote",
        "ttrl.sps_reward_mode=answer_rule_conf_weight",
        "ttrl.sps_reward_mode=answer_conf_weight",
    ]
    for needle in forbidden:
        reject(label, text, needle)


def main() -> None:
    check_runner(
        "v68 runner",
        RUNNERS["v68"],
        "count_neutral_process_sharpened_prob",
        ["ttrl.sps_direct_count_neutral_aggregation=True"],
    )
    check_runner(
        "v69 runner",
        RUNNERS["v69"],
        "pairwise_process_count_neutral_sharpened_prob",
        [
            "ttrl.sps_direct_count_neutral_aggregation=True",
            "ttrl.sps_direct_pairwise_process_preference_strength=0.8",
            "ttrl.sps_direct_pairwise_process_preference_temperature=1.0",
        ],
    )
    check_runner(
        "v75 runner",
        RUNNERS["v75"],
        "support_gated_process_sharpened_prob",
        [
            "ttrl.sps_direct_count_neutral_aggregation=False",
            "ttrl.sps_direct_pairwise_process_preference_strength=0.0",
            "ttrl.sps_direct_support_gate_strength=1.0",
            "ttrl.sps_direct_support_gate_temperature=1.0",
            "ttrl.sps_direct_support_gate_floor=0.05",
            "ttrl.sps_direct_support_gate_base_weight=1.0",
            "ttrl.sps_direct_support_gate_low_budget_weight=1.0",
            "ttrl.sps_direct_support_gate_process_weight=1.0",
        ],
    )
    check_runner(
        "v76 runner",
        RUNNERS["v76"],
        "support_mixture_process_sharpened_prob",
        [
            "ttrl.sps_direct_count_neutral_aggregation=False",
            "ttrl.sps_direct_pairwise_process_preference_strength=0.0",
            "ttrl.sps_direct_support_gate_strength=1.0",
            "ttrl.sps_direct_support_gate_temperature=1.0",
            "ttrl.sps_direct_support_gate_floor=0.05",
            "ttrl.sps_direct_support_gate_base_weight=1.0",
            "ttrl.sps_direct_support_gate_low_budget_weight=1.0",
            "ttrl.sps_direct_support_gate_process_weight=1.0",
            "ttrl.sps_direct_support_mixture_strength=0.6",
            "ttrl.sps_direct_support_mixture_confidence_threshold=0.85",
            "ttrl.sps_direct_support_mixture_effective_k_threshold=2.0",
            "ttrl.sps_direct_support_mixture_overlap_threshold=0.85",
        ],
    )
    check_runner(
        "v77 runner",
        RUNNERS["v77"],
        "support_calibrated_process_sharpened_prob",
        [
            "ttrl.sps_direct_count_neutral_aggregation=False",
            "ttrl.sps_direct_pairwise_process_preference_strength=0.0",
            "ttrl.sps_direct_support_gate_strength=0.0",
            "ttrl.sps_direct_support_gate_temperature=1.0",
            "ttrl.sps_direct_support_gate_floor=0.05",
            "ttrl.sps_direct_support_gate_base_weight=1.0",
            "ttrl.sps_direct_support_gate_low_budget_weight=1.0",
            "ttrl.sps_direct_support_gate_process_weight=1.0",
            "ttrl.sps_direct_support_mixture_strength=0.0",
            "ttrl.sps_direct_support_mixture_confidence_threshold=0.85",
            "ttrl.sps_direct_support_mixture_effective_k_threshold=2.0",
            "ttrl.sps_direct_support_mixture_overlap_threshold=0.85",
            "ttrl.sps_direct_support_confidence_cap_strength=1.0",
            "ttrl.sps_direct_support_confidence_cap_margin=0.20",
            "ttrl.sps_direct_support_confidence_cap_min=0.55",
        ],
    )
    check_runner(
        "v78 runner",
        RUNNERS["v78"],
        "support_residual_process_sharpened_prob",
        [
            "ttrl.sps_direct_count_neutral_aggregation=False",
            "ttrl.sps_direct_pairwise_process_preference_strength=0.0",
            "ttrl.sps_direct_support_gate_strength=1.0",
            "ttrl.sps_direct_support_gate_temperature=1.0",
            "ttrl.sps_direct_support_gate_floor=0.05",
            "ttrl.sps_direct_support_gate_base_weight=1.0",
            "ttrl.sps_direct_support_gate_low_budget_weight=1.0",
            "ttrl.sps_direct_support_gate_process_weight=1.0",
            "ttrl.sps_direct_support_residual_strength=0.5",
        ],
    )
    check_runner(
        "v79 runner",
        RUNNERS["v79"],
        "split_support_process_sharpened_prob",
        [
            "ttrl.sps_direct_count_neutral_aggregation=False",
            "ttrl.sps_direct_pairwise_process_preference_strength=0.0",
            "ttrl.sps_direct_support_gate_strength=0.0",
            "ttrl.sps_direct_support_residual_strength=0.0",
            "ttrl.sps_direct_split_support_strength=0.7",
            "ttrl.sps_direct_split_support_floor=0.05",
        ],
    )
    check_runner(
        "v80 runner",
        RUNNERS["v80"],
        "capacity_braked_process_sharpened_prob",
        [
            "ttrl.sps_direct_count_neutral_aggregation=False",
            "ttrl.sps_direct_pairwise_process_preference_strength=0.0",
            "ttrl.sps_direct_support_gate_strength=1.0",
            "ttrl.sps_direct_support_gate_temperature=1.0",
            "ttrl.sps_direct_support_gate_floor=0.05",
            "ttrl.sps_direct_support_gate_base_weight=1.0",
            "ttrl.sps_direct_support_gate_low_budget_weight=1.0",
            "ttrl.sps_direct_support_gate_process_weight=1.0",
            "ttrl.sps_direct_capacity_brake_strength=0.7",
            "ttrl.sps_direct_capacity_brake_min_effective_k=1.6",
            "ttrl.sps_direct_capacity_brake_confidence_threshold=0.88",
            "ttrl.sps_direct_capacity_brake_overlap_threshold=0.82",
        ],
    )
    check_runner(
        "v81 runner",
        RUNNERS["v81"],
        "basin_contrast_process_sharpened_prob",
        [
            "ttrl.sps_direct_count_neutral_aggregation=False",
            "ttrl.sps_direct_pairwise_process_preference_strength=0.0",
            "ttrl.sps_direct_support_gate_strength=1.0",
            "ttrl.sps_direct_support_gate_temperature=1.0",
            "ttrl.sps_direct_support_gate_floor=0.05",
            "ttrl.sps_direct_support_gate_base_weight=1.0",
            "ttrl.sps_direct_support_gate_low_budget_weight=1.0",
            "ttrl.sps_direct_support_gate_process_weight=1.0",
            "ttrl.sps_direct_basin_contrast_strength=0.8",
            "ttrl.sps_direct_basin_contrast_margin=0.12",
        ],
    )

    ray_text = RAY_TRAINER.read_text()
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
    ]:
        require("ray_trainer", ray_text, needle)

    utils_text = TTRL_UTILS.read_text()
    for needle in [
        "_apply_basin_contrast_calibration(",
        "count_neutral_aggregation=False",
        "pairwise_process_preference_strength=0.0",
        "pairwise_process_preference_temperature=1.0",
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
        "sps/direct_count_neutral_aggregation",
        "sps/direct_pairwise_process_preference_strength",
        "sps/direct_pairwise_process_top_preference",
        "sps/direct_support_gate_strength",
        "sps/direct_support_gate_strength_mean",
        "sps/direct_support_gate_target_overlap",
        "sps/direct_support_mixture_alpha",
        "sps/direct_support_mixture_pre_effective_K",
        "sps/direct_support_confidence_cap_alpha",
        "sps/direct_support_confidence_cap_pre_confidence",
        "sps/direct_support_residual_strength_mean",
        "sps/direct_support_residual_top_value",
        "sps/direct_split_support_strength_mean",
        "sps/direct_split_support_view_overlap",
        "sps/direct_capacity_brake_alpha",
        "sps/direct_basin_contrast_alpha",
    ]:
        require("ttrl_utils", utils_text, needle)

    config_text = CONFIG.read_text()
    for needle in [
        "sps_direct_count_neutral_aggregation: false",
        "sps_direct_pairwise_process_preference_strength: 0.0",
        "sps_direct_pairwise_process_preference_temperature: 1.0",
        "sps_direct_support_gate_strength: 0.0",
        "sps_direct_support_gate_temperature: 1.0",
        "sps_direct_support_gate_floor: 0.05",
        "sps_direct_support_gate_base_weight: 1.0",
        "sps_direct_support_gate_low_budget_weight: 1.0",
        "sps_direct_support_gate_process_weight: 1.0",
        "sps_direct_support_mixture_strength: 0.0",
        "sps_direct_support_mixture_confidence_threshold: 0.85",
        "sps_direct_support_mixture_effective_k_threshold: 2.0",
        "sps_direct_support_mixture_overlap_threshold: 0.85",
        "sps_direct_support_confidence_cap_strength: 0.0",
        "sps_direct_support_confidence_cap_margin: 0.20",
        "sps_direct_support_confidence_cap_min: 0.55",
        "sps_direct_support_residual_strength: 0.0",
        "sps_direct_split_support_strength: 0.0",
        "sps_direct_split_support_floor: 0.05",
        "sps_direct_capacity_brake_strength: 0.0",
        "sps_direct_capacity_brake_min_effective_k: 0.0",
        "sps_direct_capacity_brake_confidence_threshold: 1.0",
        "sps_direct_capacity_brake_overlap_threshold: 1.0",
        "sps_direct_basin_contrast_strength: 0.0",
        "sps_direct_basin_contrast_margin: 0.0",
    ]:
        require("ppo_trainer_ttrl.yaml", config_text, needle)

    smoke_text = V81_SMOKE.read_text()
    for needle in [
        "_apply_basin_contrast_calibration",
        "V81_BASIN_CONTRAST_CPU_SMOKE_OK",
        "weak evidence should activate calibration",
        "strong evidence should leave target unchanged",
        "strength=0 must be exact no-op",
    ]:
        require("smoke_v81_basin_contrast_cpu.py", smoke_text, needle)

    print("STRICT_PURE_SHARPENING_CONFIG_AUDIT_OK")


if __name__ == "__main__":
    main()
