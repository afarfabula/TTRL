#!/usr/bin/env python3
"""CPU-only smoke for v69 pairwise process-preference target construction."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch


REPO = Path("/opt/tiger/TTRL/verl")
sys.path.insert(0, str(REPO))

from verl.trainer.ppo.ttrl_utils import apply_direct_sharpened_ttrl_reward  # noqa: E402


class ToyTokenizer:
    def decode(self, ids, skip_special_tokens=True):  # noqa: ARG002
        return "".join(chr(int(x)) for x in ids)


@dataclass
class ToyItem:
    text: str
    batch: dict[str, torch.Tensor] = field(init=False)

    def __post_init__(self) -> None:
        prompt = torch.tensor([101, 102], dtype=torch.long)
        response = torch.tensor([ord(ch) for ch in self.text], dtype=torch.long)
        attention_mask = torch.ones(prompt.numel() + response.numel(), dtype=torch.long)
        self.batch = {
            "prompts": prompt,
            "responses": response,
            "attention_mask": attention_mask,
        }


class ToyBatch:
    def __init__(self, n_prompts: int):
        self.n_prompts = n_prompts
        self.non_tensor_batch: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return self.n_prompts


def terminal_rewards(reward_tensor: torch.Tensor, response_mask: torch.Tensor) -> list[float]:
    positions = response_mask.sum(dim=-1).long().clamp(min=1) - 1
    return [
        float(reward_tensor[row, pos].item())
        for row, pos in enumerate(positions.tolist())
    ]


def run_case(pairwise_strength: float):
    clean_1 = "Reasoning is direct. Therefore the final answer is \\boxed{1}."
    clean_1b = "Compute carefully. Thus the final answer is \\boxed{1}."
    conflicted_2 = "First I get \\boxed{3}. Wait. The final answer is \\boxed{2}."
    conflicted_2b = "A wrong attempt gives \\boxed{4}. Therefore the final answer is \\boxed{2}."
    items = [ToyItem(text) for text in [clean_1, clean_1b, conflicted_2, conflicted_2b]]

    max_len = max(item.batch["responses"].numel() for item in items)
    response_mask = torch.zeros((len(items), max_len), dtype=torch.long)
    for row, item in enumerate(items):
        response_mask[row, : item.batch["responses"].numel()] = 1

    log_probs = torch.zeros_like(response_mask, dtype=torch.float32)
    batch = ToyBatch(n_prompts=1)
    out_batch, reward_tensor, info = apply_direct_sharpened_ttrl_reward(
        batch=batch,
        gen_batch_output=items,
        n=4,
        tokenizer=ToyTokenizer(),
        ref_log_prob=log_probs,
        rollout_log_probs=log_probs,
        response_mask=response_mask,
        beta=4.0,
        target_temperature=1.0,
        length_normalize=True,
        reward_scale=1.0,
        per_sample_mode="answer_mass",
        reward_floor=0.0,
        low_budget_k=4,
        process_tail_fraction=0.5,
        process_tilt_strength=0.0,
        process_cluster_tilt_strength=0.0,
        base_contrast_strength=0.0,
        base_contrast_gate_strength=0.0,
        base_contrast_process_margin=0.0,
        target_effective_k_min=0.0,
        target_effective_k_max=0.0,
        target_effective_k_strength=0.0,
        count_neutral_aggregation=True,
        pairwise_process_preference_strength=pairwise_strength,
        pairwise_process_preference_temperature=1.0,
        mode_id=13.0 if pairwise_strength else 12.0,
    )
    return out_batch, terminal_rewards(reward_tensor, response_mask), info


def main() -> None:
    _, no_pair_rewards, no_pair_info = run_case(pairwise_strength=0.0)
    _, pair_rewards, pair_info = run_case(pairwise_strength=0.8)

    print("NO_PAIR_REWARDS=" + ",".join(f"{value:.6f}" for value in no_pair_rewards))
    print("PAIR_REWARDS=" + ",".join(f"{value:.6f}" for value in pair_rewards))
    print(f"TARGET_CONF_NO_PAIR={no_pair_info['sps/direct_target_confidence']}")
    print(f"TARGET_CONF_PAIR={pair_info['sps/direct_target_confidence']}")
    print(f"PAIRWISE_TOP_PREF={pair_info['sps/direct_pairwise_process_top_preference']}")
    print(f"PAIRWISE_PREF_STD={pair_info['sps/direct_pairwise_process_preference_std']}")

    if not all(abs(value - 0.25) < 1e-6 for value in no_pair_rewards):
        raise AssertionError("count-neutral no-pairwise target should be uniform")
    if not (pair_rewards[0] > no_pair_rewards[0] and pair_rewards[1] > no_pair_rewards[1]):
        raise AssertionError("pairwise preference did not increase clean-answer rewards")
    if not (pair_rewards[2] < no_pair_rewards[2] and pair_rewards[3] < no_pair_rewards[3]):
        raise AssertionError("pairwise preference did not decrease conflicted-answer rewards")
    if pair_info["sps/direct_pairwise_process_top_preference"] <= 0.0:
        raise AssertionError("top pairwise preference should be positive")
    if pair_info["sps/direct_target_confidence"] <= no_pair_info["sps/direct_target_confidence"]:
        raise AssertionError("pairwise preference should sharpen target confidence")

    print("V69_PAIRWISE_TARGET_CPU_SMOKE_OK")


if __name__ == "__main__":
    main()
