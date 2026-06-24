# Copyright 2025
# Licensed under the Apache License, Version 2.0
"""SPS reward for TTRL.

Replaces majority voting with a base-model (SPS-style) reward. The original
K=32 SPS finding uses the base model's *sequence logprob* to rank low-temperature
rollouts. For RL we must turn that ranking signal into a dense, well-scaled
reward, otherwise a naive per-prompt softmax over the (hundreds-large) sequence
logweights collapses onto a single rollout (effective_K ~= 1) and the GRPO
gradient becomes extremely sparse.

Two reward modes:

- "group_norm_base" (default, RL-friendly):
    r_i = (alpha * logp_base(y_i|x) - logq(y_i|x)) / len_i        # per-token logw
    reward_i = (r_i - mean_K(r)) / (std_K(r) + eps)               # group-standardized
  Dense, zero-mean within each prompt's K rollouts -> matches GRPO's group
  advantage and preserves the base-model preference ordering.

- "softmax_weight" (the original selection view; kept for ablation):
    reward_i = softmax_K(logw / weight_temperature)
  Winner-take-all; tends to collapse (effective_K ~= 1) when used as RL reward.
"""

from typing import Tuple, Dict

import torch


def compute_sps_reward(
    ref_log_prob: torch.Tensor,
    rollout_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    n: int,
    alpha: float,
    weight_temperature: float = 1.0,
    reward_mode: str = "group_norm_base",
    length_normalize: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute per-rollout SPS reward.

    Args:
        ref_log_prob: (B, R) per-token logprob under the *base* model at temp=1.0.
        rollout_log_probs: (B, R) per-token logprob under the proposal q (low-temp
            actor sampling, returned by vLLM with calculate_log_probs=True).
        response_mask: (B, R) 1 for valid response tokens.
        n: number of rollouts per prompt (K).
        alpha: 1 / proposal_temperature.
        weight_temperature: softmax temperature (softmax_weight mode only).
        reward_mode: "group_norm_base" (default) or "softmax_weight".
        length_normalize: divide sequence logweight by #response tokens.

    Returns:
        reward_tensor: (B, R) zeros except the scalar reward at the last valid token.
        info: dict with diagnostic scalars.
    """
    assert ref_log_prob.shape == rollout_log_probs.shape == response_mask.shape
    assert reward_mode in ("group_norm_base", "softmax_weight")

    mask_f = response_mask.to(torch.float32)
    base_seq = (ref_log_prob.to(torch.float32) * mask_f).sum(dim=-1)       # (B,)
    q_seq = (rollout_log_probs.to(torch.float32) * mask_f).sum(dim=-1)     # (B,)
    lengths = mask_f.sum(dim=-1).clamp(min=1.0)                            # (B,)

    logw = alpha * base_seq - q_seq                                        # (B,)
    if length_normalize:
        score = logw / lengths
    else:
        score = logw

    B = logw.shape[0]
    assert B % n == 0, f"batch size {B} must be divisible by n={n}"
    P = B // n
    score_g = score.view(P, n)

    if reward_mode == "softmax_weight":
        s = score_g / max(weight_temperature, 1e-6)
        s = s - s.max(dim=-1, keepdim=True).values
        rew_g = torch.softmax(s, dim=-1)
    else:  # group_norm_base
        mean = score_g.mean(dim=-1, keepdim=True)
        std = score_g.std(dim=-1, keepdim=True)
        rew_g = (score_g - mean) / (std + 1e-6)

    rew_flat = rew_g.reshape(B).to(torch.float32)

    valid_lens = response_mask.sum(dim=-1).long().clamp(min=1)
    reward_tensor = torch.zeros_like(rollout_log_probs, dtype=torch.float32)
    idx = (valid_lens - 1).unsqueeze(-1)
    reward_tensor.scatter_(1, idx, rew_flat.unsqueeze(-1))

    # diagnostics (softmax view for collapse monitoring, regardless of mode)
    with torch.no_grad():
        s = score_g - score_g.max(dim=-1, keepdim=True).values
        w = torch.softmax(s, dim=-1)
        eff_k = (1.0 / w.pow(2).sum(dim=-1).clamp_min(1e-12)).mean().item()
        ent = (-(w.clamp_min(1e-12).log() * w).sum(dim=-1)).mean().item()

    info = {
        "sps/reward_mode": 0.0 if reward_mode == "group_norm_base" else 1.0,
        "sps/logw_mean": float(logw.mean().item()),
        "sps/logw_std": float(logw.std().item()),
        "sps/score_mean": float(score.mean().item()),
        "sps/score_std": float(score.std().item()),
        "sps/base_seq_mean": float(base_seq.mean().item()),
        "sps/q_seq_mean": float(q_seq.mean().item()),
        "sps/len_mean": float(lengths.mean().item()),
        "sps/reward_mean": float(rew_flat.mean().item()),
        "sps/reward_std": float(rew_flat.std().item()),
        "sps/softmax_weight_entropy": float(ent),
        "sps/effective_K": float(eff_k),
    }
    return reward_tensor, info
