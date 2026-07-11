# Copyright 2025 TTRL Team (https://arxiv.org/abs/2504.16084)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import re
from typing import List
from collections import Counter
import torch
import numpy as np
from verl.utils.reward_score.ttrl_math import extract_answer, simplify_expression_string, grade


def _apply_basin_contrast_calibration(
    target_probs,
    base_probs,
    low_budget_support_probs,
    process_support_probs,
    support_probs,
    strength=0.0,
    margin=0.0,
):
    """Softly cap target top-vs-second margin by internal support margins."""
    alpha = 0.0
    target_margin = 0.0
    support_margin = 0.0
    base_margin = 0.0
    low_budget_margin = 0.0
    process_margin = 0.0
    if max(0.0, float(strength)) <= 0.0 or target_probs.numel() <= 1:
        return (
            target_probs,
            alpha,
            target_margin,
            support_margin,
            base_margin,
            low_budget_margin,
            process_margin,
        )

    strength_f = max(0.0, min(float(strength), 1.0))
    margin_f = max(0.0, float(margin))
    top2 = torch.topk(target_probs, k=2)
    top_idx = int(top2.indices[0].item())
    second_idx = int(top2.indices[1].item())
    target_top = float(target_probs[top_idx].item())
    target_second = float(target_probs[second_idx].item())
    target_margin = target_top - target_second
    base_margin = float(base_probs[top_idx].item() - base_probs[second_idx].item())
    low_budget_margin = float(
        low_budget_support_probs[top_idx].item() - low_budget_support_probs[second_idx].item()
    )
    process_margin = float(
        process_support_probs[top_idx].item() - process_support_probs[second_idx].item()
    )
    support_margin = float(support_probs[top_idx].item() - support_probs[second_idx].item())
    evidence_margin = max(
        0.0,
        (base_margin + low_budget_margin + process_margin + support_margin) / 4.0,
    )
    allowed_margin = min(1.0, evidence_margin + margin_f)
    excess_margin = max(0.0, target_margin - allowed_margin)
    alpha = strength_f * min(1.0, excess_margin)
    if alpha <= 0.0:
        return (
            target_probs,
            alpha,
            target_margin,
            support_margin,
            base_margin,
            low_budget_margin,
            process_margin,
        )

    capped_top = max(0.0, target_top - alpha * excess_margin)
    excess_mass = target_top - capped_top
    calibrated = target_probs.clone()
    calibrated[top_idx] = capped_top
    redistribution = support_probs.clone()
    redistribution[top_idx] = 0.0
    redistribution_sum = redistribution.sum()
    if float(redistribution_sum.item()) <= 1e-12:
        redistribution = torch.ones_like(calibrated)
        redistribution[top_idx] = 0.0
        redistribution_sum = redistribution.sum()
    calibrated = calibrated + excess_mass * redistribution / redistribution_sum.clamp_min(1e-12)
    calibrated = calibrated / calibrated.sum().clamp_min(1e-12)
    return (
        calibrated,
        alpha,
        target_margin,
        support_margin,
        base_margin,
        low_budget_margin,
        process_margin,
    )


def select_top_k_per_prompt(data, n_votes_per_prompt, n_samples_per_prompt):
    """
    Select the first k rollouts per prompt, used for TTRL downsampling.
    """
    assert len(data) % n_votes_per_prompt == 0, "data length must be divisible by n_votes_per_prompt"
    num_prompts = len(data) // n_votes_per_prompt

    selected_indices = []
    for i in range(num_prompts):
        start = i * n_votes_per_prompt
        selected_indices.extend(range(start, start + n_samples_per_prompt))

    return data[selected_indices]


def select_majority_first_per_prompt(data, n_votes_per_prompt, n_samples_per_prompt, tokenizer, majority_gt_list):
    """
    Select rollouts whose extracted answer matches the majority pseudo label first.
    Fill any remaining slots in original rollout order.
    """
    assert len(data) % n_votes_per_prompt == 0, "data length must be divisible by n_votes_per_prompt"
    num_prompts = len(data) // n_votes_per_prompt
    assert len(majority_gt_list) == num_prompts, "majority_gt_list length must match prompt count"

    selected_indices = []
    majority_selected_counts = []
    for i in range(num_prompts):
        start = i * n_votes_per_prompt
        prompt_indices = list(range(start, start + n_votes_per_prompt))
        majority_gt = majority_gt_list[i]
        majority_indices = []
        for idx in prompt_indices:
            data_item = data[idx]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            answer = extract_answer(response_str)
            if answer is None:
                continue
            answer = simplify_expression_string(answer)
            if answer == majority_gt:
                majority_indices.append(idx)

        chosen = majority_indices[:n_samples_per_prompt]
        if len(chosen) < n_samples_per_prompt:
            chosen_set = set(chosen)
            chosen.extend(idx for idx in prompt_indices if idx not in chosen_set)
        chosen = chosen[:n_samples_per_prompt]
        assert len(chosen) == n_samples_per_prompt
        selected_indices.extend(chosen)
        majority_selected_counts.append(min(len(majority_indices), n_samples_per_prompt))

    return data[selected_indices], np.array(majority_selected_counts, dtype=float) / float(n_samples_per_prompt)


def select_low_budget_repair_per_prompt(
    data,
    n_votes_per_prompt,
    n_samples_per_prompt,
    tokenizer,
    majority_gt_list,
    response_mask,
    low_budget_k=4,
    max_replacements=4,
):
    """
    Keep the historical first-k training order, but repair bad low-budget slots.

    This is intentionally much less aggressive than majority-first projection:
    only the first low_budget_k selected slots are eligible for replacement, and
    replacements must be parseable, non-clipped, and match the pseudo label.
    """
    assert len(data) % n_votes_per_prompt == 0, "data length must be divisible by n_votes_per_prompt"
    num_prompts = len(data) // n_votes_per_prompt
    assert len(majority_gt_list) == num_prompts, "majority_gt_list length must match prompt count"
    assert response_mask.shape[0] == len(data)

    max_response_len = response_mask.shape[-1]
    lengths_cpu = response_mask.to(torch.float32).sum(dim=-1).detach().cpu()
    low_k = max(1, min(int(low_budget_k), n_samples_per_prompt))
    max_rep = max(0, min(int(max_replacements), low_k))

    selected_indices = []
    repaired_rates = []
    low_budget_parseable_rates = []
    low_budget_clip_rates = []
    low_budget_cluster_rates = []
    available_repair_rates = []
    for i in range(num_prompts):
        start = i * n_votes_per_prompt
        prompt_indices = list(range(start, start + n_votes_per_prompt))
        chosen = list(range(start, start + n_samples_per_prompt))
        chosen_set = set(chosen)
        majority_gt = majority_gt_list[i]

        parsed = {}
        parseable = {}
        clipped = {}
        in_cluster = {}
        for idx in prompt_indices:
            data_item = data[idx]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            answer = extract_answer(response_str)
            is_parseable = answer is not None
            if is_parseable:
                answer = simplify_expression_string(answer)
            parsed[idx] = answer
            parseable[idx] = is_parseable
            clipped[idx] = bool(lengths_cpu[idx].item() >= max_response_len)
            in_cluster[idx] = bool(is_parseable and answer == majority_gt)

        repair_pool = [
            idx
            for idx in prompt_indices
            if idx not in chosen_set and parseable[idx] and (not clipped[idx]) and in_cluster[idx]
        ]
        repair_cursor = 0
        replacements = 0
        for slot in range(low_k):
            if replacements >= max_rep or repair_cursor >= len(repair_pool):
                break
            idx = chosen[slot]
            bad_low_budget = clipped[idx] or (not parseable[idx]) or (not in_cluster[idx])
            if bad_low_budget:
                chosen_set.remove(idx)
                new_idx = repair_pool[repair_cursor]
                repair_cursor += 1
                chosen[slot] = new_idx
                chosen_set.add(new_idx)
                replacements += 1

        assert len(chosen) == n_samples_per_prompt
        selected_indices.extend(chosen)
        low_budget_chosen = chosen[:low_k]
        repaired_rates.append(float(replacements) / float(low_k))
        low_budget_parseable_rates.append(float(np.mean([parseable[idx] for idx in low_budget_chosen])))
        low_budget_clip_rates.append(float(np.mean([clipped[idx] for idx in low_budget_chosen])))
        low_budget_cluster_rates.append(float(np.mean([in_cluster[idx] for idx in low_budget_chosen])))
        available_repair_rates.append(float(len(repair_pool)) / float(max(n_votes_per_prompt - n_samples_per_prompt, 1)))

    return (
        data[selected_indices],
        {
            "repaired_rate": np.array(repaired_rates, dtype=float),
            "low_budget_parseable_rate": np.array(low_budget_parseable_rates, dtype=float),
            "low_budget_clip_rate": np.array(low_budget_clip_rates, dtype=float),
            "low_budget_cluster_rate": np.array(low_budget_cluster_rates, dtype=float),
            "available_repair_rate": np.array(available_repair_rates, dtype=float),
        },
    )


def select_base_supported_repair_per_prompt(
    data,
    n_votes_per_prompt,
    n_samples_per_prompt,
    tokenizer,
    majority_gt_list,
    response_mask,
    ref_log_prob,
    low_budget_k=4,
    max_replacements=4,
    min_base_gain=0.0,
):
    """
    Repair only low-budget slots with majority samples supported by base/ref.

    v46 showed that repairing first-four slots toward raw majority alone can
    reinforce wrong self-consistent clusters. This variant keeps the same narrow
    low-budget scope, but a replacement must have better length-normalized
    base/ref support than the slot it replaces.
    """
    assert len(data) % n_votes_per_prompt == 0, "data length must be divisible by n_votes_per_prompt"
    num_prompts = len(data) // n_votes_per_prompt
    assert len(majority_gt_list) == num_prompts, "majority_gt_list length must match prompt count"
    assert response_mask.shape[0] == len(data)
    assert ref_log_prob.shape == response_mask.shape

    max_response_len = response_mask.shape[-1]
    mask_f = response_mask.to(torch.float32)
    lengths = mask_f.sum(dim=-1).clamp(min=1.0)
    base_score = ((ref_log_prob.to(torch.float32) * mask_f).sum(dim=-1) / lengths).detach().cpu()
    lengths_cpu = lengths.detach().cpu()
    low_k = max(1, min(int(low_budget_k), n_samples_per_prompt))
    max_rep = max(0, min(int(max_replacements), low_k))
    min_gain = float(min_base_gain)

    selected_indices = []
    repaired_rates = []
    low_budget_parseable_rates = []
    low_budget_clip_rates = []
    low_budget_cluster_rates = []
    available_repair_rates = []
    replacement_base_gains = []
    skipped_base_guard_rates = []
    selected_base_supports = []
    for i in range(num_prompts):
        start = i * n_votes_per_prompt
        prompt_indices = list(range(start, start + n_votes_per_prompt))
        chosen = list(range(start, start + n_samples_per_prompt))
        chosen_set = set(chosen)
        majority_gt = majority_gt_list[i]

        parsed = {}
        parseable = {}
        clipped = {}
        in_cluster = {}
        for idx in prompt_indices:
            data_item = data[idx]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            answer = extract_answer(response_str)
            is_parseable = answer is not None
            if is_parseable:
                answer = simplify_expression_string(answer)
            parsed[idx] = answer
            parseable[idx] = is_parseable
            clipped[idx] = bool(lengths_cpu[idx].item() >= max_response_len)
            in_cluster[idx] = bool(is_parseable and answer == majority_gt)

        repair_pool = sorted(
            [
                idx
                for idx in prompt_indices
                if idx not in chosen_set and parseable[idx] and (not clipped[idx]) and in_cluster[idx]
            ],
            key=lambda idx: float(base_score[idx].item()),
            reverse=True,
        )
        repair_cursor = 0
        replacements = 0
        base_gains = []
        skipped_by_base_guard = 0
        for slot in range(low_k):
            if replacements >= max_rep or repair_cursor >= len(repair_pool):
                break
            idx = chosen[slot]
            bad_low_budget = clipped[idx] or (not parseable[idx]) or (not in_cluster[idx])
            if not bad_low_budget:
                continue

            new_idx = repair_pool[repair_cursor]
            repair_cursor += 1
            base_gain = float(base_score[new_idx].item() - base_score[idx].item())
            if base_gain < min_gain:
                skipped_by_base_guard += 1
                continue

            chosen_set.remove(idx)
            chosen[slot] = new_idx
            chosen_set.add(new_idx)
            replacements += 1
            base_gains.append(base_gain)

        assert len(chosen) == n_samples_per_prompt
        selected_indices.extend(chosen)
        low_budget_chosen = chosen[:low_k]
        repaired_rates.append(float(replacements) / float(low_k))
        low_budget_parseable_rates.append(float(np.mean([parseable[idx] for idx in low_budget_chosen])))
        low_budget_clip_rates.append(float(np.mean([clipped[idx] for idx in low_budget_chosen])))
        low_budget_cluster_rates.append(float(np.mean([in_cluster[idx] for idx in low_budget_chosen])))
        available_repair_rates.append(float(len(repair_pool)) / float(max(n_votes_per_prompt - n_samples_per_prompt, 1)))
        replacement_base_gains.append(float(np.mean(base_gains)) if base_gains else 0.0)
        skipped_base_guard_rates.append(float(skipped_by_base_guard) / float(low_k))
        selected_base_supports.append(
            float(np.mean([float(base_score[idx].item()) for idx in low_budget_chosen]))
        )

    return (
        data[selected_indices],
        {
            "repaired_rate": np.array(repaired_rates, dtype=float),
            "low_budget_parseable_rate": np.array(low_budget_parseable_rates, dtype=float),
            "low_budget_clip_rate": np.array(low_budget_clip_rates, dtype=float),
            "low_budget_cluster_rate": np.array(low_budget_cluster_rates, dtype=float),
            "available_repair_rate": np.array(available_repair_rates, dtype=float),
            "replacement_base_gain": np.array(replacement_base_gains, dtype=float),
            "skipped_base_guard_rate": np.array(skipped_base_guard_rates, dtype=float),
            "selected_base_support": np.array(selected_base_supports, dtype=float),
        },
    )


def _extract_all_boxed_answers(response: str) -> list[tuple[str, int, int]]:
    """Return all balanced \boxed{...} spans in response text."""
    spans = []
    needle = "\\boxed"
    pos = 0
    while True:
        idx = response.find(needle, pos)
        if idx < 0:
            break
        brace = response.find("{", idx + len(needle))
        if brace < 0:
            pos = idx + len(needle)
            continue
        depth = 0
        end = None
        for cur in range(brace, len(response)):
            ch = response[cur]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = cur
                    break
        if end is None:
            pos = brace + 1
            continue
        spans.append((response[brace + 1 : end], idx, end + 1))
        pos = end + 1
    return spans


_REVISION_AFTER_FINAL_RE = re.compile(
    r"\b(wait|however|actually|instead|correction|mistake|wrong|revise|reconsider|not\s+correct)\b",
    re.IGNORECASE,
)


def _score_process_consistency(response: str, final_answer: str | None, clipped: bool, tail_fraction: float):
    """
    Cheap verifier-free process signal from one rollout's own text.

    It rejects outputs with conflicting boxed answers or a final answer that is
    not actually final in the response text. It does not use task ground truth.
    """
    boxed = _extract_all_boxed_answers(response)
    parseable = final_answer is not None
    simplified_final = simplify_expression_string(final_answer) if parseable else None
    simplified_boxed = []
    for answer, _, _ in boxed:
        if answer.strip():
            simplified_boxed.append(simplify_expression_string(answer))
    unique_boxed = {answer for answer in simplified_boxed if answer}
    box_conflict = len(unique_boxed) > 1

    if boxed:
        _, last_start, last_end = boxed[-1]
        tail_position = float(last_start) / float(max(len(response), 1))
        tail_ok = tail_position >= max(0.0, min(1.0, float(tail_fraction)))
        trailing_text = response[last_end:]
    else:
        tail_ok = False
        trailing_text = response
    revision_after_final = bool(_REVISION_AFTER_FINAL_RE.search(trailing_text))

    final_matches_last_box = True
    if parseable and simplified_boxed:
        final_matches_last_box = simplified_boxed[-1] == simplified_final

    consistent = (
        parseable
        and not clipped
        and bool(boxed)
        and not box_conflict
        and tail_ok
        and not revision_after_final
        and final_matches_last_box
    )
    return {
        "consistent": bool(consistent),
        "tail_ok": bool(tail_ok),
        "box_conflict": bool(box_conflict),
        "revision_after_final": bool(revision_after_final),
    }


def select_sharpened_cluster_per_prompt(
    data,
    n_votes_per_prompt,
    n_samples_per_prompt,
    tokenizer,
    majority_gt_list,
    rollout_log_probs,
    response_mask,
    ref_log_prob=None,
    selection_temperature=0.4,
    require_majority=True,
    cluster_bonus_weight=4.0,
    parseable_bonus_weight=2.0,
    nonclip_bonus_weight=1.0,
    selection_priority="score",
    contrast_count=4,
    contrast_min_cluster_ratio=0.5,
):
    """
    Select train rollouts from a sharpened answer cluster.

    The score is internal and unsupervised: prefer parseable, non-clipped
    candidates in the selected answer cluster, then rank by a PowerFlow-style
    reference-reweighted score. This changes the training distribution instead
    of adding a dense format reward.
    """
    assert len(data) % n_votes_per_prompt == 0, "data length must be divisible by n_votes_per_prompt"
    num_prompts = len(data) // n_votes_per_prompt
    assert len(majority_gt_list) == num_prompts, "majority_gt_list length must match prompt count"
    assert rollout_log_probs.shape == response_mask.shape
    if ref_log_prob is not None:
        assert ref_log_prob.shape == rollout_log_probs.shape

    mask_f = response_mask.to(torch.float32)
    lengths = mask_f.sum(dim=-1).clamp(min=1.0)
    q_seq = (rollout_log_probs.to(torch.float32) * mask_f).sum(dim=-1)
    if ref_log_prob is not None:
        base_seq = (ref_log_prob.to(torch.float32) * mask_f).sum(dim=-1)
        alpha = 1.0 / max(float(selection_temperature), 1e-6)
        score = (alpha * base_seq - q_seq) / lengths
    else:
        score = -q_seq / lengths
    score = score.detach().cpu()
    lengths_cpu = lengths.detach().cpu()
    max_response_len = response_mask.shape[-1]

    selected_indices = []
    selected_parseable_rates = []
    selected_clip_rates = []
    selected_cluster_rates = []
    selected_contrast_rates = []
    fallback_rates = []
    for i in range(num_prompts):
        start = i * n_votes_per_prompt
        prompt_indices = list(range(start, start + n_votes_per_prompt))
        majority_gt = majority_gt_list[i]
        ranked = []
        for idx in prompt_indices:
            data_item = data[idx]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            answer = extract_answer(response_str)
            parseable = answer is not None
            if parseable:
                answer = simplify_expression_string(answer)
            in_cluster = parseable and answer == majority_gt
            clipped = float(lengths_cpu[idx].item() >= max_response_len)
            cluster_bonus = 1.0 if in_cluster else 0.0
            parse_bonus = 1.0 if parseable else 0.0
            clip_bonus = 1.0 - clipped
            quality = (
                float(cluster_bonus_weight) * cluster_bonus
                + float(parseable_bonus_weight) * parse_bonus
                + float(nonclip_bonus_weight) * clip_bonus
                + float(score[idx].item())
            )
            ranked.append((quality, idx, parseable, clipped, in_cluster))

        cluster_ranked = [item for item in ranked if item[4]]
        if require_majority and cluster_ranked:
            candidate_ranked = cluster_ranked
            fallback = 0.0
        else:
            candidate_ranked = ranked
            fallback = 1.0 if require_majority else 0.0
        contrast_rate = 0.0
        if selection_priority == "nonclip_parseable_bucket":
            # Project onto the learnable support first; sharpen within each bucket.
            candidate_ranked = sorted(
                candidate_ranked,
                key=lambda item: (
                    1.0 - item[3],  # non-clipped
                    item[2],        # parseable boxed answer
                    item[4],        # majority answer cluster
                    item[0],        # reference-reweighted quality
                ),
                reverse=True,
            )
        elif selection_priority == "nonclip_parseable_contrast_bucket":
            # v26 support projection plus a small high-quality negative tail.
            # When the majority cluster is internally strong, keep most selected
            # samples as non-clipped/parseable majority positives and reserve a
            # few non-clipped/parseable non-majority samples as contrastive
            # negatives under the same pseudo label. This sharpens the train
            # distribution without using true answers.
            majority_support = sorted(
                [
                    item
                    for item in candidate_ranked
                    if (not item[3]) and item[2] and item[4]
                ],
                key=lambda item: item[0],
                reverse=True,
            )
            contrast_support = sorted(
                [
                    item
                    for item in candidate_ranked
                    if (not item[3]) and item[2] and (not item[4])
                ],
                key=lambda item: item[0],
                reverse=True,
            )
            majority_cluster_ratio = float(sum(1 for item in ranked if item[4])) / float(
                n_votes_per_prompt
            )
            max_contrast = min(
                max(int(contrast_count), 0),
                max(n_samples_per_prompt // 4, 0),
                len(contrast_support),
            )
            if (
                max_contrast > 0
                and majority_cluster_ratio >= float(contrast_min_cluster_ratio)
                and len(majority_support) >= n_samples_per_prompt - max_contrast
            ):
                chosen = (
                    majority_support[: n_samples_per_prompt - max_contrast]
                    + contrast_support[:max_contrast]
                )
                contrast_rate = float(max_contrast) / float(n_samples_per_prompt)
            else:
                candidate_ranked = sorted(
                    candidate_ranked,
                    key=lambda item: (
                        1.0 - item[3],
                        item[2],
                        item[4],
                        item[0],
                    ),
                    reverse=True,
                )
        elif selection_priority == "nonclip_parseable_cluster_short_bucket":
            # Keep v26's support projection, then use response length as a
            # sample-level capacity signal inside the selected support.
            candidate_ranked = sorted(
                candidate_ranked,
                key=lambda item: (
                    1.0 - item[3],             # non-clipped
                    item[2],                   # parseable boxed answer
                    item[4],                   # majority answer cluster
                    -float(lengths_cpu[item[1]].item()),
                    item[0],                   # reference-reweighted quality
                ),
                reverse=True,
            )
        elif selection_priority == "parseable_nonclip_cluster_bucket":
            # Favor valid answer-bearing trajectories before length support.
            # This targets mean@k: v26 already stabilizes majority voting, while
            # individual samples still lose points from invalid final answers.
            candidate_ranked = sorted(
                candidate_ranked,
                key=lambda item: (
                    item[2],        # parseable boxed answer
                    1.0 - item[3],  # non-clipped
                    item[4],        # majority answer cluster
                    item[0],        # reference-reweighted quality
                ),
                reverse=True,
            )
        else:
            candidate_ranked = sorted(candidate_ranked, key=lambda item: item[0], reverse=True)
        if selection_priority != "nonclip_parseable_contrast_bucket" or contrast_rate == 0.0:
            chosen = candidate_ranked[:n_samples_per_prompt]
        if len(chosen) < n_samples_per_prompt:
            chosen_set = {idx for _, idx, _, _, _ in chosen}
            fill_candidates = [item for item in ranked if item[1] not in chosen_set]
            if selection_priority == "nonclip_parseable_bucket":
                fill = sorted(
                    fill_candidates,
                    key=lambda item: (
                        1.0 - item[3],
                        item[2],
                        item[4],
                        item[0],
                    ),
                    reverse=True,
                )
            elif selection_priority == "nonclip_parseable_contrast_bucket":
                fill = sorted(
                    fill_candidates,
                    key=lambda item: (
                        1.0 - item[3],
                        item[2],
                        item[4],
                        item[0],
                    ),
                    reverse=True,
                )
            elif selection_priority == "nonclip_parseable_cluster_short_bucket":
                fill = sorted(
                    fill_candidates,
                    key=lambda item: (
                        1.0 - item[3],
                        item[2],
                        item[4],
                        -float(lengths_cpu[item[1]].item()),
                        item[0],
                    ),
                    reverse=True,
                )
            elif selection_priority == "parseable_nonclip_cluster_bucket":
                fill = sorted(
                    fill_candidates,
                    key=lambda item: (
                        item[2],
                        1.0 - item[3],
                        item[4],
                        item[0],
                    ),
                    reverse=True,
                )
            else:
                fill = sorted(fill_candidates, key=lambda item: item[0], reverse=True)
            chosen.extend(fill[: n_samples_per_prompt - len(chosen)])
        assert len(chosen) == n_samples_per_prompt

        selected_indices.extend(idx for _, idx, _, _, _ in chosen)
        selected_parseable_rates.append(float(np.mean([parseable for _, _, parseable, _, _ in chosen])))
        selected_clip_rates.append(float(np.mean([clipped for _, _, _, clipped, _ in chosen])))
        selected_cluster_rates.append(float(np.mean([in_cluster for _, _, _, _, in_cluster in chosen])))
        selected_contrast_rates.append(contrast_rate)
        fallback_rates.append(fallback)

    return (
        data[selected_indices],
        {
            "parseable_rate": np.array(selected_parseable_rates, dtype=float),
            "clip_rate": np.array(selected_clip_rates, dtype=float),
            "cluster_rate": np.array(selected_cluster_rates, dtype=float),
            "contrast_rate": np.array(selected_contrast_rates, dtype=float),
            "fallback_rate": np.array(fallback_rates, dtype=float),
        },
    )


def apply_direct_sharpened_ttrl_reward(
    batch,
    gen_batch_output,
    n,
    tokenizer,
    ref_log_prob,
    rollout_log_probs,
    response_mask,
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
    count_neutral_aggregation=False,
    pairwise_process_preference_strength=0.0,
    pairwise_process_preference_temperature=1.0,
    support_gate_strength=0.0,
    support_gate_temperature=1.0,
    support_gate_floor=0.05,
    support_gate_base_weight=1.0,
    support_gate_low_budget_weight=1.0,
    support_gate_process_weight=1.0,
    support_mixture_strength=0.0,
    support_mixture_confidence_threshold=0.85,
    support_mixture_effective_k_threshold=2.0,
    support_mixture_overlap_threshold=0.85,
    support_confidence_cap_strength=0.0,
    support_confidence_cap_margin=0.20,
    support_confidence_cap_min=0.55,
    support_residual_strength=0.0,
    split_support_strength=0.0,
    split_support_floor=0.05,
    capacity_brake_strength=0.0,
    capacity_brake_min_effective_k=0.0,
    capacity_brake_confidence_threshold=1.0,
    capacity_brake_overlap_threshold=1.0,
    basin_contrast_strength=0.0,
    basin_contrast_margin=0.0,
    mode_id=7.0,
):
    """
    Build a verifier-free sharpened target distribution and emit soft rewards.

    PowerFlow fits a trajectory-balance equation where log p_theta is pulled
    toward beta-scaled reference log probability up to a learned logZ.  In the
    current TTRL reward path we approximate that as an answer-level target:

        log w(y) = beta * log p_ref(y|x) - log p_rollout(y|x)
        p*(a)    = softmax_a logsumexp_{y extracts a} log w(y)

    The terminal reward for a sampled trajectory is its answer cluster's target
    probability, optionally divided by cluster count so total cluster reward
    mass matches p*(a).  If process_tilt_strength is positive, the trajectory
    score is tilted by verifier-free final-answer/process-stability features
    before answer-level aggregation.  If base_contrast_strength is positive,
    clusters that are already high-probability under the frozen base/ref model
    are contrastively penalized before the final softmax.  With a positive
    base_contrast_gate_strength, that penalty is process-conditioned: only
    clusters whose base support exceeds process support are penalized.
    If target_effective_k_strength is positive, the answer target is kept in
    an internal effective-K band by mixing or contracting against a
    base-plus-process support distribution. This constrains the target
    distribution itself, not the validation decoding or any majority label.
    If count_neutral_aggregation is enabled, answer cluster logits use
    log-mean-exp rather than log-sum-exp so repeated samples do not directly
    become a majority-count pseudo-label. This keeps the target listwise and
    process/preference based instead of cluster-size based.
    If pairwise_process_preference_strength is positive, answer logits are
    tilted by a Bradley-Terry style pairwise win rate from process-stability
    features. This moves mass toward answers whose trajectories are cleaner
    than competing answers without selecting the majority cluster as a label.
    If support_gate_strength is positive, the target is tilted by a continuous
    support distribution from base/ref probability, first-low-budget answer
    stability, and process-clean support. This gates the soft target without
    constructing or selecting a majority pseudo-label.
    If support_mixture_strength is positive, over-sharp targets are mixed back
    toward that same support distribution. This is a conservative distribution
    constraint: it cannot create a hard label, and it only activates when the
    target has already become sharper than the support evidence warrants.
    If support_confidence_cap_strength is positive, the target's top answer is
    soft-capped by the independent support distribution's confidence plus a
    margin.  The cap preserves the direct target's top answer and redistributes
    only excess mass, so it is a distribution-calibration rule rather than
    answer selection.
    If support_residual_strength is positive, the target logits receive a
    support-vs-base residual tilt log p_support(a) - log p_base(a).  This only
    changes the soft distribution: it does not select a pseudo-label and it is
    intended to amplify answers whose first-low-budget and process support is
    stronger than the frozen base prior.
    If split_support_strength is positive, first-low-budget answers and
    held-out process-clean answers form two independent support views. Their
    geometric-mean distribution softly tilts target logits, rewarding answers
    stable across both views without choosing a majority label.
    If capacity_brake_strength is positive, a stability-overconfidence brake
    activates only when the target is both too sharp and highly aligned with
    the independent support distribution. It mixes the target toward a softened
    support distribution to maintain answer capacity instead of further
    amplifying a stable basin.
    If basin_contrast_strength is positive, the target top-vs-runner-up margin
    is capped by the corresponding base/first-low-budget/process support
    margin. This uses the second basin as a contrastive reference but never
    selects it as a label; excess top mass is softly redistributed.
    Majority vote is only logged as a diagnostic.
    """
    assert len(gen_batch_output) % n == 0, "gen_batch_output length must be divisible by n"
    num_prompts = len(gen_batch_output) // n
    assert len(batch) == num_prompts, "batch length must be equal to the number of prompts"
    assert ref_log_prob.shape == rollout_log_probs.shape == response_mask.shape
    assert ref_log_prob.shape[0] == len(gen_batch_output)

    mask_f = response_mask.to(torch.float32)
    base_seq = (ref_log_prob.to(torch.float32) * mask_f).sum(dim=-1)
    q_seq = (rollout_log_probs.to(torch.float32) * mask_f).sum(dim=-1)
    lengths = mask_f.sum(dim=-1).clamp(min=1.0)
    flow_scores = float(beta) * base_seq - q_seq
    base_scores = base_seq
    if length_normalize:
        flow_scores = flow_scores / lengths
        base_scores = base_scores / lengths
    flow_scores = (flow_scores / max(float(target_temperature), 1e-6)).detach().cpu()
    base_scores = base_scores.detach().cpu()
    lengths_cpu = lengths.detach().cpu()
    max_response_len = response_mask.shape[-1]

    rewards_flat = []
    majority_ratio_list = []
    direct_target_confidence_list = []
    direct_target_entropy_list = []
    direct_target_effective_k_list = []
    direct_target_logz_list = []
    direct_unique_answer_count_list = []
    direct_parseable_rate_list = []
    direct_clip_rate_list = []
    direct_majority_target_mass_list = []
    direct_majority_agreement_list = []
    direct_base_top_confidence_list = []
    direct_base_agreement_list = []
    direct_low_budget_parseable_rate_list = []
    direct_low_budget_clip_rate_list = []
    direct_low_budget_top_mass_list = []
    direct_process_consistent_rate_list = []
    direct_process_top_support_list = []
    direct_process_tilt_mean_list = []
    direct_process_tilt_std_list = []
    direct_process_cluster_tilt_mean_list = []
    direct_process_cluster_tilt_std_list = []
    direct_process_target_top_support_list = []
    direct_base_contrast_penalty_mean_list = []
    direct_base_contrast_penalty_std_list = []
    direct_base_contrast_gate_mean_list = []
    direct_base_contrast_gate_std_list = []
    direct_effective_k_before_band_list = []
    direct_effective_k_band_alpha_list = []
    direct_effective_k_band_direction_list = []
    direct_pairwise_process_preference_mean_list = []
    direct_pairwise_process_preference_std_list = []
    direct_pairwise_process_top_preference_list = []
    direct_support_gate_strength_list = []
    direct_support_gate_top_confidence_list = []
    direct_support_gate_target_overlap_list = []
    direct_support_gate_target_agreement_list = []
    direct_support_gate_low_budget_entropy_list = []
    direct_support_gate_process_entropy_list = []
    direct_support_mixture_alpha_list = []
    direct_support_mixture_pre_confidence_list = []
    direct_support_mixture_pre_effective_k_list = []
    direct_support_mixture_pre_overlap_list = []
    direct_support_confidence_cap_alpha_list = []
    direct_support_confidence_cap_value_list = []
    direct_support_confidence_cap_pre_confidence_list = []
    direct_support_residual_strength_list = []
    direct_support_residual_mean_list = []
    direct_support_residual_std_list = []
    direct_support_residual_top_value_list = []
    direct_split_support_strength_list = []
    direct_split_support_top_confidence_list = []
    direct_split_support_target_overlap_list = []
    direct_split_support_target_agreement_list = []
    direct_split_support_view_overlap_list = []
    direct_capacity_brake_alpha_list = []
    direct_capacity_brake_pre_confidence_list = []
    direct_capacity_brake_pre_effective_k_list = []
    direct_capacity_brake_pre_overlap_list = []
    direct_capacity_brake_post_effective_k_list = []
    direct_basin_contrast_alpha_list = []
    direct_basin_contrast_target_margin_list = []
    direct_basin_contrast_support_margin_list = []
    direct_basin_contrast_base_margin_list = []
    direct_basin_contrast_low_budget_margin_list = []
    direct_basin_contrast_process_margin_list = []
    direct_basin_contrast_post_effective_k_list = []
    direct_reward_mean_list = []
    direct_reward_std_list = []
    direct_reward_nonzero_rate_list = []

    low_k = max(1, min(int(low_budget_k), n))
    for i in range(num_prompts):
        start = i * n
        answers = []
        answer_to_indices = {}
        answer_to_scores = {}
        answer_to_base_scores = {}
        answer_to_cluster_tilts = {}
        rollout_infos = []
        clipped_flags = []
        prompt_rewards = [0.0] * n
        process_tilts = [0.0] * n

        for j in range(n):
            idx = start + j
            data_item = gen_batch_output[idx]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            raw_answer = extract_answer(response_str)
            clipped = bool(lengths_cpu[idx].item() >= max_response_len)
            clipped_flags.append(1.0 if clipped else 0.0)
            process_info = _score_process_consistency(
                response_str,
                raw_answer,
                clipped=clipped,
                tail_fraction=process_tail_fraction,
            )
            answer = simplify_expression_string(raw_answer) if raw_answer is not None else None
            process_info["answer"] = answer
            process_info["parseable"] = raw_answer is not None
            process_info["clipped"] = clipped
            rollout_infos.append(process_info)
            tilt = 0.0
            if raw_answer is not None:
                tilt += 1.0
            else:
                tilt -= 1.0
            if not clipped:
                tilt += 0.5
            else:
                tilt -= 1.0
            if process_info.get("tail_ok"):
                tilt += 0.25
            if process_info.get("consistent"):
                tilt += 1.0
            if process_info.get("box_conflict"):
                tilt -= 1.0
            if process_info.get("revision_after_final"):
                tilt -= 0.75
            process_tilts[j] = tilt
            if answer is None:
                continue
            answers.append(answer)
            answer_to_indices.setdefault(answer, []).append(j)
            answer_to_scores.setdefault(answer, []).append(flow_scores[idx])
            answer_to_base_scores.setdefault(answer, []).append(base_scores[idx])

        if not answer_to_scores:
            rewards_flat.extend(prompt_rewards)
            majority_ratio_list.append(0.0)
            direct_target_confidence_list.append(0.0)
            direct_target_entropy_list.append(0.0)
            direct_target_effective_k_list.append(0.0)
            direct_target_logz_list.append(0.0)
            direct_unique_answer_count_list.append(0.0)
            direct_parseable_rate_list.append(0.0)
            direct_clip_rate_list.append(float(np.mean(clipped_flags)) if clipped_flags else 0.0)
            direct_majority_target_mass_list.append(0.0)
            direct_majority_agreement_list.append(0.0)
            direct_base_top_confidence_list.append(0.0)
            direct_base_agreement_list.append(0.0)
            direct_low_budget_parseable_rate_list.append(0.0)
            direct_low_budget_clip_rate_list.append(
                float(np.mean(clipped_flags[:low_k])) if clipped_flags else 0.0
            )
            direct_low_budget_top_mass_list.append(0.0)
            direct_process_consistent_rate_list.append(0.0)
            direct_process_top_support_list.append(0.0)
            direct_process_tilt_mean_list.append(float(np.mean(process_tilts)) if process_tilts else 0.0)
            direct_process_tilt_std_list.append(float(np.std(process_tilts)) if process_tilts else 0.0)
            direct_process_cluster_tilt_mean_list.append(0.0)
            direct_process_cluster_tilt_std_list.append(0.0)
            direct_process_target_top_support_list.append(0.0)
            direct_base_contrast_penalty_mean_list.append(0.0)
            direct_base_contrast_penalty_std_list.append(0.0)
            direct_base_contrast_gate_mean_list.append(0.0)
            direct_base_contrast_gate_std_list.append(0.0)
            direct_effective_k_before_band_list.append(0.0)
            direct_effective_k_band_alpha_list.append(0.0)
            direct_effective_k_band_direction_list.append(0.0)
            direct_pairwise_process_preference_mean_list.append(0.0)
            direct_pairwise_process_preference_std_list.append(0.0)
            direct_pairwise_process_top_preference_list.append(0.0)
            direct_support_gate_strength_list.append(0.0)
            direct_support_gate_top_confidence_list.append(0.0)
            direct_support_gate_target_overlap_list.append(0.0)
            direct_support_gate_target_agreement_list.append(0.0)
            direct_support_gate_low_budget_entropy_list.append(0.0)
            direct_support_gate_process_entropy_list.append(0.0)
            direct_support_mixture_alpha_list.append(0.0)
            direct_support_mixture_pre_confidence_list.append(0.0)
            direct_support_mixture_pre_effective_k_list.append(0.0)
            direct_support_mixture_pre_overlap_list.append(0.0)
            direct_support_confidence_cap_alpha_list.append(0.0)
            direct_support_confidence_cap_value_list.append(0.0)
            direct_support_confidence_cap_pre_confidence_list.append(0.0)
            direct_support_residual_strength_list.append(0.0)
            direct_support_residual_mean_list.append(0.0)
            direct_support_residual_std_list.append(0.0)
            direct_support_residual_top_value_list.append(0.0)
            direct_split_support_strength_list.append(0.0)
            direct_split_support_top_confidence_list.append(0.0)
            direct_split_support_target_overlap_list.append(0.0)
            direct_split_support_target_agreement_list.append(0.0)
            direct_split_support_view_overlap_list.append(0.0)
            direct_capacity_brake_alpha_list.append(0.0)
            direct_capacity_brake_pre_confidence_list.append(0.0)
            direct_capacity_brake_pre_effective_k_list.append(0.0)
            direct_capacity_brake_pre_overlap_list.append(0.0)
            direct_capacity_brake_post_effective_k_list.append(0.0)
            direct_basin_contrast_alpha_list.append(0.0)
            direct_basin_contrast_target_margin_list.append(0.0)
            direct_basin_contrast_support_margin_list.append(0.0)
            direct_basin_contrast_base_margin_list.append(0.0)
            direct_basin_contrast_low_budget_margin_list.append(0.0)
            direct_basin_contrast_process_margin_list.append(0.0)
            direct_basin_contrast_post_effective_k_list.append(0.0)
            direct_reward_mean_list.append(0.0)
            direct_reward_std_list.append(0.0)
            direct_reward_nonzero_rate_list.append(0.0)
            continue

        process_tilt_tensor = torch.as_tensor(process_tilts, dtype=flow_scores.dtype)
        parseable_positions = sorted(j for indices in answer_to_indices.values() for j in indices)
        if parseable_positions:
            parseable_tilts = process_tilt_tensor[parseable_positions]
            process_tilt_tensor = process_tilt_tensor - parseable_tilts.mean()
        process_tilt_strength_f = max(0.0, float(process_tilt_strength))

        cluster_tilt_values = []
        for answer, indices in answer_to_indices.items():
            cluster_infos = [rollout_infos[j] for j in indices]
            denom = float(max(len(cluster_infos), 1))
            consistent_rate = sum(1.0 for info in cluster_infos if info.get("consistent")) / denom
            tail_rate = sum(1.0 for info in cluster_infos if info.get("tail_ok")) / denom
            bad_rate = sum(
                1.0
                for info in cluster_infos
                if info.get("box_conflict") or info.get("revision_after_final") or info.get("clipped")
            ) / denom
            # Process-conditioned stability: unlike majority vote, this never
            # picks the largest answer cluster directly; it only tilts clusters
            # whose own trajectories are internally clean and stable.
            cluster_tilt = np.log(0.05 + consistent_rate) + 0.5 * np.log(0.05 + tail_rate) - bad_rate
            answer_to_cluster_tilts[answer] = float(cluster_tilt)
            cluster_tilt_values.append(float(cluster_tilt))
        cluster_tilt_mean = float(np.mean(cluster_tilt_values)) if cluster_tilt_values else 0.0
        cluster_tilt_std = float(np.std(cluster_tilt_values)) if cluster_tilt_values else 0.0
        process_cluster_tilt_strength_f = max(0.0, float(process_cluster_tilt_strength))

        answer_keys = list(answer_to_scores.keys())
        pairwise_preference = torch.zeros(len(answer_keys), dtype=flow_scores.dtype)
        pairwise_strength_f = max(0.0, float(pairwise_process_preference_strength))
        pairwise_temperature_f = max(float(pairwise_process_preference_temperature), 1e-6)
        if pairwise_strength_f > 0.0 and len(answer_keys) > 1:
            answer_process_scores = []
            for answer in answer_keys:
                indices = answer_to_indices[answer]
                cluster_scores = []
                for j in indices:
                    info = rollout_infos[j]
                    score = 0.0
                    if info.get("parseable"):
                        score += 0.75
                    else:
                        score -= 0.75
                    if not info.get("clipped"):
                        score += 0.50
                    else:
                        score -= 1.00
                    if info.get("consistent"):
                        score += 1.00
                    if info.get("tail_ok"):
                        score += 0.35
                    if info.get("box_conflict"):
                        score -= 1.00
                    if info.get("revision_after_final"):
                        score -= 0.75
                    cluster_scores.append(score)
                answer_process_scores.append(float(np.mean(cluster_scores)) if cluster_scores else 0.0)
            pref_values = []
            for idx, score in enumerate(answer_process_scores):
                wins = [
                    1.0 / (1.0 + np.exp(-(score - other) / pairwise_temperature_f))
                    for k, other in enumerate(answer_process_scores)
                    if k != idx
                ]
                pref_values.append(float(np.mean(wins)) if wins else 0.5)
            pref_tensor = torch.as_tensor(pref_values, dtype=flow_scores.dtype)
            pairwise_preference = pref_tensor - pref_tensor.mean()
        answer_logits = torch.stack(
            [
                (
                    torch.logsumexp(
                        torch.stack(
                            [
                                answer_to_scores[answer][pos]
                                + process_tilt_strength_f * process_tilt_tensor[j]
                                + process_cluster_tilt_strength_f
                                * (answer_to_cluster_tilts.get(answer, 0.0) - cluster_tilt_mean)
                                for pos, j in enumerate(answer_to_indices[answer])
                            ]
                        ),
                        dim=0,
                    )
                    - (
                        np.log(float(max(len(answer_to_indices[answer]), 1)))
                        if bool(count_neutral_aggregation)
                        else 0.0
                    )
                )
                for answer in answer_keys
            ]
        )
        if pairwise_strength_f > 0.0 and len(answer_keys) > 1:
            answer_logits = answer_logits + pairwise_strength_f * pairwise_preference
        base_logits = torch.stack(
            [torch.logsumexp(torch.stack(answer_to_base_scores[answer]), dim=0) for answer in answer_keys]
        )
        base_probs = torch.softmax(base_logits - base_logits.max(), dim=0)
        support_gate_strength_f = max(0.0, float(support_gate_strength))
        support_mixture_strength_f = max(0.0, min(float(support_mixture_strength), 1.0))
        support_confidence_cap_strength_f = max(
            0.0, min(float(support_confidence_cap_strength), 1.0)
        )
        support_residual_strength_f = max(0.0, float(support_residual_strength))
        split_support_strength_f = max(0.0, float(split_support_strength))
        split_support_floor_f = max(float(split_support_floor), 1e-6)
        capacity_brake_strength_f = max(0.0, min(float(capacity_brake_strength), 1.0))
        basin_contrast_strength_f = max(0.0, min(float(basin_contrast_strength), 1.0))
        basin_contrast_margin_f = max(0.0, float(basin_contrast_margin))
        support_gate_floor_f = max(float(support_gate_floor), 1e-6)
        support_temperature_f = max(float(support_gate_temperature), 1e-6)
        support_probs = torch.full_like(base_probs, 1.0 / float(len(answer_keys)))
        low_budget_support_probs = support_probs.clone()
        process_support_probs = support_probs.clone()
        if (
            support_gate_strength_f > 0.0
            or support_mixture_strength_f > 0.0
            or support_confidence_cap_strength_f > 0.0
            or support_residual_strength_f > 0.0
        ) and len(answer_keys) > 1:
            low_budget_counts = []
            for answer in answer_keys:
                count = 0.0
                for info in rollout_infos[:low_k]:
                    if info.get("parseable") and info.get("answer") == answer:
                        count += 1.0
                low_budget_counts.append(count + support_gate_floor_f)
            low_budget_support_probs = torch.as_tensor(low_budget_counts, dtype=base_probs.dtype)
            low_budget_support_probs = low_budget_support_probs / low_budget_support_probs.sum().clamp_min(1e-12)

            process_support_logits = torch.as_tensor(
                [answer_to_cluster_tilts.get(answer, 0.0) for answer in answer_keys],
                dtype=base_probs.dtype,
            )
            process_support_probs = torch.softmax(
                (process_support_logits - process_support_logits.max()) / support_temperature_f,
                dim=0,
            )

            log_terms = []
            weight_sum = 0.0
            for weight, probs in (
                (float(support_gate_base_weight), base_probs),
                (float(support_gate_low_budget_weight), low_budget_support_probs),
                (float(support_gate_process_weight), process_support_probs),
            ):
                if weight > 0.0:
                    log_terms.append(weight * probs.clamp_min(support_gate_floor_f).log())
                    weight_sum += weight
            if log_terms and weight_sum > 0.0:
                support_logits = torch.stack(log_terms).sum(dim=0) / weight_sum
                support_probs = torch.softmax(
                    (support_logits - support_logits.max()) / support_temperature_f,
                    dim=0,
                )
                answer_logits = answer_logits + support_gate_strength_f * support_probs.clamp_min(
                    support_gate_floor_f
                ).log()
        support_residual = torch.zeros_like(answer_logits)
        if support_residual_strength_f > 0.0 and len(answer_keys) > 1:
            support_residual = (
                support_probs.clamp_min(support_gate_floor_f).log()
                - base_probs.clamp_min(support_gate_floor_f).log()
            )
            support_residual = support_residual - support_residual.mean()
            answer_logits = answer_logits + support_residual_strength_f * support_residual
        split_support_probs = torch.full_like(base_probs, 1.0 / float(len(answer_keys)))
        split_low_probs = split_support_probs.clone()
        split_holdout_probs = split_support_probs.clone()
        if split_support_strength_f > 0.0 and len(answer_keys) > 1:
            low_counts = []
            holdout_counts = []
            for answer in answer_keys:
                low_count = 0.0
                for info in rollout_infos[:low_k]:
                    if info.get("parseable") and info.get("answer") == answer and not info.get("clipped"):
                        low_count += 1.0
                holdout_count = 0.0
                for info in rollout_infos[low_k:]:
                    if info.get("parseable") and info.get("answer") == answer and not info.get("clipped"):
                        holdout_count += 1.0 if info.get("consistent") else 0.25
                low_counts.append(low_count + split_support_floor_f)
                holdout_counts.append(holdout_count + split_support_floor_f)
            split_low_probs = torch.as_tensor(low_counts, dtype=base_probs.dtype)
            split_holdout_probs = torch.as_tensor(holdout_counts, dtype=base_probs.dtype)
            split_low_probs = split_low_probs / split_low_probs.sum().clamp_min(1e-12)
            split_holdout_probs = split_holdout_probs / split_holdout_probs.sum().clamp_min(1e-12)
            split_support_raw = (split_low_probs * split_holdout_probs).clamp_min(1e-12).sqrt()
            split_support_probs = split_support_raw / split_support_raw.sum().clamp_min(1e-12)
            answer_logits = answer_logits + split_support_strength_f * split_support_probs.clamp_min(
                split_support_floor_f
            ).log()
        base_contrast_strength_f = max(0.0, float(base_contrast_strength))
        base_contrast_gate_strength_f = max(0.0, float(base_contrast_gate_strength))
        if base_contrast_strength_f > 0.0 and len(answer_keys) > 1:
            if base_contrast_gate_strength_f > 0.0:
                process_support_logits = torch.as_tensor(
                    [answer_to_cluster_tilts.get(answer, 0.0) for answer in answer_keys],
                    dtype=base_probs.dtype,
                )
                process_support_probs = torch.softmax(
                    process_support_logits - process_support_logits.max(), dim=0
                )
                margin = float(base_contrast_process_margin)
                gate = torch.sigmoid(
                    base_contrast_gate_strength_f
                    * (base_probs - process_support_probs - margin)
                    * float(len(answer_keys))
                )
                # Gated contrast only subtracts suspicious base support.  It
                # never boosts low-base clusters, which was the v65 failure.
                base_excess = (base_probs - process_support_probs).clamp_min(0.0) * float(len(answer_keys))
                base_contrast_penalty = base_contrast_strength_f * gate.detach() * base_excess.detach()
                base_contrast_gate = gate.detach()
            else:
                # v65 behavior: global centered base contrast.  Kept for
                # reproducibility of the negative ablation.
                base_centered = (base_probs - (1.0 / float(len(answer_keys)))) * float(len(answer_keys))
                base_contrast_penalty = base_contrast_strength_f * base_centered.detach()
                base_contrast_gate = torch.ones_like(answer_logits)
            answer_logits = answer_logits - base_contrast_penalty
        else:
            base_contrast_penalty = torch.zeros_like(answer_logits)
            base_contrast_gate = torch.zeros_like(answer_logits)
        answer_logz = torch.logsumexp(answer_logits, dim=0)
        target_probs = torch.softmax(answer_logits - answer_logits.max(), dim=0)
        effective_k_before_band = float(
            (1.0 / target_probs.pow(2).sum().clamp_min(1e-12)).item()
        )
        band_alpha = 0.0
        band_direction = 0.0
        band_strength = max(0.0, min(float(target_effective_k_strength), 1.0))
        min_k = max(0.0, float(target_effective_k_min))
        max_k = max(0.0, float(target_effective_k_max))
        if band_strength > 0.0 and len(answer_keys) > 1 and (min_k > 0.0 or max_k > 0.0):
            process_support_logits = torch.as_tensor(
                [answer_to_cluster_tilts.get(answer, 0.0) for answer in answer_keys],
                dtype=base_probs.dtype,
            )
            support_logits = (
                base_logits
                - base_logits.max()
                + process_cluster_tilt_strength_f
                * (process_support_logits - process_support_logits.mean())
            )
            support_probs = torch.softmax(support_logits - support_logits.max(), dim=0)
            if min_k > 0.0 and effective_k_before_band < min_k:
                band_alpha = band_strength * min(
                    1.0, (min_k - effective_k_before_band) / max(min_k, 1e-6)
                )
                target_probs = (1.0 - band_alpha) * target_probs + band_alpha * support_probs
                target_probs = target_probs / target_probs.sum().clamp_min(1e-12)
                band_direction = 1.0
            elif max_k > 0.0 and effective_k_before_band > max_k:
                band_alpha = band_strength * min(
                    1.0, (effective_k_before_band - max_k) / max(effective_k_before_band, 1e-6)
                )
                target_probs = target_probs * support_probs.clamp_min(1e-12).pow(band_alpha)
                target_probs = target_probs / target_probs.sum().clamp_min(1e-12)
                band_direction = -1.0
        pre_mixture_confidence = float(target_probs.max().item())
        pre_mixture_effective_k = float(
            (1.0 / target_probs.pow(2).sum().clamp_min(1e-12)).item()
        )
        pre_mixture_overlap = float(torch.minimum(target_probs, support_probs).sum().item())
        mixture_alpha = 0.0
        if support_mixture_strength_f > 0.0 and len(answer_keys) > 1:
            conf_threshold = float(support_mixture_confidence_threshold)
            effk_threshold = max(0.0, float(support_mixture_effective_k_threshold))
            overlap_threshold = float(support_mixture_overlap_threshold)
            conf_signal = 0.0
            if conf_threshold < 1.0 and pre_mixture_confidence > conf_threshold:
                conf_signal = min(
                    1.0,
                    (pre_mixture_confidence - conf_threshold) / max(1.0 - conf_threshold, 1e-6),
                )
            effk_signal = 0.0
            if effk_threshold > 0.0 and pre_mixture_effective_k < effk_threshold:
                effk_signal = min(
                    1.0,
                    (effk_threshold - pre_mixture_effective_k) / max(effk_threshold, 1e-6),
                )
            overlap_signal = 0.0
            if overlap_threshold > 0.0 and pre_mixture_overlap < overlap_threshold:
                overlap_signal = min(
                    1.0,
                    (overlap_threshold - pre_mixture_overlap) / max(overlap_threshold, 1e-6),
                )
            mixture_alpha = support_mixture_strength_f * max(conf_signal, effk_signal) * overlap_signal
            if mixture_alpha > 0.0:
                target_probs = (1.0 - mixture_alpha) * target_probs + mixture_alpha * support_probs
                target_probs = target_probs / target_probs.sum().clamp_min(1e-12)
        support_top_confidence_for_cap = float(support_probs.max().item())
        confidence_cap_margin_f = max(0.0, float(support_confidence_cap_margin))
        confidence_cap_min_f = max(0.0, min(float(support_confidence_cap_min), 1.0))
        pre_cap_confidence = float(target_probs.max().item())
        confidence_cap_value = min(
            1.0,
            max(confidence_cap_min_f, support_top_confidence_for_cap + confidence_cap_margin_f),
        )
        confidence_cap_alpha = 0.0
        if (
            support_confidence_cap_strength_f > 0.0
            and len(answer_keys) > 1
            and pre_cap_confidence > confidence_cap_value
        ):
            top_idx_for_cap = int(torch.argmax(target_probs).item())
            capped_top = (
                (1.0 - support_confidence_cap_strength_f) * pre_cap_confidence
                + support_confidence_cap_strength_f * confidence_cap_value
            )
            excess_mass = pre_cap_confidence - capped_top
            target_probs = target_probs.clone()
            target_probs[top_idx_for_cap] = capped_top
            redistribution = support_probs.clone()
            redistribution[top_idx_for_cap] = 0.0
            redistribution_sum = redistribution.sum()
            if float(redistribution_sum.item()) <= 1e-12:
                redistribution = torch.ones_like(target_probs)
                redistribution[top_idx_for_cap] = 0.0
                redistribution_sum = redistribution.sum()
            target_probs = target_probs + excess_mass * redistribution / redistribution_sum.clamp_min(1e-12)
            target_probs = target_probs / target_probs.sum().clamp_min(1e-12)
            confidence_cap_alpha = support_confidence_cap_strength_f
        capacity_brake_pre_confidence = float(target_probs.max().item())
        capacity_brake_pre_effective_k = float(
            (1.0 / target_probs.pow(2).sum().clamp_min(1e-12)).item()
        )
        capacity_brake_pre_overlap = float(torch.minimum(target_probs, support_probs).sum().item())
        capacity_brake_alpha = 0.0
        if capacity_brake_strength_f > 0.0 and len(answer_keys) > 1:
            brake_min_k = min(
                float(len(answer_keys)),
                max(0.0, float(capacity_brake_min_effective_k)),
            )
            conf_threshold = max(0.0, min(float(capacity_brake_confidence_threshold), 1.0))
            overlap_threshold = max(0.0, min(float(capacity_brake_overlap_threshold), 1.0))
            conf_signal = 0.0
            if conf_threshold < 1.0 and capacity_brake_pre_confidence > conf_threshold:
                conf_signal = min(
                    1.0,
                    (capacity_brake_pre_confidence - conf_threshold)
                    / max(1.0 - conf_threshold, 1e-6),
                )
            effk_signal = 0.0
            if brake_min_k > 0.0 and capacity_brake_pre_effective_k < brake_min_k:
                effk_signal = min(
                    1.0,
                    (brake_min_k - capacity_brake_pre_effective_k) / max(brake_min_k, 1e-6),
                )
            overlap_signal = 0.0
            if capacity_brake_pre_overlap > overlap_threshold:
                overlap_signal = min(
                    1.0,
                    (capacity_brake_pre_overlap - overlap_threshold)
                    / max(1.0 - overlap_threshold, 1e-6),
                )
            capacity_brake_alpha = capacity_brake_strength_f * max(conf_signal, effk_signal) * overlap_signal
            if capacity_brake_alpha > 0.0:
                softened_support = support_probs.clamp_min(support_gate_floor_f).sqrt()
                softened_support = softened_support / softened_support.sum().clamp_min(1e-12)
                target_probs = (
                    (1.0 - capacity_brake_alpha) * target_probs
                    + capacity_brake_alpha * softened_support
                )
                target_probs = target_probs / target_probs.sum().clamp_min(1e-12)
        capacity_brake_post_effective_k = float(
            (1.0 / target_probs.pow(2).sum().clamp_min(1e-12)).item()
        )
        basin_contrast_alpha = 0.0
        basin_contrast_target_margin = 0.0
        basin_contrast_support_margin = 0.0
        basin_contrast_base_margin = 0.0
        basin_contrast_low_budget_margin = 0.0
        basin_contrast_process_margin = 0.0
        if basin_contrast_strength_f > 0.0 and len(answer_keys) > 1:
            (
                target_probs,
                basin_contrast_alpha,
                basin_contrast_target_margin,
                basin_contrast_support_margin,
                basin_contrast_base_margin,
                basin_contrast_low_budget_margin,
                basin_contrast_process_margin,
            ) = _apply_basin_contrast_calibration(
                target_probs=target_probs,
                base_probs=base_probs,
                low_budget_support_probs=low_budget_support_probs,
                process_support_probs=process_support_probs,
                support_probs=support_probs,
                strength=basin_contrast_strength_f,
                margin=basin_contrast_margin_f,
            )
        basin_contrast_post_effective_k = float(
            (1.0 / target_probs.pow(2).sum().clamp_min(1e-12)).item()
        )
        target_prob_by_answer = {
            answer: float(target_probs[idx].item())
            for idx, answer in enumerate(answer_keys)
        }
        top_answer_idx = int(torch.argmax(target_probs).item())
        top_answer = answer_keys[top_answer_idx]
        target_confidence = float(target_probs[top_answer_idx].item())
        target_entropy = float((-(target_probs.clamp_min(1e-12).log() * target_probs).sum()).item())
        target_effective_k = float((1.0 / target_probs.pow(2).sum().clamp_min(1e-12)).item())

        base_top_idx = int(torch.argmax(base_probs).item())
        base_top_answer = answer_keys[base_top_idx]
        base_top_confidence = float(base_probs[base_top_idx].item())
        support_top_idx = int(torch.argmax(support_probs).item())
        support_top_answer = answer_keys[support_top_idx]
        support_top_confidence = float(support_probs[support_top_idx].item())
        support_target_overlap = float(torch.minimum(target_probs, support_probs).sum().item())
        split_top_idx = int(torch.argmax(split_support_probs).item())
        split_top_answer = answer_keys[split_top_idx]
        split_support_top_confidence = float(split_support_probs[split_top_idx].item())
        split_support_target_overlap = float(torch.minimum(target_probs, split_support_probs).sum().item())
        split_support_view_overlap = float(torch.minimum(split_low_probs, split_holdout_probs).sum().item())
        low_budget_support_entropy = float(
            (-(low_budget_support_probs.clamp_min(1e-12).log() * low_budget_support_probs).sum()).item()
        )
        process_support_entropy = float(
            (-(process_support_probs.clamp_min(1e-12).log() * process_support_probs).sum()).item()
        )

        counter = Counter(answers)
        majority_gt, majority_count = counter.most_common(1)[0]
        majority_ratio = float(majority_count) / float(n)
        majority_target_mass = target_prob_by_answer.get(majority_gt, 0.0)
        reward_values = []
        trajectory_logits = None
        trajectory_probs = None
        if per_sample_mode == "trajectory_prob":
            trajectory_scores = []
            trajectory_positions = []
            for answer in answer_keys:
                for j in answer_to_indices[answer]:
                    trajectory_scores.append(
                        flow_scores[start + j]
                        + process_tilt_strength_f * process_tilt_tensor[j]
                        + process_cluster_tilt_strength_f
                        * (answer_to_cluster_tilts.get(answer, 0.0) - cluster_tilt_mean)
                    )
                    trajectory_positions.append(j)
            trajectory_logits = torch.stack(trajectory_scores)
            trajectory_probs = torch.softmax(trajectory_logits - trajectory_logits.max(), dim=0)
            for pos, j in enumerate(trajectory_positions):
                prompt_rewards[j] = float(trajectory_probs[pos].item())
        else:
            for answer, indices in answer_to_indices.items():
                answer_mass = target_prob_by_answer.get(answer, 0.0)
                if per_sample_mode in ("answer_mass", "mass_div_count"):
                    value = answer_mass / float(max(len(indices), 1))
                elif per_sample_mode == "answer_prob":
                    value = answer_mass
                else:
                    raise ValueError(f"Unknown direct sharpened per_sample_mode: {per_sample_mode}")
                for j in indices:
                    prompt_rewards[j] = value

        scale = float(reward_scale)
        floor = max(0.0, float(reward_floor))
        for j, value in enumerate(prompt_rewards):
            if value > 0.0 and floor > 0.0:
                value = max(value, floor)
            value *= scale
            prompt_rewards[j] = value
            reward_values.append(value)
        rewards_flat.extend(prompt_rewards)

        low_budget_infos = rollout_infos[:low_k]
        low_budget_parseable = [1.0 if info.get("parseable") else 0.0 for info in low_budget_infos]
        low_budget_top_mass = float(
            sum(1 for info in low_budget_infos if info.get("answer") == top_answer)
        ) / float(low_k)
        process_consistent = [1.0 if info.get("consistent") else 0.0 for info in rollout_infos]
        top_process_support = float(
            sum(1 for info in rollout_infos if info.get("consistent") and info.get("answer") == top_answer)
        ) / float(max(counter.get(top_answer, 0), 1))
        target_top_process_cluster_support = float(
            sum(1 for info in rollout_infos if info.get("consistent") and info.get("answer") == top_answer)
        ) / float(max(len(answer_to_indices.get(top_answer, [])), 1))
        rewards_np = np.array(reward_values, dtype=float)

        majority_ratio_list.append(majority_ratio)
        direct_target_confidence_list.append(target_confidence)
        direct_target_entropy_list.append(target_entropy)
        direct_target_effective_k_list.append(target_effective_k)
        direct_target_logz_list.append(float(answer_logz.item()))
        direct_unique_answer_count_list.append(float(len(answer_keys)))
        direct_parseable_rate_list.append(float(len(answers)) / float(n))
        direct_clip_rate_list.append(float(np.mean(clipped_flags)) if clipped_flags else 0.0)
        direct_majority_target_mass_list.append(majority_target_mass)
        direct_majority_agreement_list.append(float(top_answer == majority_gt))
        direct_base_top_confidence_list.append(base_top_confidence)
        direct_base_agreement_list.append(float(base_top_answer == top_answer))
        direct_low_budget_parseable_rate_list.append(float(np.mean(low_budget_parseable)))
        direct_low_budget_clip_rate_list.append(float(np.mean(clipped_flags[:low_k])) if clipped_flags else 0.0)
        direct_low_budget_top_mass_list.append(low_budget_top_mass)
        direct_process_consistent_rate_list.append(float(np.mean(process_consistent)))
        direct_process_top_support_list.append(top_process_support)
        direct_process_tilt_mean_list.append(float(np.mean(process_tilts)))
        direct_process_tilt_std_list.append(float(np.std(process_tilts)))
        direct_process_cluster_tilt_mean_list.append(cluster_tilt_mean)
        direct_process_cluster_tilt_std_list.append(cluster_tilt_std)
        direct_process_target_top_support_list.append(target_top_process_cluster_support)
        direct_base_contrast_penalty_mean_list.append(float(base_contrast_penalty.mean().item()))
        direct_base_contrast_penalty_std_list.append(float(base_contrast_penalty.std(unbiased=False).item()))
        direct_base_contrast_gate_mean_list.append(float(base_contrast_gate.mean().item()))
        direct_base_contrast_gate_std_list.append(float(base_contrast_gate.std(unbiased=False).item()))
        direct_effective_k_before_band_list.append(effective_k_before_band)
        direct_effective_k_band_alpha_list.append(float(band_alpha))
        direct_effective_k_band_direction_list.append(float(band_direction))
        direct_pairwise_process_preference_mean_list.append(float(pairwise_preference.mean().item()))
        direct_pairwise_process_preference_std_list.append(float(pairwise_preference.std(unbiased=False).item()))
        direct_pairwise_process_top_preference_list.append(
            float(pairwise_preference[top_answer_idx].item()) if len(pairwise_preference) else 0.0
        )
        direct_support_gate_strength_list.append(support_gate_strength_f)
        direct_support_gate_top_confidence_list.append(support_top_confidence)
        direct_support_gate_target_overlap_list.append(support_target_overlap)
        direct_support_gate_target_agreement_list.append(float(support_top_answer == top_answer))
        direct_support_gate_low_budget_entropy_list.append(low_budget_support_entropy)
        direct_support_gate_process_entropy_list.append(process_support_entropy)
        direct_support_mixture_alpha_list.append(float(mixture_alpha))
        direct_support_mixture_pre_confidence_list.append(pre_mixture_confidence)
        direct_support_mixture_pre_effective_k_list.append(pre_mixture_effective_k)
        direct_support_mixture_pre_overlap_list.append(pre_mixture_overlap)
        direct_support_confidence_cap_alpha_list.append(float(confidence_cap_alpha))
        direct_support_confidence_cap_value_list.append(float(confidence_cap_value))
        direct_support_confidence_cap_pre_confidence_list.append(float(pre_cap_confidence))
        direct_support_residual_strength_list.append(float(support_residual_strength_f))
        direct_support_residual_mean_list.append(float(support_residual.mean().item()))
        direct_support_residual_std_list.append(float(support_residual.std(unbiased=False).item()))
        direct_support_residual_top_value_list.append(
            float(support_residual[top_answer_idx].item()) if len(support_residual) else 0.0
        )
        direct_split_support_strength_list.append(float(split_support_strength_f))
        direct_split_support_top_confidence_list.append(split_support_top_confidence)
        direct_split_support_target_overlap_list.append(split_support_target_overlap)
        direct_split_support_target_agreement_list.append(float(split_top_answer == top_answer))
        direct_split_support_view_overlap_list.append(split_support_view_overlap)
        direct_capacity_brake_alpha_list.append(float(capacity_brake_alpha))
        direct_capacity_brake_pre_confidence_list.append(capacity_brake_pre_confidence)
        direct_capacity_brake_pre_effective_k_list.append(capacity_brake_pre_effective_k)
        direct_capacity_brake_pre_overlap_list.append(capacity_brake_pre_overlap)
        direct_capacity_brake_post_effective_k_list.append(capacity_brake_post_effective_k)
        direct_basin_contrast_alpha_list.append(float(basin_contrast_alpha))
        direct_basin_contrast_target_margin_list.append(float(basin_contrast_target_margin))
        direct_basin_contrast_support_margin_list.append(float(basin_contrast_support_margin))
        direct_basin_contrast_base_margin_list.append(float(basin_contrast_base_margin))
        direct_basin_contrast_low_budget_margin_list.append(float(basin_contrast_low_budget_margin))
        direct_basin_contrast_process_margin_list.append(float(basin_contrast_process_margin))
        direct_basin_contrast_post_effective_k_list.append(float(basin_contrast_post_effective_k))
        direct_reward_mean_list.append(float(rewards_np.mean()) if rewards_np.size else 0.0)
        direct_reward_std_list.append(float(rewards_np.std()) if rewards_np.size else 0.0)
        direct_reward_nonzero_rate_list.append(float((rewards_np > 0.0).mean()) if rewards_np.size else 0.0)

    reward_tensor = torch.zeros_like(response_mask, dtype=torch.float32)
    reward_values = torch.as_tensor(rewards_flat, dtype=torch.float32, device=reward_tensor.device)
    valid_lens = response_mask.sum(dim=-1).long().clamp(min=1)
    reward_tensor.scatter_(1, (valid_lens - 1).unsqueeze(-1), reward_values.unsqueeze(-1))

    batch.non_tensor_batch["majority_ratio_list"] = np.array(majority_ratio_list, dtype=float)
    batch.non_tensor_batch["sps_direct_target_confidence_list"] = np.array(
        direct_target_confidence_list, dtype=float
    )
    batch.non_tensor_batch["sps_direct_target_entropy_list"] = np.array(
        direct_target_entropy_list, dtype=float
    )
    batch.non_tensor_batch["sps_direct_target_effective_k_list"] = np.array(
        direct_target_effective_k_list, dtype=float
    )
    batch.non_tensor_batch["sps_direct_target_logz_list"] = np.array(
        direct_target_logz_list, dtype=float
    )
    batch.non_tensor_batch["sps_direct_unique_answer_count_list"] = np.array(
        direct_unique_answer_count_list, dtype=float
    )
    batch.non_tensor_batch["sps_direct_parseable_rate_list"] = np.array(
        direct_parseable_rate_list, dtype=float
    )
    batch.non_tensor_batch["sps_direct_clip_rate_list"] = np.array(direct_clip_rate_list, dtype=float)
    batch.non_tensor_batch["sps_direct_majority_target_mass_list"] = np.array(
        direct_majority_target_mass_list, dtype=float
    )
    batch.non_tensor_batch["sps_direct_majority_agreement_list"] = np.array(
        direct_majority_agreement_list, dtype=float
    )
    batch.non_tensor_batch["sps_direct_base_top_confidence_list"] = np.array(
        direct_base_top_confidence_list, dtype=float
    )
    batch.non_tensor_batch["sps_direct_base_agreement_list"] = np.array(
        direct_base_agreement_list, dtype=float
    )
    batch.non_tensor_batch["sps_direct_low_budget_parseable_rate_list"] = np.array(
        direct_low_budget_parseable_rate_list, dtype=float
    )
    batch.non_tensor_batch["sps_direct_low_budget_clip_rate_list"] = np.array(
        direct_low_budget_clip_rate_list, dtype=float
    )
    batch.non_tensor_batch["sps_direct_low_budget_top_mass_list"] = np.array(
        direct_low_budget_top_mass_list, dtype=float
    )
    batch.non_tensor_batch["sps_direct_process_consistent_rate_list"] = np.array(
        direct_process_consistent_rate_list, dtype=float
    )
    batch.non_tensor_batch["sps_direct_process_top_support_list"] = np.array(
        direct_process_top_support_list, dtype=float
    )
    batch.non_tensor_batch["sps_direct_process_tilt_mean_list"] = np.array(
        direct_process_tilt_mean_list, dtype=float
    )
    batch.non_tensor_batch["sps_direct_process_tilt_std_list"] = np.array(
        direct_process_tilt_std_list, dtype=float
    )
    batch.non_tensor_batch["sps_direct_process_cluster_tilt_mean_list"] = np.array(
        direct_process_cluster_tilt_mean_list, dtype=float
    )
    batch.non_tensor_batch["sps_direct_process_cluster_tilt_std_list"] = np.array(
        direct_process_cluster_tilt_std_list, dtype=float
    )
    batch.non_tensor_batch["sps_direct_process_target_top_support_list"] = np.array(
        direct_process_target_top_support_list, dtype=float
    )
    batch.non_tensor_batch["sps_direct_base_contrast_penalty_mean_list"] = np.array(
        direct_base_contrast_penalty_mean_list, dtype=float
    )
    batch.non_tensor_batch["sps_direct_base_contrast_penalty_std_list"] = np.array(
        direct_base_contrast_penalty_std_list, dtype=float
    )
    batch.non_tensor_batch["sps_direct_base_contrast_gate_mean_list"] = np.array(
        direct_base_contrast_gate_mean_list, dtype=float
    )
    batch.non_tensor_batch["sps_direct_base_contrast_gate_std_list"] = np.array(
        direct_base_contrast_gate_std_list, dtype=float
    )
    batch.non_tensor_batch["sps_direct_effective_k_before_band_list"] = np.array(
        direct_effective_k_before_band_list, dtype=float
    )
    batch.non_tensor_batch["sps_direct_effective_k_band_alpha_list"] = np.array(
        direct_effective_k_band_alpha_list, dtype=float
    )
    batch.non_tensor_batch["sps_direct_effective_k_band_direction_list"] = np.array(
        direct_effective_k_band_direction_list, dtype=float
    )
    batch.non_tensor_batch["sps_direct_pairwise_process_preference_mean_list"] = np.array(
        direct_pairwise_process_preference_mean_list, dtype=float
    )
    batch.non_tensor_batch["sps_direct_pairwise_process_preference_std_list"] = np.array(
        direct_pairwise_process_preference_std_list, dtype=float
    )
    batch.non_tensor_batch["sps_direct_pairwise_process_top_preference_list"] = np.array(
        direct_pairwise_process_top_preference_list, dtype=float
    )
    batch.non_tensor_batch["sps_direct_support_gate_strength_list"] = np.array(
        direct_support_gate_strength_list, dtype=float
    )
    batch.non_tensor_batch["sps_direct_support_gate_top_confidence_list"] = np.array(
        direct_support_gate_top_confidence_list, dtype=float
    )
    batch.non_tensor_batch["sps_direct_support_gate_target_overlap_list"] = np.array(
        direct_support_gate_target_overlap_list, dtype=float
    )
    batch.non_tensor_batch["sps_direct_support_gate_target_agreement_list"] = np.array(
        direct_support_gate_target_agreement_list, dtype=float
    )
    batch.non_tensor_batch["sps_direct_support_gate_low_budget_entropy_list"] = np.array(
        direct_support_gate_low_budget_entropy_list, dtype=float
    )
    batch.non_tensor_batch["sps_direct_support_gate_process_entropy_list"] = np.array(
        direct_support_gate_process_entropy_list, dtype=float
    )
    batch.non_tensor_batch["sps_direct_support_mixture_alpha_list"] = np.array(
        direct_support_mixture_alpha_list, dtype=float
    )
    batch.non_tensor_batch["sps_direct_support_mixture_pre_confidence_list"] = np.array(
        direct_support_mixture_pre_confidence_list, dtype=float
    )
    batch.non_tensor_batch["sps_direct_support_mixture_pre_effective_k_list"] = np.array(
        direct_support_mixture_pre_effective_k_list, dtype=float
    )
    batch.non_tensor_batch["sps_direct_support_mixture_pre_overlap_list"] = np.array(
        direct_support_mixture_pre_overlap_list, dtype=float
    )
    batch.non_tensor_batch["sps_direct_support_confidence_cap_alpha_list"] = np.array(
        direct_support_confidence_cap_alpha_list, dtype=float
    )
    batch.non_tensor_batch["sps_direct_support_confidence_cap_value_list"] = np.array(
        direct_support_confidence_cap_value_list, dtype=float
    )
    batch.non_tensor_batch["sps_direct_support_confidence_cap_pre_confidence_list"] = np.array(
        direct_support_confidence_cap_pre_confidence_list, dtype=float
    )
    batch.non_tensor_batch["sps_direct_support_residual_strength_list"] = np.array(
        direct_support_residual_strength_list, dtype=float
    )
    batch.non_tensor_batch["sps_direct_support_residual_mean_list"] = np.array(
        direct_support_residual_mean_list, dtype=float
    )
    batch.non_tensor_batch["sps_direct_support_residual_std_list"] = np.array(
        direct_support_residual_std_list, dtype=float
    )
    batch.non_tensor_batch["sps_direct_support_residual_top_value_list"] = np.array(
        direct_support_residual_top_value_list, dtype=float
    )
    batch.non_tensor_batch["sps_direct_split_support_strength_list"] = np.array(
        direct_split_support_strength_list, dtype=float
    )
    batch.non_tensor_batch["sps_direct_split_support_top_confidence_list"] = np.array(
        direct_split_support_top_confidence_list, dtype=float
    )
    batch.non_tensor_batch["sps_direct_split_support_target_overlap_list"] = np.array(
        direct_split_support_target_overlap_list, dtype=float
    )
    batch.non_tensor_batch["sps_direct_split_support_target_agreement_list"] = np.array(
        direct_split_support_target_agreement_list, dtype=float
    )
    batch.non_tensor_batch["sps_direct_split_support_view_overlap_list"] = np.array(
        direct_split_support_view_overlap_list, dtype=float
    )
    batch.non_tensor_batch["sps_direct_capacity_brake_alpha_list"] = np.array(
        direct_capacity_brake_alpha_list, dtype=float
    )
    batch.non_tensor_batch["sps_direct_capacity_brake_pre_confidence_list"] = np.array(
        direct_capacity_brake_pre_confidence_list, dtype=float
    )
    batch.non_tensor_batch["sps_direct_capacity_brake_pre_effective_k_list"] = np.array(
        direct_capacity_brake_pre_effective_k_list, dtype=float
    )
    batch.non_tensor_batch["sps_direct_capacity_brake_pre_overlap_list"] = np.array(
        direct_capacity_brake_pre_overlap_list, dtype=float
    )
    batch.non_tensor_batch["sps_direct_capacity_brake_post_effective_k_list"] = np.array(
        direct_capacity_brake_post_effective_k_list, dtype=float
    )
    batch.non_tensor_batch["sps_direct_basin_contrast_alpha_list"] = np.array(
        direct_basin_contrast_alpha_list, dtype=float
    )
    batch.non_tensor_batch["sps_direct_basin_contrast_target_margin_list"] = np.array(
        direct_basin_contrast_target_margin_list, dtype=float
    )
    batch.non_tensor_batch["sps_direct_basin_contrast_support_margin_list"] = np.array(
        direct_basin_contrast_support_margin_list, dtype=float
    )
    batch.non_tensor_batch["sps_direct_basin_contrast_base_margin_list"] = np.array(
        direct_basin_contrast_base_margin_list, dtype=float
    )
    batch.non_tensor_batch["sps_direct_basin_contrast_low_budget_margin_list"] = np.array(
        direct_basin_contrast_low_budget_margin_list, dtype=float
    )
    batch.non_tensor_batch["sps_direct_basin_contrast_process_margin_list"] = np.array(
        direct_basin_contrast_process_margin_list, dtype=float
    )
    batch.non_tensor_batch["sps_direct_basin_contrast_post_effective_k_list"] = np.array(
        direct_basin_contrast_post_effective_k_list, dtype=float
    )
    batch.non_tensor_batch["sps_direct_reward_mean_list"] = np.array(direct_reward_mean_list, dtype=float)
    batch.non_tensor_batch["sps_direct_reward_std_list"] = np.array(direct_reward_std_list, dtype=float)
    batch.non_tensor_batch["sps_direct_reward_nonzero_rate_list"] = np.array(
        direct_reward_nonzero_rate_list, dtype=float
    )

    info = {
        "sps/reward_mode": float(mode_id),
        "sps/direct_beta": float(beta),
        "sps/direct_target_temperature": float(target_temperature),
        "sps/direct_reward_scale": float(reward_scale),
        "sps/direct_process_tilt_strength": float(process_tilt_strength),
        "sps/direct_process_cluster_tilt_strength": float(process_cluster_tilt_strength),
        "sps/direct_base_contrast_strength": float(base_contrast_strength),
        "sps/direct_base_contrast_gate_strength": float(base_contrast_gate_strength),
        "sps/direct_base_contrast_process_margin": float(base_contrast_process_margin),
        "sps/direct_target_effective_k_min": float(target_effective_k_min),
        "sps/direct_target_effective_k_max": float(target_effective_k_max),
        "sps/direct_target_effective_k_strength": float(target_effective_k_strength),
        "sps/direct_count_neutral_aggregation": float(bool(count_neutral_aggregation)),
        "sps/direct_pairwise_process_preference_strength": float(
            pairwise_process_preference_strength
        ),
        "sps/direct_pairwise_process_preference_temperature": float(
            pairwise_process_preference_temperature
        ),
        "sps/direct_support_gate_strength": float(support_gate_strength),
        "sps/direct_support_gate_temperature": float(support_gate_temperature),
        "sps/direct_support_gate_floor": float(support_gate_floor),
        "sps/direct_support_gate_base_weight": float(support_gate_base_weight),
        "sps/direct_support_gate_low_budget_weight": float(support_gate_low_budget_weight),
        "sps/direct_support_gate_process_weight": float(support_gate_process_weight),
        "sps/direct_support_mixture_strength": float(support_mixture_strength),
        "sps/direct_support_mixture_confidence_threshold": float(
            support_mixture_confidence_threshold
        ),
        "sps/direct_support_mixture_effective_k_threshold": float(
            support_mixture_effective_k_threshold
        ),
        "sps/direct_support_mixture_overlap_threshold": float(
            support_mixture_overlap_threshold
        ),
        "sps/direct_support_confidence_cap_strength": float(support_confidence_cap_strength),
        "sps/direct_support_confidence_cap_margin": float(support_confidence_cap_margin),
        "sps/direct_support_confidence_cap_min": float(support_confidence_cap_min),
        "sps/direct_support_residual_strength": float(support_residual_strength),
        "sps/direct_split_support_strength": float(split_support_strength),
        "sps/direct_split_support_floor": float(split_support_floor),
        "sps/direct_capacity_brake_strength": float(capacity_brake_strength),
        "sps/direct_capacity_brake_min_effective_K": float(capacity_brake_min_effective_k),
        "sps/direct_capacity_brake_confidence_threshold": float(
            capacity_brake_confidence_threshold
        ),
        "sps/direct_capacity_brake_overlap_threshold": float(capacity_brake_overlap_threshold),
        "sps/direct_basin_contrast_strength": float(basin_contrast_strength),
        "sps/direct_basin_contrast_margin": float(basin_contrast_margin),
        "sps/direct_target_confidence": float(np.mean(direct_target_confidence_list)),
        "sps/direct_target_entropy": float(np.mean(direct_target_entropy_list)),
        "sps/direct_target_effective_K": float(np.mean(direct_target_effective_k_list)),
        "sps/direct_target_logz": float(np.mean(direct_target_logz_list)),
        "sps/direct_unique_answer_count": float(np.mean(direct_unique_answer_count_list)),
        "sps/direct_parseable_rate": float(np.mean(direct_parseable_rate_list)),
        "sps/direct_clip_rate": float(np.mean(direct_clip_rate_list)),
        "sps/direct_majority_target_mass": float(np.mean(direct_majority_target_mass_list)),
        "sps/direct_majority_agreement": float(np.mean(direct_majority_agreement_list)),
        "sps/direct_base_top_confidence": float(np.mean(direct_base_top_confidence_list)),
        "sps/direct_base_agreement": float(np.mean(direct_base_agreement_list)),
        "sps/direct_low_budget_parseable_rate": float(np.mean(direct_low_budget_parseable_rate_list)),
        "sps/direct_low_budget_clip_rate": float(np.mean(direct_low_budget_clip_rate_list)),
        "sps/direct_low_budget_top_mass": float(np.mean(direct_low_budget_top_mass_list)),
        "sps/direct_process_consistent_rate": float(np.mean(direct_process_consistent_rate_list)),
        "sps/direct_process_top_support": float(np.mean(direct_process_top_support_list)),
        "sps/direct_process_tilt_mean": float(np.mean(direct_process_tilt_mean_list)),
        "sps/direct_process_tilt_std": float(np.mean(direct_process_tilt_std_list)),
        "sps/direct_process_cluster_tilt_mean": float(np.mean(direct_process_cluster_tilt_mean_list)),
        "sps/direct_process_cluster_tilt_std": float(np.mean(direct_process_cluster_tilt_std_list)),
        "sps/direct_process_target_top_support": float(
            np.mean(direct_process_target_top_support_list)
        ),
        "sps/direct_base_contrast_penalty_mean": float(
            np.mean(direct_base_contrast_penalty_mean_list)
        ),
        "sps/direct_base_contrast_penalty_std": float(
            np.mean(direct_base_contrast_penalty_std_list)
        ),
        "sps/direct_base_contrast_gate_mean": float(
            np.mean(direct_base_contrast_gate_mean_list)
        ),
        "sps/direct_base_contrast_gate_std": float(
            np.mean(direct_base_contrast_gate_std_list)
        ),
        "sps/direct_effective_K_before_band": float(
            np.mean(direct_effective_k_before_band_list)
        ),
        "sps/direct_effective_K_band_alpha": float(
            np.mean(direct_effective_k_band_alpha_list)
        ),
        "sps/direct_effective_K_band_direction": float(
            np.mean(direct_effective_k_band_direction_list)
        ),
        "sps/direct_pairwise_process_preference_mean": float(
            np.mean(direct_pairwise_process_preference_mean_list)
        ),
        "sps/direct_pairwise_process_preference_std": float(
            np.mean(direct_pairwise_process_preference_std_list)
        ),
        "sps/direct_pairwise_process_top_preference": float(
            np.mean(direct_pairwise_process_top_preference_list)
        ),
        "sps/direct_support_gate_strength_mean": float(
            np.mean(direct_support_gate_strength_list)
        ),
        "sps/direct_support_gate_top_confidence": float(
            np.mean(direct_support_gate_top_confidence_list)
        ),
        "sps/direct_support_gate_target_overlap": float(
            np.mean(direct_support_gate_target_overlap_list)
        ),
        "sps/direct_support_gate_target_agreement": float(
            np.mean(direct_support_gate_target_agreement_list)
        ),
        "sps/direct_support_gate_low_budget_entropy": float(
            np.mean(direct_support_gate_low_budget_entropy_list)
        ),
        "sps/direct_support_gate_process_entropy": float(
            np.mean(direct_support_gate_process_entropy_list)
        ),
        "sps/direct_support_mixture_alpha": float(np.mean(direct_support_mixture_alpha_list)),
        "sps/direct_support_mixture_pre_confidence": float(
            np.mean(direct_support_mixture_pre_confidence_list)
        ),
        "sps/direct_support_mixture_pre_effective_K": float(
            np.mean(direct_support_mixture_pre_effective_k_list)
        ),
        "sps/direct_support_mixture_pre_overlap": float(
            np.mean(direct_support_mixture_pre_overlap_list)
        ),
        "sps/direct_support_confidence_cap_alpha": float(
            np.mean(direct_support_confidence_cap_alpha_list)
        ),
        "sps/direct_support_confidence_cap_value": float(
            np.mean(direct_support_confidence_cap_value_list)
        ),
        "sps/direct_support_confidence_cap_pre_confidence": float(
            np.mean(direct_support_confidence_cap_pre_confidence_list)
        ),
        "sps/direct_support_residual_strength_mean": float(
            np.mean(direct_support_residual_strength_list)
        ),
        "sps/direct_support_residual_mean": float(np.mean(direct_support_residual_mean_list)),
        "sps/direct_support_residual_std": float(np.mean(direct_support_residual_std_list)),
        "sps/direct_support_residual_top_value": float(
            np.mean(direct_support_residual_top_value_list)
        ),
        "sps/direct_split_support_strength_mean": float(np.mean(direct_split_support_strength_list)),
        "sps/direct_split_support_top_confidence": float(
            np.mean(direct_split_support_top_confidence_list)
        ),
        "sps/direct_split_support_target_overlap": float(
            np.mean(direct_split_support_target_overlap_list)
        ),
        "sps/direct_split_support_target_agreement": float(
            np.mean(direct_split_support_target_agreement_list)
        ),
        "sps/direct_split_support_view_overlap": float(np.mean(direct_split_support_view_overlap_list)),
        "sps/direct_capacity_brake_alpha": float(np.mean(direct_capacity_brake_alpha_list)),
        "sps/direct_capacity_brake_pre_confidence": float(
            np.mean(direct_capacity_brake_pre_confidence_list)
        ),
        "sps/direct_capacity_brake_pre_effective_K": float(
            np.mean(direct_capacity_brake_pre_effective_k_list)
        ),
        "sps/direct_capacity_brake_pre_overlap": float(
            np.mean(direct_capacity_brake_pre_overlap_list)
        ),
        "sps/direct_capacity_brake_post_effective_K": float(
            np.mean(direct_capacity_brake_post_effective_k_list)
        ),
        "sps/direct_basin_contrast_alpha": float(np.mean(direct_basin_contrast_alpha_list)),
        "sps/direct_basin_contrast_target_margin": float(
            np.mean(direct_basin_contrast_target_margin_list)
        ),
        "sps/direct_basin_contrast_support_margin": float(
            np.mean(direct_basin_contrast_support_margin_list)
        ),
        "sps/direct_basin_contrast_base_margin": float(
            np.mean(direct_basin_contrast_base_margin_list)
        ),
        "sps/direct_basin_contrast_low_budget_margin": float(
            np.mean(direct_basin_contrast_low_budget_margin_list)
        ),
        "sps/direct_basin_contrast_process_margin": float(
            np.mean(direct_basin_contrast_process_margin_list)
        ),
        "sps/direct_basin_contrast_post_effective_K": float(
            np.mean(direct_basin_contrast_post_effective_k_list)
        ),
        "sps/direct_reward_mean": float(np.mean(direct_reward_mean_list)),
        "sps/direct_reward_std": float(np.mean(direct_reward_std_list)),
        "sps/direct_reward_nonzero_rate": float(np.mean(direct_reward_nonzero_rate_list)),
    }
    return batch, reward_tensor, info


# === Ground Truth Manipulation ===


def apply_original_gt(batch):
    """
    Apply the original ground truth to the batch.
    """
    for i in range(len(batch)):
        data_item = batch[i]
        original_gt = data_item.non_tensor_batch["reward_model"]["original_gt"]
        data_item.non_tensor_batch["reward_model"]["ground_truth"] = original_gt

    return batch


def apply_ttrl_gt(batch, gen_batch_output, n, tokenizer):
    """
    Apply the majority vote ground truth to the batch.
    """
    assert len(gen_batch_output) % n == 0, "gen_batch_output length must be divisible by n"
    num_prompts = len(gen_batch_output) // n
    assert len(batch) == num_prompts, "batch length must be equal to the number of prompts"

    model_outputs = []  
    for i in range(num_prompts):
        start = i * n
        for j in range(n):
            data_item = gen_batch_output[start + j]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            model_outputs.append(response_str)

    majority_gt_list, majority_ratio_list = _batch_majority_vote(model_outputs, n)
    
    assert len(batch) == len(majority_gt_list), "batch length must be equal to the number of model outputs"
    
    for i in range(num_prompts):
        data_item = batch[i]
        original_gt = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        data_item.non_tensor_batch["reward_model"]["ground_truth"] = majority_gt_list[i]
        data_item.non_tensor_batch["reward_model"]["majority_gt"] = majority_gt_list[i]
        data_item.non_tensor_batch["reward_model"]["original_gt"] = original_gt

    batch.non_tensor_batch["majority_ratio_list"] = np.array(majority_ratio_list, dtype=float)
    return batch


def apply_sps_weighted_ttrl_gt(
    batch,
    gen_batch_output,
    n,
    tokenizer,
    ref_log_prob,
    rollout_log_probs,
    response_mask,
    alpha,
    length_normalize=True,
    weight_temperature=1.0,
    use_majority_fallback=False,
    gate_confidence_threshold=0.8,
    gate_majority_ratio_threshold=0.75,
    confidence_filter=False,
    confidence_weight=False,
    filter_confidence_threshold=0.8,
    filter_majority_ratio_threshold=0.75,
    weight_floor=0.25,
    clip_penalty=0.0,
    weight_power=1.0,
    answer_sharpen_beta=1.0,
    answer_sharpen_capacity=False,
    consistency_capacity=False,
    consistency_disagreement_penalty=0.5,
    low_budget_capacity=False,
    low_budget_k=4,
    low_budget_disagreement_penalty=0.35,
    base_support_capacity=False,
    base_support_temperature=1.0,
    base_support_disagreement_penalty=0.35,
    cross_view_capacity=False,
    cross_view_disagreement_penalty=0.35,
    margin_capacity=False,
    margin_capacity_floor=0.35,
    process_consistency_capacity=False,
    process_tail_fraction=0.5,
    process_disagreement_penalty=0.35,
    process_sample_weight=False,
    process_sample_positive=1.25,
    process_sample_inconsistent=0.75,
    process_sample_negative=0.35,
    process_sample_invalid=0.25,
    low_budget_negative_reward=False,
    low_budget_negative_value=0.35,
    low_budget_negative_min_support=0.65,
    low_budget_negative_process_min=0.70,
    low_budget_negative_base_min=0.65,
    low_budget_soft_reward=False,
    low_budget_soft_value=0.25,
    low_budget_rescue_reward=False,
    low_budget_rescue_value=0.15,
    low_budget_rescue_min_base=0.08,
    low_budget_rescue_min_process=0.50,
    low_budget_rescue_min_cluster_mass=0.03125,
    low_budget_rescue_min_support=0.15,
    low_budget_local_reward=False,
    low_budget_local_value=0.12,
    low_budget_local_min_mass=0.50,
    low_budget_local_min_base=0.08,
    low_budget_local_min_process=0.50,
    low_budget_local_min_support=0.15,
    independent_support_reward=False,
    independent_support_value=0.18,
    independent_support_min_base=0.12,
    independent_support_min_process=0.65,
    independent_support_min_cluster_mass=0.0625,
    independent_support_min_support=0.20,
    independent_support_entropy_gate=1.15,
    independent_support_allow_majority=True,
    independent_support_majority_margin_max=1.0,
    independent_support_competition_min=0.0,
    process_answer_reward=False,
    process_answer_value=0.16,
    process_answer_min_base=0.10,
    process_answer_min_process=0.60,
    process_answer_min_support=0.20,
    process_answer_majority_scale=0.50,
    contrastive_alt_reward=False,
    contrastive_alt_value=0.14,
    contrastive_alt_majority_penalty=0.06,
    contrastive_alt_min_base=0.08,
    contrastive_alt_min_process=0.60,
    contrastive_alt_min_support=0.18,
    contrastive_alt_sharp_min=0.92,
    contrastive_alt_effective_k_max=1.35,
    contrastive_alt_low_budget_mass_min=0.70,
    contrastive_alt_majority_support_min=0.70,
    entropy_band_capacity=False,
    entropy_band_min_weight=0.45,
    entropy_band_low_k=1.25,
    entropy_band_high_k=4.0,
    entropy_band_sharp_min=0.92,
    entropy_band_low_budget_mass_min=0.70,
    entropy_band_support_min=0.70,
    low_budget_rebalance_reward=False,
    low_budget_rebalance_alt_value=0.10,
    low_budget_rebalance_majority_penalty=0.04,
    low_budget_rebalance_sharp_min=0.92,
    low_budget_rebalance_effective_k_max=1.35,
    low_budget_rebalance_mass_min=0.70,
    low_budget_rebalance_support_min=0.70,
    low_budget_rebalance_min_base=0.06,
    low_budget_rebalance_min_process=0.55,
    low_budget_rebalance_min_support=0.12,
    process_quality_reward=False,
    process_quality_positive=0.08,
    process_quality_negative=0.10,
    process_quality_sharp_min=0.92,
    process_quality_effective_k_max=1.35,
    process_quality_mass_min=0.70,
    process_quality_majority_scale=0.25,
    process_quality_nonmajority_scale=1.0,
    process_quality_bad_majority_scale=1.0,
    process_quality_bad_nonmajority_scale=0.5,
    anti_collapse_capacity=False,
    anti_collapse_min_weight=0.45,
    anti_collapse_sharp_min=0.95,
    anti_collapse_effective_k_max=1.25,
    anti_collapse_low_budget_mass_min=0.75,
    anti_collapse_margin_gap_min=0.15,
    support_conflict_capacity=False,
    support_conflict_min_weight=0.15,
    support_conflict_sharp_min=0.90,
    support_conflict_effective_k_max=2.0,
    support_conflict_base_min=0.55,
    support_conflict_low_budget_min=0.50,
    support_conflict_process_min=0.70,
):
    """
    Apply an SPS-weighted self-consistency pseudo label to the batch.

    For each prompt, rollouts are clustered by extracted final answer. The
    pseudo label is the answer with the largest logsumexp SPS weight:
        score_i = alpha * logp_base(y_i|x) - logq(y_i|x)
        S(answer) = logsumexp(score_i / temperature for i in answer cluster)
    This stays unsupervised: original ground truth is only saved for diagnostics.
    """
    assert len(gen_batch_output) % n == 0, "gen_batch_output length must be divisible by n"
    num_prompts = len(gen_batch_output) // n
    assert len(batch) == num_prompts, "batch length must be equal to the number of prompts"
    assert ref_log_prob.shape == rollout_log_probs.shape == response_mask.shape
    assert ref_log_prob.shape[0] == len(gen_batch_output)

    mask_f = response_mask.to(torch.float32)
    base_seq = (ref_log_prob.to(torch.float32) * mask_f).sum(dim=-1)
    q_seq = (rollout_log_probs.to(torch.float32) * mask_f).sum(dim=-1)
    lengths = mask_f.sum(dim=-1).clamp(min=1.0)
    scores = alpha * base_seq - q_seq
    if length_normalize:
        scores = scores / lengths
    scores = scores.detach().cpu()

    selected_gt_list = []
    weighted_gt_list = []
    raw_majority_gt_list = []
    weighted_confidence_list = []
    majority_ratio_list = []
    unique_answer_count_list = []
    sps_override_list = []
    sps_agreement_list = []
    sps_train_weight_list = []
    answer_sharp_confidence_list = []
    answer_entropy_list = []
    answer_effective_k_list = []
    answer_logz_list = []
    majority_sharp_confidence_list = []
    consistency_capacity_list = []
    low_budget_parseable_rate_list = []
    low_budget_clip_rate_list = []
    low_budget_majority_ratio_list = []
    low_budget_majority_mass_list = []
    low_budget_agreement_list = []
    low_budget_capacity_list = []
    base_support_majority_confidence_list = []
    base_support_top_confidence_list = []
    base_support_agreement_list = []
    base_support_capacity_list = []
    cross_view_capacity_list = []
    sharp_majority_margin_list = []
    base_support_majority_margin_list = []
    low_budget_majority_margin_list = []
    margin_capacity_list = []
    process_consistency_capacity_list = []
    process_majority_support_list = []
    process_consistent_rate_list = []
    process_majority_consistent_rate_list = []
    process_tail_rate_list = []
    process_box_conflict_rate_list = []
    process_revision_after_final_rate_list = []
    process_sample_weight_list = []
    process_sample_positive_rate_list = []
    process_sample_inconsistent_rate_list = []
    process_sample_negative_rate_list = []
    process_sample_invalid_rate_list = []
    process_sample_weights_flat = []
    low_budget_negative_reward_flat = []
    low_budget_negative_rate_list = []
    low_budget_negative_gate_list = []
    low_budget_soft_reward_flat = []
    low_budget_soft_reward_mean_list = []
    low_budget_soft_reward_rate_list = []
    low_budget_soft_nonmajority_rate_list = []
    low_budget_soft_nonmajority_reward_list = []
    low_budget_soft_support_effective_k_list = []
    low_budget_rescue_reward_flat = []
    low_budget_rescue_reward_mean_list = []
    low_budget_rescue_rate_list = []
    low_budget_rescue_support_list = []
    low_budget_rescue_cluster_mass_list = []
    low_budget_local_reward_flat = []
    low_budget_local_reward_mean_list = []
    low_budget_local_rate_list = []
    low_budget_local_support_list = []
    low_budget_local_mass_list = []
    low_budget_local_agreement_list = []
    independent_support_reward_flat = []
    independent_support_reward_mean_list = []
    independent_support_rate_list = []
    independent_support_nonmajority_rate_list = []
    independent_support_value_list = []
    independent_support_effective_k_list = []
    independent_support_margin_list = []
    independent_support_competition_rate_list = []
    independent_support_majority_gate_rate_list = []
    process_answer_reward_flat = []
    process_answer_reward_mean_list = []
    process_answer_rate_list = []
    process_answer_nonmajority_rate_list = []
    process_answer_support_list = []
    process_answer_base_list = []
    process_answer_process_list = []
    contrastive_alt_reward_flat = []
    contrastive_alt_reward_mean_list = []
    contrastive_alt_rate_list = []
    contrastive_alt_majority_penalty_rate_list = []
    contrastive_alt_gate_list = []
    contrastive_alt_support_list = []
    contrastive_alt_base_list = []
    contrastive_alt_process_list = []
    entropy_band_capacity_list = []
    entropy_band_low_active_list = []
    entropy_band_high_active_list = []
    entropy_band_score_list = []
    low_budget_rebalance_reward_flat = []
    low_budget_rebalance_reward_mean_list = []
    low_budget_rebalance_alt_rate_list = []
    low_budget_rebalance_majority_penalty_rate_list = []
    low_budget_rebalance_gate_list = []
    low_budget_rebalance_support_list = []
    low_budget_rebalance_base_list = []
    low_budget_rebalance_process_list = []
    process_quality_reward_flat = []
    process_quality_reward_mean_list = []
    process_quality_positive_rate_list = []
    process_quality_negative_rate_list = []
    process_quality_bad_rate_list = []
    process_quality_collapse_gate_list = []
    anti_collapse_capacity_list = []
    anti_collapse_active_list = []
    anti_collapse_margin_gap_list = []
    support_conflict_capacity_list = []
    support_conflict_active_list = []
    support_conflict_score_list = []

    temp = max(float(weight_temperature), 1e-6)
    sharpen_beta = max(float(answer_sharpen_beta), 1e-6)
    base_support_temp = max(float(base_support_temperature), 1e-6)
    max_response_len = response_mask.shape[-1]
    for i in range(num_prompts):
        answer_to_scores = {}
        answer_to_base_scores = {}
        answers = []
        rollout_process_infos = []
        start = i * n
        for j in range(n):
            data_item = gen_batch_output[start + j]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            answer = extract_answer(response_str)
            clipped = bool(lengths[start + j].detach().cpu().item() >= max_response_len)
            process_info = _score_process_consistency(
                response_str,
                answer,
                clipped=clipped,
                tail_fraction=process_tail_fraction,
            )
            process_info["answer"] = simplify_expression_string(answer) if answer is not None else None
            process_info["parseable"] = answer is not None
            process_info["clipped"] = clipped
            rollout_process_infos.append(process_info)
            if answer is None:
                continue
            answer = simplify_expression_string(answer)
            answers.append(answer)
            answer_to_scores.setdefault(answer, []).append(scores[start + j] / temp)
            base_score = base_seq[start + j].detach().cpu()
            if length_normalize:
                base_score = base_score / lengths[start + j].detach().cpu()
            answer_to_base_scores.setdefault(answer, []).append(base_score / base_support_temp)

        if not answer_to_scores:
            selected_gt_list.append("None")
            weighted_gt_list.append("None")
            raw_majority_gt_list.append("None")
            weighted_confidence_list.append(0.0)
            majority_ratio_list.append(0.0)
            unique_answer_count_list.append(0)
            sps_override_list.append(0.0)
            sps_agreement_list.append(0.0)
            sps_train_weight_list.append(0.0)
            answer_sharp_confidence_list.append(0.0)
            answer_entropy_list.append(0.0)
            answer_effective_k_list.append(0.0)
            answer_logz_list.append(0.0)
            majority_sharp_confidence_list.append(0.0)
            consistency_capacity_list.append(0.0)
            low_budget_parseable_rate_list.append(0.0)
            low_budget_clip_rate_list.append(0.0)
            low_budget_majority_ratio_list.append(0.0)
            low_budget_majority_mass_list.append(0.0)
            low_budget_agreement_list.append(0.0)
            low_budget_capacity_list.append(0.0)
            base_support_majority_confidence_list.append(0.0)
            base_support_top_confidence_list.append(0.0)
            base_support_agreement_list.append(0.0)
            base_support_capacity_list.append(0.0)
            cross_view_capacity_list.append(0.0)
            sharp_majority_margin_list.append(0.0)
            base_support_majority_margin_list.append(0.0)
            low_budget_majority_margin_list.append(0.0)
            margin_capacity_list.append(0.0)
            process_consistency_capacity_list.append(0.0)
            process_majority_support_list.append(0.0)
            process_consistent_rate_list.append(0.0)
            process_majority_consistent_rate_list.append(0.0)
            process_tail_rate_list.append(0.0)
            process_box_conflict_rate_list.append(0.0)
            process_revision_after_final_rate_list.append(0.0)
            low_k = max(1, min(int(low_budget_k), n))
            if process_sample_weight:
                empty_weights = [float(process_sample_invalid)] * low_k + [1.0] * (n - low_k)
                process_sample_weights_flat.extend(empty_weights)
                process_sample_weight_list.append(float(np.mean(empty_weights[:low_k])))
                process_sample_invalid_rate_list.append(1.0)
            else:
                process_sample_weights_flat.extend([1.0] * n)
                process_sample_weight_list.append(1.0)
                process_sample_invalid_rate_list.append(0.0)
            process_sample_positive_rate_list.append(0.0)
            process_sample_inconsistent_rate_list.append(0.0)
            process_sample_negative_rate_list.append(0.0)
            if low_budget_negative_reward:
                low_budget_negative_reward_flat.extend([0.0] * n)
            low_budget_negative_rate_list.append(0.0)
            low_budget_negative_gate_list.append(0.0)
            if low_budget_soft_reward:
                low_budget_soft_reward_flat.extend([0.0] * n)
            low_budget_soft_reward_mean_list.append(0.0)
            low_budget_soft_reward_rate_list.append(0.0)
            low_budget_soft_nonmajority_rate_list.append(0.0)
            low_budget_soft_nonmajority_reward_list.append(0.0)
            low_budget_soft_support_effective_k_list.append(0.0)
            if low_budget_rescue_reward:
                low_budget_rescue_reward_flat.extend([0.0] * n)
            low_budget_rescue_reward_mean_list.append(0.0)
            low_budget_rescue_rate_list.append(0.0)
            low_budget_rescue_support_list.append(0.0)
            low_budget_rescue_cluster_mass_list.append(0.0)
            if low_budget_local_reward:
                low_budget_local_reward_flat.extend([0.0] * n)
            low_budget_local_reward_mean_list.append(0.0)
            low_budget_local_rate_list.append(0.0)
            low_budget_local_support_list.append(0.0)
            low_budget_local_mass_list.append(0.0)
            low_budget_local_agreement_list.append(0.0)
            if independent_support_reward:
                independent_support_reward_flat.extend([0.0] * n)
            independent_support_reward_mean_list.append(0.0)
            independent_support_rate_list.append(0.0)
            independent_support_nonmajority_rate_list.append(0.0)
            independent_support_value_list.append(0.0)
            independent_support_effective_k_list.append(0.0)
            independent_support_margin_list.append(0.0)
            independent_support_competition_rate_list.append(0.0)
            independent_support_majority_gate_rate_list.append(0.0)
            if process_answer_reward:
                process_answer_reward_flat.extend([0.0] * n)
            process_answer_reward_mean_list.append(0.0)
            process_answer_rate_list.append(0.0)
            process_answer_nonmajority_rate_list.append(0.0)
            process_answer_support_list.append(0.0)
            process_answer_base_list.append(0.0)
            process_answer_process_list.append(0.0)
            if contrastive_alt_reward:
                contrastive_alt_reward_flat.extend([0.0] * n)
            contrastive_alt_reward_mean_list.append(0.0)
            contrastive_alt_rate_list.append(0.0)
            contrastive_alt_majority_penalty_rate_list.append(0.0)
            contrastive_alt_gate_list.append(0.0)
            contrastive_alt_support_list.append(0.0)
            contrastive_alt_base_list.append(0.0)
            contrastive_alt_process_list.append(0.0)
            if low_budget_rebalance_reward:
                low_budget_rebalance_reward_flat.extend([0.0] * n)
            low_budget_rebalance_reward_mean_list.append(0.0)
            low_budget_rebalance_alt_rate_list.append(0.0)
            low_budget_rebalance_majority_penalty_rate_list.append(0.0)
            low_budget_rebalance_gate_list.append(0.0)
            low_budget_rebalance_support_list.append(0.0)
            low_budget_rebalance_base_list.append(0.0)
            low_budget_rebalance_process_list.append(0.0)
            if process_quality_reward:
                process_quality_reward_flat.extend([0.0] * n)
            process_quality_reward_mean_list.append(0.0)
            process_quality_positive_rate_list.append(0.0)
            process_quality_negative_rate_list.append(0.0)
            process_quality_bad_rate_list.append(0.0)
            process_quality_collapse_gate_list.append(0.0)
            entropy_band_capacity_list.append(1.0)
            entropy_band_low_active_list.append(0.0)
            entropy_band_high_active_list.append(0.0)
            entropy_band_score_list.append(0.0)
            anti_collapse_capacity_list.append(1.0)
            anti_collapse_active_list.append(0.0)
            anti_collapse_margin_gap_list.append(0.0)
            support_conflict_capacity_list.append(1.0)
            support_conflict_active_list.append(0.0)
            support_conflict_score_list.append(0.0)
            continue

        answer_scores = {
            answer: torch.logsumexp(torch.stack(vals), dim=0)
            for answer, vals in answer_to_scores.items()
        }
        base_answer_scores = {
            answer: torch.logsumexp(torch.stack(vals), dim=0)
            for answer, vals in answer_to_base_scores.items()
        }
        base_answer_mean_scores = {
            answer: torch.logsumexp(torch.stack(vals), dim=0) - np.log(max(len(vals), 1))
            for answer, vals in answer_to_base_scores.items()
        }
        weighted_gt = max(answer_scores.items(), key=lambda item: item[1].item())[0]
        answer_keys = list(answer_scores.keys())
        stacked = torch.stack(list(answer_scores.values()))
        probs = torch.softmax(stacked - stacked.max(), dim=0)
        answer_logz = torch.logsumexp(stacked, dim=0)
        sharp_logits = sharpen_beta * (stacked - answer_logz)
        sharp_probs = torch.softmax(sharp_logits, dim=0)

        counter = Counter(answers)
        majority_gt, majority_count = counter.most_common(1)[0]
        majority_ratio = majority_count / n
        weighted_confidence = float(probs.max().item())
        sharp_confidence = float(sharp_probs.max().item())
        sharp_entropy = float((-(sharp_probs.clamp_min(1e-12).log() * sharp_probs).sum()).item())
        sharp_effective_k = float((1.0 / sharp_probs.pow(2).sum().clamp_min(1e-12)).item())
        sharp_prob_by_answer = {
            answer: float(sharp_probs[idx].item())
            for idx, answer in enumerate(answer_keys)
        }
        base_keys = list(base_answer_scores.keys())
        base_stacked = torch.stack([base_answer_scores[answer] for answer in base_keys])
        base_probs = torch.softmax(base_stacked - base_stacked.max(), dim=0)
        base_prob_by_answer = {
            answer: float(base_probs[idx].item())
            for idx, answer in enumerate(base_keys)
        }
        base_mean_keys = list(base_answer_mean_scores.keys())
        base_mean_stacked = torch.stack([base_answer_mean_scores[answer] for answer in base_mean_keys])
        base_mean_probs = torch.softmax(base_mean_stacked - base_mean_stacked.max(), dim=0)
        base_mean_prob_by_answer = {
            answer: float(base_mean_probs[idx].item())
            for idx, answer in enumerate(base_mean_keys)
        }
        majority_sharp_confidence = sharp_prob_by_answer.get(majority_gt, 0.0)
        weighted_sharp_confidence = sharp_prob_by_answer.get(weighted_gt, 0.0)
        sharp_second_confidence = max(
            [prob for answer, prob in sharp_prob_by_answer.items() if answer != majority_gt],
            default=0.0,
        )
        sharp_majority_margin = majority_sharp_confidence - sharp_second_confidence
        base_support_gt = max(base_answer_scores.items(), key=lambda item: item[1].item())[0]
        base_support_top_confidence = max(base_prob_by_answer.values()) if base_prob_by_answer else 0.0
        base_support_majority_confidence = base_prob_by_answer.get(majority_gt, 0.0)
        base_support_second_confidence = max(
            [prob for answer, prob in base_prob_by_answer.items() if answer != majority_gt],
            default=0.0,
        )
        base_support_majority_margin = base_support_majority_confidence - base_support_second_confidence
        base_support_agreement = float(base_support_gt == majority_gt)
        base_support_capacity_value = base_support_majority_confidence
        if base_support_gt != majority_gt:
            base_support_capacity_value *= max(0.0, min(1.0, float(base_support_disagreement_penalty)))
        prompt_lengths = lengths[start : start + n]
        prompt_clip_ratio = float((prompt_lengths >= max_response_len).to(torch.float32).mean().item())
        low_k = max(1, min(int(low_budget_k), n))
        low_budget_answers = []
        low_budget_parseable = 0
        for j in range(low_k):
            data_item = gen_batch_output[start + j]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            answer = extract_answer(response_str)
            if answer is None:
                continue
            low_budget_parseable += 1
            low_budget_answers.append(simplify_expression_string(answer))
        low_counter = Counter(low_budget_answers)
        if low_counter:
            low_budget_gt, low_budget_count = low_counter.most_common(1)[0]
            low_budget_majority_ratio = float(low_budget_count) / float(low_k)
        else:
            low_budget_gt = "None"
            low_budget_majority_ratio = 0.0
        low_budget_majority_mass = float(low_counter.get(majority_gt, 0)) / float(low_k)
        low_budget_second_mass = (
            float(max([count for answer, count in low_counter.items() if answer != majority_gt], default=0))
            / float(low_k)
        )
        low_budget_majority_margin = low_budget_majority_mass - low_budget_second_mass
        low_budget_parseable_rate = float(low_budget_parseable) / float(low_k)
        low_budget_clip_rate = float(
            (prompt_lengths[:low_k] >= max_response_len).to(torch.float32).mean().item()
        )
        low_budget_agreement = float(low_budget_gt == majority_gt)
        low_budget_capacity_value = low_budget_majority_mass
        if low_budget_gt != majority_gt:
            low_budget_capacity_value *= max(0.0, min(1.0, float(low_budget_disagreement_penalty)))
        low_budget_capacity_value *= low_budget_parseable_rate * max(0.0, 1.0 - low_budget_clip_rate)
        cross_view_capacity_value = float(
            np.sqrt(max(base_support_majority_confidence, 0.0) * max(low_budget_majority_mass, 0.0))
        )
        cross_view_capacity_value *= low_budget_parseable_rate * max(0.0, 1.0 - low_budget_clip_rate)
        cross_penalty = max(0.0, min(1.0, float(cross_view_disagreement_penalty)))
        if base_support_gt != majority_gt:
            cross_view_capacity_value *= cross_penalty
        if low_budget_gt != majority_gt:
            cross_view_capacity_value *= cross_penalty
        margin_floor = max(0.0, min(1.0, float(margin_capacity_floor)))
        clipped_margins = [
            max(0.0, min(1.0, float(sharp_majority_margin))),
            max(0.0, min(1.0, float(base_support_majority_margin))),
            max(0.0, min(1.0, float(low_budget_majority_margin))),
        ]
        margin_signal = float(np.prod(clipped_margins) ** (1.0 / 3.0))
        margin_capacity_value = margin_floor + (1.0 - margin_floor) * margin_signal
        margin_capacity_value *= low_budget_parseable_rate * max(0.0, 1.0 - low_budget_clip_rate)
        support_margin = min(
            max(0.0, min(1.0, float(base_support_majority_margin))),
            max(0.0, min(1.0, float(low_budget_majority_margin))),
        )
        anti_collapse_margin_gap = max(0.0, float(sharp_majority_margin) - support_margin)
        anti_collapse_capacity_value = 1.0
        anti_collapse_active = 0.0
        if anti_collapse_capacity:
            sharp_min = max(0.0, min(1.0 - 1e-6, float(anti_collapse_sharp_min)))
            effective_k_max = max(1.0 + 1e-6, float(anti_collapse_effective_k_max))
            low_budget_mass_min = max(0.0, min(1.0 - 1e-6, float(anti_collapse_low_budget_mass_min)))
            margin_gap_min = max(0.0, min(1.0 - 1e-6, float(anti_collapse_margin_gap_min)))
            if (
                sharp_confidence >= sharp_min
                and sharp_effective_k <= effective_k_max
                and low_budget_majority_mass >= low_budget_mass_min
                and anti_collapse_margin_gap >= margin_gap_min
            ):
                sharp_excess = (sharp_confidence - sharp_min) / max(1.0 - sharp_min, 1e-6)
                k_collapse = (effective_k_max - sharp_effective_k) / max(effective_k_max - 1.0, 1e-6)
                mass_excess = (low_budget_majority_mass - low_budget_mass_min) / max(
                    1.0 - low_budget_mass_min, 1e-6
                )
                gap_excess = (anti_collapse_margin_gap - margin_gap_min) / max(
                    1.0 - margin_gap_min, 1e-6
                )
                collapse_score = max(
                    0.0,
                    min(
                        1.0,
                        float(np.prod([sharp_excess, k_collapse, mass_excess, gap_excess]) ** 0.25),
                    ),
                )
                min_weight = max(0.0, min(1.0, float(anti_collapse_min_weight)))
                anti_collapse_capacity_value = 1.0 - (1.0 - min_weight) * collapse_score
                anti_collapse_active = 1.0
        process_consistent = [
            1.0 if info["consistent"] else 0.0
            for info in rollout_process_infos
        ]
        process_majority_consistent = [
            1.0
            if info["consistent"] and info.get("answer") == majority_gt
            else 0.0
            for info in rollout_process_infos
        ]
        process_consistent_rate = float(np.mean(process_consistent)) if process_consistent else 0.0
        process_majority_support = (
            float(np.sum(process_majority_consistent)) / float(max(majority_count, 1))
            if process_majority_consistent
            else 0.0
        )
        process_majority_consistent_rate = (
            float(np.mean(process_majority_consistent)) if process_majority_consistent else 0.0
        )
        process_tail_rate = float(
            np.mean([1.0 if info["tail_ok"] else 0.0 for info in rollout_process_infos])
        )
        process_box_conflict_rate = float(
            np.mean([1.0 if info["box_conflict"] else 0.0 for info in rollout_process_infos])
        )
        process_revision_after_final_rate = float(
            np.mean([1.0 if info["revision_after_final"] else 0.0 for info in rollout_process_infos])
        )
        process_consistent_by_answer = Counter(
            info.get("answer")
            for info in rollout_process_infos
            if info.get("answer") is not None and info.get("consistent")
        )
        process_capacity_value = process_majority_support
        if process_capacity_value <= 0.0:
            process_capacity_value = max(0.0, min(1.0, float(process_disagreement_penalty))) * process_consistent_rate
        support_conflict_capacity_value = 1.0
        support_conflict_active = 0.0
        support_conflict_score = 0.0
        if support_conflict_capacity:
            sharp_min = max(0.0, min(1.0 - 1e-6, float(support_conflict_sharp_min)))
            effective_k_max = max(1.0 + 1e-6, float(support_conflict_effective_k_max))
            base_min = max(0.0, min(1.0, float(support_conflict_base_min)))
            low_budget_min = max(0.0, min(1.0, float(support_conflict_low_budget_min)))
            process_min = max(0.0, min(1.0, float(support_conflict_process_min)))
            if sharp_confidence >= sharp_min and sharp_effective_k <= effective_k_max:
                conflict_terms = [
                    1.0 if base_support_gt != majority_gt else 0.0,
                    max(0.0, (base_min - base_support_majority_confidence) / max(base_min, 1e-6)),
                    1.0 if low_budget_gt != majority_gt else 0.0,
                    max(0.0, (low_budget_min - low_budget_majority_mass) / max(low_budget_min, 1e-6)),
                    max(0.0, (process_min - process_majority_support) / max(process_min, 1e-6)),
                ]
                support_conflict_score = max(0.0, min(1.0, float(np.mean(conflict_terms))))
                if support_conflict_score > 0.0:
                    min_weight = max(0.0, min(1.0, float(support_conflict_min_weight)))
                    support_conflict_capacity_value = 1.0 - (1.0 - min_weight) * support_conflict_score
                    support_conflict_active = 1.0
        process_sample_prompt_weights = [1.0] * n
        low_budget_negative_prompt_rewards = [0.0] * n
        low_budget_soft_prompt_rewards = [0.0] * n
        low_budget_rescue_prompt_rewards = [0.0] * n
        low_budget_local_prompt_rewards = [0.0] * n
        independent_support_prompt_rewards = [0.0] * n
        process_answer_prompt_rewards = [0.0] * n
        contrastive_alt_prompt_rewards = [0.0] * n
        low_budget_rebalance_prompt_rewards = [0.0] * n
        process_quality_prompt_rewards = [0.0] * n
        sample_positive_count = 0
        sample_inconsistent_count = 0
        sample_negative_count = 0
        sample_invalid_count = 0
        if process_sample_weight:
            positive_weight = max(0.0, float(process_sample_positive))
            inconsistent_weight = max(0.0, float(process_sample_inconsistent))
            negative_weight = max(0.0, float(process_sample_negative))
            invalid_weight = max(0.0, float(process_sample_invalid))
            for j in range(low_k):
                info = rollout_process_infos[j]
                answer = info.get("answer")
                if info.get("clipped") or not info.get("parseable") or answer is None:
                    process_sample_prompt_weights[j] = invalid_weight
                    sample_invalid_count += 1
                elif answer != majority_gt:
                    process_sample_prompt_weights[j] = negative_weight
                    sample_negative_count += 1
                elif info.get("consistent"):
                    process_sample_prompt_weights[j] = positive_weight
                    sample_positive_count += 1
                else:
                    process_sample_prompt_weights[j] = inconsistent_weight
                    sample_inconsistent_count += 1
        negative_gate = (
            low_budget_majority_mass >= float(low_budget_negative_min_support)
            and process_majority_consistent_rate >= float(low_budget_negative_process_min)
            and base_support_majority_confidence >= float(low_budget_negative_base_min)
            and base_support_gt == majority_gt
            and low_budget_gt == majority_gt
        )
        negative_count = 0
        if low_budget_negative_reward and negative_gate:
            negative_value = -abs(float(low_budget_negative_value))
            for j in range(low_k):
                info = rollout_process_infos[j]
                answer = info.get("answer")
                if info.get("clipped") or not info.get("parseable") or answer is None:
                    low_budget_negative_prompt_rewards[j] = negative_value
                    negative_count += 1
                elif answer != majority_gt and not info.get("consistent"):
                    low_budget_negative_prompt_rewards[j] = negative_value
                    negative_count += 1
                elif answer != majority_gt and base_support_majority_confidence >= float(low_budget_negative_base_min):
                    low_budget_negative_prompt_rewards[j] = 0.5 * negative_value
                    negative_count += 1

        cluster_support = {}
        for answer in answer_keys:
            count_mass = float(counter.get(answer, 0)) / float(n)
            process_cluster_conf = float(process_consistent_by_answer.get(answer, 0)) / float(
                max(counter.get(answer, 0), 1)
            )
            support = (
                max(sharp_prob_by_answer.get(answer, 0.0), 0.0)
                * max(base_prob_by_answer.get(answer, 0.0), 0.0)
                * max(process_cluster_conf, 0.0)
            ) ** (1.0 / 3.0)
            support *= float(np.sqrt(max(count_mass, 0.0)))
            cluster_support[answer] = float(support)
        support_values = np.array(list(cluster_support.values()), dtype=float)
        support_sum = float(support_values.sum()) if support_values.size else 0.0
        if support_sum > 0:
            support_probs = support_values / support_sum
            soft_support_effective_k = float(1.0 / np.maximum(np.sum(support_probs ** 2), 1e-12))
        else:
            soft_support_effective_k = 0.0
        max_support = max(cluster_support.values(), default=0.0)
        soft_count = 0
        soft_nonmajority_count = 0
        soft_nonmajority_rewards = []
        if low_budget_soft_reward and max_support > 0.0:
            soft_value = abs(float(low_budget_soft_value))
            for j in range(low_k):
                info = rollout_process_infos[j]
                answer = info.get("answer")
                if info.get("clipped") or not info.get("parseable") or answer is None:
                    continue
                support_norm = max(0.0, min(1.0, cluster_support.get(answer, 0.0) / max_support))
                reward = soft_value * support_norm
                if reward <= 0.0:
                    continue
                low_budget_soft_prompt_rewards[j] = reward
                soft_count += 1
                if answer != majority_gt:
                    soft_nonmajority_count += 1
                    soft_nonmajority_rewards.append(reward)

        rescue_support = {}
        rescue_cluster_mass = {}
        for answer in answer_keys:
            count_mass = float(counter.get(answer, 0)) / float(n)
            process_cluster_conf = float(process_consistent_by_answer.get(answer, 0)) / float(
                max(counter.get(answer, 0), 1)
            )
            support = (
                max(base_prob_by_answer.get(answer, 0.0), 0.0)
                * max(process_cluster_conf, 0.0)
                * float(np.sqrt(max(count_mass, 0.0)))
            ) ** (1.0 / 3.0)
            rescue_support[answer] = float(support)
            rescue_cluster_mass[answer] = count_mass
        max_rescue_support = max(rescue_support.values(), default=0.0)
        rescue_count = 0
        rescue_rewards = []
        rescue_supports = []
        rescue_masses = []
        if low_budget_rescue_reward and max_rescue_support > 0.0:
            rescue_value = abs(float(low_budget_rescue_value))
            min_base = max(0.0, min(1.0, float(low_budget_rescue_min_base)))
            min_process = max(0.0, min(1.0, float(low_budget_rescue_min_process)))
            min_mass = max(0.0, min(1.0, float(low_budget_rescue_min_cluster_mass)))
            min_support = max(0.0, min(1.0, float(low_budget_rescue_min_support)))
            for j in range(low_k):
                info = rollout_process_infos[j]
                answer = info.get("answer")
                if (
                    info.get("clipped")
                    or not info.get("parseable")
                    or answer is None
                    or answer == majority_gt
                    or not info.get("consistent")
                ):
                    continue
                answer_base = base_prob_by_answer.get(answer, 0.0)
                answer_process = float(process_consistent_by_answer.get(answer, 0)) / float(
                    max(counter.get(answer, 0), 1)
                )
                answer_mass = rescue_cluster_mass.get(answer, 0.0)
                support_norm = max(
                    0.0,
                    min(1.0, rescue_support.get(answer, 0.0) / max(max_rescue_support, 1e-12)),
                )
                if (
                    answer_base < min_base
                    or answer_process < min_process
                    or answer_mass < min_mass
                    or support_norm < min_support
                ):
                    continue
                reward = rescue_value * support_norm
                if reward <= 0.0:
                    continue
                low_budget_rescue_prompt_rewards[j] = reward
                rescue_count += 1
                rescue_rewards.append(reward)
                rescue_supports.append(support_norm)
                rescue_masses.append(answer_mass)

        local_count = 0
        local_rewards = []
        local_supports = []
        local_mass = float(low_counter.get(low_budget_gt, 0)) / float(low_k) if low_counter else 0.0
        local_agreement = float(low_budget_gt == majority_gt) if low_counter else 0.0
        if low_budget_local_reward and low_counter and low_budget_gt != "None":
            local_value = abs(float(low_budget_local_value))
            min_local_mass = max(0.0, min(1.0, float(low_budget_local_min_mass)))
            min_base = max(0.0, min(1.0, float(low_budget_local_min_base)))
            min_process = max(0.0, min(1.0, float(low_budget_local_min_process)))
            min_support = max(0.0, min(1.0, float(low_budget_local_min_support)))
            local_process = float(process_consistent_by_answer.get(low_budget_gt, 0)) / float(
                max(counter.get(low_budget_gt, 0), 1)
            )
            local_base = base_prob_by_answer.get(low_budget_gt, 0.0)
            local_support = (
                max(local_base, 0.0)
                * max(local_process, 0.0)
                * float(np.sqrt(max(local_mass, 0.0)))
            ) ** (1.0 / 3.0)
            if (
                local_mass >= min_local_mass
                and local_base >= min_base
                and local_process >= min_process
                and local_support >= min_support
            ):
                for j in range(low_k):
                    info = rollout_process_infos[j]
                    answer = info.get("answer")
                    if (
                        info.get("clipped")
                        or not info.get("parseable")
                        or answer is None
                        or answer != low_budget_gt
                        or not info.get("consistent")
                    ):
                        continue
                    reward = local_value * min(1.0, local_support)
                    if reward <= 0.0:
                        continue
                    low_budget_local_prompt_rewards[j] = reward
                    local_count += 1
                    local_rewards.append(reward)
                    local_supports.append(local_support)

        independent_support = {}
        for answer in answer_keys:
            count_mass = float(counter.get(answer, 0)) / float(n)
            process_cluster_conf = float(process_consistent_by_answer.get(answer, 0)) / float(
                max(counter.get(answer, 0), 1)
            )
            support = (
                max(base_prob_by_answer.get(answer, 0.0), 0.0)
                * max(process_cluster_conf, 0.0)
                * float(np.sqrt(max(count_mass, 0.0)))
            ) ** (1.0 / 3.0)
            independent_support[answer] = float(support)
        independent_support_values = np.array(list(independent_support.values()), dtype=float)
        independent_support_sum = float(independent_support_values.sum()) if independent_support_values.size else 0.0
        if independent_support_sum > 0:
            independent_support_probs = independent_support_values / independent_support_sum
            independent_support_effective_k = float(
                1.0 / np.maximum(np.sum(independent_support_probs ** 2), 1e-12)
            )
        else:
            independent_support_effective_k = 0.0
        max_independent_support = max(independent_support.values(), default=0.0)
        sorted_independent_support = sorted(independent_support.values(), reverse=True)
        top_independent_support = float(sorted_independent_support[0]) if sorted_independent_support else 0.0
        second_independent_support = (
            float(sorted_independent_support[1]) if len(sorted_independent_support) > 1 else 0.0
        )
        independent_support_margin = max(0.0, top_independent_support - second_independent_support)
        independent_support_competition = (
            second_independent_support / max(top_independent_support, 1e-12)
            if top_independent_support > 0.0
            else 0.0
        )
        independent_competition_active = 0.0
        independent_majority_gate_count = 0
        independent_count = 0
        independent_nonmajority_count = 0
        independent_rewards = []
        independent_supports = []
        if independent_support_reward and max_independent_support > 0.0:
            support_value = abs(float(independent_support_value))
            min_base = max(0.0, min(1.0, float(independent_support_min_base)))
            min_process = max(0.0, min(1.0, float(independent_support_min_process)))
            min_mass = max(0.0, min(1.0, float(independent_support_min_cluster_mass)))
            min_support = max(0.0, min(1.0, float(independent_support_min_support)))
            entropy_gate = max(1.0, float(independent_support_entropy_gate))
            allow_majority = bool(independent_support_allow_majority)
            majority_margin_max = max(0.0, float(independent_support_majority_margin_max))
            competition_min = max(0.0, min(1.0, float(independent_support_competition_min)))
            independent_competition_active = float(
                independent_support_effective_k > entropy_gate
                and independent_support_margin <= majority_margin_max
                and independent_support_competition >= competition_min
            )
            for j in range(low_k):
                info = rollout_process_infos[j]
                answer = info.get("answer")
                if (
                    info.get("clipped")
                    or not info.get("parseable")
                    or answer is None
                    or not info.get("consistent")
                ):
                    continue
                answer_base = base_prob_by_answer.get(answer, 0.0)
                answer_process = float(process_consistent_by_answer.get(answer, 0)) / float(
                    max(counter.get(answer, 0), 1)
                )
                answer_mass = float(counter.get(answer, 0)) / float(n)
                support_norm = max(
                    0.0,
                    min(1.0, independent_support.get(answer, 0.0) / max(max_independent_support, 1e-12)),
                )
                if (
                    answer_base < min_base
                    or answer_process < min_process
                    or answer_mass < min_mass
                    or support_norm < min_support
                ):
                    continue
                if answer == majority_gt and not allow_majority and independent_support_effective_k <= entropy_gate:
                    continue
                if answer == majority_gt:
                    if not independent_competition_active:
                        continue
                    independent_majority_gate_count += 1
                    reward_scale = 0.5
                else:
                    reward_scale = 1.0
                reward = support_value * support_norm * reward_scale
                if reward <= 0.0:
                    continue
                independent_support_prompt_rewards[j] = reward
                independent_count += 1
                independent_rewards.append(reward)
                independent_supports.append(support_norm)
                if answer != majority_gt:
                    independent_nonmajority_count += 1

        process_answer_support = {}
        for answer in answer_keys:
            process_cluster_conf = float(process_consistent_by_answer.get(answer, 0)) / float(
                max(counter.get(answer, 0), 1)
            )
            support = (
                max(base_mean_prob_by_answer.get(answer, 0.0), 0.0)
                * max(process_cluster_conf, 0.0)
            ) ** 0.5
            process_answer_support[answer] = float(support)
        max_process_answer_support = max(process_answer_support.values(), default=0.0)
        process_answer_count = 0
        process_answer_nonmajority_count = 0
        process_answer_rewards = []
        process_answer_supports = []
        process_answer_bases = []
        process_answer_processes = []
        if process_answer_reward and max_process_answer_support > 0.0:
            pa_value = abs(float(process_answer_value))
            pa_min_base = max(0.0, min(1.0, float(process_answer_min_base)))
            pa_min_process = max(0.0, min(1.0, float(process_answer_min_process)))
            pa_min_support = max(0.0, min(1.0, float(process_answer_min_support)))
            pa_majority_scale = max(0.0, min(1.0, float(process_answer_majority_scale)))
            for j in range(low_k):
                info = rollout_process_infos[j]
                answer = info.get("answer")
                if (
                    info.get("clipped")
                    or not info.get("parseable")
                    or answer is None
                    or not info.get("consistent")
                ):
                    continue
                answer_base = base_mean_prob_by_answer.get(answer, 0.0)
                answer_process = float(process_consistent_by_answer.get(answer, 0)) / float(
                    max(counter.get(answer, 0), 1)
                )
                support_norm = max(
                    0.0,
                    min(
                        1.0,
                        process_answer_support.get(answer, 0.0)
                        / max(max_process_answer_support, 1e-12),
                    ),
                )
                if (
                    answer_base < pa_min_base
                    or answer_process < pa_min_process
                    or support_norm < pa_min_support
                ):
                    continue
                reward_scale = pa_majority_scale if answer == majority_gt else 1.0
                reward = pa_value * support_norm * reward_scale
                if reward <= 0.0:
                    continue
                process_answer_prompt_rewards[j] = reward
                process_answer_count += 1
                process_answer_rewards.append(reward)
                process_answer_supports.append(support_norm)
                process_answer_bases.append(answer_base)
                process_answer_processes.append(answer_process)
                if answer != majority_gt:
                    process_answer_nonmajority_count += 1

        contrastive_alt_count = 0
        contrastive_alt_majority_penalty_count = 0
        contrastive_alt_rewards = []
        contrastive_alt_supports = []
        contrastive_alt_bases = []
        contrastive_alt_processes = []
        contrastive_alt_gate = False
        if contrastive_alt_reward and max_process_answer_support > 0.0:
            ca_value = abs(float(contrastive_alt_value))
            ca_penalty = abs(float(contrastive_alt_majority_penalty))
            ca_min_base = max(0.0, min(1.0, float(contrastive_alt_min_base)))
            ca_min_process = max(0.0, min(1.0, float(contrastive_alt_min_process)))
            ca_min_support = max(0.0, min(1.0, float(contrastive_alt_min_support)))
            ca_sharp_min = max(0.0, min(1.0 - 1e-6, float(contrastive_alt_sharp_min)))
            ca_effective_k_max = max(1.0 + 1e-6, float(contrastive_alt_effective_k_max))
            ca_low_budget_mass_min = max(
                0.0, min(1.0, float(contrastive_alt_low_budget_mass_min))
            )
            ca_majority_support_min = max(
                0.0, min(1.0, float(contrastive_alt_majority_support_min))
            )
            contrastive_alt_gate = (
                sharp_confidence >= ca_sharp_min
                and sharp_effective_k <= ca_effective_k_max
                and low_budget_majority_mass >= ca_low_budget_mass_min
                and base_support_majority_confidence >= ca_majority_support_min
                and process_majority_support >= ca_majority_support_min
            )
            sharp_excess = (sharp_confidence - ca_sharp_min) / max(1.0 - ca_sharp_min, 1e-6)
            k_collapse = (ca_effective_k_max - sharp_effective_k) / max(
                ca_effective_k_max - 1.0, 1e-6
            )
            collapse_scale = max(0.0, min(1.0, float(np.sqrt(max(sharp_excess, 0.0) * max(k_collapse, 0.0)))))
            if contrastive_alt_gate and collapse_scale > 0.0:
                for j in range(low_k):
                    info = rollout_process_infos[j]
                    answer = info.get("answer")
                    if (
                        info.get("clipped")
                        or not info.get("parseable")
                        or answer is None
                        or answer == majority_gt
                        or not info.get("consistent")
                    ):
                        continue
                    answer_base = base_mean_prob_by_answer.get(answer, 0.0)
                    answer_process = float(process_consistent_by_answer.get(answer, 0)) / float(
                        max(counter.get(answer, 0), 1)
                    )
                    support_norm = max(
                        0.0,
                        min(
                            1.0,
                            process_answer_support.get(answer, 0.0)
                            / max(max_process_answer_support, 1e-12),
                        ),
                    )
                    if (
                        answer_base < ca_min_base
                        or answer_process < ca_min_process
                        or support_norm < ca_min_support
                    ):
                        continue
                    reward = ca_value * support_norm * collapse_scale
                    if reward <= 0.0:
                        continue
                    contrastive_alt_prompt_rewards[j] = reward
                    contrastive_alt_count += 1
                    contrastive_alt_rewards.append(reward)
                    contrastive_alt_supports.append(support_norm)
                    contrastive_alt_bases.append(answer_base)
                    contrastive_alt_processes.append(answer_process)
                if contrastive_alt_count > 0 and ca_penalty > 0.0:
                    penalty = -ca_penalty * collapse_scale
                    for j in range(low_k):
                        info = rollout_process_infos[j]
                        answer = info.get("answer")
                        if (
                            info.get("clipped")
                            or not info.get("parseable")
                            or answer is None
                            or answer != majority_gt
                        ):
                            continue
                        contrastive_alt_prompt_rewards[j] += penalty
                        contrastive_alt_majority_penalty_count += 1

        rebalance_alt_count = 0
        rebalance_majority_penalty_count = 0
        rebalance_rewards = []
        rebalance_supports = []
        rebalance_bases = []
        rebalance_processes = []
        rebalance_gate = False
        if low_budget_rebalance_reward and max_process_answer_support > 0.0:
            rb_alt_value = abs(float(low_budget_rebalance_alt_value))
            rb_penalty = abs(float(low_budget_rebalance_majority_penalty))
            rb_sharp_min = max(0.0, min(1.0 - 1e-6, float(low_budget_rebalance_sharp_min)))
            rb_effective_k_max = max(1.0 + 1e-6, float(low_budget_rebalance_effective_k_max))
            rb_mass_min = max(0.0, min(1.0, float(low_budget_rebalance_mass_min)))
            rb_support_min = max(0.0, min(1.0, float(low_budget_rebalance_support_min)))
            rb_min_base = max(0.0, min(1.0, float(low_budget_rebalance_min_base)))
            rb_min_process = max(0.0, min(1.0, float(low_budget_rebalance_min_process)))
            rb_min_support = max(0.0, min(1.0, float(low_budget_rebalance_min_support)))
            rebalance_gate = (
                sharp_confidence >= rb_sharp_min
                and sharp_effective_k <= rb_effective_k_max
                and low_budget_majority_mass >= rb_mass_min
                and base_support_majority_confidence >= rb_support_min
                and process_majority_support >= rb_support_min
            )
            sharp_excess = (sharp_confidence - rb_sharp_min) / max(1.0 - rb_sharp_min, 1e-6)
            k_collapse = (rb_effective_k_max - sharp_effective_k) / max(
                rb_effective_k_max - 1.0, 1e-6
            )
            mass_excess = (low_budget_majority_mass - rb_mass_min) / max(1.0 - rb_mass_min, 1e-6)
            collapse_scale = max(
                0.0,
                min(
                    1.0,
                    float(
                        np.prod(
                            [
                                max(sharp_excess, 0.0),
                                max(k_collapse, 0.0),
                                max(mass_excess, 0.0),
                            ]
                        )
                        ** (1.0 / 3.0)
                    ),
                ),
            )
            if rebalance_gate and collapse_scale > 0.0:
                first4_alt_exists = False
                for j in range(low_k):
                    info = rollout_process_infos[j]
                    answer = info.get("answer")
                    if (
                        info.get("clipped")
                        or not info.get("parseable")
                        or answer is None
                        or answer == majority_gt
                        or not info.get("consistent")
                    ):
                        continue
                    answer_base = base_mean_prob_by_answer.get(answer, 0.0)
                    answer_process = float(process_consistent_by_answer.get(answer, 0)) / float(
                        max(counter.get(answer, 0), 1)
                    )
                    support_norm = max(
                        0.0,
                        min(
                            1.0,
                            process_answer_support.get(answer, 0.0)
                            / max(max_process_answer_support, 1e-12),
                        ),
                    )
                    if (
                        answer_base < rb_min_base
                        or answer_process < rb_min_process
                        or support_norm < rb_min_support
                    ):
                        continue
                    first4_alt_exists = True
                    reward = rb_alt_value * support_norm * collapse_scale
                    if reward <= 0.0:
                        continue
                    low_budget_rebalance_prompt_rewards[j] = reward
                    rebalance_alt_count += 1
                    rebalance_rewards.append(reward)
                    rebalance_supports.append(support_norm)
                    rebalance_bases.append(answer_base)
                    rebalance_processes.append(answer_process)
                if first4_alt_exists and rb_penalty > 0.0:
                    penalty = -rb_penalty * collapse_scale
                    for j in range(low_k):
                        info = rollout_process_infos[j]
                        answer = info.get("answer")
                        if (
                            info.get("clipped")
                            or not info.get("parseable")
                            or answer is None
                            or answer != majority_gt
                        ):
                            continue
                        low_budget_rebalance_prompt_rewards[j] += penalty
                        rebalance_majority_penalty_count += 1

        process_quality_positive_count = 0
        process_quality_negative_count = 0
        process_quality_bad_count = 0
        process_quality_collapse_gate = False
        if process_quality_reward:
            pq_positive = abs(float(process_quality_positive))
            pq_negative = abs(float(process_quality_negative))
            pq_sharp_min = max(0.0, min(1.0 - 1e-6, float(process_quality_sharp_min)))
            pq_effective_k_max = max(1.0 + 1e-6, float(process_quality_effective_k_max))
            pq_mass_min = max(0.0, min(1.0, float(process_quality_mass_min)))
            pq_majority_scale = max(0.0, float(process_quality_majority_scale))
            pq_nonmajority_scale = max(0.0, float(process_quality_nonmajority_scale))
            pq_bad_majority_scale = max(0.0, float(process_quality_bad_majority_scale))
            pq_bad_nonmajority_scale = max(0.0, float(process_quality_bad_nonmajority_scale))
            process_quality_collapse_gate = (
                sharp_confidence >= pq_sharp_min
                and sharp_effective_k <= pq_effective_k_max
                and low_budget_majority_mass >= pq_mass_min
            )
            for j in range(low_k):
                info = rollout_process_infos[j]
                answer = info.get("answer")
                bad_process = (
                    info.get("clipped")
                    or not info.get("parseable")
                    or answer is None
                    or info.get("box_conflict")
                    or info.get("revision_after_final")
                    or not info.get("tail_ok")
                )
                if bad_process:
                    bad_scale = (
                        pq_bad_majority_scale
                        if answer == majority_gt
                        else pq_bad_nonmajority_scale
                    )
                    reward = -pq_negative * bad_scale
                    process_quality_prompt_rewards[j] = reward
                    process_quality_negative_count += 1
                    process_quality_bad_count += 1
                    continue
                if not info.get("consistent"):
                    continue
                if answer == majority_gt:
                    positive_scale = pq_majority_scale if process_quality_collapse_gate else 1.0
                else:
                    positive_scale = pq_nonmajority_scale
                reward = pq_positive * positive_scale
                if reward <= 0.0:
                    continue
                process_quality_prompt_rewards[j] = reward
                process_quality_positive_count += 1

        entropy_band_capacity_value = 1.0
        entropy_band_low_active = 0.0
        entropy_band_high_active = 0.0
        entropy_band_score = 0.0
        if entropy_band_capacity:
            band_low_k = max(1.0 + 1e-6, float(entropy_band_low_k))
            band_high_k = max(band_low_k + 1e-6, float(entropy_band_high_k))
            min_weight = max(0.0, min(1.0, float(entropy_band_min_weight)))
            sharp_min = max(0.0, min(1.0, float(entropy_band_sharp_min)))
            mass_min = max(0.0, min(1.0, float(entropy_band_low_budget_mass_min)))
            support_min = max(0.0, min(1.0, float(entropy_band_support_min)))
            low_collapse = (
                sharp_effective_k < band_low_k
                and sharp_confidence >= sharp_min
                and low_budget_majority_mass >= mass_min
                and base_support_majority_confidence >= support_min
                and process_majority_support >= support_min
            )
            high_noise = (
                sharp_effective_k > band_high_k
                or low_budget_parseable_rate < 0.75
                or low_budget_clip_rate > 0.25
            )
            if low_collapse:
                k_score = (band_low_k - sharp_effective_k) / max(band_low_k - 1.0, 1e-6)
                sharp_score = (sharp_confidence - sharp_min) / max(1.0 - sharp_min, 1e-6)
                mass_score = (low_budget_majority_mass - mass_min) / max(1.0 - mass_min, 1e-6)
                support_score = min(
                    (base_support_majority_confidence - support_min) / max(1.0 - support_min, 1e-6),
                    (process_majority_support - support_min) / max(1.0 - support_min, 1e-6),
                )
                entropy_band_score = max(
                    0.0,
                    min(
                        1.0,
                        float(
                            np.prod(
                                [
                                    max(k_score, 0.0),
                                    max(sharp_score, 0.0),
                                    max(mass_score, 0.0),
                                    max(support_score, 0.0),
                                ]
                            )
                            ** 0.25
                        ),
                    ),
                )
                entropy_band_capacity_value = 1.0 - (1.0 - min_weight) * entropy_band_score
                entropy_band_low_active = 1.0
            elif high_noise:
                k_score = (sharp_effective_k - band_high_k) / max(float(len(answer_keys)) - band_high_k, 1e-6)
                parse_score = (0.75 - low_budget_parseable_rate) / 0.75
                clip_score = (low_budget_clip_rate - 0.25) / 0.75
                entropy_band_score = max(
                    0.0,
                    min(1.0, float(max(k_score, parse_score, clip_score))),
                )
                entropy_band_capacity_value = 1.0 - (1.0 - min_weight) * entropy_band_score
                entropy_band_high_active = 1.0

        use_sps_label = True
        if confidence_filter or confidence_weight:
            use_sps_label = False
        if use_majority_fallback:
            use_sps_label = (
                weighted_gt != majority_gt
                and weighted_confidence >= gate_confidence_threshold
                and majority_ratio <= gate_majority_ratio_threshold
            )
        selected_gt = weighted_gt if use_sps_label else majority_gt

        selected_gt_list.append(selected_gt)
        weighted_gt_list.append(weighted_gt)
        raw_majority_gt_list.append(majority_gt)
        weighted_confidence_list.append(weighted_confidence)
        majority_ratio_list.append(float(majority_ratio))
        unique_answer_count_list.append(len(answer_to_scores))
        sps_override_list.append(float(weighted_gt != majority_gt and use_sps_label))
        sps_agreement_list.append(float(weighted_gt == majority_gt))
        answer_sharp_confidence_list.append(sharp_confidence)
        answer_entropy_list.append(sharp_entropy)
        answer_effective_k_list.append(sharp_effective_k)
        answer_logz_list.append(float(answer_logz.item()))
        majority_sharp_confidence_list.append(majority_sharp_confidence)
        consistency_capacity_value = 1.0
        if confidence_weight:
            if answer_sharpen_capacity:
                agreement_confidence = weighted_sharp_confidence if weighted_gt == majority_gt else 0.0
            else:
                agreement_confidence = weighted_confidence if weighted_gt == majority_gt else 0.0
            prompt_weight = max(float(majority_ratio), float(agreement_confidence))
            if consistency_capacity:
                consistency_capacity_value = max(0.0, min(1.0, float(majority_sharp_confidence)))
                if weighted_gt != majority_gt:
                    consistency_capacity_value *= max(0.0, min(1.0, float(consistency_disagreement_penalty)))
                prompt_weight = min(prompt_weight, consistency_capacity_value)
            if low_budget_capacity:
                prompt_weight = min(prompt_weight, low_budget_capacity_value)
            if base_support_capacity:
                prompt_weight = min(prompt_weight, base_support_capacity_value)
            if cross_view_capacity:
                prompt_weight = min(prompt_weight, cross_view_capacity_value)
            if margin_capacity:
                prompt_weight = min(prompt_weight, margin_capacity_value)
            if process_consistency_capacity:
                prompt_weight = min(prompt_weight, process_capacity_value)
            if anti_collapse_capacity:
                prompt_weight = min(prompt_weight, anti_collapse_capacity_value)
            if support_conflict_capacity:
                prompt_weight = min(prompt_weight, support_conflict_capacity_value)
            if entropy_band_capacity:
                prompt_weight = min(prompt_weight, entropy_band_capacity_value)
            if clip_penalty > 0:
                prompt_weight *= max(0.0, 1.0 - float(clip_penalty) * prompt_clip_ratio)
            power = max(float(weight_power), 1e-6)
            if power != 1.0:
                prompt_weight = max(0.0, min(1.0, prompt_weight)) ** power
            sps_train_weight_list.append(max(float(weight_floor), min(1.0, prompt_weight)))
        elif confidence_filter:
            keep_prompt = (
                majority_ratio >= filter_majority_ratio_threshold
                or (weighted_gt == majority_gt and weighted_confidence >= filter_confidence_threshold)
            )
            sps_train_weight_list.append(float(keep_prompt))
        else:
            sps_train_weight_list.append(1.0)
        consistency_capacity_list.append(consistency_capacity_value)
        low_budget_parseable_rate_list.append(low_budget_parseable_rate)
        low_budget_clip_rate_list.append(low_budget_clip_rate)
        low_budget_majority_ratio_list.append(low_budget_majority_ratio)
        low_budget_majority_mass_list.append(low_budget_majority_mass)
        low_budget_agreement_list.append(low_budget_agreement)
        low_budget_capacity_list.append(low_budget_capacity_value)
        base_support_majority_confidence_list.append(base_support_majority_confidence)
        base_support_top_confidence_list.append(base_support_top_confidence)
        base_support_agreement_list.append(base_support_agreement)
        base_support_capacity_list.append(base_support_capacity_value)
        cross_view_capacity_list.append(cross_view_capacity_value)
        sharp_majority_margin_list.append(sharp_majority_margin)
        base_support_majority_margin_list.append(base_support_majority_margin)
        low_budget_majority_margin_list.append(low_budget_majority_margin)
        margin_capacity_list.append(margin_capacity_value)
        anti_collapse_capacity_list.append(anti_collapse_capacity_value)
        anti_collapse_active_list.append(anti_collapse_active)
        anti_collapse_margin_gap_list.append(anti_collapse_margin_gap)
        support_conflict_capacity_list.append(support_conflict_capacity_value)
        support_conflict_active_list.append(support_conflict_active)
        support_conflict_score_list.append(support_conflict_score)
        process_consistency_capacity_list.append(process_capacity_value)
        process_majority_support_list.append(process_majority_support)
        process_consistent_rate_list.append(process_consistent_rate)
        process_majority_consistent_rate_list.append(process_majority_consistent_rate)
        process_tail_rate_list.append(process_tail_rate)
        process_box_conflict_rate_list.append(process_box_conflict_rate)
        process_revision_after_final_rate_list.append(process_revision_after_final_rate)
        process_sample_weights_flat.extend(process_sample_prompt_weights)
        process_sample_weight_list.append(float(np.mean(process_sample_prompt_weights[:low_k])))
        process_sample_positive_rate_list.append(float(sample_positive_count) / float(low_k))
        process_sample_inconsistent_rate_list.append(float(sample_inconsistent_count) / float(low_k))
        process_sample_negative_rate_list.append(float(sample_negative_count) / float(low_k))
        process_sample_invalid_rate_list.append(float(sample_invalid_count) / float(low_k))
        low_budget_negative_reward_flat.extend(low_budget_negative_prompt_rewards)
        low_budget_negative_rate_list.append(float(negative_count) / float(low_k))
        low_budget_negative_gate_list.append(float(negative_gate))
        low_budget_soft_reward_flat.extend(low_budget_soft_prompt_rewards)
        low_budget_soft_reward_mean_list.append(
            float(np.mean(low_budget_soft_prompt_rewards[:low_k])) if low_k > 0 else 0.0
        )
        low_budget_soft_reward_rate_list.append(float(soft_count) / float(low_k))
        low_budget_soft_nonmajority_rate_list.append(float(soft_nonmajority_count) / float(low_k))
        low_budget_soft_nonmajority_reward_list.append(
            float(np.mean(soft_nonmajority_rewards)) if soft_nonmajority_rewards else 0.0
        )
        low_budget_soft_support_effective_k_list.append(soft_support_effective_k)
        low_budget_rescue_reward_flat.extend(low_budget_rescue_prompt_rewards)
        low_budget_rescue_reward_mean_list.append(
            float(np.mean(low_budget_rescue_prompt_rewards[:low_k])) if low_k > 0 else 0.0
        )
        low_budget_rescue_rate_list.append(float(rescue_count) / float(low_k))
        low_budget_rescue_support_list.append(
            float(np.mean(rescue_supports)) if rescue_supports else 0.0
        )
        low_budget_rescue_cluster_mass_list.append(
            float(np.mean(rescue_masses)) if rescue_masses else 0.0
        )
        low_budget_local_reward_flat.extend(low_budget_local_prompt_rewards)
        low_budget_local_reward_mean_list.append(
            float(np.mean(low_budget_local_prompt_rewards[:low_k])) if low_k > 0 else 0.0
        )
        low_budget_local_rate_list.append(float(local_count) / float(low_k))
        low_budget_local_support_list.append(
            float(np.mean(local_supports)) if local_supports else 0.0
        )
        low_budget_local_mass_list.append(local_mass)
        low_budget_local_agreement_list.append(local_agreement)
        independent_support_reward_flat.extend(independent_support_prompt_rewards)
        independent_support_reward_mean_list.append(
            float(np.mean(independent_support_prompt_rewards[:low_k])) if low_k > 0 else 0.0
        )
        independent_support_rate_list.append(float(independent_count) / float(low_k))
        independent_support_nonmajority_rate_list.append(
            float(independent_nonmajority_count) / float(low_k)
        )
        independent_support_value_list.append(
            float(np.mean(independent_supports)) if independent_supports else 0.0
        )
        independent_support_effective_k_list.append(independent_support_effective_k)
        independent_support_margin_list.append(independent_support_margin)
        independent_support_competition_rate_list.append(independent_competition_active)
        independent_support_majority_gate_rate_list.append(
            float(independent_majority_gate_count) / float(low_k)
        )
        process_answer_reward_flat.extend(process_answer_prompt_rewards)
        process_answer_reward_mean_list.append(
            float(np.mean(process_answer_prompt_rewards[:low_k])) if low_k > 0 else 0.0
        )
        process_answer_rate_list.append(float(process_answer_count) / float(low_k))
        process_answer_nonmajority_rate_list.append(
            float(process_answer_nonmajority_count) / float(low_k)
        )
        process_answer_support_list.append(
            float(np.mean(process_answer_supports)) if process_answer_supports else 0.0
        )
        process_answer_base_list.append(
            float(np.mean(process_answer_bases)) if process_answer_bases else 0.0
        )
        process_answer_process_list.append(
            float(np.mean(process_answer_processes)) if process_answer_processes else 0.0
        )
        contrastive_alt_reward_flat.extend(contrastive_alt_prompt_rewards)
        contrastive_alt_reward_mean_list.append(
            float(np.mean(contrastive_alt_prompt_rewards[:low_k])) if low_k > 0 else 0.0
        )
        contrastive_alt_rate_list.append(float(contrastive_alt_count) / float(low_k))
        contrastive_alt_majority_penalty_rate_list.append(
            float(contrastive_alt_majority_penalty_count) / float(low_k)
        )
        contrastive_alt_gate_list.append(float(contrastive_alt_gate))
        contrastive_alt_support_list.append(
            float(np.mean(contrastive_alt_supports)) if contrastive_alt_supports else 0.0
        )
        contrastive_alt_base_list.append(
            float(np.mean(contrastive_alt_bases)) if contrastive_alt_bases else 0.0
        )
        contrastive_alt_process_list.append(
            float(np.mean(contrastive_alt_processes)) if contrastive_alt_processes else 0.0
        )
        low_budget_rebalance_reward_flat.extend(low_budget_rebalance_prompt_rewards)
        low_budget_rebalance_reward_mean_list.append(
            float(np.mean(low_budget_rebalance_prompt_rewards[:low_k])) if low_k > 0 else 0.0
        )
        low_budget_rebalance_alt_rate_list.append(float(rebalance_alt_count) / float(low_k))
        low_budget_rebalance_majority_penalty_rate_list.append(
            float(rebalance_majority_penalty_count) / float(low_k)
        )
        low_budget_rebalance_gate_list.append(float(rebalance_gate))
        low_budget_rebalance_support_list.append(
            float(np.mean(rebalance_supports)) if rebalance_supports else 0.0
        )
        low_budget_rebalance_base_list.append(
            float(np.mean(rebalance_bases)) if rebalance_bases else 0.0
        )
        low_budget_rebalance_process_list.append(
            float(np.mean(rebalance_processes)) if rebalance_processes else 0.0
        )
        process_quality_reward_flat.extend(process_quality_prompt_rewards)
        process_quality_reward_mean_list.append(
            float(np.mean(process_quality_prompt_rewards[:low_k])) if low_k > 0 else 0.0
        )
        process_quality_positive_rate_list.append(
            float(process_quality_positive_count) / float(low_k)
        )
        process_quality_negative_rate_list.append(
            float(process_quality_negative_count) / float(low_k)
        )
        process_quality_bad_rate_list.append(float(process_quality_bad_count) / float(low_k))
        process_quality_collapse_gate_list.append(float(process_quality_collapse_gate))
        entropy_band_capacity_list.append(entropy_band_capacity_value)
        entropy_band_low_active_list.append(entropy_band_low_active)
        entropy_band_high_active_list.append(entropy_band_high_active)
        entropy_band_score_list.append(entropy_band_score)

    for i in range(num_prompts):
        data_item = batch[i]
        original_gt = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        data_item.non_tensor_batch["reward_model"]["ground_truth"] = selected_gt_list[i]
        data_item.non_tensor_batch["reward_model"]["majority_gt"] = selected_gt_list[i]
        data_item.non_tensor_batch["reward_model"]["sps_weighted_gt"] = weighted_gt_list[i]
        data_item.non_tensor_batch["reward_model"]["raw_majority_gt"] = raw_majority_gt_list[i]
        data_item.non_tensor_batch["reward_model"]["original_gt"] = original_gt

    batch.non_tensor_batch["majority_ratio_list"] = np.array(majority_ratio_list, dtype=float)
    batch.non_tensor_batch["sps_selected_gt_list"] = np.array(selected_gt_list, dtype=object)
    batch.non_tensor_batch["sps_raw_majority_gt_list"] = np.array(raw_majority_gt_list, dtype=object)
    batch.non_tensor_batch["sps_weighted_confidence_list"] = np.array(weighted_confidence_list, dtype=float)
    batch.non_tensor_batch["sps_unique_answer_count_list"] = np.array(unique_answer_count_list, dtype=float)
    batch.non_tensor_batch["sps_override_list"] = np.array(sps_override_list, dtype=float)
    batch.non_tensor_batch["sps_agreement_list"] = np.array(sps_agreement_list, dtype=float)
    batch.non_tensor_batch["sps_train_weight_list"] = np.array(sps_train_weight_list, dtype=float)
    batch.non_tensor_batch["sps_answer_sharp_confidence_list"] = np.array(answer_sharp_confidence_list, dtype=float)
    batch.non_tensor_batch["sps_answer_entropy_list"] = np.array(answer_entropy_list, dtype=float)
    batch.non_tensor_batch["sps_answer_effective_k_list"] = np.array(answer_effective_k_list, dtype=float)
    batch.non_tensor_batch["sps_answer_logz_list"] = np.array(answer_logz_list, dtype=float)
    batch.non_tensor_batch["sps_majority_sharp_confidence_list"] = np.array(majority_sharp_confidence_list, dtype=float)
    batch.non_tensor_batch["sps_consistency_capacity_list"] = np.array(consistency_capacity_list, dtype=float)
    batch.non_tensor_batch["sps_low_budget_parseable_rate_list"] = np.array(low_budget_parseable_rate_list, dtype=float)
    batch.non_tensor_batch["sps_low_budget_clip_rate_list"] = np.array(low_budget_clip_rate_list, dtype=float)
    batch.non_tensor_batch["sps_low_budget_majority_ratio_list"] = np.array(low_budget_majority_ratio_list, dtype=float)
    batch.non_tensor_batch["sps_low_budget_majority_mass_list"] = np.array(low_budget_majority_mass_list, dtype=float)
    batch.non_tensor_batch["sps_low_budget_agreement_list"] = np.array(low_budget_agreement_list, dtype=float)
    batch.non_tensor_batch["sps_low_budget_capacity_list"] = np.array(low_budget_capacity_list, dtype=float)
    batch.non_tensor_batch["sps_base_support_majority_confidence_list"] = np.array(
        base_support_majority_confidence_list, dtype=float
    )
    batch.non_tensor_batch["sps_base_support_top_confidence_list"] = np.array(
        base_support_top_confidence_list, dtype=float
    )
    batch.non_tensor_batch["sps_base_support_agreement_list"] = np.array(base_support_agreement_list, dtype=float)
    batch.non_tensor_batch["sps_base_support_capacity_list"] = np.array(base_support_capacity_list, dtype=float)
    batch.non_tensor_batch["sps_cross_view_capacity_list"] = np.array(cross_view_capacity_list, dtype=float)
    batch.non_tensor_batch["sps_sharp_majority_margin_list"] = np.array(sharp_majority_margin_list, dtype=float)
    batch.non_tensor_batch["sps_base_support_majority_margin_list"] = np.array(
        base_support_majority_margin_list, dtype=float
    )
    batch.non_tensor_batch["sps_low_budget_majority_margin_list"] = np.array(
        low_budget_majority_margin_list, dtype=float
    )
    batch.non_tensor_batch["sps_margin_capacity_list"] = np.array(margin_capacity_list, dtype=float)
    batch.non_tensor_batch["sps_entropy_band_capacity_list"] = np.array(
        entropy_band_capacity_list, dtype=float
    )
    batch.non_tensor_batch["sps_entropy_band_low_active_list"] = np.array(
        entropy_band_low_active_list, dtype=float
    )
    batch.non_tensor_batch["sps_entropy_band_high_active_list"] = np.array(
        entropy_band_high_active_list, dtype=float
    )
    batch.non_tensor_batch["sps_entropy_band_score_list"] = np.array(
        entropy_band_score_list, dtype=float
    )
    batch.non_tensor_batch["sps_anti_collapse_capacity_list"] = np.array(
        anti_collapse_capacity_list, dtype=float
    )
    batch.non_tensor_batch["sps_anti_collapse_active_list"] = np.array(
        anti_collapse_active_list, dtype=float
    )
    batch.non_tensor_batch["sps_anti_collapse_margin_gap_list"] = np.array(
        anti_collapse_margin_gap_list, dtype=float
    )
    batch.non_tensor_batch["sps_support_conflict_capacity_list"] = np.array(
        support_conflict_capacity_list, dtype=float
    )
    batch.non_tensor_batch["sps_support_conflict_active_list"] = np.array(
        support_conflict_active_list, dtype=float
    )
    batch.non_tensor_batch["sps_support_conflict_score_list"] = np.array(
        support_conflict_score_list, dtype=float
    )
    batch.non_tensor_batch["sps_process_consistency_capacity_list"] = np.array(
        process_consistency_capacity_list, dtype=float
    )
    batch.non_tensor_batch["sps_process_majority_support_list"] = np.array(
        process_majority_support_list, dtype=float
    )
    batch.non_tensor_batch["sps_process_consistent_rate_list"] = np.array(
        process_consistent_rate_list, dtype=float
    )
    batch.non_tensor_batch["sps_process_majority_consistent_rate_list"] = np.array(
        process_majority_consistent_rate_list, dtype=float
    )
    batch.non_tensor_batch["sps_process_tail_rate_list"] = np.array(process_tail_rate_list, dtype=float)
    batch.non_tensor_batch["sps_process_box_conflict_rate_list"] = np.array(
        process_box_conflict_rate_list, dtype=float
    )
    batch.non_tensor_batch["sps_process_revision_after_final_rate_list"] = np.array(
        process_revision_after_final_rate_list, dtype=float
    )
    if process_sample_weight:
        gen_batch_output.non_tensor_batch["sps_sample_weight"] = np.array(
            process_sample_weights_flat, dtype=float
        )
    if low_budget_negative_reward:
        gen_batch_output.non_tensor_batch["sps_low_budget_negative_reward"] = np.array(
            low_budget_negative_reward_flat, dtype=float
        )
    if low_budget_soft_reward:
        gen_batch_output.non_tensor_batch["sps_low_budget_soft_reward"] = np.array(
            low_budget_soft_reward_flat, dtype=float
        )
    if low_budget_rescue_reward:
        gen_batch_output.non_tensor_batch["sps_low_budget_rescue_reward"] = np.array(
            low_budget_rescue_reward_flat, dtype=float
        )
    if low_budget_local_reward:
        gen_batch_output.non_tensor_batch["sps_low_budget_local_reward"] = np.array(
            low_budget_local_reward_flat, dtype=float
        )
    if independent_support_reward:
        gen_batch_output.non_tensor_batch["sps_independent_support_reward"] = np.array(
            independent_support_reward_flat, dtype=float
        )
    if process_answer_reward:
        gen_batch_output.non_tensor_batch["sps_process_answer_reward"] = np.array(
            process_answer_reward_flat, dtype=float
        )
    if contrastive_alt_reward:
        gen_batch_output.non_tensor_batch["sps_contrastive_alt_reward"] = np.array(
            contrastive_alt_reward_flat, dtype=float
        )
    if low_budget_rebalance_reward:
        gen_batch_output.non_tensor_batch["sps_low_budget_rebalance_reward"] = np.array(
            low_budget_rebalance_reward_flat, dtype=float
        )
    if process_quality_reward:
        gen_batch_output.non_tensor_batch["sps_process_quality_reward"] = np.array(
            process_quality_reward_flat, dtype=float
        )
    batch.non_tensor_batch["sps_process_sample_weight_list"] = np.array(
        process_sample_weight_list, dtype=float
    )
    batch.non_tensor_batch["sps_process_sample_positive_rate_list"] = np.array(
        process_sample_positive_rate_list, dtype=float
    )
    batch.non_tensor_batch["sps_process_sample_inconsistent_rate_list"] = np.array(
        process_sample_inconsistent_rate_list, dtype=float
    )
    batch.non_tensor_batch["sps_process_sample_negative_rate_list"] = np.array(
        process_sample_negative_rate_list, dtype=float
    )
    batch.non_tensor_batch["sps_process_sample_invalid_rate_list"] = np.array(
        process_sample_invalid_rate_list, dtype=float
    )
    batch.non_tensor_batch["sps_low_budget_negative_rate_list"] = np.array(
        low_budget_negative_rate_list, dtype=float
    )
    batch.non_tensor_batch["sps_low_budget_negative_gate_list"] = np.array(
        low_budget_negative_gate_list, dtype=float
    )
    batch.non_tensor_batch["sps_low_budget_soft_reward_mean_list"] = np.array(
        low_budget_soft_reward_mean_list, dtype=float
    )
    batch.non_tensor_batch["sps_low_budget_soft_reward_rate_list"] = np.array(
        low_budget_soft_reward_rate_list, dtype=float
    )
    batch.non_tensor_batch["sps_low_budget_soft_nonmajority_rate_list"] = np.array(
        low_budget_soft_nonmajority_rate_list, dtype=float
    )
    batch.non_tensor_batch["sps_low_budget_soft_nonmajority_reward_list"] = np.array(
        low_budget_soft_nonmajority_reward_list, dtype=float
    )
    batch.non_tensor_batch["sps_low_budget_soft_support_effective_k_list"] = np.array(
        low_budget_soft_support_effective_k_list, dtype=float
    )
    batch.non_tensor_batch["sps_low_budget_rescue_reward_mean_list"] = np.array(
        low_budget_rescue_reward_mean_list, dtype=float
    )
    batch.non_tensor_batch["sps_low_budget_rescue_rate_list"] = np.array(
        low_budget_rescue_rate_list, dtype=float
    )
    batch.non_tensor_batch["sps_low_budget_rescue_support_list"] = np.array(
        low_budget_rescue_support_list, dtype=float
    )
    batch.non_tensor_batch["sps_low_budget_rescue_cluster_mass_list"] = np.array(
        low_budget_rescue_cluster_mass_list, dtype=float
    )
    batch.non_tensor_batch["sps_low_budget_local_reward_mean_list"] = np.array(
        low_budget_local_reward_mean_list, dtype=float
    )
    batch.non_tensor_batch["sps_low_budget_local_rate_list"] = np.array(
        low_budget_local_rate_list, dtype=float
    )
    batch.non_tensor_batch["sps_low_budget_local_support_list"] = np.array(
        low_budget_local_support_list, dtype=float
    )
    batch.non_tensor_batch["sps_low_budget_local_mass_list"] = np.array(
        low_budget_local_mass_list, dtype=float
    )
    batch.non_tensor_batch["sps_low_budget_local_agreement_list"] = np.array(
        low_budget_local_agreement_list, dtype=float
    )
    batch.non_tensor_batch["sps_independent_support_reward_mean_list"] = np.array(
        independent_support_reward_mean_list, dtype=float
    )
    batch.non_tensor_batch["sps_independent_support_rate_list"] = np.array(
        independent_support_rate_list, dtype=float
    )
    batch.non_tensor_batch["sps_independent_support_nonmajority_rate_list"] = np.array(
        independent_support_nonmajority_rate_list, dtype=float
    )
    batch.non_tensor_batch["sps_independent_support_value_list"] = np.array(
        independent_support_value_list, dtype=float
    )
    batch.non_tensor_batch["sps_independent_support_effective_k_list"] = np.array(
        independent_support_effective_k_list, dtype=float
    )
    batch.non_tensor_batch["sps_independent_support_margin_list"] = np.array(
        independent_support_margin_list, dtype=float
    )
    batch.non_tensor_batch["sps_independent_support_competition_rate_list"] = np.array(
        independent_support_competition_rate_list, dtype=float
    )
    batch.non_tensor_batch["sps_independent_support_majority_gate_rate_list"] = np.array(
        independent_support_majority_gate_rate_list, dtype=float
    )
    batch.non_tensor_batch["sps_process_answer_reward_mean_list"] = np.array(
        process_answer_reward_mean_list, dtype=float
    )
    batch.non_tensor_batch["sps_process_answer_rate_list"] = np.array(
        process_answer_rate_list, dtype=float
    )
    batch.non_tensor_batch["sps_process_answer_nonmajority_rate_list"] = np.array(
        process_answer_nonmajority_rate_list, dtype=float
    )
    batch.non_tensor_batch["sps_process_answer_support_list"] = np.array(
        process_answer_support_list, dtype=float
    )
    batch.non_tensor_batch["sps_process_answer_base_list"] = np.array(
        process_answer_base_list, dtype=float
    )
    batch.non_tensor_batch["sps_process_answer_process_list"] = np.array(
        process_answer_process_list, dtype=float
    )
    batch.non_tensor_batch["sps_contrastive_alt_reward_mean_list"] = np.array(
        contrastive_alt_reward_mean_list, dtype=float
    )
    batch.non_tensor_batch["sps_contrastive_alt_rate_list"] = np.array(
        contrastive_alt_rate_list, dtype=float
    )
    batch.non_tensor_batch["sps_contrastive_alt_majority_penalty_rate_list"] = np.array(
        contrastive_alt_majority_penalty_rate_list, dtype=float
    )
    batch.non_tensor_batch["sps_contrastive_alt_gate_list"] = np.array(
        contrastive_alt_gate_list, dtype=float
    )
    batch.non_tensor_batch["sps_contrastive_alt_support_list"] = np.array(
        contrastive_alt_support_list, dtype=float
    )
    batch.non_tensor_batch["sps_contrastive_alt_base_list"] = np.array(
        contrastive_alt_base_list, dtype=float
    )
    batch.non_tensor_batch["sps_contrastive_alt_process_list"] = np.array(
        contrastive_alt_process_list, dtype=float
    )
    batch.non_tensor_batch["sps_low_budget_rebalance_reward_mean_list"] = np.array(
        low_budget_rebalance_reward_mean_list, dtype=float
    )
    batch.non_tensor_batch["sps_low_budget_rebalance_alt_rate_list"] = np.array(
        low_budget_rebalance_alt_rate_list, dtype=float
    )
    batch.non_tensor_batch["sps_low_budget_rebalance_majority_penalty_rate_list"] = np.array(
        low_budget_rebalance_majority_penalty_rate_list, dtype=float
    )
    batch.non_tensor_batch["sps_low_budget_rebalance_gate_list"] = np.array(
        low_budget_rebalance_gate_list, dtype=float
    )
    batch.non_tensor_batch["sps_low_budget_rebalance_support_list"] = np.array(
        low_budget_rebalance_support_list, dtype=float
    )
    batch.non_tensor_batch["sps_low_budget_rebalance_base_list"] = np.array(
        low_budget_rebalance_base_list, dtype=float
    )
    batch.non_tensor_batch["sps_low_budget_rebalance_process_list"] = np.array(
        low_budget_rebalance_process_list, dtype=float
    )
    batch.non_tensor_batch["sps_process_quality_reward_mean_list"] = np.array(
        process_quality_reward_mean_list, dtype=float
    )
    batch.non_tensor_batch["sps_process_quality_positive_rate_list"] = np.array(
        process_quality_positive_rate_list, dtype=float
    )
    batch.non_tensor_batch["sps_process_quality_negative_rate_list"] = np.array(
        process_quality_negative_rate_list, dtype=float
    )
    batch.non_tensor_batch["sps_process_quality_bad_rate_list"] = np.array(
        process_quality_bad_rate_list, dtype=float
    )
    batch.non_tensor_batch["sps_process_quality_collapse_gate_list"] = np.array(
        process_quality_collapse_gate_list, dtype=float
    )
    return batch


def _batch_majority_vote(model_outputs: List[str], n: int) -> tuple[List[str], List[float]]:
    """
    Used to generate the ground truth for TTRL.
    Input:
        model_outputs: list of str
        n: int
    Output:
        majority_gt_list: list of str
        majority_ratio_list: list of float
    """
    majority_gt_list = []
    majority_ratio_list = []
    assert len(model_outputs) % n == 0
    n_prompts = len(model_outputs) // n
    for i in range(n_prompts):
        prompt_outputs = model_outputs[i * n:(i + 1) * n]
        prompt_majority_gt, prompt_majority_ratio = _majority_vote(prompt_outputs)
        majority_gt_list.append(prompt_majority_gt)
        majority_ratio_list.append(prompt_majority_ratio)
        
    return majority_gt_list, majority_ratio_list


def _majority_vote(model_outputs: List[str]) -> tuple[str, float]:
    assert len(model_outputs) > 0
    model_answers = [extract_answer(generated_text) for generated_text in model_outputs]
    model_answers = [answer for answer in model_answers if answer is not None]
    model_answers = [simplify_expression_string(answer) for answer in model_answers]
    if len(model_answers) == 0:
        return "None", 0.0
    
    counter = Counter(model_answers)
    
    majority_answer, majority_count = counter.most_common(1)[0]
    majority_ratio = majority_count / len(model_outputs)
    
    return majority_answer, majority_ratio


# === Metrics Computation ===


def compute_ttrl_metrics(batch, n):
    """
    Compute the TTRL metrics.
    """
    assert len(batch) % n == 0, "batch length must be divisible by n"
    num_prompts = len(batch) // n

    # Sort the batch by the ID
    idx = sorted(range(len(batch)), key=lambda x: batch[x].non_tensor_batch["extra_info"]["index"])

    majority_reward = []
    gt_reward = []
    majority_label = []
    gt_label = []

    for i in range(len(batch)):
        data_item = batch[idx[i]]
        majority_reward.append(data_item.batch["token_level_scores"].sum().item())
        gt_reward.append(data_item.batch["token_level_scores_original"].sum().item())
        majority_label.append(data_item.non_tensor_batch["reward_model"]["majority_gt"])
        gt_label.append(data_item.non_tensor_batch["reward_model"]["original_gt"]) 

    ttrl_metrics = _batch_compute_ttrl_metrics(majority_reward, gt_reward, majority_label, gt_label, n=n)
    majority_ratio_list = batch.non_tensor_batch["majority_ratio_list"]
    majority_ratio = sum(majority_ratio_list) / len(majority_ratio_list)
    ttrl_metrics["majority_ratio"] = majority_ratio

    return ttrl_metrics


def _batch_compute_ttrl_metrics(
    majority_reward: List[float],
    gt_reward: List[float],
    majority_label: List[str],
    gt_label: List[str],
    n: int,
):
    """
    Compute the TTRL metrics for batch inputs.
    """
    assert len(majority_reward) == len(gt_reward) == len(majority_label) == len(gt_label)
    assert len(majority_reward) % n == 0
    n_prompts = len(majority_reward) // n
    ttrl_metrics = []
    for i in range(n_prompts):
        prompt_majority_reward = majority_reward[i * n:(i + 1) * n]
        prompt_gt_reward = gt_reward[i * n:(i + 1) * n]
        prompt_majority_label = majority_label[i * n:(i + 1) * n]
        prompt_gt_label = gt_label[i * n:(i + 1) * n]

        assert Counter(prompt_majority_label).most_common(1)[0][1] == n
        assert Counter(prompt_gt_label).most_common(1)[0][1] == n

        prompt_majority_label = prompt_majority_label[0]
        prompt_gt_label = prompt_gt_label[0]

        ttrl_metric = _prompt_compute_ttrl_metrics(prompt_majority_reward, prompt_gt_reward, prompt_majority_label, prompt_gt_label)
        ttrl_metrics.append(ttrl_metric)

    # Compute the average metrics
    ttrl_metrics = {k: sum(d[k] for d in ttrl_metrics) / len(ttrl_metrics) for k in ttrl_metrics[0]}

    return ttrl_metrics

def _prompt_compute_ttrl_metrics(
    majority_reward: List[float],
    gt_reward: List[float],
    majority_label: str,
    gt_label: str,
    ):    
    assert len(majority_reward) == len(gt_reward)

    hit_rate = 1.0 if grade(majority_label, gt_label) else 0.0    
    rewards_hit_rate = 0
    for estimate_reward, true_reward in zip(majority_reward, gt_reward):
        if estimate_reward == true_reward:
            rewards_hit_rate += 1
    rewards_hit_rate = rewards_hit_rate / len(majority_reward)
    
    ttrl_metric = {
        "label_accuracy": hit_rate,
        "reward_accuracy": rewards_hit_rate,
        "majority_voting_reward": sum(majority_reward) / len(majority_reward),
        "ground_truth_reward": sum(gt_reward) / len(gt_reward),
        f"pass@{len(majority_reward)}": 1.0 if sum(gt_reward) >= 1 else 0.0,
    }
    return ttrl_metric
