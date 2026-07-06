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
            continue

        answer_scores = {
            answer: torch.logsumexp(torch.stack(vals), dim=0)
            for answer, vals in answer_to_scores.items()
        }
        base_answer_scores = {
            answer: torch.logsumexp(torch.stack(vals), dim=0)
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
        process_capacity_value = process_majority_support
        if process_capacity_value <= 0.0:
            process_capacity_value = max(0.0, min(1.0, float(process_disagreement_penalty))) * process_consistent_rate

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
        process_consistency_capacity_list.append(process_capacity_value)
        process_majority_support_list.append(process_majority_support)
        process_consistent_rate_list.append(process_consistent_rate)
        process_majority_consistent_rate_list.append(process_majority_consistent_rate)
        process_tail_rate_list.append(process_tail_rate)
        process_box_conflict_rate_list.append(process_box_conflict_rate)
        process_revision_after_final_rate_list.append(process_revision_after_final_rate)

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
