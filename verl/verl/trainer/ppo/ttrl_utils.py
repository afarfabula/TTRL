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
        else:
            candidate_ranked = sorted(candidate_ranked, key=lambda item: item[0], reverse=True)
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
            else:
                fill = sorted(fill_candidates, key=lambda item: item[0], reverse=True)
            chosen.extend(fill[: n_samples_per_prompt - len(chosen)])
        assert len(chosen) == n_samples_per_prompt

        selected_indices.extend(idx for _, idx, _, _, _ in chosen)
        selected_parseable_rates.append(float(np.mean([parseable for _, _, parseable, _, _ in chosen])))
        selected_clip_rates.append(float(np.mean([clipped for _, _, _, clipped, _ in chosen])))
        selected_cluster_rates.append(float(np.mean([in_cluster for _, _, _, _, in_cluster in chosen])))
        fallback_rates.append(fallback)

    return (
        data[selected_indices],
        {
            "parseable_rate": np.array(selected_parseable_rates, dtype=float),
            "clip_rate": np.array(selected_clip_rates, dtype=float),
            "cluster_rate": np.array(selected_cluster_rates, dtype=float),
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

    temp = max(float(weight_temperature), 1e-6)
    sharpen_beta = max(float(answer_sharpen_beta), 1e-6)
    max_response_len = response_mask.shape[-1]
    for i in range(num_prompts):
        answer_to_scores = {}
        answers = []
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
            if answer is None:
                continue
            answer = simplify_expression_string(answer)
            answers.append(answer)
            answer_to_scores.setdefault(answer, []).append(scores[start + j] / temp)

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
            continue

        answer_scores = {
            answer: torch.logsumexp(torch.stack(vals), dim=0)
            for answer, vals in answer_to_scores.items()
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
        majority_sharp_confidence = sharp_prob_by_answer.get(majority_gt, 0.0)
        weighted_sharp_confidence = sharp_prob_by_answer.get(weighted_gt, 0.0)
        prompt_lengths = lengths[start : start + n]
        prompt_clip_ratio = float((prompt_lengths >= max_response_len).to(torch.float32).mean().item())

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
        if confidence_weight:
            if answer_sharpen_capacity:
                agreement_confidence = weighted_sharp_confidence if weighted_gt == majority_gt else 0.0
            else:
                agreement_confidence = weighted_confidence if weighted_gt == majority_gt else 0.0
            prompt_weight = max(float(majority_ratio), float(agreement_confidence))
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
