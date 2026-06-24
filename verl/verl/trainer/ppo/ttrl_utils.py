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

    temp = max(float(weight_temperature), 1e-6)
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
            continue

        answer_scores = {
            answer: torch.logsumexp(torch.stack(vals), dim=0)
            for answer, vals in answer_to_scores.items()
        }
        weighted_gt = max(answer_scores.items(), key=lambda item: item[1].item())[0]
        stacked = torch.stack(list(answer_scores.values()))
        probs = torch.softmax(stacked - stacked.max(), dim=0)

        counter = Counter(answers)
        majority_gt, majority_count = counter.most_common(1)[0]
        majority_ratio = majority_count / n
        weighted_confidence = float(probs.max().item())

        use_sps_label = True
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

    for i in range(num_prompts):
        data_item = batch[i]
        original_gt = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        data_item.non_tensor_batch["reward_model"]["ground_truth"] = selected_gt_list[i]
        data_item.non_tensor_batch["reward_model"]["majority_gt"] = selected_gt_list[i]
        data_item.non_tensor_batch["reward_model"]["sps_weighted_gt"] = weighted_gt_list[i]
        data_item.non_tensor_batch["reward_model"]["raw_majority_gt"] = raw_majority_gt_list[i]
        data_item.non_tensor_batch["reward_model"]["original_gt"] = original_gt

    batch.non_tensor_batch["majority_ratio_list"] = np.array(majority_ratio_list, dtype=float)
    batch.non_tensor_batch["sps_weighted_confidence_list"] = np.array(weighted_confidence_list, dtype=float)
    batch.non_tensor_batch["sps_unique_answer_count_list"] = np.array(unique_answer_count_list, dtype=float)
    batch.non_tensor_batch["sps_override_list"] = np.array(sps_override_list, dtype=float)
    batch.non_tensor_batch["sps_agreement_list"] = np.array(sps_agreement_list, dtype=float)
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
