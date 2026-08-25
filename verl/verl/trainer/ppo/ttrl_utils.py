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
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
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


def apply_ttrl_gt(batch, gen_batch_output, n, tokenizer, majority_vote_num_processes=0):
    """
    Apply the majority vote ground truth to the batch.
    """
    assert len(gen_batch_output) % n == 0, "gen_batch_output length must be divisible by n"
    num_prompts = len(gen_batch_output) // n
    assert len(batch) == num_prompts, "batch length must be equal to the number of prompts"

    response_ids = gen_batch_output.batch["responses"]
    prompt_length = gen_batch_output.batch["prompts"].shape[-1]
    valid_response_lengths = gen_batch_output.batch["attention_mask"][:, prompt_length:].sum(dim=-1).cpu().tolist()
    valid_response_ids = [response_ids[i][: int(valid_response_lengths[i])] for i in range(len(gen_batch_output))]
    model_outputs = tokenizer.batch_decode(valid_response_ids, skip_special_tokens=True)

    majority_gt_list, majority_ratio_list = _batch_majority_vote(
        model_outputs, n, num_processes=majority_vote_num_processes
    )
    
    assert len(batch) == len(majority_gt_list), "batch length must be equal to the number of model outputs"
    
    for i in range(num_prompts):
        data_item = batch[i]
        original_gt = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        data_item.non_tensor_batch["reward_model"]["ground_truth"] = majority_gt_list[i]
        data_item.non_tensor_batch["reward_model"]["majority_gt"] = majority_gt_list[i]
        data_item.non_tensor_batch["reward_model"]["original_gt"] = original_gt

    batch.non_tensor_batch["majority_ratio_list"] = np.array(majority_ratio_list, dtype=float)
    return batch


def apply_sharpened_ttrl_reward(batch, n, tokenizer, config):
    """
    Attach a selective sharpened answer-posterior reward to each sampled response.

    This keeps the original TTRL/GRPO update path intact while replacing the
    hard majority reward with a soft answer-level target distribution.
    """
    assert len(batch) % n == 0, "batch length must be divisible by n"
    num_prompts = len(batch) // n

    cfg = config.ttrl
    alpha = float(cfg.get("sharpened_alpha", 2.0))
    tau_pos = float(cfg.get("sharpened_tau_pos", 0.375))
    tau_marg = float(cfg.get("sharpened_tau_marg", 0.125))
    use_confidence = bool(cfg.get("sharpened_use_confidence", True))
    conf_temperature = max(float(cfg.get("sharpened_conf_temperature", 1.0)), 1e-6)
    reward_mode = str(cfg.get("sharpened_reward_mode", "prob"))
    soft_coef = float(cfg.get("sharpened_soft_coef", 0.25))
    negative_enable = bool(cfg.get("sharpened_negative_enable", True))
    tau_low = float(cfg.get("sharpened_tau_low", 0.125))
    negative_reward = float(cfg.get("sharpened_negative_reward", -0.25))

    response_mask = batch.batch["response_mask"].bool()
    reward_values = np.zeros(len(batch), dtype=np.float32)
    labels = []
    coverage = []
    top_probs = []
    sharpened_entropies = []
    negative_rates = []
    label_hits = []
    negative_hits = []

    original_gts = []
    decoded_outputs = []
    for i in range(len(batch)):
        data_item = batch[i]
        response_ids = data_item.batch["responses"]
        valid_len = int(data_item.batch["response_mask"].sum().item())
        response_str = tokenizer.decode(response_ids[:valid_len], skip_special_tokens=True)
        answer = extract_answer(response_str)
        if answer is not None:
            answer = simplify_expression_string(answer)
        decoded_outputs.append(answer)
        original_gts.append(data_item.non_tensor_batch["reward_model"]["original_gt"])

    if "rollout_log_probs" in batch.batch:
        log_probs = batch.batch["rollout_log_probs"].detach()
        token_counts = response_mask.sum(dim=-1).clamp(min=1)
        confidence = ((log_probs * response_mask).sum(dim=-1) / token_counts).cpu().numpy()
    else:
        confidence = np.ones(len(batch), dtype=np.float32)
    confidence = np.asarray(confidence, dtype=np.float64)

    for prompt_idx in range(num_prompts):
        start = prompt_idx * n
        end = start + n
        prompt_answers = decoded_outputs[start:end]
        prompt_conf = confidence[start:end]
        prompt_gt = original_gts[start]

        answer_weight = defaultdict(float)
        for local_idx, answer in enumerate(prompt_answers):
            if answer is None or answer == "None":
                continue
            weight = 1.0
            if use_confidence:
                weight = float(np.exp(prompt_conf[local_idx] / conf_temperature))
            answer_weight[answer] += weight

        if not answer_weight:
            labels.append("None")
            coverage.append(0.0)
            top_probs.append(0.0)
            sharpened_entropies.append(0.0)
            negative_rates.append(0.0)
            label_hits.append(0.0)
            negative_hits.append(0.0)
            continue

        total_weight = sum(answer_weight.values())
        posterior = {answer: weight / total_weight for answer, weight in answer_weight.items()}
        ranked = sorted(posterior.items(), key=lambda item: item[1], reverse=True)
        top_answer, top_prob = ranked[0]
        second_prob = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = top_prob - second_prob
        accepted = top_prob >= tau_pos and margin >= tau_marg

        if accepted:
            logits = {answer: alpha * np.log(max(prob, 1e-12)) for answer, prob in posterior.items()}
            max_logit = max(logits.values())
            exp_logits = {answer: np.exp(logit - max_logit) for answer, logit in logits.items()}
            if reward_mode == "top1_ratio":
                top_value = max(exp_logits.get(top_answer, 0.0), 1e-12)
                sharpened = {answer: min(float(value / top_value), 1.0) for answer, value in exp_logits.items()}
            elif reward_mode == "mv_anchor":
                top_value = max(exp_logits.get(top_answer, 0.0), 1e-12)
                sharpened = {
                    answer: min(float(value / top_value) * soft_coef, soft_coef)
                    for answer, value in exp_logits.items()
                }
            else:
                norm = sum(exp_logits.values())
                sharpened = {answer: value / norm for answer, value in exp_logits.items()}
        else:
            sharpened = {answer: 0.0 for answer in posterior}

        negative_answers = set()
        if negative_enable:
            for answer, prob in posterior.items():
                if answer != top_answer and prob <= tau_low:
                    negative_answers.add(answer)

        neg_count = 0
        neg_hit = 0
        prompt_majority_gt = batch[start].non_tensor_batch["reward_model"].get("majority_gt", None)
        for local_idx, answer in enumerate(prompt_answers):
            global_idx = start + local_idx
            if answer in sharpened:
                reward_values[global_idx] = float(sharpened[answer])
            majority_match = (
                reward_mode == "mv_anchor"
                and answer is not None
                and answer != "None"
                and prompt_majority_gt is not None
                and grade(answer, prompt_majority_gt)
            )
            if majority_match:
                reward_values[global_idx] = 1.0
            elif answer in negative_answers:
                reward_values[global_idx] = negative_reward
                neg_count += 1
                if not grade(answer, prompt_gt):
                    neg_hit += 1

        entropy = -sum(prob * np.log(max(prob, 1e-12)) for prob in sharpened.values() if prob > 0)
        labels.append(top_answer if accepted else "None")
        coverage.append(1.0 if accepted else 0.0)
        top_probs.append(float(top_prob))
        sharpened_entropies.append(float(entropy))
        negative_rates.append(neg_count / n)
        label_hits.append(1.0 if accepted and grade(top_answer, prompt_gt) else 0.0)
        negative_hits.append((neg_hit / neg_count) if neg_count else 0.0)

    reward_tensor = torch.zeros_like(batch.batch["responses"], dtype=torch.float32)
    response_lengths = response_mask.sum(dim=-1).clamp(min=1)
    reward_positions = response_lengths - 1
    reward_tensor[torch.arange(len(batch), device=reward_tensor.device), reward_positions] = torch.tensor(
        reward_values, device=reward_tensor.device, dtype=torch.float32
    )

    batch.batch["sharpened_token_level_scores"] = reward_tensor
    batch.non_tensor_batch["sharpened_label"] = np.repeat(np.array(labels, dtype=object), n)
    batch.non_tensor_batch["sharpened_coverage"] = np.repeat(np.array(coverage, dtype=np.float32), n)
    batch.non_tensor_batch["sharpened_top_prob"] = np.repeat(np.array(top_probs, dtype=np.float32), n)
    batch.non_tensor_batch["sharpened_entropy"] = np.repeat(np.array(sharpened_entropies, dtype=np.float32), n)
    batch.non_tensor_batch["sharpened_negative_rate"] = np.repeat(np.array(negative_rates, dtype=np.float32), n)
    batch.non_tensor_batch["sharpened_label_hit"] = np.repeat(np.array(label_hits, dtype=np.float32), n)
    batch.non_tensor_batch["sharpened_negative_hit"] = np.repeat(np.array(negative_hits, dtype=np.float32), n)
    return batch


def _batch_majority_vote(model_outputs: List[str], n: int, num_processes: int = 0) -> tuple[List[str], List[float]]:
    """
    Used to generate the ground truth for TTRL.
    Input:
        model_outputs: list of str
        n: int
    Output:
        majority_gt_list: list of str
        majority_ratio_list: list of float
    """
    assert len(model_outputs) % n == 0
    n_prompts = len(model_outputs) // n
    if num_processes and num_processes > 1:
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            results = list(executor.map(_majority_vote, (model_outputs[i * n:(i + 1) * n] for i in range(n_prompts))))
    else:
        results = [_majority_vote(model_outputs[i * n:(i + 1) * n]) for i in range(n_prompts)]

    majority_gt_list = []
    majority_ratio_list = []
    for i in range(n_prompts):
        prompt_majority_gt, prompt_majority_ratio = results[i]
        majority_gt_list.append(prompt_majority_gt)
        majority_ratio_list.append(prompt_majority_ratio)
        
    return majority_gt_list, majority_ratio_list


def _majority_vote(model_outputs: List[str]) -> tuple[str, float]:
    assert len(model_outputs) > 0
    raw_answers = [extract_answer(generated_text) for generated_text in model_outputs]
    raw_answers = [answer for answer in raw_answers if answer is not None]
    if len(raw_answers) == 0:
        return "None", 0.0

    # Avoid repeatedly running sympy simplification for duplicated answers in
    # the same vote group. Counting after canonicalization preserves semantics.
    raw_counter = Counter(raw_answers)
    counter = Counter()
    for answer, count in raw_counter.items():
        counter[simplify_expression_string(answer)] += count
    
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
