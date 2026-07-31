# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Optional, Type

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from tensordict import TensorDict
import verl.utils.torch_functional as verl_F
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.debug import marked_timer
from verl.utils.metric import (
    reduce_metrics,
)
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger

WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(
                    f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes}"
                    + "cannot be satisfied in this ray cluster"
                )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator,
    gamma=1.0,
    lam=1.0,
    num_repeat=1,
    multi_turn=False,
    norm_adv_by_std_in_grpo=True,
    config=None,
):
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator: The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.get("pf_ppo_reweight_method", "pow"),
                config.get("pf_ppo_weight_pow", 2.0),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]
        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name="cuda",
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to "cuda".
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name
        self.validation_generations_logger = ValidationGenerationsLogger()

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
            AdvantageEstimator.GRPO,
            AdvantageEstimator.GRPO_PASSK,
            AdvantageEstimator.REINFORCE_PLUS_PLUS,
            AdvantageEstimator.REMAX,
            AdvantageEstimator.RLOO,
            AdvantageEstimator.OPO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes
        if config.actor_rollout_ref.actor.strategy == "megatron":
            model_parallel_size = (
                config.actor_rollout_ref.actor.megatron.tensor_model_parallel_size
                * config.actor_rollout_ref.actor.megatron.pipeline_model_parallel_size
            )
            assert (
                n_gpus % (model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size) == 0
            ), (
                f"n_gpus ({n_gpus}) must be divisible by model_parallel_size ({model_parallel_size}) times "
                f"context_parallel_size ({config.actor_rollout_ref.actor.megatron.context_parallel_size})"
            )
            megatron_dp = n_gpus // (
                model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size
            )
            minimal_bsz = megatron_dp * config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu
        else:
            minimal_bsz = n_gpus

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % minimal_bsz == 0, (
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by minimal possible batch size "
            f"({minimal_bsz})"
        )

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            settings = {
                "actor_rollout_ref.actor": "micro_batch_size",
                "critic": "micro_batch_size",
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(
                        f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'."
                    )

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(
                        f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove "
                        f"'{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
                    )

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.actor.ppo_micro_batch_size,
                config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                "actor_rollout_ref.actor",
            )

            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "actor_rollout_ref.ref",
                )

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(
                config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu, "critic"
            )

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(
                config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model"
            )

        # Actor
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            sp_size = config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert (
                    config.actor_rollout_ref.actor.ppo_mini_batch_size
                    % config.actor_rollout_ref.actor.ppo_micro_batch_size
                    == 0
                )
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        assert config.actor_rollout_ref.actor.loss_agg_mode in [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ], f"Invalid loss_agg_mode: {config.actor_rollout_ref.actor.loss_agg_mode}"

        if config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print("NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get("ulysses_sequence_parallel_size", 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == "fsdp" and (
            config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1) > 1
            or config.actor_rollout_ref.ref.get("ulysses_sequence_parallel_size", 1) > 1
        ):
            assert config.actor_rollout_ref.model.use_remove_padding, (
                "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."
            )

        if self.use_critic and config.critic.strategy == "fsdp":
            if config.critic.get("ulysses_sequence_parallel_size", 1) > 1:
                assert config.critic.model.use_remove_padding, (
                    "When using sequence parallelism for critic, you must enable `use_remove_padding`."
                )

        if config.data.get("val_batch_size", None) is not None:
            print(
                "WARNING: val_batch_size is deprecated."
                + " Validation datasets are sent to inference engines as a whole batch,"
                + " which will schedule the memory themselves."
            )

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, (
                "validation gen temperature should be greater than 0 when enabling do_sample"
            )

        # check multi_turn with tool config
        if config.actor_rollout_ref.rollout.multi_turn.enable:
            assert (
                config.actor_rollout_ref.rollout.multi_turn.tool_config_path is not None
                or config.actor_rollout_ref.rollout.multi_turn.interaction_config_path is not None
            ), (
                "tool_config_path or interaction_config_path must be set when enabling multi_turn with tool, "
                "due to no role-playing support"
            )
            assert config.algorithm.adv_estimator in [AdvantageEstimator.GRPO], (
                "only GRPO is tested for multi-turn with tool"
            )

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files, self.config.data, self.tokenizer, self.processor
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files, self.config.data, self.tokenizer, self.processor
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []
        sample_turns = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            if "interaction_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("interaction_kwargs")
            if "agent_name" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("agent_name")
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # evaluate using reward_function
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            print(f"len reward_extra_infos_dict['reward']: {len(reward_extra_infos_dict['reward'])}")
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)
                    print(f"len reward_extra_infos_dict['{key}']: {len(reward_extra_infos_dict[key])}")

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy], config=self.config.actor_rollout_ref, role="ref"
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.trainer, "profile_steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.trainer, "profile_steps")
            assert OmegaConf.select(self.config.trainer, "worker_nsight_options") is not None, (
                "worker_nsight_options must be set when profile_steps is set"
            )
            wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                OmegaConf.select(self.config.trainer, "worker_nsight_options")
            )

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                device_name=self.device_name,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop import AgentLoopManager

            self.async_rollout_mode = True
            self.async_rollout_manager = AgentLoopManager(
                config=self.config,
                worker_group=self.actor_rollout_wg,
            )

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _compute_original_gt_rewards(self, batch: DataProto) -> torch.Tensor:
        original_non_tensor = dict(batch.non_tensor_batch)
        if "reward_model" not in original_non_tensor:
            reward_tensor, _ = compute_reward(batch, self.reward_fn)
            return reward_tensor.sum(dim=-1).detach().cpu().float()

        reward_models = []
        has_original_gt = False
        for reward_model in original_non_tensor["reward_model"]:
            reward_model_copy = dict(reward_model)
            if "original_gt" in reward_model_copy:
                reward_model_copy["ground_truth"] = reward_model_copy["original_gt"]
                has_original_gt = True
            reward_models.append(reward_model_copy)
        if not has_original_gt:
            reward_tensor, _ = compute_reward(batch, self.reward_fn)
            return reward_tensor.sum(dim=-1).detach().cpu().float()

        original_non_tensor["reward_model"] = np.asarray(reward_models, dtype=object)
        original_batch = DataProto(
            batch=batch.batch,
            non_tensor_batch=original_non_tensor,
            meta_info=batch.meta_info,
        )
        reward_tensor, _ = compute_reward(original_batch, self.reward_fn)
        return reward_tensor.sum(dim=-1).detach().cpu().float()

    def _make_chunk_state_prompts(
        self,
        batch: DataProto,
        source_correctness: Optional[torch.Tensor] = None,
    ) -> tuple[DataProto, list[int]]:
        cfg = self.config.ttrl
        n = int(cfg.n_samples_per_prompt)
        states_per_prompt = int(cfg.get("chunk_state_states_per_prompt", 1))
        max_prefix_tokens = int(cfg.get("chunk_state_max_prefix_tokens", 1024))
        boundaries = [int(x) for x in cfg.get("chunk_state_boundaries", [0, 256, 512, 768, 1024])]
        source_mode = str(cfg.get("chunk_state_source_mode", "random"))
        if not boundaries:
            boundaries = [0]

        prompt_count = len(batch) // n
        prompt_len = batch.batch["prompts"].shape[-1]
        max_prompt_length = int(self.config.data.max_prompt_length)
        pad_token_id = self.tokenizer.pad_token_id

        state_input_ids = []
        state_attention_masks = []
        source_indices = []
        source_prompt_indices = []
        source_locals = []
        state_boundaries = []
        source_response_lengths = []
        for prompt_idx in range(prompt_count):
            for state_idx in range(states_per_prompt):
                source_offset = self.global_steps + prompt_idx + state_idx
                if source_mode == "success" and source_correctness is not None:
                    prompt_scores = source_correctness[prompt_idx * n : (prompt_idx + 1) * n]
                    good_locals = torch.nonzero(prompt_scores > 0.0, as_tuple=False).flatten()
                    if good_locals.numel() > 0:
                        source_local = int(good_locals[source_offset % good_locals.numel()].item())
                    else:
                        source_local = source_offset % n
                else:
                    source_local = source_offset % n
                source_index = prompt_idx * n + source_local
                valid_prompt_len = int(batch.batch["attention_mask"][source_index, :prompt_len].sum().item())
                prompt_ids = batch.batch["prompts"][source_index, -valid_prompt_len:]
                response_mask = batch.batch["response_mask"][source_index].bool()
                valid_response_len = int(response_mask.sum().item())
                allowed_boundaries = [
                    b for b in boundaries if b <= valid_response_len and b <= max_prefix_tokens
                ]
                if not allowed_boundaries:
                    allowed_boundaries = [0]
                boundary = allowed_boundaries[(self.global_steps + prompt_idx + state_idx) % len(allowed_boundaries)]
                prefix_ids = batch.batch["responses"][source_index, :boundary]
                state_ids = torch.cat([prompt_ids, prefix_ids], dim=0)
                if state_ids.numel() > max_prompt_length:
                    state_ids = state_ids[-max_prompt_length:]
                attn = torch.ones_like(state_ids)
                state_ids, attn = verl_F.postprocess_data(
                    input_ids=state_ids.unsqueeze(0),
                    attention_mask=attn.unsqueeze(0),
                    max_length=max_prompt_length,
                    pad_token_id=pad_token_id,
                    left_pad=True,
                    truncation="left",
                )
                state_input_ids.append(state_ids.squeeze(0))
                state_attention_masks.append(attn.squeeze(0))
                source_indices.append(source_index)
                source_prompt_indices.append(prompt_idx)
                source_locals.append(source_local)
                state_boundaries.append(boundary)
                source_response_lengths.append(valid_response_len)

        input_ids = torch.stack(state_input_ids, dim=0)
        attention_mask = torch.stack(state_attention_masks, dim=0)
        position_ids = (attention_mask.cumsum(dim=-1) - 1).clamp(min=0)
        state_batch = TensorDict(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=(input_ids.shape[0],),
        )
        state_non_tensor = {}
        for key, value in batch.non_tensor_batch.items():
            state_non_tensor[key] = value[np.asarray(source_indices, dtype=np.int64)]
        state_non_tensor["chunk_state_source_index"] = np.asarray(source_indices, dtype=np.int64)
        state_non_tensor["chunk_state_source_prompt_index"] = np.asarray(source_prompt_indices, dtype=np.int64)
        state_non_tensor["chunk_state_source_local"] = np.asarray(source_locals, dtype=np.int64)
        state_non_tensor["chunk_state_boundary"] = np.asarray(state_boundaries, dtype=np.int64)
        state_non_tensor["chunk_state_source_response_len"] = np.asarray(source_response_lengths, dtype=np.int64)
        if source_correctness is not None:
            state_non_tensor["chunk_state_source_original_correct"] = (
                source_correctness[np.asarray(source_indices, dtype=np.int64)].numpy().astype(np.float32)
            )
        state_proto = DataProto(batch=state_batch, non_tensor_batch=state_non_tensor)
        state_proto.meta_info = {
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "recompute_log_prob": False,
            "do_sample": True,
        }
        return state_proto, source_indices

    def _compute_chunk_state_diag_metrics(
        self,
        full_batch: DataProto,
        state_prompts: DataProto,
        scores: torch.Tensor,
    ) -> dict:
        cfg = self.config.ttrl
        candidates = int(cfg.get("chunk_state_candidates", 8))
        num_states = len(state_prompts)
        score_matrix = scores.view(num_states, candidates).float()
        metrics = {}

        boundaries = torch.as_tensor(state_prompts.non_tensor_batch["chunk_state_boundary"], dtype=torch.float32)
        source_response_lens = torch.as_tensor(
            state_prompts.non_tensor_batch["chunk_state_source_response_len"], dtype=torch.float32
        )
        metrics.update(
            {
                "chunk_state_diag/boundary_mean": boundaries.mean().item(),
                "chunk_state_diag/boundary_min": boundaries.min().item(),
                "chunk_state_diag/boundary_max": boundaries.max().item(),
                "chunk_state_diag/boundary_zero_ratio": (boundaries == 0).float().mean().item(),
                "chunk_state_diag/source_response_len_mean": source_response_lens.mean().item(),
                "chunk_state_diag/probe_score_std": score_matrix.std(unbiased=False).item(),
                "chunk_state_diag/state_probe_mean_max": score_matrix.mean(dim=-1).max().item(),
                "chunk_state_diag/state_probe_mean_min": score_matrix.mean(dim=-1).min().item(),
                "chunk_state_diag/state_all_positive_ratio": (score_matrix.min(dim=-1).values > 0.0).float().mean().item(),
                "chunk_state_diag/state_all_negative_ratio": (score_matrix.max(dim=-1).values <= 0.0).float().mean().item(),
                "chunk_state_diag/state_mixed_ratio": (
                    (score_matrix.max(dim=-1).values > 0.0) & (score_matrix.min(dim=-1).values <= 0.0)
                ).float().mean().item(),
            }
        )

        if not bool(cfg.get("chunk_state_diag_enable", False)):
            return metrics

        pseudo_reward_tensor, _ = compute_reward(full_batch, self.reward_fn)
        pseudo_rewards = pseudo_reward_tensor.sum(dim=-1).detach().cpu().float()

        original_rewards = self._compute_original_gt_rewards(full_batch)

        source_indices = torch.as_tensor(state_prompts.non_tensor_batch["chunk_state_source_index"], dtype=torch.long)
        source_prompt_indices = torch.as_tensor(
            state_prompts.non_tensor_batch["chunk_state_source_prompt_index"], dtype=torch.long
        )
        n = int(cfg.n_samples_per_prompt)
        prompt_count = len(full_batch) // n
        pseudo_prompt = pseudo_rewards.view(prompt_count, n)
        original_prompt = original_rewards.view(prompt_count, n)
        selected_pseudo = pseudo_rewards[source_indices]
        selected_original = original_rewards[source_indices]
        selected_prompt_original_mean = original_prompt[source_prompt_indices].mean(dim=-1)
        selected_prompt_original_pass = (original_prompt[source_prompt_indices].max(dim=-1).values > 0.0).float()
        selected_prompt_pseudo_mean = pseudo_prompt[source_prompt_indices].mean(dim=-1)

        probe_mean = score_matrix.mean(dim=-1)
        correct_mask = selected_original > 0.0
        wrong_mask = ~correct_mask
        metrics.update(
            {
                "chunk_state_diag/source_pseudo_acc_mean": selected_pseudo.mean().item(),
                "chunk_state_diag/source_original_acc_mean": selected_original.mean().item(),
                "chunk_state_diag/prompt_pseudo_mean": selected_prompt_pseudo_mean.mean().item(),
                "chunk_state_diag/prompt_original_mean": selected_prompt_original_mean.mean().item(),
                "chunk_state_diag/prompt_original_pass": selected_prompt_original_pass.mean().item(),
                "chunk_state_diag/probe_mean_source_original_correct": (
                    probe_mean[correct_mask].mean().item() if correct_mask.any() else 0.0
                ),
                "chunk_state_diag/probe_mean_source_original_wrong": (
                    probe_mean[wrong_mask].mean().item() if wrong_mask.any() else 0.0
                ),
                "chunk_state_diag/source_original_correct_ratio": correct_mask.float().mean().item(),
            }
        )
        return metrics

    def _dump_chunk_state_diag_jsonl(
        self,
        state_prompts: DataProto,
        scores: torch.Tensor,
    ) -> dict:
        path = self.config.ttrl.get("chunk_state_diag_jsonl", None)
        if not path:
            return {}

        candidates = int(self.config.ttrl.get("chunk_state_candidates", 8))
        score_matrix = scores.view(len(state_prompts), candidates).float().detach().cpu()
        boundaries = np.asarray(state_prompts.non_tensor_batch["chunk_state_boundary"], dtype=np.int64)
        source_indices = np.asarray(state_prompts.non_tensor_batch["chunk_state_source_index"], dtype=np.int64)
        source_prompt_indices = np.asarray(
            state_prompts.non_tensor_batch["chunk_state_source_prompt_index"], dtype=np.int64
        )
        source_locals = np.asarray(state_prompts.non_tensor_batch["chunk_state_source_local"], dtype=np.int64)
        source_response_lens = np.asarray(
            state_prompts.non_tensor_batch["chunk_state_source_response_len"], dtype=np.int64
        )
        source_original = state_prompts.non_tensor_batch.get("chunk_state_source_original_correct", None)
        if source_original is not None:
            source_original = np.asarray(source_original, dtype=np.float32)

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        rows = 0
        with open(path, "a", encoding="utf-8") as f:
            for state_idx in range(len(state_prompts)):
                state_scores = score_matrix[state_idx]
                row = {
                    "global_step": int(self.global_steps),
                    "state_index": int(state_idx),
                    "source_index": int(source_indices[state_idx]),
                    "source_prompt_index": int(source_prompt_indices[state_idx]),
                    "source_local": int(source_locals[state_idx]),
                    "boundary": int(boundaries[state_idx]),
                    "source_response_len": int(source_response_lens[state_idx]),
                    "source_original_correct": (
                        float(source_original[state_idx]) if source_original is not None else None
                    ),
                    "probe_mean": float(state_scores.mean().item()),
                    "probe_max": float(state_scores.max().item()),
                    "probe_min": float(state_scores.min().item()),
                    "probe_positive_count": int((state_scores > 0.0).sum().item()),
                    "probe_scores": [float(x) for x in state_scores.tolist()],
                    "all_negative": bool((state_scores <= 0.0).all().item()),
                    "all_positive": bool((state_scores > 0.0).all().item()),
                }
                row["mixed"] = (not row["all_negative"]) and (not row["all_positive"])
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows += 1
        return {"chunk_state_diag/jsonl_rows": float(rows)}

    def _combine_state_and_completion_prompts(
        self,
        state_prompts: DataProto,
        chunk_output: DataProto,
        max_completion_tokens: int,
    ) -> DataProto:
        prompt_len = state_prompts.batch["input_ids"].shape[-1]
        response_mask = chunk_output.batch["response_mask"].bool()
        combined_ids = []
        combined_masks = []
        candidates = len(chunk_output) // len(state_prompts)
        for i in range(len(chunk_output)):
            source_idx = i // candidates
            state_valid_len = int(state_prompts.batch["attention_mask"][source_idx].sum().item())
            state_ids = state_prompts.batch["input_ids"][source_idx, -state_valid_len:]
            chunk_len = int(response_mask[i].sum().item())
            chunk_ids = chunk_output.batch["responses"][i, :chunk_len]
            combined = torch.cat([state_ids, chunk_ids], dim=0)
            if combined.numel() > prompt_len:
                combined = combined[-prompt_len:]
            attn = torch.ones_like(combined)
            combined, attn = verl_F.postprocess_data(
                input_ids=combined.unsqueeze(0),
                attention_mask=attn.unsqueeze(0),
                max_length=prompt_len,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation="left",
            )
            combined_ids.append(combined.squeeze(0))
            combined_masks.append(attn.squeeze(0))

        input_ids = torch.stack(combined_ids, dim=0)
        attention_mask = torch.stack(combined_masks, dim=0)
        position_ids = (attention_mask.cumsum(dim=-1) - 1).clamp(min=0)
        prompt_batch = TensorDict(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=(len(chunk_output),),
        )
        non_tensor_batch = {}
        for key, value in state_prompts.non_tensor_batch.items():
            non_tensor_batch[key] = np.repeat(value, len(chunk_output) // len(state_prompts), axis=0)
        proto = DataProto(batch=prompt_batch, non_tensor_batch=non_tensor_batch)
        proto.meta_info = {
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "recompute_log_prob": False,
            "do_sample": True,
            "kwargs": {"n": 1, "max_tokens": int(max_completion_tokens)},
        }
        return proto

    def _build_probe_reward_batch(
        self,
        state_prompts: DataProto,
        chunk_output: DataProto,
        probe_output: DataProto,
    ) -> DataProto:
        candidates = len(chunk_output) // len(state_prompts)
        prompt_ids = []
        prompt_masks = []
        prompt_position_ids = []
        responses = []
        response_masks = []
        chunk_mask = chunk_output.batch["response_mask"].bool()
        probe_mask = probe_output.batch["response_mask"].bool()
        max_response_len = chunk_output.batch["responses"].shape[-1] + probe_output.batch["responses"].shape[-1]

        for i in range(len(chunk_output)):
            source_idx = i // candidates
            prompt_ids.append(state_prompts.batch["input_ids"][source_idx])
            prompt_masks.append(state_prompts.batch["attention_mask"][source_idx])
            prompt_position_ids.append(state_prompts.batch["position_ids"][source_idx])

            chunk_len = int(chunk_mask[i].sum().item())
            probe_len = int(probe_mask[i].sum().item())
            response_ids = torch.cat(
                [
                    chunk_output.batch["responses"][i, :chunk_len],
                    probe_output.batch["responses"][i, :probe_len],
                ],
                dim=0,
            )
            padded = torch.full(
                (max_response_len,),
                self.tokenizer.pad_token_id,
                dtype=response_ids.dtype,
                device=response_ids.device,
            )
            padded[: response_ids.numel()] = response_ids
            mask = torch.zeros((max_response_len,), dtype=state_prompts.batch["attention_mask"].dtype, device=response_ids.device)
            mask[: response_ids.numel()] = 1
            responses.append(padded)
            response_masks.append(mask)

        prompt_ids = torch.stack(prompt_ids, dim=0)
        prompt_masks = torch.stack(prompt_masks, dim=0)
        prompt_position_ids = torch.stack(prompt_position_ids, dim=0)
        responses = torch.stack(responses, dim=0)
        response_masks = torch.stack(response_masks, dim=0)
        input_ids = torch.cat([prompt_ids, responses], dim=-1)
        attention_mask = torch.cat([prompt_masks, response_masks], dim=-1)
        delta_position_id = torch.arange(1, responses.shape[-1] + 1, device=prompt_ids.device).unsqueeze(0)
        response_position_ids = prompt_position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([prompt_position_ids, response_position_ids], dim=-1)

        reward_batch = TensorDict(
            {
                "prompts": prompt_ids,
                "responses": responses,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "response_mask": response_masks,
            },
            batch_size=(len(chunk_output),),
        )
        non_tensor_batch = {
            key: np.repeat(value, candidates, axis=0) for key, value in state_prompts.non_tensor_batch.items()
        }
        return DataProto(batch=reward_batch, non_tensor_batch=non_tensor_batch)

    def _repeat_non_tensor_like(self, source: DataProto, target: DataProto, repeat_times: int) -> DataProto:
        if len(target) == len(source) * repeat_times:
            target.non_tensor_batch = {
                key: np.repeat(value, repeat_times, axis=0) for key, value in source.non_tensor_batch.items()
            }
        return target

    def _build_chunk_state_response_span(
        self,
        chunk_output: DataProto,
        probe_output: DataProto | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        cfg = self.config.ttrl
        actor_span = str(cfg.get("chunk_state_actor_span", "chunk"))
        chunk_responses = chunk_output.batch["responses"]
        chunk_mask = chunk_output.batch["response_mask"]
        if actor_span == "chunk":
            metrics = {
                "chunk_state_actor_span/mode_chunk": 1.0,
                "chunk_state_actor_span/response_len_mean": chunk_mask.sum(dim=-1).float().mean().item(),
                "chunk_state_actor_span/truncated_ratio": 0.0,
            }
            return chunk_responses, chunk_mask, metrics
        if actor_span != "chunk_probe":
            raise ValueError(f"Unsupported ttrl.chunk_state_actor_span={actor_span!r}")
        if probe_output is None:
            raise ValueError("probe_output is required when ttrl.chunk_state_actor_span='chunk_probe'")

        max_response_len = int(self.config.data.max_response_length)
        pad_token_id = self.tokenizer.pad_token_id
        probe_responses = probe_output.batch["responses"]
        probe_mask = probe_output.batch["response_mask"]
        responses = []
        response_masks = []
        truncated = []
        response_lengths = []
        for idx in range(len(chunk_output)):
            chunk_len = int(chunk_mask[idx].sum().item())
            probe_len = int(probe_mask[idx].sum().item())
            span = torch.cat(
                [
                    chunk_responses[idx, :chunk_len],
                    probe_responses[idx, :probe_len],
                ],
                dim=0,
            )
            was_truncated = int(span.numel() > max_response_len)
            if was_truncated:
                span = span[:max_response_len]
            padded = torch.full(
                (max_response_len,),
                pad_token_id,
                dtype=chunk_responses.dtype,
                device=chunk_responses.device,
            )
            mask = torch.zeros(
                (max_response_len,),
                dtype=chunk_mask.dtype,
                device=chunk_mask.device,
            )
            padded[: span.numel()] = span
            mask[: span.numel()] = 1
            responses.append(padded)
            response_masks.append(mask)
            truncated.append(was_truncated)
            response_lengths.append(span.numel())

        responses = torch.stack(responses, dim=0)
        response_masks = torch.stack(response_masks, dim=0)
        metrics = {
            "chunk_state_actor_span/mode_chunk_probe": 1.0,
            "chunk_state_actor_span/response_len_mean": float(np.mean(response_lengths)) if response_lengths else 0.0,
            "chunk_state_actor_span/truncated_ratio": float(np.mean(truncated)) if truncated else 0.0,
        }
        return responses, response_masks, metrics

    def _apply_chunk_state_teacher_anchor(
        self,
        full_batch: DataProto,
        state_prompts: DataProto,
        chunk_output: DataProto,
    ) -> dict:
        cfg = self.config.ttrl
        candidates = int(cfg.get("chunk_state_candidates", 8))
        anchor_idx = int(cfg.get("chunk_state_teacher_anchor_candidate_index", 0))
        chunk_size = int(cfg.get("chunk_state_chunk_size", 256))
        if anchor_idx < 0 or anchor_idx >= candidates:
            raise ValueError(
                f"chunk_state_teacher_anchor_candidate_index={anchor_idx} outside [0, {candidates})"
            )

        source_indices = np.asarray(state_prompts.non_tensor_batch["chunk_state_source_index"], dtype=np.int64)
        boundaries = np.asarray(state_prompts.non_tensor_batch["chunk_state_boundary"], dtype=np.int64)
        full_response_mask = full_batch.batch["response_mask"].bool()
        replacements = 0
        anchor_lengths = []
        for state_idx, (source_index, boundary) in enumerate(zip(source_indices, boundaries)):
            valid_response_len = int(full_response_mask[source_index].sum().item())
            if boundary >= valid_response_len:
                continue
            anchor_len = min(chunk_size, valid_response_len - int(boundary))
            if anchor_len <= 0:
                continue
            target_idx = state_idx * candidates + anchor_idx
            chunk_output.batch["responses"][target_idx].fill_(self.tokenizer.pad_token_id)
            chunk_output.batch["response_mask"][target_idx].zero_()
            anchor_tokens = full_batch.batch["responses"][source_index, int(boundary) : int(boundary) + anchor_len]
            chunk_output.batch["responses"][target_idx, :anchor_len] = anchor_tokens
            chunk_output.batch["response_mask"][target_idx, :anchor_len] = 1
            replacements += 1
            anchor_lengths.append(anchor_len)

        state_count = len(state_prompts)
        return {
            "chunk_state_teacher_anchor/enabled": 1.0,
            "chunk_state_teacher_anchor/replaced_ratio": replacements / max(state_count, 1),
            "chunk_state_teacher_anchor/mean_len": float(np.mean(anchor_lengths)) if anchor_lengths else 0.0,
            "chunk_state_teacher_anchor/candidate_index": float(anchor_idx),
        }

    def _build_chunk_actor_batch(
        self,
        state_prompts: DataProto,
        chunk_output: DataProto,
        scores: torch.Tensor,
        probe_output: DataProto | None = None,
    ) -> tuple[DataProto, dict]:
        cfg = self.config.ttrl
        num_states = len(state_prompts)
        candidates = int(cfg.get("chunk_state_candidates", 8))
        alpha = float(cfg.get("chunk_state_alpha", 2.0))
        eps = float(cfg.get("chunk_state_eps", 0.05))
        skip_uniform = bool(cfg.get("chunk_state_skip_uniform", False))
        skip_all_negative = bool(cfg.get("chunk_state_skip_all_negative", False))

        score_matrix = scores.view(num_states, candidates).float()
        raw_weights = torch.pow(score_matrix + eps, alpha)
        weight_sums = raw_weights.sum(dim=-1, keepdim=True)
        uniform = torch.full_like(raw_weights, 1.0 / candidates)
        weights = torch.where(weight_sums > 0, raw_weights / weight_sums.clamp(min=1e-12), uniform)
        score_max = score_matrix.max(dim=-1).values
        score_min = score_matrix.min(dim=-1).values
        informative = ((score_max - score_min) > float(cfg.get("chunk_state_min_informative_gap", 0.0))).float()
        if skip_uniform:
            keep_state = informative.bool()
        else:
            keep_state = torch.ones(num_states, dtype=torch.bool)
        if skip_all_negative:
            keep_state &= score_max > 0.0
        keep_indices = []
        for state_idx in range(num_states):
            if keep_state[state_idx]:
                keep_indices.extend(range(state_idx * candidates, (state_idx + 1) * candidates))
        if not keep_indices:
            keep_indices = list(range(len(chunk_output)))

        repeated_state_prompts = state_prompts.repeat(repeat_times=candidates, interleave=True)
        kept_states = repeated_state_prompts[keep_indices]
        responses, response_mask, span_metrics = self._build_chunk_state_response_span(
            chunk_output=chunk_output,
            probe_output=probe_output,
        )
        responses = responses[keep_indices]
        response_mask = response_mask[keep_indices]
        flat_weights = weights.reshape(-1)[keep_indices]
        powerflow_flat_weights = flat_weights * candidates

        prompt_ids = kept_states.batch["input_ids"]
        prompt_mask = kept_states.batch["attention_mask"]
        input_ids = torch.cat([prompt_ids, responses], dim=-1)
        attention_mask = torch.cat([prompt_mask, response_mask], dim=-1)
        response_len = responses.shape[-1]
        delta_position_id = torch.arange(1, response_len + 1, device=prompt_ids.device).unsqueeze(0)
        response_position_ids = kept_states.batch["position_ids"][..., -1:] + delta_position_id
        position_ids = torch.cat([kept_states.batch["position_ids"], response_position_ids], dim=-1)
        actor_batch = TensorDict(
            {
                "prompts": prompt_ids,
                "responses": responses,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "response_mask": response_mask,
                "old_log_probs": torch.zeros_like(responses, dtype=torch.float32),
                "advantages": torch.zeros_like(responses, dtype=torch.float32),
                "chunk_weights": flat_weights.to(dtype=torch.float32),
                "powerflow_chunk_weights": powerflow_flat_weights.to(dtype=torch.float32),
                "boxed_reward": torch.zeros_like(responses, dtype=torch.float32),
            },
            batch_size=(len(keep_indices),),
        )
        response_lengths = response_mask.sum(dim=-1).clamp(min=1).long()
        actor_batch["boxed_reward"][torch.arange(len(keep_indices)), response_lengths - 1] = score_matrix.reshape(-1)[
            keep_indices
        ].to(dtype=torch.float32)
        actor_proto = DataProto(
            batch=actor_batch,
            non_tensor_batch=kept_states.non_tensor_batch,
            meta_info={
                "temperature": self.config.actor_rollout_ref.rollout.temperature,
                "multi_turn": self.config.actor_rollout_ref.rollout.multi_turn.enable,
            },
        )
        metrics = {
            "chunk_state/num_states": float(num_states),
            "chunk_state/num_candidates": float(len(chunk_output)),
            "chunk_state/num_actor_samples": float(len(actor_proto)),
            "chunk_state/positive_ratio": score_matrix.mean().detach().item(),
            "chunk_state/informative_ratio": informative.mean().detach().item(),
            "chunk_state/kept_state_ratio": keep_state.float().mean().detach().item(),
            "chunk_state/target_entropy": (-(weights * torch.log(weights.clamp(min=1e-12))).sum(dim=-1).mean()).detach().item(),
            "chunk_state/weight_max": weights.max().detach().item(),
            "chunk_state/weight_min": weights.min().detach().item(),
            "chunk_state/powerflow_weight_mean": powerflow_flat_weights.mean().detach().item(),
            "chunk_state/powerflow_weight_max": powerflow_flat_weights.max().detach().item(),
            "chunk_state/powerflow_weight_min": powerflow_flat_weights.min().detach().item(),
        }
        metrics.update(span_metrics)
        return actor_proto, metrics

    def _run_chunk_state_training_step(self, full_batch: DataProto, metrics: dict, timing_raw: dict):
        cfg = self.config.ttrl
        candidates = int(cfg.get("chunk_state_candidates", 8))
        chunk_size = int(cfg.get("chunk_state_chunk_size", 256))
        probe_max_tokens = int(cfg.get("chunk_state_probe_max_tokens", self.config.data.max_response_length))
        max_model_len = int(self.config.actor_rollout_ref.rollout.max_model_len)
        max_prompt_len = int(self.config.data.max_prompt_length)
        source_mode = str(cfg.get("chunk_state_source_mode", "random"))
        source_correctness = None
        if source_mode == "success" or bool(cfg.get("chunk_state_diag_enable", False)):
            source_correctness = self._compute_original_gt_rewards(full_batch)

        with marked_timer("chunk_state_make_states", timing_raw, color="cyan"):
            state_prompts, _ = self._make_chunk_state_prompts(full_batch, source_correctness=source_correctness)
            if source_correctness is not None:
                n = int(cfg.n_samples_per_prompt)
                prompt_count = len(full_batch) // n
                original_prompt = source_correctness.view(prompt_count, n)
                source_indices = torch.as_tensor(
                    state_prompts.non_tensor_batch["chunk_state_source_index"], dtype=torch.long
                )
                metrics["chunk_state_source/selected_original_acc_mean"] = source_correctness[
                    source_indices
                ].float().mean().item()
                metrics["chunk_state_source/prompt_original_pass"] = (
                    original_prompt.max(dim=-1).values > 0.0
                ).float().mean().item()
                metrics["chunk_state_source/prompt_original_mean"] = original_prompt.mean(dim=-1).mean().item()

        with marked_timer("chunk_state_chunks", timing_raw, color="red"):
            chunk_prompts = deepcopy(state_prompts)
            chunk_prompts.meta_info["kwargs"] = {"n": candidates, "max_tokens": chunk_size}
            chunk_output = self.actor_rollout_wg.generate_sequences(chunk_prompts)
            chunk_output = self._repeat_non_tensor_like(state_prompts, chunk_output, candidates)
            if bool(cfg.get("chunk_state_teacher_anchor_enable", False)):
                metrics.update(self._apply_chunk_state_teacher_anchor(full_batch, state_prompts, chunk_output))
            if "timing" in chunk_output.meta_info:
                for key, value in chunk_output.meta_info["timing"].items():
                    timing_raw[f"chunk_state_chunks/{key}"] = value
                chunk_output.meta_info.pop("timing", None)

        with marked_timer("chunk_state_probe", timing_raw, color="red"):
            probe_tokens = max(1, min(probe_max_tokens, max_model_len - max_prompt_len))
            probe_prompts = self._combine_state_and_completion_prompts(
                state_prompts=state_prompts,
                chunk_output=chunk_output,
                max_completion_tokens=probe_tokens,
            )
            probe_output = self.actor_rollout_wg.generate_sequences(probe_prompts)
            probe_output.non_tensor_batch = probe_prompts.non_tensor_batch
            if "timing" in probe_output.meta_info:
                for key, value in probe_output.meta_info["timing"].items():
                    timing_raw[f"chunk_state_probe/{key}"] = value
                probe_output.meta_info.pop("timing", None)

        with marked_timer("chunk_state_score", timing_raw, color="yellow"):
            probe_batch = self._build_probe_reward_batch(state_prompts, chunk_output, probe_output)
            reward_tensor, _ = compute_reward(probe_batch, self.reward_fn)
            scores = reward_tensor.sum(dim=-1).detach().cpu()
            if bool(cfg.get("chunk_state_teacher_anchor_enable", False)):
                anchor_idx = int(cfg.get("chunk_state_teacher_anchor_candidate_index", 0))
                anchor_score = float(cfg.get("chunk_state_teacher_anchor_score", 1.0))
                score_matrix = scores.view(len(state_prompts), candidates)
                score_matrix[:, anchor_idx] = torch.maximum(
                    score_matrix[:, anchor_idx],
                    torch.full_like(score_matrix[:, anchor_idx], anchor_score),
                )
                scores = score_matrix.reshape(-1)
                metrics["chunk_state_teacher_anchor/score_floor"] = anchor_score
            metrics.update(self._compute_chunk_state_diag_metrics(full_batch, state_prompts, scores))
            metrics.update(self._dump_chunk_state_diag_jsonl(state_prompts, scores))

        with marked_timer("chunk_state_build_actor_batch", timing_raw, color="blue"):
            actor_batch, chunk_metrics = self._build_chunk_actor_batch(
                state_prompts,
                chunk_output,
                scores,
                probe_output=probe_output,
            )
            metrics.update(chunk_metrics)
            actor_batch.meta_info["global_token_num"] = torch.sum(actor_batch.batch["attention_mask"], dim=-1).tolist()

        if self.config.actor_rollout_ref.actor.get("powerflow_enable", False):
            with marked_timer("chunk_state_ref", timing_raw, color="olive"):
                if not self.ref_in_actor:
                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(actor_batch)
                else:
                    ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(actor_batch)
                actor_batch = actor_batch.union(ref_log_prob)

        with marked_timer("update_actor", timing_raw, color="red"):
            actor_output = self.actor_rollout_wg.update_actor(actor_batch)
        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
        metrics.update(actor_output_metrics)
        return actor_batch

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        repeat_sampling_sglang_grpo = (
            self.config.actor_rollout_ref.rollout.name == "sglang"
            and self.config.actor_rollout_ref.rollout.multi_turn.enable
        )

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                do_profile = (
                    self.global_steps in self.config.trainer.profile_steps
                    if self.config.trainer.profile_steps is not None
                    else False
                )
                if do_profile:
                    self.actor_rollout_wg.start_profile()
                    if self.use_reference_policy:
                        self.ref_policy_wg.start_profile()
                    if self.use_critic:
                        self.critic_wg.start_profile()
                    if self.use_rm:
                        self.rm_wg.start_profile()

                metrics = {}
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]

                non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]

                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                if "interaction_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("interaction_kwargs")
                if "agent_name" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("agent_name")
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                if repeat_sampling_sglang_grpo:
                    uids_for_prompts = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                    batch.non_tensor_batch["uid"] = uids_for_prompts
                    gen_batch.non_tensor_batch["uid"] = uids_for_prompts
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    assert np.array_equal(batch.non_tensor_batch["uid"], gen_batch.non_tensor_batch["uid"]), (
                        "UIDs must be identical for SGLang rollout"
                    )

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if self.config.get("ttrl", {}).get("enable", False):
                            from verl.trainer.ppo.ttrl_utils import select_top_k_per_prompt, apply_ttrl_gt

                            powerflow_no_majority = self.config.ttrl.get("powerflow_no_majority", False)
                            if powerflow_no_majority:
                                # Match the original PowerFlow dataflow: expand prompts first and
                                # let vLLM sample one response per expanded prompt, instead of
                                # asking vLLM for n responses per original prompt.
                                batch.non_tensor_batch["uid"] = np.array(
                                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                                )
                                batch = batch.repeat(
                                    repeat_times=self.config.ttrl.n_samples_per_prompt,
                                    interleave=True,
                                )
                                gen_batch = gen_batch.repeat(
                                    repeat_times=self.config.ttrl.n_samples_per_prompt,
                                    interleave=True,
                                )
                                rollout_n = 1
                            else:
                                rollout_n = self.config.ttrl.n_votes_per_prompt
                            gen_batch.meta_info["global_steps"] = self.global_steps
                            gen_batch.meta_info["kwargs"] = {"n": rollout_n}
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                            assert len(gen_batch_output) == len(batch) * rollout_n

                            if not powerflow_no_majority:
                                batch = apply_ttrl_gt(
                                    batch,
                                    gen_batch_output,
                                    self.config.ttrl.n_votes_per_prompt,
                                    self.tokenizer,
                                    self.config.ttrl.get("majority_vote_num_processes", 0),
                                )
                                gen_batch_output = select_top_k_per_prompt(gen_batch_output, self.config.ttrl.n_votes_per_prompt, self.config.ttrl.n_samples_per_prompt)

                            expected_samples = 1 if powerflow_no_majority else self.config.ttrl.n_samples_per_prompt
                            assert len(gen_batch_output) == len(batch) * expected_samples
                        else:
                            if not self.async_rollout_mode:
                                gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                            else:
                                # vllm should set async_rollout_mode to enable async rollout
                                # sglang turns on async_rollout_mode by default
                                gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    if not repeat_sampling_sglang_grpo and not (
                        self.config.get("ttrl", {}).get("enable", False)
                        and self.config.ttrl.get("powerflow_no_majority", False)
                    ):
                        batch.non_tensor_batch["uid"] = np.array(
                            [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                        )
                        # repeat to align with repeated responses in rollout
                        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch:
                        batch.batch["response_mask"] = compute_response_mask(batch)

                    if (
                        self.config.get("ttrl", {}).get("enable", False)
                        and self.config.ttrl.get("chunk_state_enable", False)
                    ):
                        batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                        self._run_chunk_state_training_step(batch, metrics, timing_raw)

                        rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                        if rollout_data_dir:
                            metrics["chunk_state/rollout_dump_skipped"] = 1.0

                        if (
                            self.val_reward_fn is not None
                            and self.config.trainer.test_freq > 0
                            and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                        ):
                            with marked_timer("testing", timing_raw, color="green"):
                                val_metrics: dict = self._validate()
                                if is_last_step:
                                    last_val_metrics = val_metrics
                            metrics.update(val_metrics)

                        metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                        if "step" in timing_raw:
                            metrics.update(
                                compute_throughout_metrics(
                                    batch=batch,
                                    timing_raw=timing_raw,
                                    n_gpus=self.config.trainer.n_gpus_per_node,
                                )
                            )
                        logger.log(data=metrics, step=self.global_steps)
                        self.global_steps += 1
                        if is_last_step:
                            pprint(f"Final validation metrics: {last_val_metrics}")
                            progress_bar.close()
                            return
                        progress_bar.update(1)
                        continue

                    if (
                        self.config.get("ttrl", {}).get("enable", False)
                        and self.config.ttrl.get("sharpened_enable", False)
                        and not self.config.ttrl.get("powerflow_no_majority", False)
                    ):
                        from verl.trainer.ppo.ttrl_utils import apply_sharpened_ttrl_reward

                        batch = apply_sharpened_ttrl_reward(
                            batch,
                            self.config.ttrl.n_samples_per_prompt,
                            self.tokenizer,
                            self.config,
                        )
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        use_rollout_log_probs_as_old = self.config.actor_rollout_ref.rollout.get(
                            "use_rollout_log_probs_as_old", False
                        )
                        if use_rollout_log_probs_as_old and "rollout_log_probs" in batch.batch.keys():
                            batch.batch["old_log_probs"] = batch.batch["rollout_log_probs"]
                            batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                        else:
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                            entropy_agg = agg_loss(
                                loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode
                            )
                            old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            rollout_old_log_probs = batch.batch["rollout_log_probs"]
                            actor_old_log_probs = batch.batch["old_log_probs"]
                            attention_mask = batch.batch["attention_mask"]
                            responses = batch.batch["responses"]
                            response_length = responses.size(1)
                            response_mask = attention_mask[:, -response_length:]

                            rollout_probs = torch.exp(rollout_old_log_probs)
                            actor_probs = torch.exp(actor_old_log_probs)
                            rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                            rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
                            rollout_probs_diff_max = torch.max(rollout_probs_diff)
                            rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
                            rollout_probs_diff_std = torch.std(rollout_probs_diff)
                            metrics.update(
                                {
                                    "training/rollout_probs_diff_max": rollout_probs_diff_max.detach().item(),
                                    "training/rollout_probs_diff_mean": rollout_probs_diff_mean.detach().item(),
                                    "training/rollout_probs_diff_std": rollout_probs_diff_std.detach().item(),
                                }
                            )

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        powerflow_enabled = self.config.actor_rollout_ref.actor.get("powerflow_enable", False)
                        if powerflow_enabled:
                            batch.batch["boxed_reward"] = reward_tensor
                            batch.batch["token_level_scores"] = torch.zeros_like(reward_tensor)
                            batch.batch["token_level_rewards"] = torch.zeros_like(reward_tensor)
                            batch.batch["advantages"] = torch.zeros_like(reward_tensor)
                            batch.batch["returns"] = torch.zeros_like(reward_tensor)
                            metrics["train/powerflow_observed_reward"] = (
                                reward_tensor.sum(dim=-1).float().mean().detach().item()
                            )
                            metrics["train/powerflow_no_majority"] = float(
                                self.config.ttrl.get("powerflow_no_majority", False)
                            )
                        else:
                            if (
                                self.config.get("ttrl", {}).get("enable", False)
                                and self.config.ttrl.get("sharpened_enable", False)
                                and "sharpened_token_level_scores" in batch.batch
                            ):
                                reward_tensor = batch.batch["sharpened_token_level_scores"]
                                metrics["train/sharpened_reward"] = (
                                    reward_tensor.sum(dim=-1).float().mean().detach().item()
                                )
                                for key in [
                                    "sharpened_coverage",
                                    "sharpened_top_prob",
                                    "sharpened_entropy",
                                    "sharpened_negative_rate",
                                    "sharpened_label_hit",
                                    "sharpened_negative_hit",
                                ]:
                                    if key in batch.non_tensor_batch:
                                        metrics[f"train/{key}"] = float(np.mean(batch.non_tensor_batch[key]))
                            batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        if not powerflow_enabled:
                            # compute rewards. apply_kl_penalty if available
                            if self.config.algorithm.use_kl_in_reward:
                                batch, kl_metrics = apply_kl_penalty(
                                    batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                                )
                                metrics.update(kl_metrics)
                            else:
                                batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                            # compute advantages, executed on the driver process

                            norm_adv_by_std_in_grpo = self.config.algorithm.get(
                                "norm_adv_by_std_in_grpo", True
                            )  # GRPO adv normalization factor

                            batch = compute_advantage(
                                batch,
                                adv_estimator=self.config.algorithm.adv_estimator,
                                gamma=self.config.algorithm.gamma,
                                lam=self.config.algorithm.lam,
                                num_repeat=self.config.actor_rollout_ref.rollout.n,
                                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                                multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                                config=self.config.algorithm,
                            )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    if self.config.get("ttrl", {}).get("enable", False) and not self.config.ttrl.get("powerflow_no_majority", False):
                        from verl.trainer.ppo.ttrl_utils import apply_original_gt, compute_ttrl_metrics
                        batch = apply_original_gt(batch)
                        reward_tensor_original, reward_extra_infos_dict_original = compute_reward(batch, self.reward_fn)
                        batch.batch["token_level_scores_original"] = reward_tensor_original
                        # Compute ttrl metrics
                        ttrl_metrics = compute_ttrl_metrics(batch, self.config.ttrl.n_samples_per_prompt)
                        for key, value in ttrl_metrics.items():
                                metrics.update({f"train/{key}": value})

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
                            print(batch.batch.keys())
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                    # validate
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    ):
                        with marked_timer("testing", timing_raw, color="green"):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    esi_close_to_expiration = should_save_ckpt_esi(
                        max_steps_duration=self.max_steps_duration,
                        redundant_time=self.config.trainer.esi_redundant_time,
                    )
                    if self.config.trainer.save_freq > 0 and (
                        is_last_step
                        or self.global_steps % self.config.trainer.save_freq == 0
                        or esi_close_to_expiration
                    ):
                        if esi_close_to_expiration:
                            print("Force saving checkpoint: ESI instance expiration approaching.")
                        with marked_timer("save_checkpoint", timing_raw, color="green"):
                            self._save_checkpoint()

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)
                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if do_profile:
                    self.actor_rollout_wg.stop_profile()
                    if self.use_reference_policy:
                        self.ref_policy_wg.stop_profile()
                    if self.use_critic:
                        self.critic_wg.stop_profile()
                    if self.use_rm:
                        self.rm_wg.stop_profile()

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return
