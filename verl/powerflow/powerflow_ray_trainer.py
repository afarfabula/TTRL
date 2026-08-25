# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
PowerFlow Ray Trainer that extends RayPPOTrainer with PowerFlow-specific components.
"""

import json
import os
import time
import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint
from typing import Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
try:
    from verl.experimental.dataset.sampler import AbstractCurriculumSampler
except ModuleNotFoundError:
    class AbstractCurriculumSampler:
        pass
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
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
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
try:
    from verl.utils.rollout_skip import RolloutSkip
except ModuleNotFoundError:
    RolloutSkip = None
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger

from verl.trainer.ppo.ray_trainer import RayPPOTrainer, Role, WorkerType, apply_kl_penalty, compute_advantage, compute_response_mask


class RayPowerFlowTrainer(RayPPOTrainer):
    """
    PowerFlow trainer that uses the PowerFlow advantage estimator.
    The main difference is in the advantage estimation which is registered
    as 'powerflow' in powerflow_adv_estimator.py
    """

    def __init__(self, *args, **kwargs):
        config = kwargs.get("config", args[0] if args else None)
        original_adv_estimator = None
        if config is not None and str(config.algorithm.adv_estimator) == "grpo_ref":
            original_adv_estimator = config.algorithm.adv_estimator
            config.algorithm.adv_estimator = AdvantageEstimator.GRPO
        try:
            super().__init__(*args, **kwargs)
        finally:
            if original_adv_estimator is not None:
                config.algorithm.adv_estimator = original_adv_estimator

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
        for key in ["multi_modal_data", "raw_prompt", "tools_kwargs", "interaction_kwargs", "agent_name"]:
            if key in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append(key)
        return batch.pop(batch_keys=batch_keys_to_pop, non_tensor_batch_keys=non_tensor_batch_keys_to_pop)
    
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

        current_epoch = self.global_steps // len(self.train_dataloader)

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            if RolloutSkip is None:
                raise RuntimeError("rollout.skip_rollout=True requires verl.utils.rollout_skip.RolloutSkip")
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0
        profile_steps = OmegaConf.select(self.config, "global_profiler.steps")
        profile_continuous_steps = bool(
            OmegaConf.select(self.config, "global_profiler.profile_continuous_steps", default=False)
        )

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in profile_steps
            if profile_steps is not None
            else False
        )
        next_step_profile = False

        print(f"[Training] Starting training loop: {self.config.trainer.total_epochs} epochs, {self.total_training_steps} total steps")
        
        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            print(f"[Training] Starting epoch {epoch + 1}/{self.config.trainer.total_epochs}")
            for batch_dict in self.train_dataloader:
                print(f"[Step {self.global_steps}] Starting training step...")
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    should_start_profile = (
                        not prev_step_profile and curr_step_profile
                        if profile_continuous_steps
                        else curr_step_profile
                    )
                    if should_start_profile and hasattr(self, "_start_profiling"):
                        self._start_profiling(should_start_profile)
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                is_last_step = self.global_steps >= self.total_training_steps
                step_start_time = time.time()
                with marked_timer("step", timing_raw):
                    # generate a batch
                    gen_start = time.time()
                    print(f"[Step {self.global_steps}] ▶ Generating sequences...")
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)

                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)
                    gen_duration = time.time() - gen_start
                    print(f"[Step {self.global_steps}] ✓ Sequence generation completed ({gen_duration:.1f}s)")

                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    batch.batch["token_level_scores"]=torch.zeros_like(batch.batch["response_mask"], dtype=torch.float32)
                    batch.batch["token_level_rewards"]=torch.zeros_like(batch.batch["response_mask"], dtype=torch.float32)
                    batch.batch["advantages"]=torch.zeros_like(batch.batch["response_mask"], dtype=torch.float32)
                    batch.batch["returns"]=torch.zeros_like(batch.batch["response_mask"], dtype=torch.float32)
                    
                    reward_start = time.time()
                    print(f"[Step {self.global_steps}] ▶ Computing rewards...")
                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(
                                data=batch, config=self.config, tokenizer=self.tokenizer
                            )
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
                        batch.batch["boxed_reward"] = reward_tensor
                    reward_duration = time.time() - reward_start
                    print(f"[Step {self.global_steps}] ✓ Reward computation completed ({reward_duration:.1f}s)")


                    # Operating Mode Selection:
                    # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
                    # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
                    #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                    if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
                        from verl.trainer.ppo.rollout_corr_helper import apply_rollout_correction

                        apply_rollout_correction(
                            batch=batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                    else:  # Recompute old_log_probs
                        logprob_start = time.time()
                        print(f"[Step {self.global_steps}] ▶ Computing log probabilities...")
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
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
                                rollout_probs = torch.exp(batch.batch["rollout_log_probs"])
                                actor_probs = torch.exp(batch.batch["old_log_probs"])
                                rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                                rollout_probs_diff = torch.masked_select(
                                    rollout_probs_diff, batch.batch["response_mask"].bool()
                                )
                                metrics.update(
                                    {
                                        "training/rollout_probs_diff_max": rollout_probs_diff.max().detach().item(),
                                        "training/rollout_probs_diff_mean": rollout_probs_diff.mean().detach().item(),
                                        "training/rollout_probs_diff_std": rollout_probs_diff.std().detach().item(),
                                    }
                                )
                        logprob_duration = time.time() - logprob_start
                        print(f"[Step {self.global_steps}] ✓ Log probabilities completed ({logprob_duration:.1f}s)")

                    assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                    if self.use_reference_policy:
                        # compute reference log_prob
                        ref_start = time.time()
                        print(f"[Step {self.global_steps}] ▶ Computing reference policy log probabilities...")
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)
                        ref_duration = time.time() - ref_start
                        print(f"[Step {self.global_steps}] ✓ Reference policy log probabilities completed ({ref_duration:.1f}s)")

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        actor_start = time.time()
                        print(f"[Step {self.global_steps}] ▶ Updating actor policy...")
                        with marked_timer("update_actor", timing_raw, color="red"):
                            rollout_config = self.config.actor_rollout_ref.rollout
                            batch.meta_info["multi_turn"] = rollout_config.multi_turn.enable
                            # TODO: Make "temperature" single source of truth from generation.
                            batch.meta_info["temperature"] = rollout_config.temperature
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)
                        actor_duration = time.time() - actor_start
                        print(f"[Step {self.global_steps}] ✓ Actor policy update completed ({actor_duration:.1f}s)")

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, {}, timing_raw, rollout_data_dir)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    print(f"[Step {self.global_steps}] Running validation...")
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)
                    print(f"[Step {self.global_steps}] Validation completed")

                # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                # Check if the conditions for saving a checkpoint are met.
                # The conditions include a mandatory condition (1) and
                # one of the following optional conditions (2/3/4):
                # 1. The save frequency is set to a positive value.
                # 2. It's the last training step.
                # 3. The current step number is a multiple of the save frequency.
                # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    print(f"[Step {self.global_steps}] Saving checkpoint...")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()
                    print(f"[Step {self.global_steps}] Checkpoint saved")

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in profile_steps
                        if profile_steps is not None
                        else False
                    )
                    should_stop_profile = (
                        curr_step_profile and not next_step_profile
                        if profile_continuous_steps
                        else curr_step_profile
                    )
                    if should_stop_profile and hasattr(self, "_stop_profiling"):
                        self._stop_profiling(should_stop_profile)
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # Format step duration for display
                if steps_duration >= 60:
                    minutes = int(steps_duration // 60)
                    seconds = steps_duration % 60
                    duration_str = f"{minutes}m {seconds:.1f}s"
                else:
                    duration_str = f"{steps_duration:.1f}s"

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
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                total_step_duration = time.time() - step_start_time
                print(f"[Step {self.global_steps}] ═══ Step completed in {duration_str} ═══ Progress: {self.global_steps}/{self.total_training_steps}")
                self.global_steps += 1

                if OmegaConf.select(self.config, "actor_rollout_ref.actor.profiler.tool") == "torch_memory":
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)
