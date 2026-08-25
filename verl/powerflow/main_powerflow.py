#!/usr/bin/env python3
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

"""Main training script for PowerFlow algorithm."""

import os
import socket

import hydra
import ray
from omegaconf import OmegaConf

from verl.trainer.ppo.reward import load_reward_manager
from verl.utils.device import is_cuda_available


def _ensure_transformers_vision2seq_alias() -> None:
    """Keep the original refcode compatible with newer transformers releases."""
    import transformers

    if hasattr(transformers, "AutoModelForVision2Seq"):
        return
    fallback = getattr(transformers, "AutoModelForImageTextToText", None)
    if fallback is None:
        fallback = getattr(transformers, "AutoModel", None)
    if fallback is not None:
        transformers.AutoModelForVision2Seq = fallback


@hydra.main(config_path="config", config_name="powerflow_trainer", version_base=None)
def main(config):
    run_powerflow(config)


def run_powerflow(config) -> None:
    _ensure_transformers_vision2seq_alias()

    if not ray.is_initialized():
        # this is for local ray cluster
        default_runtime_env = {
            "env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN", "VLLM_LOGGING_LEVEL": "WARN"}
        }
        ray_init_kwargs = OmegaConf.to_container(config.ray_kwargs.get("ray_init", {}), resolve=True) or {}
        no_runtime_env = ray_init_kwargs.pop("no_runtime_env", False)
        if no_runtime_env:
            for key, value in default_runtime_env["env_vars"].items():
                os.environ.setdefault(key, value)
            os.environ.setdefault("RAY_USAGE_STATS_ENABLED", "0")
            ray_init_kwargs.pop("runtime_env", None)
        else:
            runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
            runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
            ray_init_kwargs = {**ray_init_kwargs, "runtime_env": OmegaConf.to_container(runtime_env, resolve=True)}
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**ray_init_kwargs)

    try:
        global_profiler = OmegaConf.select(config, "global_profiler")
        if (
            is_cuda_available
            and global_profiler is not None
            and global_profiler.tool == "nsys"
            and OmegaConf.select(global_profiler, "steps") is not None
            and len(OmegaConf.select(global_profiler, "steps")) > 0
        ):
            nsight_options = OmegaConf.to_container(
                global_profiler.global_tool_config.nsys.controller_nsight_options
            )
            runner = TaskRunner.options(runtime_env={"nsight": nsight_options}).remote()
        else:
            runner = TaskRunner.remote()
        ray.get(runner.run.remote(config))
    finally:
        if ray.is_initialized():
            ray.shutdown()


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:
    def run(self, config):
        _ensure_transformers_vision2seq_alias()

        # print initial config
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")

        pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
        OmegaConf.resolve(config)

        # download the checkpoint from hdfs
        local_path = copy_to_local(config.actor_rollout_ref.model.path)

        # instantiate tokenizer
        from verl.utils import hf_processor, hf_tokenizer

        tokenizer = hf_tokenizer(local_path)
        processor = hf_processor(local_path, use_fast=True)  # used for multimodal LLM, could be none

        from verl.single_controller.ray import RayWorkerGroup

        # define worker classes
        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            assert config.critic.strategy in {"fsdp", "fsdp2"}

            # Use PowerFlow custom worker instead of standard worker
            from powerflow.powerflow_fsdp_worker import PowerFlowActorRolloutRefWorker
            from verl.workers.fsdp_workers import CriticWorker  # , ActorRolloutRefWorker

            ActorRolloutRefWorker = PowerFlowActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker

            ray_worker_group_cls = RayWorkerGroup

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
            Role.Critic: ray.remote(CriticWorker),
        }

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        # we should adopt a multi-source reward function here
        # - for rule-based rm, we directly call a reward score
        # - for model-based rm, we call a model
        # - for code related prompt, we send to a sandbox if there are test cases
        # - finally, we combine all the rewards together
        # - The reward type depends on the tag of the data
        if config.reward_model.enable:
            if config.reward_model.strategy in {"fsdp", "fsdp2"}:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        # reference model
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        reward_fn = load_reward_manager(
            config,
            tokenizer,
            0,
            max_resp_len=config.data.max_response_length,
            overlong_buffer_cfg=config.reward_model.overlong_buffer,
        )

        # Note that we always use function-based RM for validation
        val_reward_fn = load_reward_manager(
            config,
            tokenizer,
            1,
            max_resp_len=config.data.max_response_length,
            overlong_buffer_cfg=config.reward_model.overlong_buffer,
        )
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        from powerflow.powerflow_ray_trainer import RayPowerFlowTrainer

        trainer = RayPowerFlowTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
        )
        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    main()
