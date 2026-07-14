# TTRL Math500 Qwen2.5-Math-7B Paper-Style B32 Run

## Summary

This run was created to reproduce the original TTRL Qwen2.5-Math math500 training setup more closely than the earlier `bs8/310` run.  The target comparison point was the reported `mean@16=83.4`.

Final result:

- Final `val-core/math/acc/mean@16`: `0.8275`
- Final `val-core/math/acc/maj@16/mean`: `0.8530`
- Final `val-core/math/acc/best@16/mean`: `0.8850`
- Peak `val-core/math/acc/mean@16`: `0.8300` at step `140`
- Completed progress: `150/150`
- Wall-clock progress line: `5:11:28<00:00, 124.59s/it`
- Plain non-validation step average: `108.894s`
- Plain non-validation step median: `108.628s`

Conclusion:

- The run reproduced the reported number to the same range: peak `mean@16=83.0%`, within `0.4` points of `83.4%`.
- The earlier `bs8/310` result was lower mainly because the training semantics were not aligned with the paper-style run. This run used `batch_size=32`, `10` epochs, and natural dataloader step count.

## Key Paths

- Persistent run script: `/mlx_devbox/users/quyanyi/playground/TTRL/verl/run_records/ttrl_math500_paper_b32_20260714_101500.sh`
- Run script copy in this record: `run_script.sh`
- Full runtime log: `/tmp/ttrl_b200/logs/ttrl_math500_paper_b32_20260714_101500.log`
- Checkpoint directory: `/tmp/ttrl_b200/checkpoints/ttrl_math500_paper_b32_20260714_101500/global_step_150`
- Experiment record directory: `/mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/ttrl_math500_paper_b32_20260714_101500`

Large artifacts were not copied into git:

- Raw log size: `121092624` bytes
- Checkpoint size: `86G`
- Checkpoint file count under `global_step_150`: `26`

## Artifact Hashes

- Log SHA256: `e4d0df3dc6bbfd76c9780794b5d017df1bbf36666946734cb72bf5c8ab223b88`
- Run script SHA256: `393f706ea2797a5fdfc45d247895d869b036c0b546905db74db821ea785b1a91`

## Configuration

The run used the local B200 launcher with paper-style training semantics:

- Model: `/models/Qwen2.5-Math-7B`
- Data:
  - `/mlx_devbox/users/quyanyi/playground/TTRL/verl/data/MATH-TTT/train.parquet`
  - `/mlx_devbox/users/quyanyi/playground/TTRL/verl/data/MATH-TTT/test.parquet`
- `data.train_batch_size=32`
- `trainer.total_epochs=10`
- `trainer.total_training_steps=None`
- Actual computed training steps: `150`
- `ttrl.n_votes_per_prompt=64`
- `ttrl.n_samples_per_prompt=32`
- `data.max_prompt_length=1024`
- `data.max_response_length=3072`
- `actor_rollout_ref.rollout.temperature=1.0`
- `actor_rollout_ref.rollout.val_kwargs.n=16`
- `actor_rollout_ref.rollout.val_kwargs.temperature=0.6`
- `actor_rollout_ref.rollout.val_kwargs.top_p=0.95`
- `actor_rollout_ref.actor.ppo_mini_batch_size=1`
- `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2`
- `actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2`
- `actor_rollout_ref.ref.fsdp_config.param_offload=True`
- `trainer.test_freq=20`
- `trainer.save_freq=2000000`
- `trainer.resume_mode=disable`
- `trainer.val_before_train=False`

Infra retained without changing the training semantics:

- Persistent venv under `/mlx_devbox/users/quyanyi/playground/.venvs/ttrl_b200`
- Short Ray/runtime directory: `/tmp/pb32`
- Logs/checkpoints under `/tmp/ttrl_b200`
- vLLM rollout with `FLASH_ATTN`
- Fused Triton kernels enabled
- Rule-based math reward with `reward_manager=prime`
- Rollout logprob reuse enabled

## Validation Metrics

See `validation_metrics.csv` for machine-readable metrics.

| step | mean@16 | maj@16 | best@16 | testing_s | step_s |
| ---: | ------: | -----: | ------: | --------: | -----: |
| 20 | 0.763 | 0.825 | 0.912 | 277.822 | 362.003 |
| 40 | 0.801 | 0.849 | 0.911 | 284.416 | 380.657 |
| 60 | 0.810 | 0.846 | 0.899 | 289.357 | 393.418 |
| 80 | 0.824 | 0.852 | 0.889 | 289.366 | 398.215 |
| 100 | 0.826 | 0.853 | 0.881 | 287.431 | 406.777 |
| 120 | 0.827 | 0.853 | 0.884 | 292.022 | 404.558 |
| 140 | 0.830 | 0.853 | 0.887 | 293.124 | 423.055 |
| 150 | 0.8275 | 0.853 | 0.885 | 290.875 | 435.625 |

## Final Training Step

Final step `150` metrics:

- `training/global_step=150`
- `training/epoch=9`
- `timing_s/step=435.625`
- `timing_s/gen=30.560`
- `timing_s/reward=22.985`
- `timing_s/ref=12.075`
- `timing_s/update_actor=37.433`
- `timing_s/testing=290.875`
- `perf/throughput=230.484`
- `perf/total_num_tokens=803236`
- `train/majority_voting_reward=0.896`
- `train/ground_truth_reward=0.848`

## Health Checks

End-state checks:

- Training process exited.
- GPUs were released: all 8 GPUs reported `0` MiB used after completion.
- `/tmp` remained safe: `3.5T` total, `377G` used, `2.9T` available, `12%`.
- No fatal markers in the final 1MB of log:
  - `RayTaskError=False`
  - `OutOfMemoryError=False`
  - `CUDA out of memory=False`
  - `No space left=False`
  - `Error executing job=False`
  - `RuntimeError:=False`

## Notes

- Natural step count was `150`, not `160`, because this dataloader used `15` batches per epoch for `500` rows with `batch_size=32`, across `10` epochs.
- The earlier `bs8/310` run had final `mean@16=0.7495`, while this paper-style run reached final `0.8275` and peak `0.8300`.
- The eval chain is therefore considered valid; the main previous gap came from training semantics, not validation logic.
