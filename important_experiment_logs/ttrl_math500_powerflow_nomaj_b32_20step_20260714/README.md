# TTRL Math500 PowerFlow No-Majority B32 20-Step Probe

## Purpose

This run is the aligned comparison requested after the first PowerFlow smoke test. It uses the paper-style TTRL training scale for the first 20 steps, while replacing only the training loss / pseudo-label path:

- `batch_size=32`
- `rollout.n=32`
- `20` training steps
- PowerFlow actor loss
- no majority-vote pseudo label
- math500 validation unchanged

This is the fairer comparison against paper-style majority TTRL step 20.

## Implementation

The implementation is the default-off PowerFlow branch added in the local TTRL repo:

- actor injects `proj_z`
- `proj_z` is FSDP-wrapped
- vLLM weight sync filters `proj_z`
- `ttrl.powerflow_no_majority=True` skips majority pseudo-label generation
- actor update uses PowerFlow trajectory-balance residual
- `powerflow_use_boxed_reward=False`, so rule reward is not part of the loss
- rule reward is still logged as `train/powerflow_observed_reward`

Loss form:

```text
delta = log_z + avg_log_prob(policy) - beta * avg_log_prob(ref)
loss = mean(clipped_importance_weight * delta^2)
```

## Run Config

- Model: `/models/Qwen2.5-Math-7B`
- Data:
  - `/mlx_devbox/users/quyanyi/playground/TTRL/verl/data/MATH-TTT/train.parquet`
  - `/mlx_devbox/users/quyanyi/playground/TTRL/verl/data/MATH-TTT/test.parquet`
- `data.train_batch_size=32`
- `trainer.total_training_steps=20`
- `ttrl.powerflow_no_majority=True`
- `ttrl.n_samples_per_prompt=32`
- `actor_rollout_ref.actor.powerflow_enable=True`
- `actor_rollout_ref.actor.powerflow_use_boxed_reward=False`
- `actor_rollout_ref.actor.powerflow_beta_coef=4.0`
- `actor_rollout_ref.actor.powerflow_init_ref_log_prob=0.36`
- `actor_rollout_ref.actor.ppo_mini_batch_size=1`
- `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2`
- `actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2`
- `actor_rollout_ref.ref.fsdp_config.param_offload=True`
- `actor_rollout_ref.rollout.val_kwargs.n=16`
- `trainer.test_freq=20`
- `trainer.save_freq=20`

Run script:

```text
/mlx_devbox/users/quyanyi/playground/TTRL/verl/run_records/ttrl_math500_powerflow_nomaj_b32_20step_20260714.sh
```

Runtime log:

```text
/tmp/ttrl_b200/logs/ttrl_math500_powerflow_nomaj_b32_20step_20260714.log
```

Checkpoint:

```text
/tmp/ttrl_b200/checkpoints/ttrl_math500_powerflow_nomaj_b32_20step_20260714/global_step_20
```

## Results

Final validation at step 20:

```text
val-core/math/acc/mean@16 = 0.7295
val-core/math/acc/maj@16/mean = 0.817
val-core/math/acc/best@16/mean = 0.910
```

Step 20 timing:

```text
timing_s/step = 382.258
timing_s/gen = 14.945
timing_s/reward = 5.283
timing_s/ref = 15.326
timing_s/update_actor = 48.441
timing_s/testing = 278.416
timing_s/save_checkpoint = 19.698
```

Ordinary training step timing:

```text
plain step avg = 87.222s
plain step median = 87.546s
plain step count = 19
```

Selected metrics:

```text
step5  powerflow_loss=0.189  observed_reward=0.588
step10 powerflow_loss=0.145  observed_reward=0.674
step14 powerflow_loss=0.132  observed_reward=0.679
step19 powerflow_loss=0.076  observed_reward=0.562
step20 powerflow_loss=2.454  observed_reward=0.645
```

## Comparison

Relevant baselines on the same math500 validation path:

```text
raw base:                         mean@16=0.467
PowerFlow no-majority bs8 step20: mean@16=0.534
PowerFlow no-majority b32 step20: mean@16=0.7295
majority TTRL paper-style step20: mean@16=0.763
```

Interpretation:

- Aligning to `batch_size=32` makes the PowerFlow run much stronger than the bs8 smoke test.
- It still underperforms majority-TTRL at the same step budget by about `3.35` points of `mean@16`.
- It is not GRPO. The configured trainer may still carry GRPO fields, but the actor update uses PowerFlow loss when `powerflow_enable=True`.
- The no-majority, no-boxed-reward variant has some useful signal but is not strong enough to replace majority pseudo-label TTRL directly.

## Health

- Run completed `20/20`.
- `global_step_20` checkpoint was saved.
- No fatal markers in final log tail:
  - `RayTaskError=False`
  - `Error executing job=False`
  - `RuntimeError:=False`
  - `OutOfMemoryError=False`
  - `CUDA out of memory=False`
  - `No space left=False`

## Next

The most promising follow-up is not to replace majority-TTRL with pure PowerFlow. Better directions:

- Use PowerFlow as an auxiliary loss with a small coefficient alongside majority-TTRL.
- Try `powerflow_use_boxed_reward=True` only as a sanity check; it is no longer verifier-free.
- Add an internal correctness/capacity gate before using PowerFlow alone.
