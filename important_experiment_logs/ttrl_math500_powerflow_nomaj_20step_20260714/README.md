# TTRL Math500 PowerFlow No-Majority 20-Step Probe

## Purpose

This probe tested whether the PowerFlow loss from `/mlx_devbox/users/quyanyi/playground/refcode/PowerFlow` can be directly grafted into the local TTRL project as a training loss that does not require majority-vote pseudo labels.

The experiment was intentionally short: 20 training steps on Qwen2.5-Math-7B with the same math500 validation path used by the paper-style TTRL runs.

## Implementation Summary

The local TTRL code now has default-off PowerFlow switches:

- `actor_rollout_ref.actor.powerflow_enable`
- `actor_rollout_ref.actor.powerflow_beta_coef`
- `actor_rollout_ref.actor.powerflow_init_ref_log_prob`
- `actor_rollout_ref.actor.powerflow_proj_layers`
- `actor_rollout_ref.actor.powerflow_use_boxed_reward`
- `actor_rollout_ref.actor.powerflow_on_policy`
- `ttrl.powerflow_no_majority`

When enabled:

1. The actor model gets a `proj_z` head for log partition estimation.
2. `proj_z` is FSDP-wrapped and filtered from vLLM weight sync.
3. The trainer can skip TTRL majority pseudo-label generation.
4. Actor update uses the PowerFlow trajectory-balance residual instead of PPO/GRPO advantage loss:

```text
delta = log_z + avg_log_prob(policy) - beta * avg_log_prob(ref)
loss = mean(clipped_importance_weight * delta^2)
```

This probe used `powerflow_use_boxed_reward=False`, so the rule reward was not part of the loss. Rule reward was still computed and logged as `train/powerflow_observed_reward` for diagnostics.

## Run Config

- Model: `/models/Qwen2.5-Math-7B`
- Data:
  - `/mlx_devbox/users/quyanyi/playground/TTRL/verl/data/MATH-TTT/train.parquet`
  - `/mlx_devbox/users/quyanyi/playground/TTRL/verl/data/MATH-TTT/test.parquet`
- `trainer.total_training_steps=20`
- `data.train_batch_size=8`
- `ttrl.powerflow_no_majority=True`
- `ttrl.n_samples_per_prompt=32`
- `actor_rollout_ref.actor.powerflow_enable=True`
- `actor_rollout_ref.actor.powerflow_use_boxed_reward=False`
- `actor_rollout_ref.actor.powerflow_beta_coef=4.0`
- `actor_rollout_ref.actor.powerflow_init_ref_log_prob=0.36`
- `actor_rollout_ref.actor.use_kl_loss=True`
- `actor_rollout_ref.actor.kl_loss_coef=0.0`
- `actor_rollout_ref.rollout.val_kwargs.n=16`
- `trainer.test_freq=20`
- `trainer.save_freq=20`

Run script:

```text
/mlx_devbox/users/quyanyi/playground/TTRL/verl/run_records/ttrl_math500_powerflow_nomaj_20step_20260714.sh
```

Runtime log:

```text
/tmp/ttrl_b200/logs/ttrl_math500_powerflow_nomaj_20step_20260714.log
```

Checkpoint:

```text
/tmp/ttrl_b200/checkpoints/ttrl_math500_powerflow_nomaj_20step_20260714/global_step_20
```

## Results

Final validation at step 20:

```text
val-core/math/acc/mean@16 = 0.534
val-core/math/acc/maj@16/mean = 0.675
val-core/math/acc/best@16/mean = 0.883
```

Timing:

```text
progress = 20/20 15:11<00:00, 45.56s/it
plain step avg = 30.087s
plain step median = 29.277s
step20 total = 339.183s
step20 testing = 292.665s
step20 save_checkpoint = 18.625s
```

Selected training metrics:

```text
step1  powerflow_loss=1.768  observed_reward=0.383
step5  powerflow_loss=7.630  observed_reward=0.270
step10 powerflow_loss=0.819  observed_reward=0.480
step15 powerflow_loss=4.096  observed_reward=0.422
step20 powerflow_loss=1.967  observed_reward=0.406
```

No fatal markers were found in the final log tail:

- `RayTaskError=False`
- `Error executing job=False`
- `RuntimeError:=False`
- `OutOfMemoryError=False`
- `CUDA out of memory=False`
- `No space left=False`

## Interpretation

The integration works mechanically:

- `proj_z` is created and FSDP-wrapped.
- vLLM rollout does not try to load `proj_z`.
- `ttrl.powerflow_no_majority=True` skips majority pseudo-label generation.
- `actor/powerflow_loss` is logged and the run finishes 20 steps.

The algorithm result is not competitive with majority TTRL at the same validation budget:

- Raw base on this validation chain: `mean@16=0.467`
- PowerFlow no-majority 20 step: `mean@16=0.534`
- Paper-style majority TTRL step20: `mean@16=0.763`

So this verifier-free PowerFlow variant improves over raw base but is much weaker than majority pseudo-label TTRL on math500.

## Next Ideas

Do not directly replace majority TTRL with this loss for math500.

Useful follow-ups:

- Try `powerflow_use_boxed_reward=True` as a sanity check. That is no longer verifier-free, but it verifies whether the PowerFlow residual can exploit correctness signals.
- Add a conservative internal correctness signal rather than pure ref/policy flow matching.
- Tune beta/init offset/lr only after confirming the loss is directionally useful with some correctness signal.
- If staying verifier-free, monitor whether `best@16` stays high while `mean@16` lags; that suggests the loss is not improving low-budget sample quality.
