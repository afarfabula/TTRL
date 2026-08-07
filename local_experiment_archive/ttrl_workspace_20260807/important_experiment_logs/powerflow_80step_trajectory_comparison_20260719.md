# Original Major Vote vs Original PowerFlow 80-step Math500 Trajectory

This note compares the two trajectories that should be kept separate:

- **Original TTRL major-vote / majority pseudo-label trajectory**: the paper-style B32 run under the TTRL `verl.trainer.main_ppo` path.
- **Original PowerFlow repo trajectory**: the 80-step run launched from the original `/refcode/PowerFlow` repo.

This document intentionally does **not** compare two PowerFlow variants. It also does **not** use the later TTRL-native vendored PowerFlow run as the TTRL side.

## Run Identity

| Item | Original major vote | Original PowerFlow repo |
|---|---|---|
| Run ID | `ttrl_math500_paper_b32_20260714_101500` | `orig_powerflow_math500_b32_rollout32_80step_20260715` |
| Entry path | `/mlx_devbox/users/quyanyi/playground/TTRL/verl` | `/mlx_devbox/users/quyanyi/playground/refcode/PowerFlow` |
| Launcher / record | `/mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/ttrl_math500_paper_b32_20260714_101500/run_script.sh` | `/mlx_devbox/users/quyanyi/playground/refcode/PowerFlow/run_records/orig_powerflow_math500_b32_rollout32_80step_20260715.sh` |
| Main training path | `python -m verl.trainer.main_ppo` | `python -m powerflow.main_powerflow` |
| Algorithm semantics | Original TTRL majority pseudo-label / major vote | PowerFlow loss, no-majority |
| Model | `/models/Qwen2.5-Math-7B` | `/models/Qwen2.5-Math-7B` |
| Train batch | 32 prompts | 32 prompts |
| Validation samples | `val_kwargs.n=16` | `val_kwargs.n=16` |
| Validation cadence | every 20 steps | every 20 steps |

Notes:

- The major-vote run is a 150-step paper-style run. This document compares its first 80 steps to the 80-step PowerFlow run.
- The major-vote metrics are persisted in `validation_metrics.csv`.
- The original PowerFlow `/tmp` log was later cleaned, so its values below are recovered from preserved session log excerpts.

## Accuracy Trajectory

Metric naming:

- `maj@16` is the majority-vote/MV accuracy over 16 validation samples.
- `best@16` is the pass@16-style metric: at least one of the 16 validation samples is correct.

| Step | TTRL mean@16 | TTRL MV / maj@16 | TTRL pass@16 / best@16 | PowerFlow mean@16 | PowerFlow MV / maj@16 | PowerFlow pass@16 / best@16 | PowerFlow - TTRL mean |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | 0.763 | 0.825 | 0.912 | 0.709500 | 0.814176 | 0.913520 | -0.0535 |
| 40 | 0.801 | 0.849 | 0.911 | 0.791500 | 0.857432 | 0.922316 | -0.0095 |
| 60 | 0.810 | 0.846 | 0.899 | 0.824875 | 0.869012 | 0.920836 | +0.0149 |
| 80 | 0.824 | 0.852 | 0.889 | 0.843125 | 0.880192 | 0.927924 | +0.0191 |

Summary:

- Major vote starts stronger at step 20: `0.763` vs PowerFlow `0.7095`.
- PowerFlow catches up by step 60 and is ahead by step 80.
- At step 80, PowerFlow is about `+1.9` points higher on `mean@16`, `+2.8` points higher on MV / `maj@16`, and `+3.9` points higher on pass@16 / `best@16`.
- TTRL's pass@16 / `best@16` trends down from `0.912` to `0.889`, while PowerFlow's pass@16 / `best@16` trends up from `0.9135` to `0.9279`.

## Timing Trajectory

The two code paths report validation-step `timing_s/step` differently enough that direct comparison needs care:

- The major-vote `validation_metrics.csv` records `step_s` for validation steps, which includes validation time.
- The original PowerFlow log reports train-step `timing_s/step`, validation as `timing_s/testing`, and checkpoint save separately.

For the major-vote run, the table includes a train-only estimate:

`Major vote train_only_s = step_s - testing_s`.

| Step | Major vote logged step_s | Major vote train_only_s | Major vote val_s | PowerFlow train step_s | PowerFlow val_s | PowerFlow save_s | PowerFlow throughput |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | 362.003 | 84.181 | 277.822 | 91.156 | 302.995 | - | 1355.230 |
| 40 | 380.657 | 96.241 | 284.416 | 84.616 | 294.443 | - | 1267.154 |
| 60 | 393.418 | 104.061 | 289.357 | 86.297 | 292.804 | - | 1280.408 |
| 80 | 398.215 | 108.849 | 289.366 | 79.225 | 294.104 | 18.476 | 1226.485 |

Timing summary over the compared validation points:

| Metric | Major vote | PowerFlow original | Difference |
|---|---:|---:|---:|
| Avg logged validation-step `step_s` | 383.573s | 85.323s | Not directly comparable; major vote includes validation |
| Avg train-only selected step_s | 98.333s | 85.323s | Major vote train path is about 1.15x slower |
| Avg validation time | 285.240s | 296.086s | Major vote validation is about 10.8s faster |
| Step 80 checkpoint save | not saved in major-vote run at step 80 | 18.476s | not comparable |

The persistent major-vote CSV does not include per-validation-step throughput. The PowerFlow throughput is listed from the original log excerpt for reference.

## Trajectory Interpretation

Accuracy:

- Major vote is better early, especially at step 20.
- PowerFlow has a stronger upward slope through step 80.
- By step 80, PowerFlow is ahead in `mean@16` and `maj@16`, and materially ahead in `best@16`.

Timing:

- Major vote train-only selected steps are roughly `84-109s`.
- PowerFlow train selected steps are roughly `79-91s`.
- Validation time is close between the two; major vote is slightly faster in this recorded run.

## Evidence Notes

Major-vote values are from:

- `/mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/ttrl_math500_paper_b32_20260714_101500/validation_metrics.csv`
- `/mlx_devbox/users/quyanyi/playground/TTRL/important_experiment_logs/ttrl_math500_paper_b32_20260714_101500/README.md`

Original PowerFlow values are from preserved session artifacts containing original run log lines for:

- `/tmp/ttrl_b200/logs/orig_powerflow_math500_b32_rollout32_80step_20260715.log`
- `step:20`
- `step:40`
- `step:60`
- `step:80`

The `/tmp/ttrl_b200` raw logs were no longer present when this document was corrected, so this file records the recovered/persisted metrics rather than re-parsed raw logs.
