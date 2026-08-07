# Sharpened MV-Anchor Soft0.02 Pilot - 2026-07-31

## Run

- Run id: `ttrl_sharpened_grpo_b32_r32_v64_150step_paperstyle_seedfix_mvanchor_soft002_20260731`
- Script: `/mlx_devbox/users/quyanyi/playground/TTRL/verl/run_records/ttrl_sharpened_grpo_b32_r32_v64_150step_paperstyle_seedfix_mvanchor_soft002_20260731.sh`
- Model: `/models/Qwen2.5-Math-7B`
- Data: `/mlx_devbox/users/quyanyi/playground/TTRL/verl/data/MATH-TTT`
- Python: `/mlx_devbox/users/quyanyi/playground/.venvs/ttrl_b200/bin/python`
- Hardware: 8x B200 via `mlx worker login`
- Train semantics: B32/R32/V64, `VAL_N=16`, 150-step scheduler, validation every 20 steps
- Infra guard: `ACTOR_USE_DYNAMIC_BSZ=False`; rollout old-logprob reuse kept because `training/rollout_probs_diff_*` stayed `0.000`

## Algorithm

This is the sharpened GRPO variant anchored to majority-vote labels:

- `ttrl.sharpened_enable=True`
- `ttrl.sharpened_alpha=2.0`
- `ttrl.sharpened_tau_pos=0.375`
- `ttrl.sharpened_tau_marg=0.125`
- `ttrl.sharpened_use_confidence=True`
- `ttrl.sharpened_reward_mode=mv_anchor`
- `ttrl.sharpened_soft_coef=0.02`
- `ttrl.sharpened_negative_enable=False`

## Validation Trajectory

| step | mean@16 | maj@16 | best@16 | current MV mean | current MV maj | current MV best | delta mean | delta maj | delta best |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 0.757 | 0.821 | 0.911 | 0.760 | 0.820 | 0.901 | -0.003 | +0.001 | +0.010 |
| 40 | 0.803 | 0.847 | 0.904 | 0.799 | 0.843 | 0.912 | +0.004 | +0.004 | -0.008 |
| 60 | 0.829 | 0.862 | 0.896 | 0.821 | 0.854 | 0.898 | +0.008 | +0.008 | -0.002 |
| 80 | 0.836 | 0.864 | 0.892 | 0.828 | 0.854 | 0.897 | +0.008 | +0.010 | -0.005 |
| 100 | 0.836 | 0.865 | 0.894 | n/a | n/a | n/a | n/a | n/a | n/a |
| 120 | 0.842 | 0.864 | 0.892 | n/a | n/a | n/a | n/a | n/a | n/a |
| 140 | 0.844 | 0.864 | 0.888 | n/a | n/a | n/a | n/a | n/a | n/a |
| 150 | 0.842 | 0.867 | 0.882 | n/a | n/a | n/a | n/a | n/a | n/a |

Historical MV step80 was `0.824 / 0.852 / 0.889`, so this run is `+0.012 / +0.012 / +0.003` at step80 against that record.

## Timing Notes

Validation steps include about 270s of testing and should not be compared to ordinary training step time.

Selected ordinary steps after warmup:

| step | step_s | update_actor_s | ref_s | gen_s | total_tokens | throughput |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 40 | 353.336 | 35.372 | 11.716 | 21.592 | 767356 | 271.468 |
| 53 | 68.922 | 29.270 | 9.904 | 18.892 | 663160 | 1202.743 |
| 60 | 342.660 | 31.706 | 10.546 | 19.440 | 674234 | 245.956 |
| 67 | 82.674 | 37.679 | 12.354 | 21.499 | 798991 | 1208.049 |
| 73 | 73.696 | 32.150 | 10.722 | 19.719 | 771898 | 1309.261 |
| 80 | 359.849 | 38.897 | 12.677 | 24.469 | 846700 | 294.116 |
| 100 | 349.111 | 32.961 | 10.874 | 21.528 | 738062 | 264.265 |

Step80 decision: continue to step100 because mean/maj beat both the current MV run and the historical MV trajectory, although best@16 is slightly lower than the current MV run.

Step100 decision: continue current run to step120 to keep GPUs active and check for late improvement, but treat the metric as a plateau for algorithm search because mean@16 stayed at 0.836 from step80 to step100. Next candidate should keep the same semantic infra and change only the sharpened loss, likely with a more conservative soft coefficient or higher confidence gate.

Step120 decision: continue current run to step150 because mean@16 improved from
0.836 to 0.842. Do not switch to the prepared soft0.01 pilot unless the final
step150 metric regresses or the user asks to stop this run.

Step140 decision: continue to final step150. Mean@16 improved again to 0.844,
maj@16 stayed at 0.864, and best@16 declined to 0.888. This remains the best
mean@16 trajectory in the current TTRL-native line so far.

Final step150:

- `val-core/math/acc/mean@16=0.842125`
- `val-core/math/acc/maj@16/mean=0.866578`
- `val-core/math/acc/best@16/mean=0.881822`
- Checkpoint: `/tmp/ttrl_b200/checkpoints/ttrl_sharpened_grpo_b32_r32_v64_150step_paperstyle_seedfix_mvanchor_soft002_20260731/global_step_150`
- Wall time: `4:01:31`
- Final validation step timing: `testing=274.928s`, `save_checkpoint=19.062s`, `step=370.505s`

Interpretation: the run peaked on mean@16 at step140 (`0.844`) and ended at
`0.842125`; maj@16 continued to improve to `0.866578`, while best@16 fell to
`0.881822`. This is better than the current MV baseline on mean/maj, but it has
not reached the target `0.85` mean@16.
