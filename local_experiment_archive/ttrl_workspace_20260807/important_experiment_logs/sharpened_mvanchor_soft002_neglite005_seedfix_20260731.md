# Sharpened MV-Anchor Soft0.02 Negative-Lite Pilot - 2026-07-31

## Run

- Run id: `ttrl_sharpened_grpo_b32_r32_v64_150step_paperstyle_seedfix_mvanchor_soft002_neglite005_20260731`
- Script: `/mlx_devbox/users/quyanyi/playground/TTRL/verl/run_records/ttrl_sharpened_grpo_b32_r32_v64_150step_paperstyle_seedfix_mvanchor_soft002_neglite005_20260731.sh`
- Model: `/models/Qwen2.5-Math-7B`
- Data: `/mlx_devbox/users/quyanyi/playground/TTRL/verl/data/MATH-TTT`
- Python: `/mlx_devbox/users/quyanyi/playground/.venvs/ttrl_b200/bin/python`
- Hardware: 8x B200 via `mlx worker login`
- Train semantics: B32/R32/V64, `VAL_N=16`, 150-step scheduler, validation every 20 steps
- Infra guard: `ACTOR_USE_DYNAMIC_BSZ=False`; actor/ref/rollout dynamic batch disabled
- vLLM: `FLASH_ATTN`, `use_rollout_log_probs_as_old=True`, diff monitor required to stay zero

## Algorithm

This pilot starts from the strongest completed conservative run, soft0.02
mv-anchor, and adds a light negative reward for low-posterior non-top answers.
The majority-vote anchor remains priority: majority matches keep reward `1.0`
and cannot be overwritten by the negative branch.

- `ttrl.sharpened_enable=True`
- `ttrl.sharpened_alpha=2.0`
- `ttrl.sharpened_tau_pos=0.375`
- `ttrl.sharpened_tau_marg=0.125`
- `ttrl.sharpened_use_confidence=True`
- `ttrl.sharpened_reward_mode=mv_anchor`
- `ttrl.sharpened_soft_coef=0.02`
- `ttrl.sharpened_negative_enable=True`
- `ttrl.sharpened_tau_low=0.125`
- `ttrl.sharpened_negative_reward=-0.05`

## Validation Trajectory

| step | mean@16 | maj@16 | best@16 | decision |
| ---: | ---: | ---: | ---: | --- |
| 20 | 0.764 | 0.832 | 0.915 | Pass gate; continue to step40/60/80. Beats MV step20 0.760/0.820/0.901 and soft0.02 step20 0.757/0.821/0.911. |
| 40 | 0.796 | 0.843 | 0.893 | Continue to step60. Roughly matches MV step40 0.799/0.843/0.912 on mean/maj but lower best; below soft0.02 step40 0.803/0.847/0.904. |
| 60 | 0.816 | 0.855 | 0.890 | Continue to step80 for full pilot trajectory. Mean/best trail MV step60 0.821/0.854/0.898 and soft0.02 step60 0.829/0.862/0.896; maj is effectively MV-aligned. |
| 80 | 0.824 | 0.855 | 0.890 | Stop after step80. Trails soft0.02 step80 0.836/0.864/0.892 and MV step80 0.828/0.854/0.897 on mean/best; light negative reward has no observed benefit. |

## Decision

Negative-lite `-0.05` is not a winner under the aligned B32/R32/V64 setup.
It looked promising at step20, but by step80 it underperformed the completed
soft0.02 run and did not improve over MV on mean/best. Do not continue this
configuration or use it as the next base. If negative rewards are revisited,
make them weaker or sparser and require a fresh step20 gate.

## Timing Notes

- Step20 validation step: `timing_s/testing=274.069`, `timing_s/step=362.757`.
- Step40 validation step: `timing_s/testing=269.873`, `timing_s/step=348.913`.
- Step60 validation step: `timing_s/testing=275.237`, `timing_s/step=348.885`.
- Step80 validation step: `timing_s/testing=270.786`, `timing_s/step=354.954`.
- Recent non-validation steps before step20 were about 80-96s, with actor update
  about 39-47s, ref about 13-15s, and gen about 19-22s.
- Recent non-validation steps before step40 were about 72-92s, with shorter
  response lengths around 600-780 tokens and actor update about 31-44s.
- Recent non-validation steps before step60 were about 68-89s, with response
  lengths around 520-750 tokens and actor update about 28-41s.
- Recent non-validation steps before step80 were about 72-88s, with response
  lengths around 545-775 tokens and actor update about 31-41s.
- `training/rollout_probs_diff_max/mean/std` stayed `0.000` through step20.
- `training/rollout_probs_diff_max/mean/std` stayed `0.000` through step40.
- `training/rollout_probs_diff_max/mean/std` stayed `0.000` through step60.
- `training/rollout_probs_diff_max/mean/std` stayed `0.000` through step80.
- Actor dynamic batch remains disabled because the current implementation is not
  proven equivalent to static token/loss normalization.
