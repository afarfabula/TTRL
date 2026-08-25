# Sharpened MV-Anchor Soft0.01 Pilot - 2026-07-31

## Run

- Run id: `ttrl_sharpened_grpo_b32_r32_v64_150step_paperstyle_seedfix_mvanchor_soft001_20260731`
- Script: `/mlx_devbox/users/quyanyi/playground/TTRL/verl/run_records/ttrl_sharpened_grpo_b32_r32_v64_150step_paperstyle_seedfix_mvanchor_soft001_20260731.sh`
- Model: `/models/Qwen2.5-Math-7B`
- Data: `/mlx_devbox/users/quyanyi/playground/TTRL/verl/data/MATH-TTT`
- Python: `/mlx_devbox/users/quyanyi/playground/.venvs/ttrl_b200/bin/python`
- Hardware: 8x B200 via `mlx worker login`
- Train semantics: B32/R32/V64, `VAL_N=16`, 150-step scheduler, validation every 20 steps
- Infra guard: `ACTOR_USE_DYNAMIC_BSZ=False`; actor/ref/rollout dynamic batch disabled
- vLLM: `FLASH_ATTN`, `use_rollout_log_probs_as_old=True`, diff monitor required to stay zero

## Algorithm

This is the conservative sharpened GRPO variant anchored to majority-vote labels:

- `ttrl.sharpened_enable=True`
- `ttrl.sharpened_alpha=2.0`
- `ttrl.sharpened_tau_pos=0.375`
- `ttrl.sharpened_tau_marg=0.125`
- `ttrl.sharpened_use_confidence=True`
- `ttrl.sharpened_reward_mode=mv_anchor`
- `ttrl.sharpened_soft_coef=0.01`
- `ttrl.sharpened_negative_enable=False`

## Validation Trajectory

| step | mean@16 | maj@16 | best@16 | decision |
| ---: | ---: | ---: | ---: | --- |
| 20 | 0.766 | 0.835 | 0.903 | Continue to step40; better than soft0.02 at step20 on mean/maj and aligned with MV gate. |
| 40 | 0.802 | 0.848 | 0.901 | Continue to step60/80; slightly above current MV step40 on mean/maj, roughly tied with soft0.02 step40. |
| 60 | 0.818 | 0.850 | 0.898 | Below soft0.02 step60 on mean/maj; continue only to collect step80 pilot trajectory while keeping GPUs active. |
| 80 | 0.825 | 0.856 | 0.889 | Stop/switch after recording; clearly below soft0.02 step80 `0.836/0.864/0.892`. |

## Timing Notes

Early ordinary steps were slower because response lengths stayed high. By step19
the run reached `75.392s` ordinary step time with `update_actor=35.473s`,
`ref=11.374s`, `gen=18.457s`, and `training/rollout_probs_diff_* = 0.000`.

Step20 validation included `testing=273.800s` and total `step=360.497s`.

Observed ordinary steps after validation:

| step | step_s | update_actor_s | ref_s | gen_s | rollout_diff_max |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 30 | 88.271 | 41.277 | 13.065 | 20.595 | 0.000 |
| 31 | 87.834 | 41.718 | 13.158 | 21.802 | 0.000 |
| 32 | 85.992 | 42.669 | 13.516 | 19.875 | 0.000 |
| 33 | 93.765 | 44.765 | 14.093 | 22.717 | 0.000 |
| 34 | 77.652 | 35.978 | 11.550 | 19.502 | 0.000 |
| 35 | 73.998 | 34.906 | 11.085 | 18.533 | 0.000 |
| 36 | 76.241 | 34.090 | 10.967 | 20.665 | 0.000 |
| 37 | 77.769 | 35.684 | 11.484 | 19.619 | 0.000 |
| 38 | 71.311 | 31.414 | 10.170 | 18.465 | 0.000 |
| 39 | 77.353 | 35.618 | 11.467 | 19.048 | 0.000 |
| 50 | 73.618 | 31.784 | 10.341 | 19.752 | 0.000 |
| 53 | 67.310 | 27.859 | 9.157 | 18.705 | 0.000 |
| 56 | 68.620 | 29.905 | 9.712 | 18.491 | 0.000 |
| 57 | 70.215 | 31.043 | 10.092 | 17.917 | 0.000 |
| 58 | 73.742 | 31.717 | 10.260 | 21.050 | 0.000 |
| 59 | 74.949 | 33.372 | 10.743 | 19.505 | 0.000 |
| 70 | 78.956 | 34.734 | 11.140 | 21.654 | 0.000 |
| 71 | 72.927 | 33.016 | 9.777 | 18.355 | 0.000 |
| 72 | 70.913 | 30.950 | 10.011 | 19.740 | 0.000 |
| 76 | 72.019 | 31.176 | 10.174 | 19.165 | 0.000 |
| 77 | 85.867 | 40.536 | 12.882 | 21.181 | 0.000 |
| 78 | 81.452 | 36.926 | 11.812 | 20.855 | 0.000 |
| 79 | 79.763 | 37.852 | 12.090 | 19.378 | 0.000 |

Step40 validation included `testing=271.242s` and total `step=348.933s`.
Step60 validation included `testing=273.884s` and total `step=346.344s`.
Step80 validation included `testing=273.389s` and total `step=354.253s`.

## Infra Semantics Guard

Dynamic batch remains disabled for this run. Current actor dynamic batching is
not semantics-safe for reported experiments because it changes microbatch
partitioning and scales the microbatch scalar loss by sample count rather than
by a shared fixed-batch token denominator. For variable response lengths, this
can change the effective GRPO token weights and therefore the update.

Allowed speedups for the next ablation must preserve the fixed prompt/sample
set, rewards, advantages, optimizer step count, scheduler phase, and effective
loss denominator. Acceptable candidates are fixed microbatch increases,
kernel/runtime improvements such as flash attention and CUDA graph, and
rollout-old-logprob reuse only while the tensor diff monitor stays zero.
