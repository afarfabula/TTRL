# TTRL MV B32/R32/V64 Infra Semantics Audit - 2026-07-31

## Current baseline gate

Active run:

- Run id: `ttrl_majvote_b32_r32_v64_150step_paperstyle_seedfix_20260731`
- Script: `verl/run_records/ttrl_majvote_b32_r32_v64_150step_paperstyle_seedfix_20260731.sh`
- Model: `/models/Qwen2.5-Math-7B`
- Data: `MATH-TTT`
- Training config: `B32/R32/V64`, `VAL_N=16`, `total_training_steps=150`
- Dynamic batch: disabled for actor/ref/rollout logprob in the baseline config

Step 20 validation is aligned with the historical paper-style MV trajectory:

| run | mean@16 | maj@16 | best@16 |
| --- | ---: | ---: | ---: |
| historical paper-style step20 | 0.763 | 0.825 | 0.912 |
| current strict 150-step step20 | 0.760 | 0.820 | 0.901 |

The earlier forced 20-step checks are not valid baseline comparisons because
they changed `trainer.total_training_steps` to 20 and therefore changed the
cosine scheduler phase.

## Dynamic batch decision

Actor dynamic batch is not treated as a semantics-preserving infra optimization
for this MV baseline.

Reason from code audit:

- Actor update computes each microbatch scalar loss first, then scales it by
  `len(micro_batch) / ppo_mini_batch_size` when `use_dynamic_bsz=True`
  (`verl/workers/actor/dp_actor.py`).
- This weighting is only strictly equivalent for sequence-level averaging
  assumptions. For `token-mean` style losses, different microbatches can have
  different valid-token counts, so changing the microbatch partition can change
  the effective token weights.
- Therefore actor dynamic batch can change training semantics unless rewritten
  to use the same global denominator as the non-dynamic path and verified by a
  fixed-batch loss/gradient equivalence test.

Decision:

- Keep `actor_rollout_ref.actor.use_dynamic_bsz=False` for baseline and pilot
  runs.
- Do not use actor dynamic batch as a speed knob.
- Treat the current actor dynamic-batch implementation as semantics-unsafe for
  reported experiments, not merely as a low-confidence optimization. It changes
  the microbatch partition and scales by sample count instead of a shared
  effective token denominator, so it can change GRPO token weighting when
  response lengths differ.
- Updated `examples/ttrl/Qwen2.5-Math/math500_7b_8gpu.sh` so the public
  8-GPU launcher now defaults `ACTOR_USE_DYNAMIC_BSZ=False`; dynamic batching
  must be explicitly opted in for a separate ablation.
- Reconsider only after implementing an equivalence test that checks fixed
  batch loss, response mask, advantage grouping, effective denominator, and
  one-step parameter delta.

Required standard before actor dynamic batch can be used in reported runs:

- Packing may change execution grouping only; it must not change prompt/sample
  order, GRPO group membership, reward values, advantage normalization, loss
  numerator, or effective loss denominator.
- The implementation should accumulate token-level loss numerators and valid
  token denominators across the original static mini-batch, then apply the same
  global normalization as the static path. Scaling a microbatch scalar loss by
  sample count is not sufficient when response lengths differ.
- A fixed-batch A/B must pass before enabling it: same seed and same batch,
  static vs dynamic must match scalar loss, per-layer grad norms, selected
  parameter gradients, and one optimizer-step parameter deltas within expected
  bf16/fp32 numerical tolerance.
- Until that proof exists, dynamic batch is classified as a semantics-changing
  optimization and is disallowed for baseline or algorithm claims.

## Semantics-safe or low-risk infra candidates

These can be tested after the current baseline has enough validation points:

1. Increase fixed actor microbatch size from 2 to 4 if memory allows.
   - This keeps the same mini-batch samples, optimizer step count, scheduler,
     prompt grouping, rewards, advantages, and loss function.
   - It reduces gradient accumulation splits inside the same mini-batch.
   - It still needs a 20-step A/B because floating point accumulation order can
     change slightly.

2. Keep rollout logprobs as old logprobs when the diff monitor stays zero.
   - Current run logs `training/rollout_probs_diff_max=0.000`,
     `training/rollout_probs_diff_mean=0.000`, and
     `training/rollout_probs_diff_std=0.000` through step20.
   - This is already instrumented in `verl/trainer/ppo/ray_trainer.py`.

3. Keep vLLM FLASH_ATTN and existing CUDA/NCCL/cache settings.
   - These are kernel/runtime choices and do not alter rollout sampling
     semantics by themselves.
   - They should remain hard-verified in logs rather than assumed.

4. Increase fixed ref/rollout logprob microbatch sizes only if memory allows.
   - These are forward-only paths.
   - They should be checked by comparing ref/old logprob tensors or existing
     diff metrics, not by validation accuracy alone.

## Disallowed for next pilot

Do not enable these in the next MV-aligned pilot:

- Actor dynamic batch without an equivalence rewrite and fixed-batch proof.
- Any change to `total_training_steps`, warmup ratio/style, or validation
  frequency when comparing step20 to historical MV.
- Any rollout sample filtering, reordering, repair, answer selection, or
  sharpened reward branch while the experiment is supposed to be pure MV.
- Any validation-time best-of or answer-selection trick as a replacement for
  `mean@16`.

## Next planned pilot after baseline checkpoint

If the current 150-step run remains aligned at step40, the first speed pilot
should be:

- Same script and semantic config as current strict MV baseline.
- Only change `MICRO_BATCH_SIZE=4`.
- Keep `ACTOR_USE_DYNAMIC_BSZ=False`.
- Keep `DATA_TRAIN_BATCH_SIZE=32`, `N_SAMPLES_PER_PROMPT=32`,
  `N_VOTES_PER_PROMPT=64`, `VAL_N=16`, and total training steps at 150.
- Run to step20 and compare `mean@16/maj@16/best@16`, ordinary step time,
  `update_actor`, `ref`, `gen`, and memory.

