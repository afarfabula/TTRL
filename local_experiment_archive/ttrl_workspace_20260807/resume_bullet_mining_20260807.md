# Resume Bullet Mining - 2026-08-07

## Current Machine

- Visible GPU: 1x NVIDIA A100-SXM4-80GB.
- This is not enough for the historical 8xH100/B200 TTRL training runs, but it is enough for log mining, result curation, small vLLM/inference tests, evaluator/debug runs, and single-GPU ablations on smaller models.

## High-Confidence Historical Work

### 1. TTRL / verl Math500 Reproduction And Runtime Optimization

Evidence:

- Repo: `/mlx_devbox/users/quyanyi/playground/TTRL/verl`
- Frameworks: verl, Ray, FSDP, vLLM, FlashAttention2, Triton fused kernels.
- Model: `/models/Qwen2.5-Math-7B`
- Task/data: Math500 through local `MATH-TTT` train/test parquet.
- Main record: `TTRL/important_experiment_logs/ttrl_math500_paper_b32_20260714_101500/README.md`
- Metrics CSV: `TTRL/important_experiment_logs/ttrl_math500_paper_b32_20260714_101500/validation_metrics.csv`

Key numbers:

- Earlier misaligned `bs8/310` run: final `mean@16=0.7495`.
- Paper-style run: final `mean@16=0.8275`, peak `mean@16=0.8300` at step 140.
- Target paper-style comparison: reported `mean@16=0.834`, reproduced within 0.4 points.
- Final `maj@16=0.8530`, final `best@16=0.8850`.
- Non-validation step average: `108.894s`; median: `108.628s`.

Infra optimization evidence:

- `TTRL/verl/TTRL_MATH500_INFRA_PROGRESS_ZH.md`
- Heavy offload baseline around `165.5908s/step`, with `update_actor=63.6186s`, `gen=50.5859s`, `ref=27.0555s`, `old_log_prob=14.319s`.
- Micro-batch/runtime tuning moved stable steps into roughly `118-131s`.
- Rollout-logprob reuse reduced `old_log_prob` from roughly `8s` to `0.02-0.04s`, with exact diff monitor `training/rollout_probs_diff_max/mean/std = 0.000`.
- Fused Triton path reached a representative complete step of `110.269s` with `gen=47.424s`, `ref=9.167s`, `update_actor=44.266s`.
- Later paper-style B200 run had final-step non-validation components around `gen=30.560s`, `ref=12.075s`, `update_actor=37.433s`.

Resume angle:

- AI infra / systems bullet: debugged and optimized a distributed RL training stack, hard-verified FlashAttention2, tuned vLLM/FSDP memory placement, eliminated redundant actor logprob computation, and reproduced a paper-style Math500 TTRL result.
- Strongest metric: accuracy recovery from `74.95%` to `83.0%` peak after aligning training semantics, plus old-logprob path from about `8s` to near zero.

Draft bullet:

- Reproduced and optimized TTRL math reasoning training for Qwen2.5-Math-7B on verl/Ray/FSDP/vLLM, fixing data/config drift and runtime bottlenecks to recover Math500 `mean@16` from `74.95%` to `83.0%` peak, within 0.4 pp of the reference setup, while removing redundant old-logprob recomputation (`~8s -> <0.04s/step`).

### 2. TTRL Algorithm Experiments: SPS / Confidence-Weighted Internal Feedback

Evidence:

- Repo: `/mlx_devbox/users/quyanyi/playground/refcode/TTRL`
- Main handoff: `refcode/TTRL/TTRL_SPS_HANDOFF.md`
- Frameworks: verl, Ray, FSDP, vLLM, TTRL reward plumbing.
- Models/tasks:
  - Qwen3-4B on MATH-TTT/Math500, `mean@4` validation.
  - Qwen2.5-Math-7B on MATH-TTT/Math500, longer 184-step validation.

Key numbers:

- v3 hard confidence filter: `mean@4=0.8184104628`, `best@4=0.8619939638`, `maj@4=0.8218933602`.
- v4 continuous confidence weighting + clip penalty: `mean@4=0.8294768612`, `best@4=0.8731448692`, `maj@4=0.8319597586`; +1.11 pp over v3.
- v11 rule confidence weight with lower capacity floor: `mean@4=0.8370221328`, `best@4=0.8826277666`, `maj@4=0.8413963783`; +0.75 pp over v4.
- Qwen2.5-Math-7B v14 184-step: `mean@4=0.8435613682`, `best@4=0.8692736419`, `maj@4=0.8476639839`; representative training steps around `34-39s` before final validation.

Algorithmic content:

- Used internal rollout signals only for training feedback: majority pseudo-labels, SPS agreement confidence, rollout/base/proposal logprob, response clip ratio.
- Fixed an important implementation bug where prompt weights were logged but not applied to `token_level_scores`.
- Replaced hard filtering with continuous confidence weights and clip-aware weighting.
- Explored majority/self-consistency and rule-confidence variants.

Resume angle:

- Algorithm bullet: designed unsupervised internal-feedback weighting for math RL training, improving short-run Math500 validation without external labels in training.
- Infra + algorithm bullet: implemented reward plumbing, validation harness, static checks, and runbooks across failure-prone Ray/GPU environments.

Draft bullet:

- Developed confidence-weighted internal feedback for TTRL math RL, combining majority pseudo-labels, SPS agreement, and rollout quality signals to improve Qwen3 Math500 `mean@4` from `81.84%` to `83.70%`, and scaled the best variant to Qwen2.5-Math-7B with `84.36% mean@4` after 184 steps, using only model-internal signals for training.

### 3. Chunk-Level Search-State TTRL / PowerFlow-Style Targets

Evidence:

- Repo: `/mlx_devbox/users/quyanyi/playground/TTRL/verl`
- Audit: `TTRL/important_experiment_logs/chunk_level_search_state_ttrl_24h_audit_20260801.md`
- Plan: `chunk_level_search_state_ttrl_24h_goal.md`
- Frameworks: verl, Ray, FSDP, vLLM, PowerFlow-style actor update.
- Model/task: Qwen2.5-Math-7B, MATH-TTT/Math500.

Key positive infra results:

- Implemented chunk-state construction from on-policy full rollouts, chunk resampling/probing/scoring, and weighted chunk actor updates.
- Support-anchor no-probe target gave dense training (`num_actor_samples=128`) and very low overhead:
  - chunk score around `0.001s`
  - chunk ref around `0.84-1.02s` after warmup
  - update_actor around `2.6-3.3s`
- 20-step support-anchor run was a negative algorithm result:
  - `mean@16=0.43725`
  - `maj@16=0.558596`
  - `best@16=0.83514`
- Later variants improved some negative pilots but did not pass the MV baseline gate; e.g. suffix support v28 got `mean@16=0.529875`, `maj@16=0.667382`, `best@16=0.882996`.

Resume angle:

- This is strong as a research/engineering exploration bullet if phrased as "built and evaluated a new training substrate" rather than "improved final accuracy".
- Best claim: built a chunk-level search-state training pipeline and isolated target-quality as the bottleneck, with actor update overhead reduced from full-response update scale to a few seconds.

Draft bullet:

- Built a chunk-level search-state training path in verl that converts full on-policy rollouts into state-conditional continuation targets, supports PowerFlow-style weighted updates, and reduced chunk actor-update overhead to `~3s/step`; ran 20-step Math500 pilots that isolated target quality, not systems throughput, as the limiting factor.

### 4. Full-Rollout PowerFlow / Weighted NLL Diagnostics

Evidence:

- Progress doc: `TTRL/important_experiment_logs/full_rollout_ttrl_12h_progress_20260802.md`
- Model/task: Qwen2.5-Math-7B on MATH-TTT/Math500.

Key numbers:

- Posterior sharpening with PowerFlow squared-delta: `mean@16=0.45325`, `maj@16=0.57668`, `best@16=0.83677`.
- Oracle correctness with PowerFlow squared-delta: `mean@16=0.398875`, `maj@16=0.503472`, `best@16=0.814348`.
- Oracle correctness with weighted NLL: `mean@16=0.69275`, `maj@16=0.797464`, `best@16=0.904338`.
- Interpretation: full-response target transfer can work; the squared-delta full-response objective was the main failure point.

Resume angle:

- This is useful as an algorithm-debugging bullet: designed oracle diagnostics that falsified a loss-function hypothesis.

Draft bullet:

- Designed oracle-target diagnostics for full-rollout distribution matching, showing that a PowerFlow squared-delta objective failed even with ground-truth oracle targets (`39.9% mean@16`), while weighted NLL recovered to `69.3% mean@16`; used the result to redirect method design away from the failing objective.

### 5. Chunked Power Sampling / Training-Free Inference-Time Scaling

Evidence:

- Repo: `/mlx_devbox/users/quyanyi/playground/Chunked-Power-Sampling`
- README: `Chunked-Power-Sampling/README.md`
- Handoff: `Chunked-Power-Sampling/llm_experiments/SPS_EXPERIMENT_HANDOFF.md`
- Frameworks: vLLM, multiprocessing GPU workers, offline evaluators.
- Model: Qwen2.5-Math-7B.
- Tasks: MATH500, HumanEval, GPQA, Olympiad/AIME/AMC variants.

Key numbers:

- MATH500 greedy: `68.8%`.
- SPS K=32: `76.4%`, around `240.7s` total on 4 GPUs.
- SPS K=16: `75.2%`, around `125.7s` total.
- MCMC: `71.6%`, around `1023.6s` total.
- Chunked PS MATH500 repeat5: selected `77.84%` mean, majority `79.32%` mean, oracle/pass@32 `80.64%`, `122-139s/run`.
- HumanEval chunked PS: selected around `61.71%`, majority `61.59%`, oracle `63.78%`, about `208s`.
- GPQA had high oracle but weak selection: selected `34.85%`, majority `35.35-37.88%`, oracle `62.12-64.65%`.

Resume angle:

- Inference systems/algorithm bullet: implemented vLLM-only chunked search with prune/branch, batched multi-benchmark workers, and offline evaluation.
- Best metric: MATH500 training-free improvement from greedy `68.8%` to `77.84%` selected / `79.32%` majority.

Draft bullet:

- Implemented a vLLM-only Chunked Power Sampling system for training-free inference-time scaling, using chunk-level prune/branch search and batched GPU workers to improve Qwen2.5-Math-7B MATH500 accuracy from `68.8%` greedy to `77.84%` selected / `79.32%` majority, while exposing oracle-selection gaps on GPQA for follow-up reranking work.

### 6. Qwen3-30B-A3B Expert Sampling / MoE Routing Experiments

Evidence:

- Logs: `TTRL/important_experiment_logs/qwen3_30b_a3b_*_summary.json`
- Framework: vLLM.
- Model: `/tmp/Qwen3-30B-A3B-Base`
- Tasks: MATH-TTT/Math500 and GPQA-TTT.

Key numbers:

- Base Math500 full n=16: `mean_acc=0.3725`, `pass@16=0.72`.
- Base GPQA n=16 maxlen 4096: `mean_acc=0.2838`, `pass@16=0.9040`.
- Expert sampling variants often reduced mean accuracy on full Math500, e.g. k5 pool32 stochastic full `mean_acc=0.3245`, pass@16 `0.71`; k7 pool16 full `mean_acc=0.345875`, pass@16 `0.712`.

Resume angle:

- Good for "negative result / evaluation harness" but weaker than the TTRL and Chunked PS bullets.
- Can support a story about evaluating MoE expert-sampling hypotheses and finding that naive expert sampling hurts mean accuracy while preserving high pass@k.

Draft bullet:

- Built vLLM evaluation harnesses for Qwen3-30B-A3B MoE expert-sampling experiments on Math500/GPQA, measuring mean/pass@k tradeoffs and showing that naive stochastic expert selection reduced Math500 mean accuracy despite high pass@16, guiding the project away from unsupported routing changes.

## What This Single A100 Can Do Next

### Feasible Immediately

1. Result curation and resume artifact polish.
   - Build a concise table of runs, paths, framework/model/task, and metric deltas.
   - Extract 3-5 strongest bullets with exact evidence paths.
   - No GPU needed.

2. Re-run small vLLM inference-time scaling experiments.
   - Use Qwen2.5-Math-7B or smaller models if available locally.
   - Limit to 50-100 Math500 examples, n=4/8/16.
   - Compare greedy, SPS, answer-marginalized selection, and majority/oracle.

3. Offline analysis of existing outputs.
   - For Chunked PS and Qwen3 outputs, compute answer-cluster margins, oracle gaps, error categories, and reranking upper bounds.
   - This can produce an algorithm bullet without needing new 8-GPU training.

4. Unit/smoke tests for algorithm code.
   - Validate parsers, target construction, weighting math, and loss masks.
   - Run single-step or CPU/GPU-mini tests with tiny models.

5. Single-GPU SFT/LoRA-style toy runs.
   - Possible for 0.5B-1.5B models with short sequence length.
   - Useful for verifying a training objective, not for final Math500 claims.

### Not Feasible On One A100

- Reproducing 8xH100/B200 TTRL full runs with Qwen2.5-Math-7B and `n_votes=64`, `n_samples=32`.
- Full 150-step or 184-step TTRL benchmarks with comparable throughput.
- 30B MoE full n=16 evaluation unless using very constrained TP/offload or slow settings.

## Best Resume Direction

For AI infra roles, lead with:

1. TTRL/verl distributed training reproduction and runtime optimization.
2. vLLM Chunked Power Sampling inference-time scaling.
3. Building robust experiment infrastructure in Ray/FSDP/vLLM under flaky GPU worker conditions.

For algorithm roles, lead with:

1. SPS/confidence-weighted internal feedback improvements.
2. Chunk-level search-state TTRL substrate and negative-result diagnosis.
3. Full-rollout objective diagnostics proving weighted NLL works where squared-delta fails.

## Strongest Final Bullets To Refine

- Reproduced and optimized TTRL math reasoning training for Qwen2.5-Math-7B on verl/Ray/FSDP/vLLM, fixing data/config drift and runtime bottlenecks to recover Math500 `mean@16` from `74.95%` to `83.0%` peak, within 0.4 pp of the reference setup, while removing redundant old-logprob recomputation (`~8s -> <0.04s/step`).

- Developed confidence-weighted internal feedback for TTRL math RL, combining majority pseudo-labels, SPS agreement, and rollout quality signals to improve Qwen3 Math500 `mean@4` from `81.84%` to `83.70%`, and scaled the best variant to Qwen2.5-Math-7B with `84.36% mean@4` after 184 steps, using only model-internal signals for training.

- Implemented a vLLM-only Chunked Power Sampling system for training-free inference-time scaling, using chunk-level prune/branch search and batched GPU workers to improve Qwen2.5-Math-7B MATH500 accuracy from `68.8%` greedy to `77.84%` selected / `79.32%` majority, while exposing oracle-selection gaps on GPQA for follow-up reranking work.

- Built a chunk-level search-state training path in verl that converts full on-policy rollouts into state-conditional continuation targets, supports PowerFlow-style weighted updates, and reduced chunk actor-update overhead to `~3s/step`; ran 20-step Math500 pilots that isolated target quality, not systems throughput, as the limiting factor.

## Recommended Next A100 Work Plan

1. Produce a one-page run table from the evidence above.
2. Add an offline answer-clustering/reranking analysis for Chunked PS using existing candidate outputs.
3. If Qwen2.5-Math-7B is locally usable on this A100, run a 50-problem MATH500 answer-marginalized SPS smoke to see whether selection can improve without 8 GPUs.
4. Convert the best three bullets into AI-infra and algorithm resume variants with quantified impact and no overclaiming.
