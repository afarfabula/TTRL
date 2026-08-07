# 2602.02443 Expert-Sample Reproduction Log

Date: 2026-08-05

## Goal

Reproduce the reported gains from `Certain Head, Uncertain Tail: Expert-Sample for Test-Time Scaling in Fine-Grained MoE` (`2602.02443`) on the local B200 environment.

The public abstract/search snippets report the headline result on `Qwen3-30B-A3B-Instruct` + `GPQA-Diamond` with `32` parallel samples: `pass@32` rises from `85.4%` to `91.9%`, and verifier Best-of-N accuracy rises from `59.1%` to `62.6%`.

## Local Availability

- Available model: `/tmp/Qwen3-30B-A3B-Base`
- Missing model for headline reproduction: `Qwen3-30B-A3B-Instruct`
- Available datasets:
  - MATH500/MATH-TTT: `/mlx_devbox/users/quyanyi/playground/TTRL/verl/data/MATH-TTT/test.parquet`
  - GPQA-Diamond/GPQA-TTT: `/mlx_devbox/users/quyanyi/playground/TTRL/verl/data/GPQA-TTT/test.json`

Because the Instruct checkpoint was not present locally, the completed runs below are Base-model transfer checks, not the strict headline setting.

## Implementation State

Patched local vLLM router:

`/mlx_devbox/users/quyanyi/playground/.venvs/ttrl_b200/lib/python3.11/site-packages/vllm/model_executor/layers/fused_moe/router/fused_topk_router.py`

Implemented env-gated Expert-Sample:

- `VLLM_EXPERT_SAMPLE_ENABLE=1`
- `VLLM_EXPERT_SAMPLE_KEEP_HEAD`
- `VLLM_EXPERT_SAMPLE_POOL`
- `VLLM_EXPERT_SAMPLE_TAU`
- `VLLM_EXPERT_SAMPLE_STOCHASTIC=1` for per-forward stochastic Gumbel tail sampling
- `VLLM_EXPERT_SAMPLE_DEBUG=1` for early route-hit logging

Hard execution checks:

- `Using AttentionBackendEnum.FLASH_ATTN backend`
- `Using FlashAttention version 4`
- `Using TRITON Unquantized MoE backend`
- Debug smoke confirmed actual router hit:
  `VLLM_EXPERT_SAMPLE routing active tokens=1536 topk=8 keep_head=5 sample_count=3 pool_size=32 tau=0.5 stochastic=1`

Persistent environment issue:

- DeepGEMM found but not usable because `libnvrtc.so.13` is missing; vLLM falls back to Triton MoE.

## Result Summary

### MATH500 / Qwen3-30B-A3B-Base / n=16

| Run | total | mean_acc | pass@16 | elapsed_s | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| Base | 500 | 37.25% | 72.00% | 3062 | baseline |
| ES deterministic keep7/pool16/tau0.5 | 500 | 34.59% | 71.20% | 4214 | negative |
| ES stochastic keep5/pool32/tau0.5 | 500 | 32.45% | 71.00% | 4199 | negative |
| ES stochastic keep5/pool32/tau0.5 | 100 | 37.63% | 79.00% | 820 | early-subset false positive; full run did not hold |

### GPQA-Diamond / Qwen3-30B-A3B-Base / n=16

| Run | total | mean_acc | pass@16 | elapsed_s | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| Base | 198 | 28.38% | 90.40% | 1054 | baseline |
| ES stochastic keep5/pool32/tau0.5 | 198 | 28.88% | 89.39% | 1417 | pass@16 negative, mean slightly positive |

## Conclusion

The current local Base-model experiments do not reproduce the paper's reported pass@n gain.

The most important correction found during debugging was that deterministic tail sampling was not enough for test-time scaling: it made the selected tail experts effectively fixed for the same router state. Adding `VLLM_EXPERT_SAMPLE_STOCHASTIC=1` produced strong early-subset gains on MATH500, but the gain vanished on full MATH500 and did not improve GPQA-Diamond pass@16.

Strict reproduction is still blocked by setup mismatch:

- The paper headline setting uses `Qwen3-30B-A3B-Instruct`, but only `Qwen3-30B-A3B-Base` is present locally.
- The paper headline metric is `GPQA-Diamond pass@32` and verifier Best-of-N, while the completed local checks used `pass@16` without verifier BoN.
- Full paper access from this machine failed repeatedly due TLS EOF / remote disconnect on arXiv, OpenReview, and text proxy endpoints; only public search snippets were accessible.

## Next Required Steps

1. Put `Qwen3-30B-A3B-Instruct` on this machine.
2. Run GPQA-Diamond baseline with `n=32`, `max_model_len=4096`, and the same GPQA evaluator.
3. Run Expert-Sample with `keep_head=5`, `pool=32`, stochastic tail sampling, `n=32`.
4. Add verifier/Best-of-N evaluation if the paper's exact verifier is available; otherwise report pass@32 separately and do not claim verifier reproduction.
5. Re-fetch the full paper or source when network access permits and audit exact sampling distribution, head/tail cutoff, temperature, and verifier protocol.
