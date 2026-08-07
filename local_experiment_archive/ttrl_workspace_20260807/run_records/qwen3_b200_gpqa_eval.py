import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _build_prompt(tokenizer, question: str) -> str:
    messages = [
        {
            "role": "user",
            "content": (
                question
                + "\n\nAnswer with the letter A, B, C, or D. "
                + "Put the final answer in the last line."
            ),
        }
    ]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def main() -> None:
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/tmp/Qwen3-30B-A3B-Base")
    parser.add_argument("--data", default=str(_repo_root() / "data/GPQA-TTT/test.json"))
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=1536)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--enforce-eager", action="store_true")
    args = parser.parse_args()

    out_json = Path(args.out_json)
    out_jsonl = Path(args.out_jsonl)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    print(f"loading dataset {args.data}", flush=True)
    records_raw = json.loads(Path(args.data).read_text(encoding="utf-8"))
    if args.limit > 0:
        records_raw = records_raw[: args.limit]
    print(f"dataset rows: {len(records_raw)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    prompts = []
    records = []
    for idx, row in enumerate(records_raw):
        prompt = _build_prompt(tokenizer, row["prompt"])
        token_len = len(tokenizer.encode(prompt, add_special_tokens=False))
        if token_len + args.max_tokens > args.max_model_len:
            raise RuntimeError(
                f"row {idx} prompt tokens {token_len} + max_tokens {args.max_tokens} "
                f"exceeds max_model_len {args.max_model_len}"
            )
        prompts.append(prompt)
        records.append(
            {
                "index": idx,
                "id": row.get("id", str(idx)),
                "data_source": row.get("source", "gpqa"),
                "answer": row["answer"],
                "prompt": row["prompt"],
                "prompt_tokens": token_len,
            }
        )

    extra_kwargs = {
        "attention_config": {"backend": "FLASH_ATTN"},
        "kernel_config": {"moe_backend": "triton", "enable_flashinfer_autotune": False},
        "enable_prefix_caching": False,
        "enable_chunked_prefill": True,
        "disable_hybrid_kv_cache_manager": True,
        "async_scheduling": False,
    }
    if os.environ.get("KV_CACHE_MEMORY_BYTES"):
        extra_kwargs["kv_cache_memory_bytes"] = int(os.environ["KV_CACHE_MEMORY_BYTES"])

    print(f"loading vLLM model {args.model}", flush=True)
    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        tensor_parallel_size=args.tp,
        dtype="bfloat16",
        trust_remote_code=False,
        gpu_memory_utilization=0.65,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_model_len,
        max_num_seqs=max(args.batch_size, 1),
        enforce_eager=args.enforce_eager,
        **extra_kwargs,
    )

    sys.path.insert(
        0, "/mlx_devbox/users/quyanyi/playground/Chunked-Power-Sampling/llm_experiments"
    )
    from grader_utils.gpqa_grader import grade_answer, parse_answer_gpqa

    sampling = SamplingParams(
        n=args.n,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=-1,
        max_tokens=args.max_tokens,
    )

    sample_correct = 0
    sample_total = 0
    prompt_correct = 0
    prompt_total = 0
    start = time.time()
    with out_jsonl.open("w", encoding="utf-8") as fout:
        for begin in range(0, len(prompts), args.batch_size):
            batch_prompts = prompts[begin : begin + args.batch_size]
            outputs = llm.generate(batch_prompts, sampling)
            for offset, output in enumerate(outputs):
                base_rec = records[begin + offset]
                any_correct = False
                for sample_idx, sample in enumerate(output.outputs):
                    rec = dict(base_rec)
                    text = sample.text
                    pred = parse_answer_gpqa(text)
                    acc = bool(grade_answer(pred, rec["answer"]))
                    rec.update(
                        {
                            "sample_index": sample_idx,
                            "response": text,
                            "acc": acc,
                            "pred": pred,
                        }
                    )
                    any_correct = any_correct or acc
                    sample_correct += int(acc)
                    sample_total += 1
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                prompt_correct += int(any_correct)
                prompt_total += 1
            fout.flush()
            elapsed = time.time() - start
            print(
                f"progress {prompt_total}/{len(prompts)} "
                f"mean_acc={sample_correct / max(sample_total, 1):.4f} "
                f"pass@{args.n}={prompt_correct / max(prompt_total, 1):.4f} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    summary = {
        "model": args.model,
        "data": args.data,
        "total": prompt_total,
        "sample_total": sample_total,
        "correct": prompt_correct,
        "sample_correct": sample_correct,
        "acc": sample_correct / max(sample_total, 1),
        "mean_acc": sample_correct / max(sample_total, 1),
        f"pass@{args.n}": prompt_correct / max(prompt_total, 1),
        "pass_at_n": prompt_correct / max(prompt_total, 1),
        "n": args.n,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len,
        "tp": args.tp,
        "batch_size": args.batch_size,
        "enforce_eager": args.enforce_eager,
        "expert_sample": {
            "enabled": os.environ.get("VLLM_EXPERT_SAMPLE_ENABLE", "0") == "1",
            "keep_head": os.environ.get("VLLM_EXPERT_SAMPLE_KEEP_HEAD"),
            "pool": os.environ.get("VLLM_EXPERT_SAMPLE_POOL"),
            "tau": os.environ.get("VLLM_EXPERT_SAMPLE_TAU"),
            "seed": os.environ.get("VLLM_EXPERT_SAMPLE_SEED"),
        },
        "elapsed_s": time.time() - start,
        "out_jsonl": str(out_jsonl),
    }
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
