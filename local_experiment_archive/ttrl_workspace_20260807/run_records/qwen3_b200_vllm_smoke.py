import os

from vllm import LLM, SamplingParams


def main() -> None:
    model = "/tmp/Qwen3-30B-A3B-Base"
    enforce_eager = os.environ.get("ENFORCE_EAGER", "true").lower() in {"1", "true", "yes"}
    tp_size = int(os.environ.get("TP_SIZE", "2"))
    attention_backend = os.environ.get("ATTENTION_BACKEND")
    attention_config = {"backend": attention_backend} if attention_backend else None
    extra_kwargs = {}
    if os.environ.get("DISABLE_PREFIX_CACHE", "").lower() in {"1", "true", "yes"}:
        extra_kwargs["enable_prefix_caching"] = False
    if os.environ.get("DISABLE_CHUNKED_PREFILL", "").lower() in {"1", "true", "yes"}:
        extra_kwargs["enable_chunked_prefill"] = False
    if os.environ.get("DISABLE_ASYNC_SCHEDULING", "").lower() in {"1", "true", "yes"}:
        extra_kwargs["async_scheduling"] = False
    if os.environ.get("DISABLE_HYBRID_KV", "").lower() in {"1", "true", "yes"}:
        extra_kwargs["disable_hybrid_kv_cache_manager"] = True
    if os.environ.get("KV_CACHE_MEMORY_BYTES"):
        extra_kwargs["kv_cache_memory_bytes"] = int(os.environ["KV_CACHE_MEMORY_BYTES"])
    kernel_config = {}
    if os.environ.get("MOE_BACKEND"):
        kernel_config["moe_backend"] = os.environ["MOE_BACKEND"]
    if os.environ.get("ENABLE_FLASHINFER_AUTOTUNE"):
        kernel_config["enable_flashinfer_autotune"] = (
            os.environ["ENABLE_FLASHINFER_AUTOTUNE"].lower() in {"1", "true", "yes"}
        )
    if kernel_config:
        extra_kwargs["kernel_config"] = kernel_config
    print(f"loading vllm model {model}", flush=True)
    llm = LLM(
        model=model,
        tensor_parallel_size=tp_size,
        dtype="bfloat16",
        trust_remote_code=False,
        gpu_memory_utilization=0.65,
        max_model_len=2048,
        max_num_batched_tokens=2048,
        enforce_eager=enforce_eager,
        attention_config=attention_config,
        **extra_kwargs,
    )
    print("loaded", flush=True)
    prompts = [
        "Solve: What is 17 + 25? Put the final answer in \\boxed{}.",
        "Compute 12*13. Put the final answer in \\boxed{}.",
    ]
    outputs = llm.generate(
        prompts,
        SamplingParams(temperature=0.6, top_p=0.95, max_tokens=64, n=1),
    )
    for output in outputs:
        print("PROMPT", output.prompt[:80].replace("\n", " "))
        print("OUT", output.outputs[0].text.replace("\n", " ")[:300])


if __name__ == "__main__":
    main()
