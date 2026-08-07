import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    model_path = "/tmp/Qwen3-30B-A3B-Base"
    print(f"loading transformers model {model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        trust_remote_code=False,
        attn_implementation="sdpa",
    )
    print("loaded", flush=True)
    prompt = "Solve: What is 17 + 25? Put the final answer in \\boxed{}."
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(text)


if __name__ == "__main__":
    main()
