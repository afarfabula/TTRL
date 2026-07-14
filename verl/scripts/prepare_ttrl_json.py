#!/usr/bin/env python
"""Convert TTRL JSON data into verl-compatible parquet files."""

import argparse
import os

import datasets


def make_map_fn(split: str, fallback_source: str):
    def process_fn(example, idx):
        data_source = example.get("source") or fallback_source
        question = example["prompt"]
        solution = example["answer"]
        raw_id = example.get("id") or example.get("unique_id") or idx

        return {
            "data_source": data_source,
            "prompt": [{"role": "user", "content": question}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": solution},
            "extra_info": {"split": split, "index": f"{data_source}-{raw_id}"},
        }

    return process_fn


def convert_split(data_dir: str, split: str, fallback_source: str) -> str:
    json_path = os.path.join(data_dir, f"{split}.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(json_path)

    dataset = datasets.load_dataset("json", data_files=json_path, split="train")
    dataset = dataset.map(function=make_map_fn(split, fallback_source), with_indices=True)

    parquet_path = os.path.join(data_dir, f"{split}.parquet")
    dataset.to_parquet(parquet_path)
    return parquet_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=os.path.join("data", "MATH-TTT"))
    parser.add_argument("--source", default=None)
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    fallback_source = args.source or os.path.basename(data_dir)
    for split in args.splits:
        parquet_path = convert_split(data_dir, split, fallback_source)
        print(f"Wrote {parquet_path}")


if __name__ == "__main__":
    main()
