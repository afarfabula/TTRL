#!/usr/bin/env python
"""Fail fast if a local Hugging Face model directory is incomplete."""

import argparse
import json
import os
import sys


def die(message: str) -> None:
    print(f"MODEL_NOT_READY: {message}", file=sys.stderr)
    raise SystemExit(1)


def check_required_file(model_dir: str, filename: str) -> None:
    path = os.path.join(model_dir, filename)
    if not os.path.isfile(path):
        die(f"missing {path}")
    if os.path.getsize(path) <= 0:
        die(f"empty {path}")


def check_no_incomplete_downloads(model_dir: str) -> None:
    download_dir = os.path.join(model_dir, ".cache", "huggingface", "download")
    if not os.path.isdir(download_dir):
        return

    bad = []
    for name in os.listdir(download_dir):
        if name.endswith(".incomplete"):
            bad.append(os.path.join(download_dir, name))
    if bad:
        preview = "\n".join(bad[:8])
        die(f"download cache still has incomplete files:\n{preview}")


def check_safetensors(model_dir: str) -> None:
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    single_path = os.path.join(model_dir, "model.safetensors")

    if os.path.isfile(index_path):
        with open(index_path) as f:
            index = json.load(f)

        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            die(f"invalid or empty weight_map in {index_path}")

        shard_names = sorted(set(weight_map.values()))
        missing = [name for name in shard_names if not os.path.isfile(os.path.join(model_dir, name))]
        if missing:
            die(f"missing shard files: {missing}")

        empty = [name for name in shard_names if os.path.getsize(os.path.join(model_dir, name)) <= 0]
        if empty:
            die(f"empty shard files: {empty}")

        total_size = index.get("metadata", {}).get("total_size")
        if isinstance(total_size, int) and total_size > 0:
            actual_size = sum(os.path.getsize(os.path.join(model_dir, name)) for name in shard_names)
            if actual_size < total_size:
                die(f"shard files are incomplete: total bytes {actual_size} < expected tensor bytes {total_size}")
        return

    if os.path.isfile(single_path) and os.path.getsize(single_path) > 0:
        return

    die(f"missing {index_path} or {single_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    args = parser.parse_args()

    model_dir = os.path.abspath(args.model_dir)
    if not os.path.isdir(model_dir):
        die(f"missing model directory {model_dir}")

    for filename in ["config.json", "tokenizer_config.json", "tokenizer.json"]:
        check_required_file(model_dir, filename)
    check_safetensors(model_dir)

    print(f"MODEL_READY: {model_dir}")


if __name__ == "__main__":
    main()
