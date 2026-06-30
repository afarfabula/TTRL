#!/usr/bin/env python3
import argparse
import csv
import re
import statistics
from pathlib import Path


METRIC_KEYS = [
    "timing_s/step",
    "timing_s/gen",
    "timing_s/generate_sequences",
    "timing_s/sps_generate_sequences_call",
    "timing_s/sps_build_score_batch",
    "timing_s/sps_base_logprob",
    "timing_s/sps_response_mask",
    "timing_s/sps_reuse_ref_union",
    "timing_s/sps_apply_weighted_gt",
    "timing_s/sps_compute_reward",
    "timing_s/sps_selection",
    "timing_s/sps_metrics_update",
    "timing_s/old_log_prob",
    "timing_s/ref",
    "timing_s/update_actor",
    "perf/total_num_tokens",
    "perf/throughput",
    "response_length/mean",
    "response_length/clip_ratio",
]


def parse_float_metric(block: str, key: str):
    match = re.search(re.escape(key) + r":(-?\d+(?:\.\d+)?)", block)
    return float(match.group(1)) if match else None


def parse_steps(log_path: Path):
    text = log_path.read_text(errors="replace")
    rows = []
    for match in re.finditer(r"step:(\d+) - .*?(?=\nstep:|\Z)", text, re.S):
        block = match.group(0)
        row = {"step": int(match.group(1))}
        for key in METRIC_KEYS:
            row[key] = parse_float_metric(block, key)
        rows.append(row)
    return rows


def mean_present(rows, key):
    values = [row[key] for row in rows if row.get(key) is not None]
    return statistics.mean(values) if values else None


def summarize_gpu_csv(path: Path):
    if not path or not path.exists():
        return {}
    utils = []
    mem_used = []
    power = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row = {key.strip(): value.strip() for key, value in row.items()}
            util_key = next((key for key in row if key.startswith("utilization.gpu")), None)
            mem_key = next((key for key in row if key.startswith("memory.used")), None)
            power_key = next((key for key in row if key.startswith("power.draw")), None)
            try:
                if util_key:
                    utils.append(float(row[util_key].split()[0]))
                if mem_key:
                    mem_used.append(float(row[mem_key].split()[0]))
                if power_key:
                    power.append(float(row[power_key].split()[0]))
            except (KeyError, ValueError):
                continue
    out = {}
    if utils:
        out["gpu_util_mean_pct"] = statistics.mean(utils)
        out["gpu_util_min_pct"] = min(utils)
    if mem_used:
        out["gpu_mem_used_mean_mib"] = statistics.mean(mem_used)
        out["gpu_mem_used_max_mib"] = max(mem_used)
    if power:
        out["gpu_power_mean_w"] = statistics.mean(power)
    return out


def fmt(value):
    if value is None:
        return "NA"
    return f"{value:.3f}"


def main():
    parser = argparse.ArgumentParser(description="Summarize TTRL throughput from a Ray TaskRunner log.")
    parser.add_argument("task_log", type=Path)
    parser.add_argument("--gpu-csv", type=Path, default=None)
    parser.add_argument("--last", type=int, default=10)
    args = parser.parse_args()

    rows = parse_steps(args.task_log)
    if not rows:
        raise SystemExit(f"No step metrics found in {args.task_log}")

    window = rows[-args.last :]
    total_tokens = sum(row["perf/total_num_tokens"] for row in window if row["perf/total_num_tokens"] is not None)
    total_step_s = sum(row["timing_s/step"] for row in window if row["timing_s/step"] is not None)
    whole_machine_tps = total_tokens / total_step_s if total_tokens and total_step_s else None

    print(f"task_log={args.task_log}")
    print(f"steps={window[0]['step']}-{window[-1]['step']}")
    for key in METRIC_KEYS:
        print(f"{key}={fmt(mean_present(window, key))}")
    print(f"whole_machine_tokens_per_s={fmt(whole_machine_tps)}")

    gpu_summary = summarize_gpu_csv(args.gpu_csv) if args.gpu_csv else {}
    for key, value in gpu_summary.items():
        print(f"{key}={fmt(value)}")


if __name__ == "__main__":
    main()
