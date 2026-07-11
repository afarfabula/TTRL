#!/usr/bin/env bash
set -euo pipefail

KNOWN_BAD_WORKERS="${KNOWN_BAD_WORKERS:-987816 989057}"
TARGET_RUNNER="/opt/tiger/TTRL/verl/examples/ttrl/worker_run_sps_basin_contrast_process_sharpened_prob_qwen25_math_7b_50step_v81_strict_n4.sh"

tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT

if [ -n "${WORKER_LIST_FIXTURE:-}" ]; then
  cp "$WORKER_LIST_FIXTURE" "$tmp"
  list_exit=0
else
  list_exit=0
  NO_COLOR=1 TERM=dumb mlx worker list > "$tmp" 2>&1 || list_exit=$?
fi
cat "$tmp"

python3 - "$tmp" "$KNOWN_BAD_WORKERS" "$TARGET_RUNNER" "$list_exit" <<'PY'
import sys
from pathlib import Path

list_path = Path(sys.argv[1])
known_bad = {item for item in sys.argv[2].replace(",", " ").split() if item}
runner = sys.argv[3]
list_exit = int(sys.argv[4])
lines = [line.rstrip("\n") for line in list_path.read_text(errors="ignore").splitlines()]
workers = []
for line in lines[1:]:
    parts = line.split()
    if len(parts) < 6:
        continue
    worker_id, cpu, mem, gpu, gpu_type, pod_ip = parts[:6]
    if not worker_id.isdigit():
        continue
    workers.append(
        {
            "id": worker_id,
            "gpu": gpu,
            "gpu_type": gpu_type,
            "pod_ip": pod_ip,
            "known_bad": worker_id in known_bad,
        }
    )

list_text = "\n".join(lines)
list_error = list_exit != 0 or "status code:500" in list_text or "failed to list gpu worker" in list_text
healthy = [
    worker
    for worker in workers
    if worker["gpu"] == "8"
    and worker["gpu_type"] == "NVIDIA-B200"
    and worker["pod_ip"]
    and worker["pod_ip"] != "-"
    and not worker["known_bad"]
]

print(f"KNOWN_BAD_WORKERS={','.join(sorted(known_bad)) if known_bad else 'NA'}")
print(f"WORKER_LIST_EXIT={list_exit}")
print(f"WORKER_LIST_STATUS={'ERROR' if list_error else 'OK'}")
print(f"WORKER_COUNT={len(workers)}")
for worker in workers:
    print(
        "WORKER "
        f"id={worker['id']} gpu={worker['gpu']} gpu_type={worker['gpu_type']} "
        f"pod_ip={worker['pod_ip']} known_bad={int(worker['known_bad'])}"
    )

if list_error:
    print("READINESS=FAIL")
    print("REASON=worker list command failed; worker control plane is unhealthy")
    raise SystemExit(2)

if len(workers) == 0:
    print("READINESS=FAIL")
    print("REASON=no worker is listed")
    raise SystemExit(2)

if len(workers) > 1:
    print("READINESS=FAIL")
    print("REASON=multiple workers are listed; keep exactly one worker before running v81")
    raise SystemExit(2)

if not healthy:
    print("READINESS=FAIL")
    print("REASON=no healthy non-known-bad 8x NVIDIA-B200 worker is listed")
    raise SystemExit(2)

print("READINESS=PASS")
print(f"SELECTED_WORKER={healthy[0]['id']}")
print("NEXT_COMMAND=NO_COLOR=1 TERM=dumb mlx worker login " + healthy[0]["id"])
print("RUNNER_AFTER_LOGIN=bash " + runner)
PY
