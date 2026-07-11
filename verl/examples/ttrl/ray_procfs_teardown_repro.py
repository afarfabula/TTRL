#!/usr/bin/env python3
"""Minimal Ray teardown/procfs repro for MLX workers."""

import argparse
import json
import os
import socket
import time

import ray


def proc_health(label: str) -> dict:
    health = {
        "label": label,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "proc_self": os.path.exists("/proc/self"),
        "proc_meminfo": os.path.exists("/proc/meminfo"),
    }
    try:
        health["proc_count"] = len(os.listdir("/proc"))
    except Exception as exc:  # pragma: no cover - this is diagnostic output.
        health["proc_count_error"] = repr(exc)
    print("PROC_HEALTH", json.dumps(health, sort_keys=True), flush=True)
    return health


@ray.remote
def ping_task() -> dict:
    return proc_health("inside_task")


@ray.remote
class PingActor:
    def ping(self) -> dict:
        return proc_health("inside_actor")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["explicit", "implicit"], required=True)
    parser.add_argument("--num-cpus", type=int, default=2)
    args = parser.parse_args()

    proc_health("before_ray_init")
    ray.init(
        num_cpus=args.num_cpus,
        runtime_env={
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "false",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "WARN",
            }
        },
    )
    print("RAY_ADDRESS", ray.get_runtime_context().gcs_address, flush=True)
    proc_health("after_ray_init")

    task_result = ray.get(ping_task.remote())
    print("TASK_RESULT", json.dumps(task_result, sort_keys=True), flush=True)

    actor = PingActor.remote()
    actor_result = ray.get(actor.ping.remote())
    print("ACTOR_RESULT", json.dumps(actor_result, sort_keys=True), flush=True)

    if args.mode == "explicit":
        print("CALLING_RAY_SHUTDOWN", flush=True)
        ray.shutdown()
        proc_health("after_explicit_ray_shutdown")
    else:
        print("SKIPPING_RAY_SHUTDOWN_FOR_IMPLICIT_EXIT", flush=True)

    print("SCRIPT_DONE", args.mode, flush=True)


if __name__ == "__main__":
    main()
