#!/usr/bin/env bash
set -euo pipefail

READINESS="/opt/tiger/TTRL/verl/examples/ttrl/check_worker_readiness_for_v81.sh"
tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

write_fixture() {
  local name="$1"
  local body="$2"
  printf '%s\n' "$body" > "$tmpdir/$name.txt"
}

run_case() {
  local name="$1"
  local fixture="$2"
  local expected_status="$3"
  local expected_snippet="$4"
  local output
  local status

  set +e
  output="$(WORKER_LIST_FIXTURE="$tmpdir/$fixture.txt" bash "$READINESS" 2>&1)"
  status=$?
  set -e
  printf '%s\n' "CASE=$name STATUS=$status"
  printf '%s\n' "$output"
  if [ "$status" -ne "$expected_status" ]; then
    printf '%s\n' "CASE=$name EXPECTED_STATUS=$expected_status GOT=$status"
    exit 1
  fi
  if ! grep -Fq "$expected_snippet" <<<"$output"; then
    printf '%s\n' "CASE=$name MISSING_SNIPPET=$expected_snippet"
    exit 1
  fi
}

header="id        cpu    mem    gpu    gpuType      podIP                                 port   createdAt            webshell"

write_fixture "api500" "[MlxClient] ListWorkder, [HttpGet]URL:/api/v1/workspaces/57226/workers/detail/ status code:500, resp:<h1>Server Error (500)</h1>
failed to list gpu worker, err: [MlxClient] ListWorker, [HttpGet]URL:/api/v1/workspaces/57226/workers/detail/ status code:500, resp:<h1>Server Error (500)</h1>"
write_fixture "none" "$header"
write_fixture "known_bad" "$header
989057    248    3800   8      NVIDIA-B200  fdbd:dccd:cde2:2131:0:e665:d5ca:9aea  11360  2026-07-08T10:35:46  https://example.invalid/"
write_fixture "multi" "$header
987900    248    3800   8      NVIDIA-B200  fdbd:dccd:cde2:2131:0:e665:d5ca:9aeb  9615   2026-07-07T09:59:56  https://example.invalid/
987901    248    3800   8      NVIDIA-B200  fdbd:dccd:cde2:2131:0:e665:d5ca:9aec  9616   2026-07-07T09:59:57  https://example.invalid/"
write_fixture "healthy" "$header
987900    248    3800   8      NVIDIA-B200  fdbd:dccd:cde2:2131:0:e665:d5ca:9aeb  9615   2026-07-07T09:59:56  https://example.invalid/"

run_case "worker_list_api500" "api500" 2 "REASON=worker list command failed; worker control plane is unhealthy"
run_case "no_worker" "none" 2 "REASON=no worker is listed"
run_case "known_bad_single" "known_bad" 2 "REASON=no healthy non-known-bad 8x NVIDIA-B200 worker is listed"
run_case "multiple_workers" "multi" 2 "REASON=multiple workers are listed; keep exactly one worker before running v81"
run_case "healthy_single" "healthy" 0 "READINESS=PASS"

echo "V81_WORKER_READINESS_GUARD_SMOKE_OK"
