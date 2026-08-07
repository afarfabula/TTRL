# TTRL Workspace Experiment Archive - 2026-08-07

This is a slim archive of the local experiment workspace from:

`/mlx_devbox/users/quyanyi/playground/TTRL`

It is stored in the `afarfabula/TTRL` fork so the experiment records are not
committed into the upstream `PRIME-RL/TTRL` repository.

## Included

- Planning and progress notes from the workspace root.
- Resume-oriented result mining notes.
- `important_experiment_logs/*.md` experiment writeups and audits.
- `*_summary.json`, `*_val_metrics.json`, CSV metric files, and small JSONL
  diagnostics useful for result reconstruction.
- `run_records/` launchers and small helper scripts.
- The paper-style TTRL Math500 record directory with README, metrics, manifest,
  and launcher copy.

## Omitted From Git

Raw stdout/stderr logs and per-sample model-output JSONL files are intentionally
omitted from this slim archive because they do not materially improve the resume
bullet mining evidence and make the git commit too large.

The omitted files are still indexed for local recovery:

- `raw_log_manifest.tsv`: raw `.log` files with sha256, byte size, and original
  local path.
- `large_artifact_manifest.tsv`: omitted `*_outputs.jsonl` and diagnostic JSONL
  files with sha256, byte size, and original local path.

No checkpoints, dependency caches, or model weights are included.

## Remote Safety

- Original source worktree: `/mlx_devbox/users/quyanyi/playground/TTRL`
- Original source remote: `https://github.com/PRIME-RL/TTRL.git`
- Archive destination: `/mlx_devbox/users/quyanyi/playground/refcode/TTRL`
- Archive destination remote: `https://github.com/afarfabula/TTRL.git`

No commit is required in the upstream source worktree for this archive.
