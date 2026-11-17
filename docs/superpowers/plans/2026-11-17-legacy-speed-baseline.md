# Legacy speed baseline implementation plan

> **For agentic workers:** Use the authorized direct development branch. Execute
> scoped implementation tasks with tests and independent review.

**Goal:** Queue a bounded original GIPO baseline after the current experiment.
**Architecture:** Legacy trainer lifecycle hooks, a dedicated official-entry
queue adapter, and a CPU-only exactly-once predecessor watcher.
**Tech Stack:** Python, Ray, DeepSpeed, Git, existing server queue.
**Spec:** `docs/superpowers/specs/2026-11-17-legacy-speed-baseline-design.md`

## Global constraints

- Work on `codex/weaver-original95-runnable`; never change the active runtime.
- Eval0, GIPO, no world model; keep legacy replay/GAE/rendering semantics.
- Save outputs under `/mnt/data/lcx1/dong/WEAVER`, preserve original files.
- Publish and verify the exact commit before installing the authorized handoff.

## Review focus

- A timed boundary saves only after all ranks update and publication completes.
- A failure cannot become a success merely because checkpoint files exist.
- No cleanup can target another experiment, other user, or the shared MPS server.
- Waiting for training alone must not race the predecessor's final evaluation.
- An existing output/queue identity cannot silently change configuration.

## Task 1 — bounded legacy training lifecycle

Files: `rl/ds_libero_ppo_discrete.py`, new `rl/legacy_baseline_control.py`, and CPU
tests. Add `--baseline-output-dir`, `--max-train-seconds`, `--ray-temp-dir` and
`--stop-file`. With bounded mode enabled, record arguments/placements/per-update
timing, check duration or stop flag at completed update/publication boundaries,
save final rank0 inference weights into `checkpoints/latest`, and write
`driver_complete.json` before normal Ray shutdown. Records are inside the provided
output directory; Ray uses a short recorded `/mnt/data/lcx1/ray/<hash>` path.
Timing excludes initialization/Replay
fill and records those separately. Do not add an env-step budget in this task.

- [x] Write and run failing CPU tests for boundary decisions and final-save order.
- [x] Implement opt-in lifecycle hooks and verify defaults remain compatible.
- [x] Verify compilation and relevant CPU tests.

## Task 2 — dedicated queue adapter

Files: new `weaver_run/reference_site/legacy_baseline.py`,
`legacy_queue_entry.py`, CPU tests. Parent owns small dispatch edits in `cli.py`
and `launch.py`. Adapter supports a check-only invocation and actual submission
through official `run.sh legacy-baseline --id LABEL --training-seconds 7200`.
Use immutable `WEAVER/.legacy-runtime/<commit>` checkout and a dedicated
`legacy_speed_baseline` record. Reuse queue helpers but not WEAVER acceptance.
The supervisor validates the frozen source/arguments, runs the legacy entry,
audits completion plus saved artifacts, and performs exact run-owned cleanup.

- [x] Test exact argv, validation, source identity, and duplicate identity behavior.
- [x] Implement queue adapter and supervisor with failure/cleanup tests.
- [ ] Wire official dispatch and run CPU admission on the server.

## Task 3 — review, publish and install continuation

Files: operational helper/evidence under `analysis/legacy_baseline_20261117`
and server `WEAVER/pending/legacy-gipo-baseline-20261117-a`.

- [x] Independently review diff, lifecycle and compatibility; resolve findings.
- [ ] Commit, publish and verify the exact branch and server development HEAD.
- [ ] Install an idempotent CPU-only watcher for the exact current queue job.
- [ ] Verify it is waiting, no baseline GPU processes started, and current
      experiment still runs from `16f75f9`.
- [ ] Report configuration, differences, duration and durable status paths.
