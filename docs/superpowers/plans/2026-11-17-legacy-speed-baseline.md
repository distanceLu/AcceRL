# Legacy speed baseline implementation plan

> **For agentic workers:** Use the authorized direct development branch. Execute
> scoped implementation tasks with tests and independent review.

**Goal:** Queue an original GIPO baseline after the current experiment and
measure time to 98/99% rollout success.
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

- A stopping boundary saves only after all ranks update and publication completes.
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
through official `run.sh legacy-baseline --id LABEL`. The initial two-hour
duration was superseded by the success-target settings in Task 4.
Use immutable `WEAVER/.legacy-runtime/<commit>` checkout and a dedicated
`legacy_speed_baseline` record. Reuse queue helpers but not WEAVER acceptance.
The supervisor validates the frozen source/arguments, runs the legacy entry,
audits completion plus saved artifacts, and performs exact run-owned cleanup.

- [x] Test exact argv, validation, source identity, and duplicate identity behavior.
- [x] Implement queue adapter and supervisor with failure/cleanup tests.
- [x] Wire official dispatch and run CPU admission on the server.

## Task 3 — review, publish and install continuation

Files: operational helper/evidence under `analysis/legacy_baseline_20261117`
and server `WEAVER/pending/legacy-gipo-baseline-20261117-a`.

- [x] Independently review diff, lifecycle and compatibility; resolve findings.
- [x] Commit, publish and verify the exact branch and server development HEAD.
- [x] Install an idempotent CPU-only watcher for the exact current queue job.
- [x] Verify it is waiting, no baseline GPU processes started, and current
      experiment still runs from `16f75f9`.
- [x] Report configuration, differences, duration and durable status paths.

## Task 4 — user correction: stop on 99% rather than two hours

The user replaced the previously selected two-hour measurement with training
to 98/99% success. Stop the existing CPU continuation only after verifying its
waiting state and exact PID/UID/birth, then rebind its unpublished target after
testing/publishing. Do not change or stop the predecessor training.

- [x] Add an exact same-task real-success snapshot, exclude Eval, require a full
      1000-episode window, latch first98/first99 as episodes are recorded so a
      crossing followed by a decline between update boundaries cannot be lost.
- [x] Stop after first99 at a completed update/publication boundary; save
      counts, steps, launch/loop times and checkpoint version. Preserve manual
      stop and update-cap reasons without claiming target achievement.
- [x] Freeze target99/window1000/no duration in adapter/queue arguments. Use an
      update-progress watchdog rather than a two-hour wall-clock kill.
- [x] Test partial-window/malformed/NaN/989vs990 cases, milestone retention,
      positive-duration compatibility, healthy >2h and stalled-update behavior.
- [ ] Review, commit, verify GitHub/server, rebind one continuation and confirm
      waiting state, no baseline submission and unchanged active runtime.
