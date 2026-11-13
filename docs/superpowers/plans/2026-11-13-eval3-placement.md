# Eval3 placement implementation plan

> **For agentic workers:** Use superpowers:executing-plans. The user explicitly requested direct edits on the existing branch; do not create a worktree or source copy.

**Goal:** Use three online Eval workers on Policy replicas 1,2,3 (physical GPU5–7 at the reference site), with a 300-episode window.

**Architecture:** Resolve and validate online Eval routing and the workers×100 window before creating actors. Keep the existing 1000-entry non-Eval statistics. Record the resolved routing/window for startup acceptance. Keep the training algorithm and running process unchanged.

**Tech Stack:** Python, existing Ray actors, argparse, JSON recipe, CPU unittest/AST fixtures.

**Spec:** User's approved Eval3 / window300 / GPU5–7 proposal in this conversation. Alternatives Eval2 and Eval4 were examined; Eval3 removes Eval competition from the real-collection replica.

## Global constraints

- Edit `codex/weaver-original95-runnable` directly, commit and push tested changes; no source copy.
- Do not stop, restart, hot-edit or submit the current GPU run in this task.
- Preserve real/imagination sampling, learning, RNG, Value computation, optimizer and model allocation.
- Online window is workers×100; non-Eval window remains 1000. This is not a frozen-policy evaluation.

## Review focus

- Reject missing/out-of-range explicit replica assignments instead of silently routing onto the real-collection card.
- Keep worker IDs separate from target replica IDs; do not change worker seed identities to route requests.
- Derive the minimum evaluation sample gate from the actual window; reject an unreachable explicit threshold.
- Exercise the real StatsActor methods: Eval300 must not truncate real or imagined statistics to300.
- Validate the actual main actor arguments, including reference GPU IDs and portable four-card allocations.

## Task 1: routing, independent window, recipe and regression tests

- [x] Add failing CPU tests for config validation, actual StatsActor retention, main worker routing and threshold eligibility at300.
- [x] Implement pure configuration resolution in `evaluation_threshold.py`; wire main, StatsActor and telemetry in `train.py`.
- [x] Update recipe Eval count/routing, automatic threshold gate and the managed shell default. Update README.
- [x] Run targeted tests, then the existing cost/fresh CPU acceptance runner and relevant startup/launcher suites.

## Task 2: review and versioning

- [x] Record the existing P236–265 baseline and the four GPU acceptance metrics in the documentation.
- [x] Review the diff and verify no training math or running source changes.
- [x] Clearly distinguish CPU acceptance from the next GPU run's real queue/Value/throughput/supply measurements.

## Verification and delivery

Final CPU acceptance: 442 tests, zero failures/errors/skips, CUDA not initialized; focused configuration/startup tests: 22 passed. Independent review's disabled-Eval routing issue was reproduced, fixed and rechecked; no outstanding actionable findings. Actual Ray/CUDA behavior and the four performance outcomes require a future authorized run.

Delivery uses a normal commit and push on the existing branch, followed by a remote-head check. Generated test reports and the delivery receipt stay in the experiment workspace rather than the source repository.
