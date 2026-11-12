# Critic dependency pruning implementation plan

> **For agentic workers:** Execute inline with test-driven-development, then request a fresh review of the complete change.

**Goal:** Skip original critic micro batches that do not contribute to any selected GAE target, without changing selected samples or normalization.

**Architecture:** Build the original full-segment replica placement and micro boundaries first. Retain a whole original micro when any member is required, return ordered sparse current Values, and calculate the original GAE on the selected suffix. Scheduler demand counts only retained observations.

**Tech stack:** Python, NumPy, existing Ray actor interfaces and unittest CPU fixtures.

**Spec:** Research approved by the user's request to implement and push; evidence and full design are in `analysis/weaver_value_research_20261112` in the parent experiment workspace. Recent P491–590 replay predicts 81,395 → 65,031 refresh micro calls; this is not measured GPU speedup.

## Constraints

- Edit the existing `codex/weaver-original95-runnable` branch directly as requested; no new worktree or training snapshot.
- Keep sampled IDs/order, 144:432, rank shards, rewards/logits/masks, current batch global normalization and float64 GAE recurrence.
- Keep the original replica placement and each retained micro's companions, order and padding; no Value-only, EMA, cache or concurrency changes.
- Keep version pins, cancellation/failure cleanup and successful consumption ownership. Missing current dependencies fail, never use generation Values.
- Feature defaults off in generic library use and is explicitly enabled in the maintained recipe, with a documented rollback.
- No new training is authorized by this task. CPU/fixture validation does not certify OpenVLA/CUDA numerical equality.

## Review focus

- Nonterminal bootstrap must remain; terminal placeholder is not a measured model prediction.
- Explicit retained batch boundaries, including batch1, cannot be silently repacked.
- Wrong version, missing/nonfinite/duplicate results cancel the reservation before training.
- Cancellation during admission/registration must release exactly actual retained demand.
- Skipped unused predictions no longer receive the original finite-output check; successful-path equivalence is the claim.

## Tasks

- [x] Add failing tests for dependency masks, original micro preservation and sparse suffix reconstruction; implement in `critic_refresh_plan.py`, integrate `SegmentReplay.materialize`.
- [x] Add failing actor/dispatcher tests for explicit batch boundaries, actual demand and cancellation/failure; extend `policy_service_dispatch.py` and Value endpoints.
- [x] Wire Replay sampling and feature configuration; test actual AST-extracted Replay/actor paths against the full-refresh reference and verify recipe scope.
- [x] Run targeted and full CPU checks, server CPU overlay/serialization checks if reachable, and a fresh code review; resolve findings.
- [x] Document scope and limits, commit only related changes, push the maintained GitHub branch and verify remote HEAD.

## Execution record

- Initial parent: `10914ab908ce280de45e41f48fc1bd3f8d07fee6`; branch and worktree clean.
- Ruling: work directly in the existing maintained repository, following explicit user/AGENTS instructions over the skill's worktree default.
- Ruling: retain a legacy full-refresh rollback; generic tests/configurations remain unchanged unless the pruning switch is enabled.
- TDD: the first seven dependency tests failed on missing planner/batch-boundary behavior; after implementation they passed. The actual Replay wiring test then failed because no sparse boundaries were sent; wiring the planner made it pass.
- Final local acceptance: 400 tests, zero failures/errors/skips; 208 Python files compiled, Python 3.14 / PyTorch 2.9.1+cpu, CUDA uninitialized. The first runner invocation lacked `WEAVER_TEST_UPSTREAM`; rerun used the bundled pinned vendor checkout.
- Server acceptance: 61 targeted CPU tests plus 12 cloudpickle checks, Python 3.11.13 / PyTorch 2.12.0+cu130; CUDA uninitialized, Ray not started. Changed modules and actual actor AST methods were overlaid in memory against the same parent commit; no deployment/source copy. An earlier full server-suite attempt hit the 180-second SSH budget and is not a pass.
- The unrelated historical `test_execution_control_cpu` module has identical two failures/nine errors on both parent and candidate (12 tests); it expects later parent-budget APIs absent from the restored orange implementation and is outside the existing acceptance module list. No unrelated scheduler behavior was changed to satisfy it.
- Fresh read-only reviewer found no definite correctness or requirement issue, including four-replica/global576 dependency, GAE and async-cleanup checks. Additional cancellation, failed-micro, direct-endpoint, rollback and malformed-position tests are included in the final acceptance run.
- Validation summaries and commands: `docs/validation/2026-11-12-critic-dependency-pruning.json`; detailed logs remain in the experiment workspace analysis directory, outside the maintained source tree.
