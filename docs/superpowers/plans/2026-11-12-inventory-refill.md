# Inventory-driven refill implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Execute in the maintained branch as explicitly requested by the user.

**Goal:** Spend existing production credit before a projected eligible-stock shortage, including the first batch after the next World publication.

**Architecture:** Replay projects distinct and legal-use capacity without sampling. A bounded CPU feedback loop replaces an expiring production plan; the serialized producer alone accounts credits, leases, versions and seed readiness. Macro scheduling consumes that producer's actual readiness, including already-running work.

**Tech Stack:** Python, asyncio, existing Ray actors; CPU tests with actual Replay/producer classes.

**Spec:** User-approved design in `docs/inventory-refill.md`.

## Global Constraints

- Edit `codex/weaver-original95-runnable` directly; no copied source tree, no new launcher.
- Do not modify/stop/restart the current GPU run. No new GPU training in this task.
- Keep 144:432, global576/1024, 5:1, age16/3, usecap4, sampler, reward, GAE and behavior logits unchanged.
- Generation credits remain capped and funded by actual successful global draws; no minted credit, no refund of executed work.
- One earliest uncovered cohort (432 imagined chunks maximum); replace plans, never accumulate them.
- Version/revision/TTL fences, deduplicated in-flight output, publication checks and cleanup remain.
- No GPU calls, extra actor, per-micro RPC or CUDA synchronization on the new fast path.

## Review Focus

- Bootstrap fill must remain possible before the loop is armed; stale control must close speculation without disabling reactive rescue.
- ACK races must conservatively under-dispatch rather than count a chunk as both admitted and pending.
- Publication interleavings and late RPC replies must not revive an old-version plan or retain stale macro readiness.
- Four active leases remain runnable even when no fifth seed can start.
- Seed protection must yield to the global byte cap and mandatory World windows without advancing stratification counters.

### Task 1: Project demand and estimate complete-cohort lead time

Files: new `weaver_run/engine/inventory_refill.py`, `test_inventory_refill_cpu.py`.

- [x] RED: fixtures for P29 world-age cliff, distinct vs uses, reserved samples, terminal publication pairs, six-boundary horizon, no RNG mutation.
- [x] Implement `project_refill(store, phase, timing, producer, now)` with capacity envelope, explicit world-version projection, measured work-time deadlines and bounded earliest deficit.
- [x] GREEN: test projection and timing with literal expectations; preserve random Replay draws.

### Task 2: Unify producer admission and seed retention

Files: `pipeline_budget.py`, `pipeline_runtime.py`, tests above.

- [x] RED: 1728-credit pause accepts bounded plan, caps credit/output, TTL/revisions/version changes, inflight readiness, cancellation/zero output, reserve under frame pressure.
- [x] Implement `set_refill_plan`, `refill_status` and `production_admission`; credit and lease owner remains PipelineBudget. Record bounded end-to-end production timing. Protect at most one existing candidate per six strata during optional pending eviction, with forced fallback under mandatory byte pressure.
- [x] GREEN: real producer state transitions plus prior budget/pressure/seed/integration suites.

### Task 3: Wire bounded feedback and macro readiness

Files: `inventory_refill.py`, `train.py`, `cycle_service_controller.py`, `cost_scheduler_runtime.py`, `pipeline_runtime.py`, recipe pipeline JSON.

- [x] RED: actual Replay actor loop activation after fill, reserve/publish/stop lifecycle, delayed transport, stale producer state, active-only readiness in macro.
- [x] Implement one sequential feedback loop, two-second refresh/phase wakeups, visible degraded fallback; no overlapping remote retries. Propagate producer revision/version/expiry and in-flight state to macro.
- [x] Enable candidate in maintained recipe; generic PipelineBudgetConfig defaults remain off.
- [x] GREEN: run new integration tests and canonical CPU acceptance suite.

### Task 4: Review, validate and publish code

- [x] Fresh independent read-only code review, fix important findings with RED/GREEN regressions.
- [x] Server CPU validation with CUDA hidden and bounded execution; serialization and compile checks. Explicitly separate CPU evidence from Ray/GPU acceptance.
- [x] Document behavior, limitations and results. Commit and push existing branch; verify GitHub SHA. No training launch.
