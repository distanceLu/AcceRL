# Inventory-driven refill

The producer credit ledger is an upper bound, not an inventory signal. The candidate replaces post-bootstrap hysteresis-only production with short-lived, Replay-derived demand. It forecasts through the next joint publication and the first subsequent Policy draw. At each projected draw it checks both distinct chunks and remaining legal uses, including reserved uses and age at the projected Policy/World versions. Capacity estimates are necessary feasible envelopes, not promises about future random draws; Replay remains the final authority.

Only the earliest uncovered demand can authorize at most 432 new imagined chunks. Complete-cohort production latency includes seed admission, Policy/World service, and Replay ACK; measured reserve-to-publication work estimates exclude stockout waiting. Unknown timing is explicitly reported and uses immediate bounded refill rather than delaying demand on an invented throughput estimate. Existing credit, frame, lease and version caps still apply. Plans replace prior plans with monotonic revisions and expire after six seconds; phase changes wake the two-second CPU loop. A stale or failed controller stops speculative generation and reports degradation, while the original reactive shortage feedback remains available.

Production readiness distinguishes the ability to start a new seed from work already in flight. Both Policy and World service can be needed by active imagination even when all four lease slots are occupied. Macro scheduling uses the producer's accepted plan/readiness rather than treating forecast demand alone as runnable work. Stale/version-mismatched state is not actionable.

Optional seed eviction protects one existing candidate for each success/failure x early/middle/late bucket by keeping its containing episode reference (at most six references, often two; remaining starts in those episodes stay intact). This is a soft reserve, not extra fabricated seeds. The existing selector, position buckets and admission counters are unchanged; altered retention can change the available seed distribution. It cannot override active pins, the 8 GiB frame cap, minimum World windows, or collector progress. Mandatory pressure can release it and records `seed_reserve_forced_releases`.

Startup fill is unchanged until Replay is ready and the feedback loop is explicitly armed. Publication order and stale-resident rejection remain unchanged. No new GPU forward, actor, micro RPC, `.item()` or device synchronization is introduced. The learning recipe (144 real / 432 imagined, global576/1024, 5:1, age16/3, usecap4, sampler, rewards, GAE and behavior logits) is preserved, but earlier production changes the asynchronous sample pool and may change learning trajectories. The .25 generation-credit rate and four-use cap do not fund an arbitrary increase in unique data, or guarantee lower mean reuse. CPU tests cannot establish GPU speed or learning equivalence.

## Control lifecycle and measurement

- `inventory_refill.py`: at most six projected draws, exact integral prefix capacity flow; deadlines use observed ordinary/joint reserve-to-publication time. The earliest uncovered prefix is funded only within measured production lead time. Pending ACKs remain deducted until reconciled; they are not trainable stock.
- Production lead uses first seed demand/admission to last Replay ACK for completed 432-chunk cohorts, including service queues and intervening seed waits. Before that evidence exists, `lead_method=seed_latency_scaled_proxy` explicitly labels a conservative four-worker approximation; no history means immediate bounded refill. These are wall times, not pure CUDA times or throughput guarantees.
- The existing async Replay actor owns one feedback task/lock. It refreshes every two seconds and on reservation/publication/commit edges. The driver's existing once-per-update inventory read also settles the same-version producer receipt before macro weight selection. There is no new GPU actor. Timed-out outstanding control RPCs are tracked; a second controller RPC is not submitted over them.
- General owner-state expiry and `seed_admission_expires_at` are separate. An ACK cannot extend a plan's right to start seeds. Existing leases remain runnable after its plan expires. Readiness for real collection previews legal frame reclamation without evicting data, respects evaluation/World tickets and the minimum World cohort, then actual acquisition rechecks real bytes.
- `fresh_inventory_refill` records target versions, projected distinct/use capacity, deadline, bounded quota, producer receipt, timing method and CPU/RPC `control_wall_seconds`. `fresh_inventory_refill_degraded` makes loss of speculation visible; original `fresh_replay_wait` records reactive fallback. State polls refresh runnable services without repeating macro weight search.

## Enable, rollback and validation

Generic `PipelineBudgetConfig` defaults are `inventory_refill_enabled=false`, `seed_reserve_per_bucket=0`. The maintained `weaver_run/recipe/pipeline_config.json` explicitly selects `true` / `1`. To restore prior production/retention behavior for a future process, set both back to `false` / `0`, test and commit; launch through the same root `./run.sh`. Running processes retain their original commit.

```bash
cd weaver_run/engine
WEAVER_TEST_UPSTREAM="$(cd ../vendor/weaver && pwd)" python -B validate_cost_scheduler_candidate.py
```

The new tests cover expiry/revision/version fences, the post-publication age cliff, credit conservation, actual Replay activation/shutdown, delayed transport, active leases, legal frame reclamation and pinned World data. Validation receipts are in `docs/validation/2026-11-12-inventory-refill.json`. Real Ray/CUDA ordering, complete-cycle throughput, publication waste and same-stage learning results remain unverified; this code change does not start or alter a GPU run.
