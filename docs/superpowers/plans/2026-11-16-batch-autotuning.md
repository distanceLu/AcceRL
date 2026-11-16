# Memory-aware batch autotuning implementation plan

> **For agentic workers:** Use executing-plans with bounded independent implementation/review tasks. Tasks below record their own RED/GREEN evidence.

**Goal:** Add an opt-in, bounded model-free calibration that selects critic refresh and Policy micro-batches using measured throughput and GPU memory headroom while keeping global batch 576.

**Architecture:** One driver controller selects a shared configuration for four ranks at complete update boundaries. Calibrate refresh batches 2/4/8 first, then training micro/GAS 8/18,16/9,24/6, using six-update windows (first observation warmup, remaining measurements). Freeze after the finite search. Missing/invalid telemetry or predicted insufficient headroom skips expansion. Existing runs/default settings remain unchanged.

**Tech Stack:** Python, Ray, PyTorch, DeepSpeed ZeRO2; pure CPU unit tests plus runtime receipts.

**Spec:** User-approved proposal in this conversation; implementation decisions below complete that proposal.

## Global constraints
- Direct edits on codex/weaver-original95-runnable in this development repository, per user AGENTS.md.
- No restart, stop, GPU benchmark or changes to the active pinned experiment.
- Model-free only, Real4, online Eval0, four train ranks, global576, local144. Fixed validation stays greedy batch1/EGL.
- Defaults off; explicit --batch-autotune enables finite calibration in the first normal training updates, not disposable hidden training or replay probes.
- No counter rebasing, partial update retry or automatic per-rank OOM recovery. An unexpected CUDA error follows existing whole-run failure handling.
- Memory reserve max(15% total,8GiB), expansion estimate adds 1.25 times measured activation growth; prediction is a guard, not proof of fit.
- Loss preserves original contiguous groups of eight samples before combining into larger micros. Value loss remains sample mean; Policy/entropy/KL keep the original group-of-eight masked-token weights.
- Every candidate window spans six complete updates, preserving modulo18 cumulative microstep boundaries for all supported GAS values.
- Report measured micro/GAS, refresh batch, phase duration, real latency, whole-device sampled memory, skip reason and frozen selection in TensorBoard/SQLite/JSON.
- No performance claim until actual GPU calibration runs.

## Review focus
- Rank divergence or GAS change at an invalid cumulative boundary must fail before backward.
- Padded/partial action chunks must retain old group-of-eight objective weights.
- Missing telemetry, cold-start peaks, allocator caches and co-resident services must not masquerade as free capacity.
- Checkpoint validation is excluded from calibration; action batch1 and published critic version remain exact.
- Queue/runner/check arguments and final receipt audit must carry the same feature flag.

## Task 1: Pure configuration and finite controller
Files: engine/batch_autotune.py and test_batch_autotune_cpu.py.
- [x] RED tests: finite refresh-first selection, warmup exclusion, real latency rejection, absent memory refusal, reserve guard, worst-card bound, frozen best, six-update safe transitions.
- [x] Implement strict opt-in config, memory model, measured windows, immutable decision receipts; use runtime metrics with normalized refresh seconds per observation and fixed576 training update time, plus action latency constraints. Full update time is recorded but stage-normalized score avoids changing refresh population confounding selection.
- [x] GREEN targeted tests.

## Task 2: Safe execution and invariant tests
Files: policy_joint_batch_mode.py, policy_batch_receipts.py, new batch_autotune_runtime.py and batch_loss.py, train.py, corresponding CPU tests.
- [x] RED tests: installed DeepSpeed CPU boundary simulation across all candidate and rollback blocks; malformed rank receipts; partial-mask objective invariance.
- [x] Implement driver configuration RPCs before epoch submission; four-rank agreement in the existing barrier, running cumulative receipt checks, and preserved loss weighting.
- [x] Implement cooperative model-free critic refresh using configurable chunks and exact version/order; no extra samples or World services.
- [x] Wire bounded per-update stats and controller report/TensorBoard hooks.
- [x] GREEN relevant unit and integration tests.

## Task 3: Managed launch and reproducibility
Files: launch.py, check_runtime.py, runner.py, reference_site/queue_wrapper entry points, docs/model-free.md, launch CPU tests.
- [x] RED: opt-in threaded through actual managed command paths, reject unsupported mode/Eval, defaults unchanged.
- [x] Implement --batch-autotune, metadata propagation and adaptive receipt audit.
- [x] GREEN launch checks and targeted regression suites; run broader CPU suite if feasible and report any existing environment failures.

## Task 4: Review and publish
- [x] Independent correctness review of final diff, address material findings and rerun affected checks.
- [ ] Commit related files; push same GitHub branch and verify exact remote SHA. Synchronize server development only, leave active pinned runtime untouched.
- [ ] Report implemented behavior, test evidence and lack of GPU speedup measurement.
