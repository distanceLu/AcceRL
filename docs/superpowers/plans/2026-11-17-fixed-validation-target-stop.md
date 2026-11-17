# Fixed-validation target stop implementation plan

> **For agentic workers:** Use superpowers:subagent-driven-development to implement and review the bounded tasks below.

**Goal:** Save and stop model-free updates on the first complete 50/50 fixed validation, then execute normal final assessment.

**Architecture:** Add a distinct target decision to the CPU controller and runtime boundary. Reuse the final snapshot/cleanup path, extend fail-closed completion receipts, and safely update the existing delayed launch to the new pinned commit.

**Tech Stack:** Python, Ray/DeepSpeed orchestration, unittest, Git, existing SSH and experiment queue.

**Spec:** `docs/superpowers/specs/2026-11-17-fixed-validation-target-stop-design.md`.

## Global constraints

- Edit directly in `codex/weaver-original95-runnable`, as required by workspace AGENTS.md.
- Do not change active `.git-runtime` or claim old historical weights are recoverable.
- Preserve 50 fixed cases, greedy, batch1, EGL, existing horizons and global training batch.
- No partial or wrong-version evaluation can count as target completion.
- All server artifacts belong on the allocated data disk.

## Review focus

- P0 already full: no replay fill, collection, or training must be required.
- First later full result: no next cohort or optimizer update may begin.
- Saved/final policy version must equal the successful evaluated version.
- Target completion must survive strict runner/queue audits and reach final evaluation.
- Pending successor must retain exact predecessor identity and launch only once with the new commit.

## Tasks

- [x] Controller/runtime: add `target_reached`, expose the decision, preserve pause on success; write failing CPU tests before implementation.
- [x] Driver: handle P0 success before worker launch/fill; retain final save and exact final version; exercise the actual main with inert transports. Fresh review additionally required metadata-only initialization of the real coordinator before recording P0, now tested through the real completion validator without acquiring a lease.
- [x] Completion and supervisors: validate target history/version/budget evidence and permit the distinct natural stop reason; add rejection and P0 tests.
- [ ] Operations: verify and gracefully stop the old run, retain real provenance, synchronize/push tested development code, and rebind the delayed helper without touching active runtime files.
- [ ] Verification: run focused controller/runtime/driver/receipt/queue and autotune regressions locally and on server; perform CPU admission, independent review, commit and verify GitHub/server identity.
- [ ] Observe current saved checkpoint and final-evaluation transition, and verify successor controller state and pinned configuration.
