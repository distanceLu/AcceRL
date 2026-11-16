# Model-free budget stopping and periodic validation implementation plan

> **For agentic workers:** Use superpowers:executing-plans or subagent-driven-development to implement and verify each owned component.

**Goal:** Gracefully stop model-free training at its real collection budget or a fixed-validation plateau, with permanent online Eval0 and overwrite-only checkpoints.

**Architecture:** The driver admits a synchronized update only when the complete global real cohort is eligible. Budget and manual stop decisions occur before this admission, so no rank enters a collective while another is waiting for unavailable Replay. At a validation boundary, stop new collection leases, drain existing episodes, keep models/optimizers/Replay resident, and run finite fixed cases against an unchanged published Policy.

**Tech stack:** Python, Ray actor RPC, existing EGL LIBERO environments, existing CPU unittest harnesses.

**Spec:** User's approved choice in this chat: overwrite saving; budget stopping; performance early stopping; low-frequency fixed validation; permanent Eval0. This document records implementation defaults chosen by the implementer.

## Constraints and choices

- Edit the existing development branch directly; never modify a pinned running experiment or launch training during this task.
- Model-free only; preserve the existing model-based training behavior.
- Budget is consumed simulator steps, including reset settling; wait for outstanding leases before declaring exhaustion. An unusable small remainder is valid exhaustion.
- Defaults: validation at P0, then every 100000 newly consumed budget steps, and final weights; 50 existing fixed cases, existing fixed seeds, greedy, batch1, EGL, action horizon220.
- Plateau: no improvement of at least one success for three completed periodic validations; do not stop before300000 consumed steps. Parameters configurable. No statistical-significance claim.
- Save latest weights by atomic overwrite. Track best score in metadata only; do not retain best/milestone weights. Existing separate initial snapshot remains.
- Validation creates no training Replay entries or training success statistics. Failure or interruption cannot become a completed validation score.
- Graceful automatic reasons must be accepted by driver, managed runner, and queue with budget/validation evidence. Legacy manual-only contracts remain strict.

## Approaches

Continuous online Eval contradicts the chosen resource policy. Process stop/restart for every validation loses optimizer/Replay state because checkpoints are weights-only. Use finite validation while the live training state stays resident, with collection paused at episode admission and training paused at a synchronized update boundary.

## Review focus

1. Exhaustion during initial fill or waiting for a cohort must save and exit rather than hang.
2. Last in-flight reservations may refund unused steps; do not mistake reserved budget for consumed budget.
3. No update starts without a complete eligible cohort, and no stop interrupts an active collective.
4. Validation must use the same cases and one frozen Policy version; cleanup/resume on error must be explicit.
5. Runner/queue must accept supported automatic completion without accepting malformed or failed runs.

## Tasks

- [x] Add CPU-tested collection admission pause, exhaustion accounting, and a pure configurable plateau controller.
- [x] Add CPU-tested finite fixed-case EGL validation helper using existing inference replicas and exact protocol identities.
- [x] Add driver readiness gate, P0/periodic/final validation, telemetry, and completion evidence; test actual driver control flow with inert model boundaries.
- [x] Forward configuration through managed launch and teach runner/queue automatic completion contracts; retain final evaluation and overwrite artifacts.
- [x] Run relevant CPU regression suite, independent code review, syntax/whitespace checks, and document results.
- [ ] Commit and push reviewed source via the existing authenticated server workflow, synchronize development checkout only, verify pinned runtime remains unchanged.

## Verification evidence

- Relevant engine regression selection: 559 tests passed with CUDA uninitialized; includes actual driver budget/plateau/manual waiting paths and a mutation proving the pre-update gate matters.
- Additional review fix: validation/drain duration no longer consumes the Replay supply timeout; its failing regression was observed before the fix, then all8 runtime tests passed.
- Launch/configuration/completion18 and queue admission16 tests passed. Python AST parsing and Bash syntax/whitespace checks passed.
- Initial-state allocation disabled in model-free is now rejected before model creation. CPU startup fixtures use the actual managed recipe.
- GPU model loading, EGL execution through the new temporary actor and first distributed optimizer update remain for the next authorized training experiment. This task does not launch one.
