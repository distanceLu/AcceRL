# EGL evaluation and critic admission implementation plan

> **For agentic workers:** Use superpowers:subagent-driven-development for independent tasks; root integrates and reviews.

**Goal:** Default fixed GPU evaluation to verified EGL and defer new online Eval episodes during current-critic refresh, then push the requested branch.

**Architecture:** A shared renderer bootstrap preserves CUDA UUID binding while initializing the simulator with an explicit EGL mapping. A broker-owned episode admission gate uses existing refresh demand and admits only at episode boundaries.

**Tech Stack:** Python, asyncio, Ray, PyTorch, MuJoCo/robosuite, Git.

**Spec:** `docs/superpowers/specs/2026-11-15-egl-eval-critic-admission.md`.

## Global constraints

- Edit existing branch directly as requested; no worktree, no new training.
- No installed-library edits, no new data under `/home`, no shared MPS changes.
- Preserve policy/environment semantics and in-flight work.
- Push is explicitly authorized; never force-push.

## Review focus

- Import failure restores CUDA visibility; EGL maps to the exact model GPU.
- Eval's first reset is also gated; permitted episodes survive a later refresh start.
- Refresh counter nesting/failure/cancellation cannot deadlock admission.
- Repeated refresh updates cannot reset the continuous-wait safeguard.
- Renderer source fingerprints and existing CPU admission remain valid.

## Tasks

- [x] EGL: regression tests, shared evaluator bootstrap, managed and standalone evaluation integration, reference fingerprint updates where required, documentation.
- [x] Eval admission: behavioral tests, broker gate and telemetry, worker integration before every reset, managed configuration and documentation.
- [x] Integration: focused tests, portable acceptance, independent concurrency/device review, bounded server smoke.
- [ ] Delivery: update plan with evidence, commit changes, push exact branch, verify GitHub SHA, synchronize clean idle server checkout and report restart scope.

## Execution record

Starting revision: `c2f8a746de484bf6ea7de4aebf77b04b809e9aae`.
User's direct-edit request overrides skill defaults for new worktrees and additional design approval.

2026-11-15 verification: local/server CPU acceptance each pass 466 tests, with
zero failures/errors/skips and identical engine source fingerprints. Parallel
evaluation tests pass 112 cases. Independent review found no blocking issues.
Four actual GPU evaluator runtimes pass construction and reset/action checks
with NVIDIA EGL on their allocated PCI identities. An initial failure exposed
two libEGL installations loaded in the same process; isolated device discovery
resolves the reproduced failure without changing installed libraries.
An actual zero-GPU Ray broker smoke verifies nested refresh release and stop.
No test processes or MPS clients remain. No new training was started.

Evidence: workspace `analysis/egl_eval_gate_implementation_20261115/` contains
CPU receipts, `egl_smoke.json`, `ray_gate_smoke.json`, the initial A/B diagnosis,
source staging hashes and `final_resources.json`. Delivery receipt is written
there as `publication.json` after the authorized push and server fast-forward.
