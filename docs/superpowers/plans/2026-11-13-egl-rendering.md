# EGL Rendering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable verified per-worker NVIDIA EGL rendering in the managed training recipe.

**Architecture:** A small CPU-importable rendering module resolves device identity and constructs Ray runtime environments. Real/Eval verify their active context and record receipts; the managed recipe selects EGL.

**Tech Stack:** Python, Ray, LIBERO/robosuite, ctypes EGL, NVIDIA driver.

**Spec:** `docs/superpowers/specs/2026-11-13-egl-rendering-design.md`

## Global Constraints

- Work directly on `codex/weaver-original95-runnable`, as instructed by AGENTS.md.
- CPU physics; existing horizons, worker counts and learning rules unchanged.
- No installed-library edits, automatic software fallback or data under /home.
- Preserve the active training process and its immutable source snapshot.

## Review Focus

- EGL enumeration order differs from physical/CUDA order: match UUID/PCI.
- Ray imports simulator before actor initialization: set backend in runtime_env.
- MPS model actors use UUID CUDA masks: never inherit render device selectors.
- Context silently chooses another card/software renderer: validate active display.
- Concurrent Real/Eval startup and hard resets: verify four Ray workers and clean shutdown.

### Task 1: Explicit rendering configuration and device identity

Files: create `weaver_run/engine/environment_rendering.py` and `test_environment_rendering_cpu.py`.

Interfaces: backend configuration; pure `select_render_bindings(bindings, gpu_inventory, egl_devices)`; discovery; `render_worker_options(render_bindings, replica, worker_label, environ=None)`; environment preparation and active-renderer verification.

- [x] Write failing tests for reordered EGL devices, missing/ambiguous mapping, hidden CUDA, per-worker routing, stale selector removal and software/wrong-card rejection.
- [x] Run CPU tests and observe expected missing-feature failures.
- [x] Implement device discovery, runtime env, data-disk cache and renderer receipts.
- [x] Run the tests successfully.

### Task 2: Training wiring and managed default

Files: `weaver_run/engine/train.py`, `weaver_run/recipe/env.json`, launcher/check path if required; startup contract tests and operational documentation.

- [x] Extend actual-main startup tests to verify Real/Eval placement and Imagine exclusion.
- [x] Configure imports early; discover render mappings; apply actor options; verify environment contexts.
- [x] Set the managed recipe to EGL; retain explicit OSMesa rollback.
- [x] Run relevant CPU regression suites and compile changed sources.

### Task 3: Server validation and publication

Files: bounded validation tools/results under `analysis/egl_integration_20261113`; acceptance notes in repository docs.

- [x] Run four fixed-action Ray environment workers with CUDA hidden; check NVIDIA vendor, target PCI/UUID, reset/step images and teardown.
- [x] Obtain an independent diff review and resolve material findings.
- [ ] Commit, push and verify the same GitHub branch; fast-forward the clean server development checkout.
- [ ] Report separately the deployed default and the still-running process backend.

## Execution record

- Initial local/server HEAD: 04f36a3; both development trees clean. Active run remains fb59cd2.
- Ruling: execute the already-authorized integration directly; no new worktree or repeated design approval, per user branch instructions and “接入EGL”.
- Local broad acceptance first stopped before tests because WEAVER_TEST_UPSTREAM was not configured. Server acceptance with the bundled vendor path passed all 451 tests; no product-code change was made for that environment issue.
- Renderer tests observed six missing-module failures, then passed. Actual-main routing test observed missing EGL actor options before wiring, then passed; 28 focused tests pass.
- Independent review found no material defect. Four-worker Ray smoke passed in 47.75 seconds, with consistent physical and image hashes across GPUs and all environment cleanup calls successful.
- A real Ray model actor imported train under the production MPS UUID mask successfully; no EGL selector or GL context leaked into that actor. Torch CUDA initialized on the model side, while environment workers retained CUDA-hidden operation.
