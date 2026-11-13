# EGL rendering for Real and online Eval

The managed recipe selects `ACCERL_ENV_RENDER_BACKEND=egl`. Physics remains CPU MuJoCo. This change does not alter the 150-action Real horizon, 220-action Eval horizon, Real1/Imagine4/Eval3 counts, rewards, sampling or training schedule.

## Placement and startup

Real and online Eval render on the physical card serving their Policy inference requests. With the current allocation this is Real0 on GPU4 and Eval0/1/2 on GPU5/6/7. Other four-card allocations are resolved from the existing Policy replica bindings.

The driver matches NVIDIA UUID/PCI identity to EGL's DRM device enumeration. EGL indices are not assumed to equal physical GPU or CUDA indices. Only environment workers receive `MUJOCO_EGL_DEVICE_ID`; their `CUDA_VISIBLE_DEVICES` stays empty, so TensorFlow preprocessing and simulation do not acquire tensor-compute GPUs. Model workers keep their existing MPS UUID bindings without an EGL selector.

Every environment worker checks its active GL vendor and EGL display's physical PCI identity after construction. Missing hardware rendering or wrong-card placement fails startup, rather than silently falling back to software. The driver awaits those receipts before proceeding with distributed model-group initialization.

Receipts are saved under `<ACCERL_EXPERIMENT_ROOT>/rendering/real-0.json` and `eval-0.json` through `eval-2.json`, and recorded in telemetry as `environment_rendering`. Shader caches are under the same directory's `cache/` subdirectory. Models remain resident when environments are idle; no workload throttling is introduced here.

## Rollback and lifecycle

To select software rendering for a later run, change `ACCERL_ENV_RENDER_BACKEND`, `MUJOCO_GL` and `PYOPENGL_PLATFORM` in `weaver_run/recipe/env.json` to `osmesa`, then commit/push/synchronize that recipe through the normal managed entry. Shell environment overrides are not authoritative: managed launchers reload the committed recipe.

The portable CPU admission check explicitly continues to use OSMesa. The separate saved-checkpoint evaluation utility is unchanged; it is not the online Eval worker path.

Changing source or environment variables cannot change existing OpenGL contexts. The running job retains its pinned source. A newly submitted managed training uses the new recipe; do not claim the old process has switched. Existing checkpoints are weights-only, so restarting is not an exact optimizer/Replay continuation.

## Validation (2026-11-13)

- Eight rendering configuration/placement/startup tests, plus twenty existing startup/Eval tests, pass locally.
- The server CPU acceptance suite passes 451 tests with zero failures, errors or skips; CUDA initialization and real Ray imports are guarded out of that suite.
- A separate private Ray instance ran four environment workers on GPUs4–7: three fixed initial states per worker, eight fixed actions per state (96 actions and 12 resets total). All used NVIDIA EGL on their assigned physical cards; Torch CUDA remained uninitialized and TensorFlow exposed no GPU to environment workers.
- For each fixed initial state, physics and policy-image hashes matched across all four cards, including hard resets. This checks placement consistency across EGL workers, not pixel identity with OSMesa or unchanged learned-policy performance.
- A separate Ray model actor imported the actual training module with its existing MPS UUID mask. Import succeeded, the numeric EGL selector was absent, and no GL context was created. That model-side import initialized Torch CUDA, as permitted for model processes; the four environment processes above kept CUDA uninitialized.
- The bounded validation creates no Policy/World model weights and does not start or restart training. Earlier standalone CPU/EGL timing evidence is separate from this integration acceptance; full training throughput remains to be measured in an EGL training run.

Experiment artifacts are in the workspace's `analysis/egl_integration_20261113/` and the server data-disk directory recorded by its `stage.json`.
