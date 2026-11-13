# Compact TensorBoard metrics

`CaptureSummaryWriter.TENSORBOARD_TAGS` contains 13 common tags. Two explicit
backend-specific loss slots keep each run's allowlist at 15 possible curves.
A curve appears only when its producer supplies a measurement:

| Purpose | Tags |
| --- | --- |
| Evaluation | `Eval/_Global/Average_Return`, `Eval/_Global/Average_Episode_Length` |
| Real rollout | `Rollout/_Global/Average_Return`, `Rollout/_Global/Average_Episode_Length` |
| Policy loss | `Loss/Policy`, `Loss/Value` |
| World loss slots | Native: `Loss/World_Flow`, `Loss/World_RM`; legacy: `Loss/CtrlWorld`, `Loss/Reward_Model` |
| Policy distribution | `Metrics/Entropy`, `Metrics/KL_Divergence` (unweighted) |
| Phase time | `Performance/policy_train_time`, `Performance/ctrl_train_time`, `Performance/sync_time` |
| Replay inventory | `System/Replay_Buffer_Size_Total`, `System/WM_Replay_Buffer_Size_Total` |

All supplied scalars still enter SQLite before TensorBoard filtering, with their
original tag, value, step and optional wall time. Configuration, milestones,
per-task duplicates, learning rates, detailed diagnostics and unknown tags stay
in telemetry. `Loss/Total`, weighted `Loss/KL` and weighted `Loss/Entropy` are
audit-only; raw entropy remains visible even when `ent_coef=0`. Anomaly events
and native `world_backend_losses` events are unchanged. No rule drops arbitrary
flat signals or single points: an allowed zero throughout warmup is retained.

World training scalars and timings are recorded on the existing successful
`trained_ctrl` / `trained_reward` flags, outside the periodic Policy log guard.
Unique/fresh sample diagnostics and Ctrl gradients require a Ctrl update;
Reward diagnostics require a Reward update. Shared WM preparation/wait times
require either update. Missing fields and ordinary Policy-only steps produce
no World observations, rather than zeros or stale measurements. Real measured
zeros remain valid. Buffer inventory and actual held-out evaluations keep their
existing sampling schedules; the latter remain in telemetry.

Native World flow is the microstep mean of the explicit `flow/Total Loss`
audit-log values. The engine reuses their existing scalar conversions and returns
the mean through the existing bridge response. Native RM is the microstep mean
of `native_half_mse_loss` (dense targets), or `native_half_brier_loss` (legacy
native RM). `combine_wm_metrics` then takes the equal rank mean of both losses.
RM already uses the global cohort denominator: uneven local masks must **not**
be weighted again, nor may rank 0 stand in for the other ranks. Missing values
on any rank yield no component observation. Actual measured zeros are retained.

For the current native objective (RM coefficient 1, critic/decoder/temporal
terms disabled), aggregate CtrlWorld equals flow + RM up to floating-point
rounding. The aggregate remains in SQLite but is excluded from native
TensorBoard. Native World with an external Reward trainer uses World_Flow plus
Reward_Model; nonnative backends retain CtrlWorld/Reward_Model as appropriate.
No extra model forward, GPU readback loop, collective or RPC is introduced.
Existing TensorBoard event files are not rewritten.

Focused writer verification (from `weaver_run/engine`; no training module,
Torch or Ray imports):

```sh
python -m unittest test_metric_logging_cpu -v
```

`test_fresh_native_reward_cpu` additionally exercises the actual native objective
on CPU Torch: four ranks, two microsteps, uneven masks including an entirely
masked rank 0, exact RM normalization, loss decomposition, and one flow scalar
conversion per microstep. The engine/bridge tests cover streaming and updates.

## Candidate CPU validation and logs

Use explicit module lists rather than repository-wide discovery, which also
collects upstream tests with unrelated dependencies. Run the following from the
repository root in PowerShell with CPU Torch and NumPy installed. Separate
processes keep startup-test import stubs out of the native arithmetic tests.
The bundled upstream path supplies the pinned model AST; it downloads no model.

```powershell
$repo = (Get-Location).Path
$env:WEAVER_TEST_UPSTREAM = Join-Path $repo 'weaver_run/vendor/weaver'
$logs = Join-Path $env:TEMP ('weaver-candidate-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
New-Item -ItemType Directory -Path $logs | Out-Null
Push-Location (Join-Path $repo 'weaver_run/engine')
try {
    python -m unittest test_metric_logging_cpu test_weaver_online_bridge_cpu test_fresh_native_reward_cpu test_weaver_online_engine_cpu test_weaver_integrated_reward_cpu -v *> (Join-Path $logs 'metrics-native.log')
    if ($LASTEXITCODE -ne 0) { throw "Metrics/native tests failed; see $logs" }
    python -m unittest test_main_startup_cpu test_integrated_main_cpu -v *> (Join-Path $logs 'startup.log')
    if ($LASTEXITCODE -ne 0) { throw "Startup tests failed; see $logs" }
} finally { Pop-Location }
python -m unittest weaver_run.diagnostics.test_rm_report weaver_run.diagnostics.test_rm_probe weaver_run.diagnostics.test_learning_metrics weaver_run.diagnostics.test_panel -v *> (Join-Path $logs 'diagnostics.log')
if ($LASTEXITCODE -ne 0) { throw "Diagnostics tests failed; see $logs" }
Write-Output "Candidate CPU logs: $logs"
```

For additional topology regression, run `python -m unittest
test_backend_topology_cpu -v` from `weaver_run/engine` and retain its output
separately. Its existing
`CapturePayloadTest.test_backend_targets_are_reservations_only_and_defaults_keep_tuple5`
fixture lacks `Policy.sample`, so that test fails before exercising metrics.
Do not count this as a green suite or silently skip it. Startup fixtures may
also warn about reloading NumPy. CPU checks do not constitute GPU/queued-job
acceptance and do not start or modify any live training process.
