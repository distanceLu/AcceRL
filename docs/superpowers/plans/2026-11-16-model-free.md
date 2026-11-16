# Model-free switch implementation

## Scope and design

User authorizes direct edits on `codex/weaver-original95-runnable`; no new training or interruption of the live run. Preserve the model-based default. A managed `--model-free` switch selects a Policy-only service plan, four real collectors and four Policy replicas on the existing four-card topology. Online Eval defaults to zero in this mode; explicit four maps one worker to each replica. CPU physics and EGL rendering remain unchanged.

The Policy keeps the current two-camera input, proprio setting, pretrained + distilled checkpoint load order, GIPO, optimizer settings, action horizon and initial-state budget. Model-free means real environment rewards only: no World/Reward model owners, no imagination workers, no WM replay/window projection, no World checkpoint dependency. The Policy Value critic and current-value GAE refresh stay active. Fresh replay gets an explicit real-only mode; mixed replay retains its existing invariants.

A metadata-only real collection coordinator owns global initial-state/budget leases. It does not retain images, World windows, or imagination seeds. Policy checkpoints, publication, telemetry and fixed checkpoint evaluation remain available. Cost scheduling and imagination inventory refill are disabled in the first model-free recipe; this change does not claim to implement adaptive sharing.

## Steps

1. CPU configuration, real-only replay, and metadata collection coordinator with failing behavioral tests first.
2. Integrate switch into actual train parser/startup, collector ingestion, first fill, update/publication, metrics and checkpoint paths. Test actual startup AST with inert model boundaries.
3. Expose managed site/portable launch configuration; skip World assets and World-only admission in model-free mode. Preserve explicit run/git metadata and no-Eval behavior.
4. Run CPU regressions, inspect two-camera checkpoint compatibility, independent code review and fixes.
5. Commit reviewed changes and report exact validation and remaining GPU smoke-test limitation. Do not launch a run.

## Acceptance

- Mode off retains prior recipe and tests.
- Mode on creates exactly four Policy trainers and four Policy inference owners, zero World/Reward owners, zero imagination workers or WM replay actors.
- Four real workers bind renderer and Policy to replicas 0..3. Eval0 allocates none; Eval4 maps 0..3.
- A real-only batch reaches readiness, refreshes GAE and commits sample usage without imagined data or World versions.
- Stop/checkpoint paths work for Policy-only kinds; no World assets are required.
- Task0, two cameras, action horizon and existing initial Policy weights remain explicit in artifacts.

## Validation evidence

- 2026-11-16: the existing `validate_cost_scheduler_candidate.modules_to_run()` suite plus the four new model-free startup/foundation/resources/completion modules: **502 tests passed**, CUDA not initialized. Pinned upstream supplied via `WEAVER_TEST_UPSTREAM=weaver_run/vendor/weaver`.
- Managed/portable launch tests: **12 passed**. Queue admission regressions: **16 passed**. Fixed-evaluation package regressions: **112 passed** (implementer run).
- Independent review reproduced and resolved a serial-role configuration failure before actor creation and a missing Policy-only `monolith_steps` final-audit field.
- A separate complete-main CPU test passed six Policy epochs, including step five, per-epoch logging, version publication, periodic save and final stop/save. Final independent review found no remaining P1/P2 issues.
- The broader historical 107-test selection had 20 errors also reproduced unchanged from an isolated HEAD archive: obsolete trajectory/JIT options, an AST fixture missing a pre-existing helper, and a fake Policy missing `sample`. Those unrelated fixtures were not rewritten.
- No training launch, interruption, GPU model load or production benchmark was performed. GPU startup/first-update and throughput validation remain future experiment work.
