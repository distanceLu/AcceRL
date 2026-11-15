# EGL evaluation and critic-refresh Eval admission

The user explicitly requested direct source edits on `codex/weaver-original95-runnable` and a GitHub push. No new training is requested. The existing plan and successful P637 renderer experiment supply the design; proceed without another approval round.

## Intent

Fixed-checkpoint GPU evaluation must use NVIDIA EGL by default, matching Real and online Eval. Preserve checkpoint/case/seed/action/preprocessing semantics. Map each evaluator's EGL device by its allocated GPU UUID/PCI and verify the active renderer. Handle robosuite's numeric-EGL-versus-CUDA-UUID import guard without modifying installed libraries or changing the model GPU. Preserve an explicit OSMesa rollback path for diagnostics and CPU-only admission checks.

During current-critic refresh demand, stop admitting new online Eval episodes. The first and all later episodes wait before reset. Already admitted episodes run normally; refresh never waits for Eval to drain. Release admission when pending refresh reaches zero. Do not pause Real, Imagine, collectives, publication, or frozen evaluation.

## Chosen implementation boundaries

Reuse the current scheduler broker's `refresh_remaining`, including all current-critic requests already represented by that counter. Add a configurable online-Eval gate, enabled in the managed recipe, with a 360-second maximum continuous deferral safeguard and observable waits/admissions/safeguard triggers. Keep the generic gate disabled when its configuration is absent for compatibility. Cancellation, broker failure and user stop must not strand waiters. A permit is never revoked after admission.

Alternatives already considered: per-kernel preemption, stopping in-flight episodes, PID, and changes to Real supply are unnecessary for this experiment. A fixed sleep would not track actual refresh completion. Backend-only configuration without verified EGL placement previously failed on mixed CUDA UUID/EGL ordinal checks, so the evaluator needs the validated bootstrap helper.

## Validation

Test reset admission, normal release, nested refresh demand, continuous-refresh safeguard, cancellation/failure/stop behavior, and observability. Test EGL bootstrap environment restoration on success/failure, exact UUID/PCI selection and software/wrong-device rejection. Preserve portable CPU checks and immutable evaluation case semantics. Run focused and existing CPU acceptance suites, an independent code review, and a bounded server runtime smoke without training. Commit, push and verify the exact remote branch revision; synchronize the idle server development checkout if clean.
