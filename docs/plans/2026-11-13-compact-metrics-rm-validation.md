# Compact metrics and independent RM validation

User-approved scope: reduce TensorBoard noise, log World metrics only for real World updates, check RM state discrimination on independent real samples, and assess optimization using fixed-weight outcomes, time to target and complete-cycle throughput. Do not add alternate horizontal axes or change learning.

Approach: explicit compact TensorBoard selection with raw SQLite evidence retained; do not heuristically delete all temporarily flat safety signals. Use the existing frozen-policy collector and native RM encoder, with a separate finite diagnostic through run.sh. Never score historical scalar loss as an independent prediction measurement.

- [x] Test and implement World-update gating and compact TensorBoard selection; retain audit events and genuine low-frequency failures.
- [x] Test a CPU RM report with calibration/test initial-state separation, equal sample pairing across checkpoints, prediction distributions, success/failure and temporal groups, and constant-reference errors.
- [x] Add a finite RM-only diagnostic using orange150 P594/W118 and refill P275/W55, two independently collected frozen-policy panels on the existing 16 fixed initial states. Cross-score actual observations; do not generate imagined frames or update weights. Fit constants on eight calibration initial states and report on eight disjoint test initial states, grouping both policies by initial state.
- [x] Reuse 220-action collector; RM targets use observed first-150-action outcome and gamma_action=0.99**(1/8). Policy evaluation keeps220. Missing/rare failure cases must remain explicit limitations.
- [x] Add run.sh diagnose-rm / diagnose-rm-status, idempotent queue submission and CPU checkpoint admission. GPU work starts only with exclusive queue ownership. User chose to keep current training running and wait for resources. No new source copy; queued diagnostic checks the published development commit before execution.
- [ ] Verify, review, commit and push the existing branch; sync development without changing the active Git runtime. Record task outcomes separately from training fit and preserve unavailable comparisons rather than inventing them.

Tests: no ordinary-step World zero-padding; raw non-TensorBoard metrics survive; constant/perfect/reversed RM predictions; uneven episode length; disjoint initial states; duplicate/missing cross-model pairs;150-boundary target alignment and censor handling; queue idempotence and exact owner checks. GPU acceptance is separate from CPU tests.

No optimizer, GAE, sampling, seeds, quotas or runtime parameter changes. Existing TensorBoard event files are historical evidence; current-run display changes need a separate CPU presentation path, never a hot patch of the training process.
