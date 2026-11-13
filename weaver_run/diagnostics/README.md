# Frozen-input learning diagnostics

This package contains offline measurement primitives and a finite diagnostic
runner. Training does not import the measurement code. It never changes the
running training job, its samples, rewards, weights, or optimizer state.

On the reference server, submit **behind** the current manual training:

```sh
./run.sh diagnose --id learning-diagnostic-20261109-a
./run.sh diagnose-status --id learning-diagnostic-20261109-a
```

Submission is idempotent. It verifies the published maintenance-branch commit,
checkpoint identity, and saved four-rank EMA on CPU. The existing GPU4–7 queue
controls when it starts. The diagnostic additionally checks its own PID/UID/
birth, exclusive queue ownership, and existing idle MPS service before loading
models. It does not request a stop, change MPS, or place a deadline on training.
The diagnostic itself has a four-hour limit and preserves partial failures.

The fixed protocol compares original P451/W90 against restored P457/W91 on
16 predetermined task-0 initial states. It collects fresh stochastic episodes
and N+1 states, cross-scores both World/RM models on both real panels with the
same measured actions and two fixed noise seeds, and reports 8/16/32-action
errors separately. RM on actual observations is distinct from RM on generated
observations. State0 has explicit repeated-reset padding, not invented history.

For each frozen Policy it samples up to 144 real chunks and generates up to
108 complete four-chunk imagined trajectories. If real coverage is smaller,
the diagnostic reports a smaller batch while keeping the 1:3 ratio. It saves
inputs, behavior logits, action tokens, masks, full GAE suffixes, fixed EMA,
selection, and source gradient vectors. Gradients use the saved micro size of
8, with a separate frozen forward prepass for values and no optimizer step.
The prepass/gradient-forward value difference is reported. First-micro source
additivity is checked against the direct joint objective (5% relative L2
diagnostic BF16 rejection threshold, not a numerical equivalence claim).

Actor, critic, and regularizer geometry is reported for policy and value
parameters separately: the actual model detaches features into its value head.
These fresh on-policy gradients cannot reconstruct historical stale-behavior
gradients or prove the cause of a learning-curve difference. The finite panel
is not a replacement for a full success-rate evaluation.

Run the CPU mathematical checks from the repository root:

```sh
python -m unittest weaver_run.diagnostics.test_learning_metrics weaver_run.diagnostics.test_panel -v
```

`learning_metrics.py` provides:

- `paired_wm_metrics`: aligned predictions and true subsequent observations
  under **the same recorded real actions**. Report 8/16/32-action horizons
  separately; physical EEF XYZ error is distinct from image error and from
  flow-matching training loss. Image/state errors alone do not prove task fidelity.
- `paired_rm_metrics`: predictions paired to observed episode outcomes using
  the actual discounted-success target, including a censor mask. Report both
  frame-weighted and equal-episode MSE. RM scores are not calibrated binary
  success probabilities. Evaluate RM on true observations separately from RM
  on generated observations; the latter combines WM and RM error.
- `source_losses`: exact source split of the restored JIT micro objective,
  preserving masks, original token denominators and chunk-weighted value loss.
  Actor, critic and regularizer remain separate. It does **not** recompute GAE
  or advantage normalization. Supply the real fixed values from the replay panel.
- `loss_gradients`: autograd without optimizer updates or writes to `.grad`.
  Include identical named parameter coordinates, including zero contributions.
- `gradient_geometry`: full-vector norms, cosine, angle, and source projection
  on the joint gradient. Sum all micros/ranks first using the original GAS/DP
  scaling. Never average per-micro cosines. A zero vector has no defined angle.

Required integration data: frozen checkpoint identity, exact input tensors,
real/imagined IDs and parents, action tokens, behavior logits, action masks,
complete GAE suffixes and bootstrap states, reward/version provenance, fixed
normalization statistics, and the original micro/rank assignment. A historical
ledger containing only IDs and scalar advantages cannot reconstruct these.

The currently running legacy October-30 trainer has a different normalization
path. Its gradient adapter must be checked against that path before use; this
JIT-specific adapter must not silently be treated as its executed objective.

Use fresh episodes collected after checkpoint freezing for independent error
measurement. Fix initial states and action/noise seeds, cross-score both World
models on both Policy panels, and group uncertainty by episode, not by frame.
Same training-task initial states are not held-out tasks. Fresh panels measure
the frozen models, not an exact reconstruction of an old asynchronous update.

CPU test success validates mathematical and orchestration contracts only.
GPU results exist only when the queued diagnostic has actually executed and
written a complete result; queued status is not a passed OpenVLA/CUDA check.

## Frozen RM state-discrimination report (CPU only)

On the reference server, queue the finite GPU collection/scoring job through the
same entrypoint:

```sh
./run.sh diagnose-rm --id rm-state-20261113-a
./run.sh diagnose-rm-status --id rm-state-20261113-a
```

It waits behind active training and never requests its stop. It cross-scores
orange150 P594/W118 and inventory-refill P275/W55 on 32 newly collected real
episodes (two frozen policies, the same 16 initial states). The four-hour budget
applies to diagnostic execution, not queue waiting or the active training.
Checkpoint admission is CPU-only; GPU models require exclusive queue ownership.
The live training runtime is untouched. The diagnostic runs from the published
development commit and fails closed if that checkout changes before execution;
keep it at the submitted commit while the job is pending. It never updates model
weights, produces imagined data, or inserts samples into training Replay.
Completion requires `result.json` with `complete=true`; queued status does not
mean any RM prediction has yet been measured.

`rm_report.rm_state_report` consumes already-scored independent real samples;
it imports NumPy and the standard library only, with no model loading or I/O.
Cross-score every frozen RM on the **same** `(episode_id, frame_index)` panel.
Each row supplies `model`, `episode_id`, `initial_state_id`, `frame_index`,
`episode_length`, `success`, `target`, and `prediction`; optional `policy_panel`
is retained. Scores and discounted targets must be finite numbers in `[0,1]`.

```python
from weaver_run.diagnostics.rm_report import rm_state_report

STATES = [0, 3, 6, 8, 11, 13, 18, 23, 27, 29, 31, 37, 39, 42, 47, 49]
report = rm_state_report(
    rows,
    calibration_initial_states=STATES[::2],
    test_initial_states=STATES[1::2],
)
```

The complete signature is
`rm_state_report(rows, calibration_ids=None, test_ids=None, *, calibration_initial_states=None, test_initial_states=None, valid=None, include_oracle=False)`.
Choose either the two episode-ID arguments or the two initial-state keywords.
Both modes require exhaustive, disjoint partitions and prohibit any initial
state from crossing the split. For the two-policy protocol, episode IDs are
`<policy>:state<id>`; both policy episodes for a state stay together. The
`split` manifest records the state-to-episode mapping and sample support.

For real 150-action-prefix evaluation of complete collected episodes, use
`episode_length=min(N,150)`, frames `1..min(N,150)`, and
`success=full_episode_success and N<=150`. Successful targets are
`(.99**(1/8))**(episode_length-frame_index)`; other observed-prefix targets are
zero. The report accepts frame zero as well, following the repository's
N-action/N+1-state convention. It does not infer these outcomes from scores.
Fresh collection after checkpoint freezing and exclusion from training Replay
remain the caller's responsibility; rows alone cannot prove independence.

Each model has four headline values for each of `frame_weighted` and
`equal_episode`: MSE, signed bias (prediction minus target), calibration-constant
MSE, and `skill_vs_constant = 1 - MSE/constant_MSE`. Each constant is fitted
**only** on calibration targets under its own weighting. Zero reference error
makes skill undefined (`null`). Groups never refit their reference constants.
The compact `groups` table covers success/failure, early/middle/late, and their
crosses; empty groups have `no_support`, counts, and null metrics. Time groups
are right-closed thirds of the evaluated horizon (for 150 actions: 1–50,
51–100, 101–150), using actual frame indices, not the rank of sampled frames.
Equal-episode weights give each represented episode equal total weight within
each group. `prediction_distribution` provides population mean/std and weighted
empirical-CDF p05/p25/p50/p75/p95, without quantile interpolation.

`success_failure` uses episode-mean prediction scores for descriptive AUC
(half credit for ties) and a success-minus-failure score gap. A missing class is
`no_support`; fewer than two initial states in either class is `limited_support`.
RM scores are discounted outcomes, **not calibrated binary probabilities**.
`progression` removes each successful episode's prediction and target means,
then compares centered MSE against a flat-prediction reference. It needs at
least two valid frames per trace, and flat targets have no progression skill.
No Pearson significance or frame-independent uncertainty is claimed; episodes
sharing an initial state are also related. Optional `held_out_oracle` is labeled
non-deployable and is only a lower bound for the **constant predictor class**.

The optional `valid` argument is a boolean vector aligned with input rows.
All models must agree on keys, metadata, targets, and masks; duplicates or
episodes without any valid support fail explicitly. `raw_rows` retains the
normalized input schema, policy panel, split, and validity; masked nonfinite
numbers become JSON `null`. The result supports `json.dumps(..., allow_nan=False)`.
No TensorBoard metrics are emitted.

```sh
python -m unittest weaver_run.diagnostics.test_rm_report -v
```
