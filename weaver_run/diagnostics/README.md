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
