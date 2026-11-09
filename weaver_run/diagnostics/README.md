# Frozen-input learning diagnostics

This package contains offline measurement primitives. The training entry point
does not import it. It does not change sampling, rewards, GAE, EMA, weights,
optimizer state, or GPU scheduling, and it is not an end-to-end GPU runner.

Run the CPU mathematical checks from the repository root:

```sh
python -m unittest weaver_run.diagnostics.test_learning_metrics -v
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

CPU test success validates mathematical contracts only. No OpenVLA, native
World/RM, CUDA, multi-rank gradient, or live-training measurement is claimed.
