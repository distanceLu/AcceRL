# Original GIPO speed baseline

This baseline executes `rl/ds_libero_ppo_discrete.py`, the actual trainer called
by `run_ppo_training_example.sh`. It does not invoke the newer WEAVER model-free
training engine. The teacher's requested changes are Eval0 and GIPO.

For the first bounded measurement, the user selected:

| Setting | Value |
| --- | --- |
| GPUs | Physical 4–7; three exclusive training ranks, one inference replica |
| Task | LIBERO spatial task 0 |
| Policy input | External and wrist images; no proprio input |
| Initialization | Current spatial pretrained checkpoint plus the same distilled `student_state_dict` |
| Real / imagine / online Eval | 10 / 0 / 0 |
| Training batch | micro12 × GAS16 × three ranks = 576 action chunks |
| Clip mode | GIPO; sigma1 matches the legacy loss and diagnostics |
| Renderer | Original OSMesa software rendering |
| Episode horizon | Original spatial limit of 220 actions |
| Duration | 7200 seconds from entry to the training update loop |
| Final save | Inference weights, once at the completed update/publication boundary |

The user explicitly selected task/checkpoint/four-GPU alignment and a two-hour
duration. The original shell example uses a different configuration: it uses
two GPUs, object tasks, another checkpoint, proprio input, and global batch252.

## Submission and isolation

On the reference server, the authorized entry is:

```bash
/mnt/data/lcx1/dong/WEAVER/run.sh legacy-baseline \
  --id legacy-gipo-baseline-20261117-a --training-seconds 7200
```

The CPU-only `--check-only` path constructs and validates the frozen record
without submitting a production job. A clean, published maintenance-branch
commit is required for a real submission. The baseline runs from an immutable
`WEAVER/.legacy-runtime/<commit>` checkout. The currently active `.git-runtime`
is never switched by this command.

For the requested continuation, the CPU watcher `after_experiment.py` waits for
the exact predecessor queue job, source commit and run ID, including its final
evaluation acceptance and owned-process cleanup. It also checks that the queue
and allocated GPUs are idle, then invokes the official entry once. A failed
predecessor or ambiguous submission is retained as a failure for inspection.

Outputs and checkpoints stay under the experiment directory on `/mnt/data/lcx1`.
Ray uses a recorded short `/mnt/data/lcx1/ray/<hash>` directory so its Unix socket
paths fit Linux's limit; an existing or redirected directory is refused. GPU
arguments and environment use the verified UUIDs for physical cards 4–7, including
under the existing MPS daemon. The frozen record contains the full command,
initialization paths, code commit and baseline differences.

## Measurements and interpretation

`training/driver_metadata.json` records arguments, actual actor placement and
startup context. `training/updates.jsonl` records completed update and weight
publication durations. `training/driver_status.json` records the current phase.
`training/driver_complete.json` is written only after final weights save succeeds;
the outer `queue_receipt.json` records child exit, finite CPU tensor checks,
artifact hashes and
scoped process cleanup before declaring overall success.

Report both end-to-end wall time and update-loop time. The two-hour bound excludes
model loading and initial Replay fill, and permits the already running update,
publication and final save to finish. A stalled or failed operation is a fault,
not a successful duration stop. Saved weights do not provide exact optimizer,
Replay or RNG continuation.

Training samples/s is global batch576 times successful updates per second.
These are consumed **action chunks**, not new simulator actions. Legacy Replay
retains trajectories and can reuse them repeatedly. Its simulator-action and
episode counters reflect completed episodes and exclude reset settling actions
and in-flight episodes; retain that scope when comparing with the current
model-free run's budget counters.

## Differences remaining after input alignment

This is an original-implementation baseline, not an isolated GPU-layout
ablation. In addition to three-trainer/one-inference placement, it retains:

- Ten rollout workers instead of four, distributed across three Replay buffers.
- OSMesa instead of EGL, and a 220-action horizon instead of 150.
- Worker-local sequential initial-state selection instead of the shared shuffled
  allocation queue.
- The original trajectory sampling/reuse and old-value GAE path, without FRESH
  policy-age/use-count admission or the new current-critic refresh stage.
- Original action-chunk handling, learning-rate warmup, and NCCL weight broadcast.
- Eval0 without periodic fixed validation in this short speed measurement.

Faster updates alone do not establish faster learning or equal data efficiency.
Use produced data, consumed data, sample reuse, update timing and reward together
when reporting the eventual result. A controlled placement-only comparison would
require a separate experiment with the other mechanisms matched.
