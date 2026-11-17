# Original GIPO speed baseline

This baseline executes `rl/ds_libero_ppo_discrete.py`, the actual trainer called
by `run_ppo_training_example.sh`. It does not invoke the newer WEAVER model-free
training engine. The teacher's requested changes are Eval0 and GIPO.

The user selected the following comparison, then replaced the initial two-hour
limit with a success target:

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
| Stop target | First >=99% over a complete rolling window of 1000 real episodes; save at the next completed update boundary |
| Milestones | Latch first >=98% and >=99% as episodes are recorded, with counts and elapsed times |
| Time limit | Disabled; two hours is no longer a stopping condition |
| Final save | Inference weights, once at the completed update/publication boundary |

The user explicitly selected task/checkpoint/four-GPU alignment and subsequently
requested learning to 98/99% instead of stopping after two hours.
The original shell example uses a different configuration: it uses
two GPUs, object tasks, another checkpoint, proprio input, and global batch252.

## Submission and isolation

On the reference server, the authorized entry is:

```bash
/mnt/data/lcx1/dong/WEAVER/run.sh legacy-baseline \
  --id legacy-gipo-baseline-20261117-a --training-seconds 0 \
  --target-success-rate 0.99 --target-success-window 1000
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

Report both end-to-end wall time from driver launch and time since updates began.
Each completed episode is recorded by the training statistics actor; it latches
the first >=98% and >=99% windows, including the episode-observation timestamp.
A window smaller than 1000 cannot trigger a target. These latches survive a later
decline in the rolling window. At the next completed training/publication boundary,
the driver records the milestones and first >=99% triggers save and stop with
`success_target_reached`. Milestone time and checkpoint-boundary time are recorded
separately; a boundary step does not identify the policy that generated every
episode. This is the training actor's episode-arrival order, not a frozen
checkpoint's success probability. The same-task window may contain trajectories
from several policy versions, matching the earlier training-curve comparison.

The old 30000-update cap and explicit stop requests still apply, and their
completion reasons must not be relabeled as target achievement. Startup,
stalled-update and final-save fault bounds remain; a healthy run has no two-hour
time limit. Saved weights do not provide exact optimizer, Replay or RNG continuation.

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
- Eval0 without periodic fixed validation in this legacy baseline.

Faster updates alone do not establish faster learning or equal data efficiency.
Use produced data, consumed data, sample reuse, update timing and reward together
when reporting the eventual result. A controlled placement-only comparison would
require a separate experiment with the other mechanisms matched.
