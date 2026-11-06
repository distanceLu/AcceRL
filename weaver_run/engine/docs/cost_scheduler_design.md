# Cost-accounted cooperative scheduler candidate

This candidate extends the full FRESH learning rules. It has not been deployed
or run against production CUDA/Ray/DDP. CPU validation is a protocol and method
integration check, not evidence of hardware speedup or numerical equivalence.

## Accounting contract

Work is charged in reference-device execution seconds, not request count. At
dispatch, reserve estimated cost divided by the service weight. At completion,
correct the reservation by `(observed-estimated)/dispatch_weight`. The original
dispatch weight is retained even if the controller changes weights meanwhile.
The first implementation uses execution **wall-time proxies**, excluding
admission queue delay. These proxies include owning-backend input preparation
and host/device synchronization; they are not measured SM seconds or FLOPs.
Four-rank training is charged once using the sum of the rank execution intervals.
Temporary resource saturation does not reactivate an idle service: completion
refunds remain effective while its queue is continuously backlogged.

The four macro services are Policy training, World/RM training, Policy inference,
and World inference. Policy inference is subdivided into real actions, imagined
actions/bootstrap, critic refresh, and Eval. Child weights sum to the parent
weight; independent replicas share those accounts without forming a gang.
Logical admission lanes limit submitted work. They do not change MPS, partition
SMs, preempt CUDA kernels, guarantee a GPU percentage, or require a card drain.
A fully arrived training gang reserves one future lane on each required card
when the training gate is available. Existing work finishes; other lanes remain
usable. This prevents staggered inference backfill from starving the gang.

Initial costs in example profiles are explicit bootstrap priors, based on the
existing FRESH timing scale, not isolated service benchmarks. Shape-keyed EWMA
completion observations replace them. Full per-request histories are bounded;
aggregate cost/progress counters are retained. Runtime work IDs must remain
globally unique throughout the process lifetime. The core also supports strict
monotonic identity namespaces when a producer needs bounded exact replay defense.

## Backend boundaries

The initial allowlist admits a whole optimizer update for Policy ZeRO2 and native
World DDP. Every rank must acknowledge the same sequence, published versions,
source identity, recipe and completed prior communication. All ranks are released
together. A granted group must drain even after a peer failure; new groups stop.
Only each owner's stream is synchronized, not unrelated inference contexts.

DeepSpeed backward/step runs on every Policy micro. Native DDP also performs
distributed finite-loss checks inside `no_sync`. Neither backend is assumed safe
to pause at arbitrary micro boundaries. FSDP and TP adapters are rejected until
their actual communication boundaries have been reviewed and validated.
This coarse initial training quantum can limit latency and fairness; finer
backend stages are a separate admission capability, not a configuration shortcut.

## Policy execution

The same model serves four request classes. Critic jobs are split into bounded
micros (initially at most two observations), ordered results are retained, and
the event loop yields between micros. All observations for a refreshed segment
retain the exact published Policy version. The async publication adapter closes
admission of new refresh jobs, waits for existing pins to drain, then applies
weights. New bootstrap requests bind the newly published version. Refresh does
not sample actions. Real budget-censored endpoints use this same scheduled
interface, including its scheduler-disabled fallback.

Action batches retain original behavior logits, action masks and exact request
IDs. Telemetry metadata is selected by ID after reordering, rather than consumed
from a FIFO belonging to a different request order. Cancellation never preempts
an already submitted execution. Changing action batch order may change assignment
of stochastic RNG draws; bitwise trajectory equivalence is not claimed.

World inference is still single-request in the native bridge. Its generation
and context initialization are cost-accounted under different shape keys;
initialization never counts as a generated Replay chunk. Release/cleanup bypasses
the work budget so exhausted budgets cannot prevent freeing resident handles.
CPU simulation, publication and checkpoint I/O remain separately observable
control work. This candidate does not implement native World batching.
Cancellation retains ownership of outstanding control RPCs until their replies
are reconciled. Unused inference grants are released with zero successful work;
an initialized context whose reply is cancelled releases its resident handle.
Refresh registration cancellation retains its pin until the bounded job drains.

## Macro control and feedback

Replay supplies a read-only inventory of disjoint sample-ID buckets. Next-batch
deficits use distinct eligible chunks, while complete-cycle deficits use remaining
exposure credits and project version expiry/use caps. Reserved and completed uses
are separate. Expired samples and another source cannot fill a missing quota.

The forecast estimates an end-to-end progress-capacity envelope to the next 5:1 boundary, including
current/future critic observations, real/imagined production, Policy/World updates
and publication dependencies. It uses responses measured at the **complete weight
vector and ordinary/joint context**, never linear extrapolation from weight alone.
These rates share complete update wall-clock windows. Their implied durations
are combined by maximum, not added as independent phase costs. For example,
1353 critic observations and one update completed in a common 146-second window
imply a 146-second envelope, not 292 seconds. This is not a phase-level critical
path simulation, an isolated execution-time estimate, or a speedup guarantee.
Progress windows count admitted chunks and successful completed updates, not merely
RPC attempts. Initial action demand conversion is explicit: one Policy observation
per real chunk and 1.25 per imagined chunk for the present four-chunk branch plus
one bootstrap. Unknown cost/response remains unknown.
Before a cohort is selected, current critic work uses the observed per-update
estimate or remains unknown; an empty refresh queue is not proof of zero demand.
Replay sends monotonic, versioned inventory snapshots after reserve, commit,
cancel and publication. Reserved uses cannot be reused in the forecast.

Candidate search is bounded. Positive weight bounds, L1 step limits, cooldown,
improvement hysteresis and the switching penalty constrain changes. Measured-only
mode holds when candidate responses are missing; adaptive mode permits explicit
bounded probes after stable baseline observations. An unmeasured probe has no
predicted speedup. Phase changes emit advisory plans; actual weight changes
occur at update_cycle only after closing the previous measurement and reading
authoritative Replay inventory. A probe needs the configured number of accepted
ordinary and joint response samples and a finite baseline forecast. Its hold
also prevents early measured switches. Publication edges cannot discard the
second joint response or change weights mid-window. Probe telemetry retains
the measured baseline while leaving the unmeasured candidate prediction null.

Learning constraints remain global576 with real144/imag432, usecap4, Policy age16,
World/RM age3, current-critic ordered GAE, original generation-time rewards, and
Policy:World/RM5:1. The controller cannot change these rules or stop the run.

## Validation and deployment boundary

`00_disabled` is the FRESH control; `01_fixed` enables cost admission and the four
Policy classes; `02_measured` adds measured-only macro decisions; `03_adaptive`
also permits bounded response probes. Use a new experiment root. The launcher
does not submit a queue job or stop an existing run.

CPU serializer export can be checked separately with
`python validate_scheduler_serialization.py --cloudpickle-root <existing-tool-path>`.
The local development check uses an isolated, pinned cloudpickle3.1.2 source copy
with its license and SHA recorded under `analysis/scheduler_dev_tools`; it is not
installed into the training environments or a runtime dependency of this scheduler.
It tests dynamic function/class export, not an actual Ray actor deployment.

Before production, validate actual Ray async calls, all four GPU ranks, collective
drain on failure, numerical behavior, version/ACK/release receipts, memory peaks,
and complete 5:1-cycle throughput. Compare tail latency, useful admitted chunks,
freshness and successful updates together. No speedup follows from CPU tests.

## Related implementations

- Linux CFS: cost/weight-normalized virtual runtime and idle-service placement:
  https://kernel.org/doc/html/latest/scheduler/sched-design-CFS.html
- Gavel, OSDI2020: measured effective throughput and allocation feedback;
  https://www.usenix.org/conference/osdi20/presentation/narayanan-deepak
  and https://github.com/stanford-futuredata/gavel
- Ray actor concurrency documentation:
  https://docs.ray.io/en/latest/ray-core/actors/async_api.html

These provide established components and references. The current code is a new
integration for this workload; it is not claimed to introduce a new fair-queuing
algorithm or to reproduce Gavel's cluster scheduler or reported results.
