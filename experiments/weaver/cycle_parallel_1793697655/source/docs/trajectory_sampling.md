# Complete imagined trajectories with trainer JIT GAE

This candidate extends the frozen 247-file JIT parent, SHA
`d35246f411cddb37921c58197032fb10c7c3956ea02874260341ee43a1f647d7`.
The running experiment is unchanged. The new sampling protocol is opt-in:
`FreshConfig.policy_sampling = "imagined_trajectories"`.

## What changed

The admitted production recipe uses four Policy chunks per native imagined
branch (normally 32 actions). Uniformly choose 108 **eligible complete branches**
and keep all four chunks of each branch on one rank. Four ranks each receive
27 branches / 108 imagined chunks and 36 real chunks. Global batch stays 576,
with 144 real / 432 imagined. Rank ownership uses the existing contiguous
coordinator shards; no new cross-rank context RPC is introduced.

Real IDs retain their original uniform eligible-chunk draw. Place large selected
real segment groups first, using the least-filled rank; split a real group only
when no rank has enough remaining real slots. Real targets retain the **entire
original suffix**, including unselected future rewards and critic values.
This first candidate does not sample variable-length complete real episodes or
silently replace full-suffix GAE by short-window GAE.

Every imagined member must pass the original Policy age <=16, World/RM age <=3
and successful uses + reserved uses <4. Reservation adds one lease to every
member. A commit containing part of an imagined branch is rejected before any
mutation. Rank commits consume entire branches; cancellation releases only
uncommitted leases. Context-only/bootstrap states never consume Replay uses.
The new mode rejects non-four-chunk imagined segments rather than silently
discarding an ending or padding it. Startup admits only native horizon-only
four-chunk generation; other imaginary termination recipes need a separate
variable-length packing design.

## Boundaries and targets

Storage preserves N actions/rewards and N+1 actual decision-boundary states.
These are Policy chunk boundaries, not every raw simulator or video frame.
The existing collector stores the observation after the last executed action,
before any reset. No synthetic last-state repetition is permitted.

For a true terminal last chunk, its TD residual is `r - V(s)`: no endpoint
critic forward is performed. For a nonterminal truncated last chunk, it is
`r + gamma * V(actual_next_state) - V(s)`. The final executed chunk remains in
the policy loss. Missing endpoint/flag geometry fails validation. Existing
finite LIBERO task-horizon and early budget-censor semantics are preserved.

The JIT planner now checks mask lengths and requires all four imagined members
on the rank. Configuration, rank source quotas and complete membership are
checked inside the existing all-rank preparation failure vote. The common
rank-state check includes the sampling protocol.

Detached GAE, GIPO/PPO/SAPO objectives, original behavior logits, generation-time
rewards and actual-action masks are unchanged. Value loss remains chunk
weighted. Actor/KL/entropy retain valid-token weights from the sampled shard's
original micro groups through reverse-order planning. Advantage EMA includes
selected chunks only, uses one frozen global mean/variance per accumulation
window, and commits after a successful all-rank optimizer boundary. The mean
drift variance term remains in place.

## Supply and scheduling

Immediate shortage counts complete eligible imagined branches times four.
Thus 480 individually eligible chunks scattered over 160 incomplete branches
do not falsely satisfy the 432-chunk quota.

H2 forecasting performs integral max flow in **branch units**, with a quota of
108 per future update. Each branch's remaining uses are the minimum across its
members, and future age eligibility is the AND across all members. Forecast
results convert back to chunk units for the existing production controller;
new supply is always a multiple of four. Additional telemetry reports
`sampling_unit_chunks`, `eligible_complete_trajectories` and
`new_trajectories_needed`. This remains a feasible capacity envelope, not a
promise about future random draws.

The macro controller receives only complete eligible imagined groups. Project
each group with its oldest member versions and tightest member use budget;
mark the inventory bucket's sampling unit as four. The macro's minimum new
imagined supply rounds up to complete branches. These are conservative capacity
projections, not per-member consumption history. Existing macro cycle capacity
is still a necessary supply bound; it is not an exact future matching solver.
Authoritative production admission, credits, in-flight limits and sample
commit ownership remain unchanged. One global draw grants 432 imagined credits,
not four times that number. Inference refresh debt stays zero for JIT.

Replay selection and forecasting remain CPU metadata operations. JIT GAE and
EMA retain the existing trainer-device tensors. This change adds no per-chunk
remote request or GPU-to-CPU scalar synchronization. Images/context still come
from CPU Replay; it is not a fully GPU-resident pipeline.

## Evidence and limits

Run `python validate_trajectory_candidate.py` for the parent regression suite
plus new trajectory tests. The separate `validate_trajectory_gloo_cpu.py` uses
four real CPU Gloo processes and the actual JIT runtime with a toy optimizer.
It checks two global576 windows, global EMA count576, unequal real context
lengths, no shared imagined branch, one-rank optimizer skip, and collective
rejection of a one-rank fragmented branch. Reports are under `validation/`.
Neither runner validates Ray, ZeRO2, OpenVLA, NCCL, CUDA memory or wall-time gain.

With all 108 imagined branches nonterminal, post-initialization imagined work
is 432 selected states plus 108 bootstrap states =540 state forwards, excluding
activation-checkpoint recomputation. Imagined suffix D/U is one by construction;
real suffixes can still duplicate across ranks. EMA initialization still needs
its extra no-grad prepass.

The new sampler deliberately increases temporal correlation and changes
trajectory eligibility, rank source composition and mask-dependent micro
normalizers. Equal global batch/EMA does **not** establish numerical equivalence
to scattered-chunk sampling or guarantee learning quality. Any future GPU
comparison must measure full 5:1 cycles after warmup, real and imagined new
admissions, shortage, D/U, sample age/uses, peak memory and fixed-checkpoint
paired success rates. The current long run has not been restarted or modified.

Candidate launch profiles, only for a separately authorized isolated experiment:

```text
bash run_trajectory.sh 00_chunks
bash run_trajectory.sh 01_imagined
```

Both require `ACCERL_EXPERIMENT_ROOT`; neither submits a queue job. Default
configuration and old JIT profiles continue to use the original chunk sampler.
