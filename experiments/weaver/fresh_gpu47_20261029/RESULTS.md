# Checking the apparent twenty-hour result

Data snapshot: 2026-10-30 10:22:24 Asia/Shanghai, current Policy update 454.
Historical TensorBoard events and current scalar / episode records were read on
CPU without model RPCs, GPU collection or training changes. Full exported curves
are in `evidence/baseline_curves.csv` and `evidence/current_curves.csv`.

## The comparison is real, with a specific metric

Both columns below use the logged **online greedy task-0 Eval success rate**.
They do not mix stochastic real collection with Eval.

| At first Eval >=95% | Historical CTRL run | Current WEAVER FRESH |
| --- | ---: | ---: |
| Time from first TensorBoard Event | 151.5589 h (6.315 days) | 17.6183 h |
| Completed Policy updates | 2,410 | 349 |
| Logged success | 95.3% | 95.0% (950/1,000) |
| Completed real training action steps | 79,960 | 138,032 |
| Completed real training episodes | 936 | 1,479 |
| Completed Eval episodes | 200,874 | 15,560 |
| Completed Eval action steps | 17,682,680 | 1,489,526 |

Historical run:
`1791343661_OpenVLA_DS_gipo_DISCRETE_task0_ctrl_wm_same_param`.
Current run:
`1793244803_OpenVLA_IntegratedWEAVER_PolicyWorldRM_Real1_Imag4_Budget1M_Manual_gpu47_fresh`.

The historical origin is its first event at 2026-10-07 11:27:41.242 BJT, because
an independently preserved runner start is unavailable. The current first event
is 2026-10-29 11:33:23.837 BJT; its runner started 27.69 seconds earlier. Using the
runner clock gives 17.6260 hours instead. Float32 threshold comparisons allow
small rounding tolerance: 0.949999988 represents 950/1,000 rather than a failure
to reach a literal 0.95.

The user's approximate “twenty hours versus seven days” observation is therefore
supported. The precise observed time-to-first-crossing ratio is **8.60x**.
The historical series has no multi-hour logging gaps that explain away those six
days: its largest gap is about 746 seconds, and cumulative recorded iteration
time is close to its 294-hour total wall time.

## What was checked for metric errors

- All 4,694 historical task-specific Eval `Success_Rate` points equal
  `Eval/_Global/Average_Return` exactly at matching steps.
- All 454 current Eval points can be reconstructed from actual environment
  episode-success events and cumulative Eval-episode counters. Maximum absolute
  disagreement is 2.96e-8 (float32 rounding), with no mismatches. The same check
  passes for all 454 stochastic real-rollout points.
- In the frozen current source, Eval uses deterministic actions and environment
  `info.is_success`, not the learned reward-head score. The underlying wrapper's
  success termination is separate from its 220-step horizon truncation.
- Current Eval workers report to statistics; their frames are not admitted to
  Policy Replay. Weight application validates a payload hash before ACK, and the
  publication protocol checks all-rank version/hash/ACK/release consistency.

These checks found no metric-substitution or counter-reconstruction error. They
do not prove every model, simulator, gradient or learning decision is correct.

## Why 95% is not a fixed-checkpoint result

At the current first crossing, the last 1,000 Eval completions span **60.31 minutes**
and Policy versions **328 through 348**. **151 episodes themselves span more than
one Policy version**. A model publication can happen during an ongoing episode.
Thus the online value logged beside P349 is not an evaluation of P349 alone.

The nearest twenty-hour record (P397) is 95.1%. The series later peaks at 96.0%;
the latest common export (P454) is **94.0%**, while stochastic real rollout is
95.1%. The last 100 Eval values average **94.986%**, range **93.7–96.0%**.
Ten consecutive >=95% logged points first appear at P365–374, but only 420 new
episodes enter the 1,000-episode window across those ten points. They are not ten
independent replications.

Eval and training use the same task-0 benchmark pool of 50 initial states. Current
Eval workers each cycle through that pool from index zero; real collection uses
a global shuffled schedule. This is not held-out-state or all-task LIBERO
generalization. Repeated initial states, overlapping windows and changing policies
make an ordinary independent-binomial confidence interval an inadequate estimate
of the uncertainty of a fixed final policy.

A saved, fixed initial/final checkpoint comparison on a balanced paired state
plan remains the appropriate next verification. The existing source contains
`FROZEN_POLICY_EVAL.md` and the corresponding evaluator. No new GPU evaluation was
started as part of publication.

## Where the observed improvement comes from

There is a useful descriptive identity:

```
wall-clock ratio = update-count ratio * average time-per-update ratio
8.60             = (2410 / 349)      * (226.40 s / 181.73 s)
                 = 6.91             * 1.25
```

The averages include initialization, filling, collection, refreshing, training
and publication until the first threshold. They are not isolated kernel timings.
The much larger observed change is **fewer Policy updates before reaching the
threshold**, rather than an eightfold improvement in update execution speed.
This algebra is not a causal attribution: the training and evaluation protocols
also changed.

The following mechanisms are plausible contributors, with executable paths and
runtime evidence rather than isolated effectiveness ablations:

1. **Real reward anchoring.** Every Policy batch now has 25% real chunks, with real
   environment success rewards, alongside 75% imagined chunks. The former
   imagination-only Policy path could learn solely from a changing model reward.
2. **Fresher optimization targets.** Replay stores ordered segments and recomputes
   values/GAE with the current critic. It caps Policy age at 16, World/RM age at 3
   and successful reuse at four. Behavior logits remain the original denominator;
   reward relabeling and imagined-state regeneration are not claimed.
3. **A different reward objective.** The native RM learns discounted eventual
   success for successful real episodes and zero for completed failures, without
   the old giant inverse-frequency binary weights. This can distribute signal
   further from the last successful frame. It is an objective change, not a proof
   that prior instantaneous-success labels were erroneous or that the new score
   is calibrated.
4. **Better seed coverage.** Outcome/position buckets and explicit early-context
   repetition admit states that previously could not seed an imagination branch.
   Their independent contribution has not been measured.
5. **Different initialization and execution.** FRESH uses native WEAVER initialized
   from 3,200 offline updates, four Policy training ranks, warmup100 and the
   selected-frame/KV/last-chunk/cache execution path. The historical same-name
   source uses one Policy training rank and warmup500. The new configuration also
   replaces the older World/RM architecture and resource layout.

The current first-loop receipts verify real consumption, quotas, current-critic
refreshes, successful updates and publications. Earlier actual telemetry also
showed refreshed advantages changing. Those are mechanism checks, not evidence
that any one feature caused the success increase. All four FRESH features were
enabled together. The cost-accounting scheduler was not deployed, and no current
SM Active measurement supports attributing this result to that scheduler or to
specific GPU-counter improvements.

## What this comparison does not establish

- **Better true sample efficiency:** at the threshold FRESH has used **1.726x as
  many real training action steps**. It reaches the threshold faster in online
  wall time, but not with fewer observed real training actions. Eval simulator
  work is separately much lower, as the table shows; it must not be silently
  included in or excluded from a differently named budget.
- **End-to-end training from scratch:** both runs use pretrained Policy assets.
  Current WEAVER offline pretraining is outside the 17.62-hour online clock.
  The old and new Policy path identifiers match, but historical weight bytes were
  not preserved with a launch-bound SHA, so exact initial-weight equality is not
  established. Small early online windows are not balanced initial-policy tests.
- **A controlled infrastructure speedup:** World/Reward, learning rules, warmup,
  topology and evaluation work differ. The old protocol reconstruction relies on
  a historical same-name source snapshot without a launch-bound source SHA;
  current running source is byte-bound to its manifest. The old measured curve
  and counters themselves are complete.
- **Stable 95% or broad generalization:** current online Eval fluctuates around
  95%, includes changing versions and repeatedly visits training initial states.

The defensible statement is: **this FRESH run reached 95% online rolling task-0
Eval in about 17.6 hours, versus 151.6 hours in the historical run, with fewer
Policy updates but more real training actions.** It is a promising run comparison
that still needs fixed-checkpoint evaluation and controlled ablations for causal
claims.

Machine-readable details: `evidence/baseline_findings.json`,
`evidence/current_findings.json`, `evidence/baseline_success_crosscheck.json`.
`evidence/plot_comparison.py` regenerates the plot from the exported CSVs.
