# Legacy GIPO speed baseline

The user authorized one baseline based on `run_ppo_training_example.sh`, with
Eval0 and GIPO, after the current model-free experiment and its final evaluation.
The purpose is to measure the original training implementation's speed and
time to 98/99% training rollout success.

Two configurations were offered: literal two-GPU object-task example, or the
legacy engine with the current spatial task/checkpoints and four GPUs. The user
explicitly selected the latter: physical
GPUs 4–7, three training ranks, one inference replica, spatial task 0, the same
two-image/no-proprio pretrained plus distilled initialization, global batch 576
(micro12/GAS16/world3), 10 real workers, Eval0, GIPO sigma1 diagnostics matching
the legacy loss. Keep the legacy OSMesa rendering, replay, action-chunk treatment
and old-value GAE; document these differences instead of claiming an isolated
placement ablation. The user subsequently replaced the two-hour measurement
with first-observed 99% success over a complete latest-1000 real episode window.
Record first 98% and 99% times separately; stop at the completed update/publication
boundary after 99%. No two-hour limit remains. Record launch and update-loop
elapsed times. Preserve the existing update cap, explicit stop and fault bounds.

Do not start before `model-free-autobatch-20261116-a` has completed training,
final evaluation, and owned-process cleanup successfully. Do not modify its
`.git-runtime` checkout or stop its processes. Use the official server `run.sh`
for submission, a separate immutable checkout, an exact published commit, and
data-disk output directories. Preserve the user's overwrite checkpoint choice.

Add only operational instrumentation to the original trainer: bounded stopping,
an explicit output/temp directory, final weights saving, per-update timings,
placement/argument records and completion evidence. Default legacy behavior
without the new bounded-run option remains unchanged. No FRESH replay, world
models or online evaluation are introduced. Final weights are not advertised as
an exact optimizer/replay resume snapshot.

The queue supervisor owns error propagation and scoped cleanup, and may report
success only after the child exits successfully, final weights exist, and all
owned processes are gone. Queueing is exactly once; an ambiguous submission or
failed predecessor is recorded for inspection rather than retried blindly.

Alternatives: directly running the old shell would bypass version/queue rules
and miss final saving; reusing the new WEAVER model-free trainer would not test
the teacher's original implementation. The selected route retains the legacy
trainer with a small dedicated lifecycle adapter.
