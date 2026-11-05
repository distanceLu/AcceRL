# WEAVER cycle feedback + parallel trainers: run 1793697655

This archive preserves the exact 267-file deployed source for:

`1793697655_OpenVLA_IntegratedWEAVER_PolicyWorldRM_Real1_Imag4_Budget1M_Manual_gpu47_fresh`

Run: `weaverintegrated_1103_172031_a7557d7d`; job: `ed10ac94daba48fe87102040a4463940`.
Started 2026-11-03 17:20:31 BJT. The source SHA256 is
`9a0d2a8247fc055760fa5076f4d1e7278586eb8f3c09a87581794a7e4a56b91b`.

## Contents

- [source/](source/): byte-exact runtime source and original manifest.
- [queue_wrapper/](queue_wrapper/): the 17 pinned deployment wrapper files.
- [run_identity.json](run_identity.json): run, TensorBoard, profile and hash binding.
- [evidence/](evidence/): existing CPU admission, runtime configuration and final training records.
- [REPRODUCTION.md](REPRODUCTION.md): external dependencies and reproduction limits.

The active scheduler profile was [cycle_03_feedback_parallel.json](source/examples/cycle_03_feedback_parallel.json):
two independent four-rank training groups may be admitted together; logical per-card
inflight capacity is three. Complete-cycle feedback changes service weights with
bounded probes, hold periods and rollback. Logical admission is not an SM allocation
or a guarantee of simultaneous CUDA execution.

This version includes JIT current-critic GAE inside the Policy trainer, advantage
statistics EMA, complete imagined-trajectory sampling and the earlier funded-inventory
stall fix. Each Policy update uses 144 real and 432 imagined chunks. Each fifth
Policy update includes one World/RM update (global batch 1024). Version ages and the
four-use cap remain enforced. It trains LIBERO-Spatial task 0 over 50 initial states,
not 50 distinct tasks.

**This exact run used 10 online Eval workers.** The later local Eval2 edit and later
pipeline-efficiency/allocator versions are not part of this snapshot. The README
inside `source/` is an unmodified historical candidate document; its pre-deployment
wording is preserved for byte identity. This outer document and the runtime evidence
describe the deployed run.

## Recorded outcome

Training stopped at **P451/W90**, with boundary saving and CPU audit accepted and
owned training processes cleaned up. The user then canceled the fixed final
evaluation after **37/200** episode records. The evaluation was incomplete and the
queue's historical `failed` state is retained. This archive makes no final success-rate
or causal speedup claim. These results do not describe the earlier P868 checkpoint.

The evidence files are historical observations with their own timestamps and scope;
they are not tests re-run during publication. Publication verified file hashes and
Python syntax without importing models or launching GPU work.

```bash
python experiments/weaver/cycle_parallel_1793697655/verify_snapshot.py
```

Model checkpoints, private datasets, replay contents, raw frames, event files and
databases are not included. Publishing this archive does not modify a running job.
