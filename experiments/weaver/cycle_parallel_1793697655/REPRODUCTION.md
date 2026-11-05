# Reproduction boundary

The frozen runtime is an experiment snapshot, not a replacement for the repository's
minimal examples. Its AcceRL runtime base is commit
`a0a0dfa969a9f1a8f5e85af0bca0b5facab5b5d8`, available through the historical
`codex/ipc-policy-world-fastest-20261022` branch.

Use a separate checkout of that base. The already-published sibling archive
[fresh_gpu47_20261029](../fresh_gpu47_20261029/REPRODUCTION.md) supplies `base_overlay/`,
`host_support/` and historical environment provenance. All 19 support-file hashes
were compared to the server during this publication and matched. See
[runtime_dependencies.json](runtime_dependencies.json) for the explicit scope of that
check. Do not run historical support CLIs as experiment entrypoints.

For this run, the source entrypoints are `source/run_cycle_parallel.sh` and
`source/runner.py`, with profile `source/examples/cycle_03_feedback_parallel.json`.
The wrapper is separately pinned under `queue_wrapper/`. Review the command-line
contracts and resource ownership before launching any new experiment. This archive
itself only supports an inert integrity check by `verify_snapshot.py`.

The original absolute paths, GPU UUIDs, machine identity, environment boundaries and
ownership guards are intentionally retained. Policy/control and native World/RM use
separate Python environments. The original base, LIBERO assets, pretrained Policy,
external WEAVER source/encoders and the offline native3200 bundle are required; they
are not included here. The native3200 bundle SHA256 is
`6b45705115034914803224c3653533e30236fffdac544b46400b5af68bfd3443`.
Installed DeepSpeed boundary source hashes also matter, beyond version strings.

On another host, create a new candidate copy, adapt its paths and resource pins,
rebuild its manifest and perform relevant admission checks. Such a port has a new
source identity. Historical CPU reports can require parent fixtures or the original
workspace; the small integrity verifier does not stand in for Ray/DDP/GPU admission.
No optimizer/replay checkpoint is redistributed, and no exact training resume is
promised. Respect original model and dataset access/license conditions.
