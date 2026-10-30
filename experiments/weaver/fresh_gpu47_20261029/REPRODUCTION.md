# Runtime provenance and reproduction boundary

The repository default branch now contains minimal examples. This experiment ran
against the historical full AcceRL checkout at
`a0a0dfa969a9f1a8f5e85af0bca0b5facab5b5d8`, not against those minimal examples.
That base is also reachable through the previously published
`codex/ipc-policy-world-fastest-20261022` branch. Obtain a **separate checkout** of
that base and apply `base_overlay/` there when reconstructing the runtime.
Keep this archive unchanged.

`source/run_fresh.sh` and `source/runner.py` are the FRESH orchestration sources;
the active model profile is `source/examples/fresh_04_real.json`. Read their
arguments and isolation contracts before attempting a new launch. Publication
does not launch, enqueue, stop or modify a training job.

The frozen sources retain real machine paths and resource guards. In particular:

- the AcceRL runtime root is `/mnt/data/lcx1/dong/AcceRL-scheduler`;
- the CPU supervisor imports the three helpers from
  `/mnt/data/lcx1/dong/need_data/archive/` using absolute paths;
- the Policy/control and native World/RM run in separate Python environments and
  exchange bounded CPU NumPy messages. Do not merge their dependency environments;
- `policy_joint_batch_mode.py` verifies hashes of installed DeepSpeed boundary
  sources; matching a package version string alone is insufficient;
- the profile records external WEAVER source revision
  `3e41e90738fea9a7b142db6b2fd3d9ae72b7a82e`, external encoders, a normalizer,
  pretrained Policy assets and the offline WEAVER bundle. The bundle SHA256 is
  `6b45705115034914803224c3653533e30236fffdac544b46400b5af68bfd3443`;
- task0 and its 50 LIBERO initial states are part of the environment assets.

The package versions observed in both environments are in
`runtime_provenance.json`. They record the actual environment, not a promise that
an arbitrary pip installation recreates its native libraries, patched boundary
sources, encoders or model assets. Model weights and third-party datasets are not
redistributed here; retain their original access and license conditions.

For another machine, make a new candidate copy and change paths, GPU identifiers,
environment locations and resource ownership there. Rebuild its manifest and
repeat the relevant admission checks. Such a port is a new source identity; it
must not be represented as the byte-exact original run.

The `host_support/` modules are dependency provenance. Their historical command
line entrypoints can write reports, launch old experiments or signal exactly
identified processes. **Do not execute those historical CLIs to start FRESH.**
The serial helper clears `CUDA_VISIBLE_DEVICES` at import and belongs only in CPU
supervision. `base_overlay` also contains an old training entrypoint and a data
generator; neither is the entrypoint for this run.

CPU acceptance reports validate their named candidate revision and scope.
The 178-test parent report is not a claim that all 187 frozen files were re-tested
as a complete GPU job. Later two-environment CPU contracts and real startup /
first-loop receipts supply distinct evidence. The synthetic native precheck is a
transport/DDP/IPC test, not a learning-effectiveness result. No additional GPU
profiling or model tests were run for this publication.

The current cost-accounting scheduler candidate is absent from `source/`; it was
not active during the reported success curve. Source files may retain optional
profiling utilities from earlier work; their presence does not imply a new SM
measurement was taken. Old-run SM samples cannot establish current FRESH SM Active.
