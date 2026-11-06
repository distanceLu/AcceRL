# WEAVER optimizations: finite equivalence and performance validation

This document is a plan, not a completed GPU result. The helper and CPU tests do not establish CUDA numerical equivalence, a speedup, stability of a new full training run, or a Policy success-rate gain. Current v5 training is immutable and remains separate from this candidate.

## Implemented helper and CPU scope

`weaver_feature_cache.py` provides full-window CPU admission; selection of actual16 frames from the40-frame training window; real episode/frame/camera/content identities; a process-local frozen-encoder identity namespace; conservative byte-and-entry-bounded CPU LRU; and a mutation guard. Default cache budget is64MiB, max8192 entries. Owned tensor payload and conservative key/object storage are charged; this is not a process RSS or allocator-reservation limit. Transient tensors remain bounded by the current micro.

Native32-context/8-future geometry selects memory `[1,6,11,16,21,26]` and main `[30..39]`. State/action/dt/label/terminal/window validation still covers all40 original frames. Frames skipped for image encoding are not discarded from those checks. Preserve the exact original FP32 byte normalization `/255,*2,-1`, inverse to encoder `[0,1]`, and bilinear192x320 preprocessing order; do not silently replace the roundtrip with `/255` alone.

Canonical IDs are `{episode}:window{start}` for training and `{episode}:seed{start}` for real imagination context. Global frame identity is start+local index. Explicit frame_start and request_lineage.seed_start/episode_id, when present, must agree. A retained identity changing pixel content is an error, not a cache hit. Raw canonical uint8 CPU bytes are SHA256-hashed. Normalized float inputs retain the legacy path and must not masquerade as canonical byte cache entries.

`FeatureNamespace.encoder_sha256` is the digest of admitted encoder identity metadata and loaded scope, not a claim that the helper independently rescanned all encoder weights. The caller includes the admitted source bundle receipt, source/encoder ID and this loaded object scope. Run and preprocessing digests are separate. WM/RM publication does not change frozen encoder parameters. Same-value small frozen normalization-buffer copies are admitted only with exact shape/dtype/device/value equality; changed parameters, buffer values or module registration invalidate the guard. Small buffer CPU snapshots are capped at1MiB; larger changed buffers fail closed. Direct unsafe external `.data` writes bypassing Torch version counters violate the immutable-source contract and cannot be disproved cheaply without rehashing the actual weights.

The pinned SD3Encoder calls `latent_dist.mode()`, not posterior.sample(), and the selected feature path has no sampling RNG. Frozen encoders with active dropout, training batch normalization or stochastic RReLU are rejected. CUDA batch-size/order changes may still change floating-point rounding. Do not claim bitwise equality from the source-level RNG observation.

Current cache hit implementation avoids encoder execution; unless the backend adds a lazy image-materialization path, it still normalizes/transfers/resizes the selected image tensor before cache lookup. Report this honestly. The CPU tests verify full40 versus selected16 feature/loss/downstream-gradient equivalence with deterministic actual Torch tensors; actual pinned SD3Encoder forward is also exercised using a tiny CPU VAE double and exact reshape adapter. That is not full SD3 inference or GPU validation.

## Explicit actual input fixture

Collect a **bounded real input fixture** while taking no new simulation or GPU action. Store it as a `.pt` containing only safe primitive containers and CPU tensors so `torch.load(map_location='cpu', weights_only=True, mmap=True)` can load it. Maximum size128MiB; batch1..4. A batch4 fixture may repeat one real window, but must declare that repetition and is only a compute-equivalence fixture, never four independent quality examples.

```text
protocol: weaver_optimization_fixture_v1
source_proof:
  run_id: exact source training run
  observed_ns: actual receipt timestamp
  observations_source: episode_frame_store
  label_source: libero_env_step
  no_imagined_frames: true
  schema_sha256: fingerprint(exact admitted schema)
windows:
  - obs: CPU uint8 [40,2,3,224,224]
    states: CPU finite float [40,8]
    actions: CPU finite float [40,7]
    success: CPU bool [40] from actual LIBERO info.is_success
    done/truncated/mask/action_mask/dt: actual complete aligned fields
    episode_id/sample_id/instruction/schema/label_source: exact actual metadata
fixture_policy:                 # required if windows repeat
  allow_repeated_windows: true
  purpose: compute_equivalence_only
```

Preserve a separate small source receipt tying the fixture's SHA256 to real episode IDs/window offsets and the existing frame store; a metadata claim is not itself independent evidence of real provenance. If the original global1024 RM cohort is available, preserve its exact counts and fixed class weights, and use them unchanged in both variants. Otherwise explicitly label a local fixed-weight comparison and do not call it a recreation of the original global update. Keep all same-RNG variants on identical windows, state/action values, labels and class weights.

`weaver_optimization_admission.py --config CONFIG [--fixture FILE --fixture-sha256 SHA --source-run-id RUN]` performs only bounded CPU admission. It does not load model weights, initialize CUDA, create a claim, execute a benchmark or expose a pretend `--execute`. `cpu_admission_passed` is separate from `comparison_passed` (null). The root agent's actual GPU executor must implement the following plan before any execution is requested.

## Mandatory execution gate, once, at most20 minutes

Only after the current training has saved and **all of its exact PID/UID/birth/run-marker processes have exited**, accept a caller-created exclusive probe claim. No automatic stop, restart, retry, new claim, or extended deadline is part of this probe.

The claim records the probe run ID, source training run, caller PID/UID/birth identity, exact candidate/config/source-manifest SHA, fixture SHA, admitted native3200 bundle SHA, GPU4 UUID `GPU-c43d6671-0979-b38d-9eae-4fe0a9d71c8c`, claimed_ns and one absolute deadline. Deadline is at most claimed_ns+1200s; CPU audit, model load, all cases, report and cleanup share it. Current time cannot create a fresh budget. CPU-only admission is allowed while training runs; actual GPU execution is not.

Before loading CUDA, verify exact MPS server/control UID/birth, the existing pipe and100% caps. Require two bounded read-only checks of no MPS clients and no GPU4 compute clients. Leave GPU0–3 and other users untouched. Use only GPU4, the admitted WEAVER interpreter, exact CVD UUID and current MPS pipe. No environment installation, Nsight or driver changes.

Run one owned foreground GPU worker under an independent bounded parent watchdog. Retain PID/UID/birth/run marker and parent-death protection. OOM or any failed case preserves the complete exception/resource receipt and stops comparisons; no fallback, smaller micro, new seed, restart or automatic retry. Cleanup signals only the exact owned process identities. Verify all owned processes gone, existing MPS retained and no owned MPS client; allow bounded unregister latency and two idle observations inside the same absolute budget. Cleanup failure means failure, even if numerical cases passed.

## A. Native training equivalence, same actual3200 model

Load the actual single native3200 model and freeze exactly the same encoders/critic; WM+RM trainable scope, FP32 parameters/BF16 autocast,32 activation-checkpoint blocks, compile=false. Do not build an optimizer or perform an update: this comparison checks features, native objective and every trainable gradient on one identical micro. Actual DDP/optimizer correctness remains the separate established native training path.

Record exact loaded inventory and tensor hashes once. Save/restore CPU RNG and GPU RNG before every variant; assert no VAE-encoding sampling-RNG change. Train-mode SPRINT indices/history/noise/timesteps must therefore follow the same RNG stream. Use identical labels and RM class weights.

1. Baseline: full40 image encode and original main/memory gathering.
2. Selected16: exact compact frame gathering, no cache.
3. Selected16 cold cache.
4. Selected16 warm cache, same admitted image/encoder identity.

Compare every selected image-feature tensor, memory tensor and state/action/task/label input. Compare native loss and every individual native loss component, including flow state/two cameras, half-Brier and frozen critic diagnostic. Zero gradients before every variant and compare **all403 WM+RM parameter gradients**, not only a norm or a selected layer. Respect None-versus-zero gradients for missing-class head skips. Store per-family count, maximum absolute error, relative L2 and cosine, with per-tensor outliers. Compare CPU/GPU RNG states after forward/backward. A tiny loss difference alone is not sufficient.

Use separate tolerances for preprocessing/encoded features, native scalar loss, and BF16 mixed-precision gradients. The executor must print chosen tolerances before running, reject nonfinite values and report actual maxima; do not loosen them after observing a failure. Exact CPU toy equality does not justify zero-tolerance CUDA assertions. Timing covers complete preparation+encoder+forward+backward and separately timed phases with explicit CUDA synchronization at boundaries, no profiling collector. Record peak allocated/reserved after resetting peaks and the cache's actual counters/byte limits. These phase times are controlled single-model micro measurements, not promises for the full16-model concurrent training system.

## B. Native uncached versus KV generation

From the same fixed actual32-frame real context and identical proposed actions, compare horizons1/4/8. Reset exact CPU/GPU RNG state for each uncached/cached pair; use identical noise seeds and solver settings. Keep configured denoising steps16 and the same diffusion-forcing/pyramid schedule. Instrument and report the actual FlowWM forward/evaluation count: a configured16-step staggered schedule must not be mislabeled as exactly16 calls if its actual count differs.

Compare **all generated future latents, all predicted states and all raw/bounded native reward scores**, not only the final image. Check output shapes, action/frame alignment, served model version and fixed-horizon terminal=false. Compare all memory/history inputs and cache invalidation after a WM/RM publication. Establish that no weight-dependent KV cache survives publication; frozen image features may survive only because their encoder/preprocessing identity is unchanged. Report finite checks and numerical tolerances separately for each horizon.

## C. Decode only what the consumer uses

On identical precomputed latent/state/score trajectories, compare the original complete decode output to optimized last-frame/last-chunk decoding exactly at every frame actually consumed by Policy and reward logic. Keep image size/range/camera order and chunk-to-control-step semantics identical. Report equality/error for the selected RGBs and verify latent/state/score outputs are unchanged. Do not report this as eliminating all VAE decode if consumed pixels still require it. A final-frame-only success comparison must not hide altered intermediate latent trajectories.

## D. Cache/resident transport semantics

Measure cold/warm hits, misses, evictions, entry count and conservative resident bytes. Exercise repeated real windows, overlapping seed/window IDs and distinct episodes. Exact retained frame identity plus changed pixel bytes must fail; changed encoder/preprocessing/run identity must not reuse features. Verify same-value small buffer publication refresh, actual encoder mutation rejection and no encoder GPU tensors retained in CPU LRU.

Count actual private-RPC pickle bytes for old opaque state versus resident-session references and CPU observations, including session creation/release. Logical payload size is not measured physical PCIe/NVLink/network bandwidth. Exercise bounded sessions, byte eviction, stale/unknown/released session errors and publication handling. Ensure request failures release their state and no stale resident state enters another episode. Inference state stays private to the native worker and is not exposed as a CUDA tensor across interpreters.

## Required final result

The final report must include input/config/source/bundle hashes; actual interpreter/Torch; loaded scope; real/repeated fixture coverage; all case names and exact RNG/NFE/tolerances; numerical maxima; timings and memory; cache/resident/CPU-wire statistics; failure stack if any; absolute budget used; owned identities; MPS pre/post receipts; cleanup.all_owned_exited; and independent flags for admission/numerical/performance/cleanup completeness. Missing cases or timeout are partial/failure, not a fabricated pass. Preserve the current training source and its source hashes throughout.
