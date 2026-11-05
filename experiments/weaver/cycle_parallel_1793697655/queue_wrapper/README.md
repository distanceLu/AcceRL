# JIT GPU deployment

Manual-stop training, independent final save/audit and 200 paired evaluation.
Same frozen 247-file JIT candidate, explicit jit_03_fused_ema, H2 and cost03.
No calendar or four-hour cutoff; those CLI arguments are rejected.
Native tiny preflight checks transport/DDP/IPC, not JIT semantics. Actual formal
P1/P2 then P5/P6 must establish OpenVLA/ZeRO2/NCCL, EMA and publication acceptance.
Legacy stop_control and measurement helpers are retained only for historical CPU
regression coverage; neither is an admitted runtime stop mechanism in this entry.
