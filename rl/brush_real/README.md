# Brush real-data asynchronous RL

This entry keeps the original asynchronous actor-critic/policy-inference/world-
model layout, but removes Libero, learned reward training, and online world-
model training. Every rollout and evaluator trajectory starts from a sampled
strictly synchronized point in the real Brush recordings. Ctrl-World is loaded
once in inference actors, checked to have zero trainable parameters, and never
broadcast or optimized. No Libero VAE-decoder checkpoint is accepted or loaded.

Run the data/checkpoint preflight without allocating GPUs:

```bash
PREFLIGHT_ONLY=1 bash rl/brush_real/run_brush_real_async.sh
```

Start training:

```bash
bash rl/brush_real/run_brush_real_async.sh
```

`SmokeDenseReward` in `reward.py` is called exactly once per 10 Hz low-level
OpenVLA action, so the default 32-step imagined trajectory has 32 reward calls.
It currently returns a constant `1.0`; replace its `__call__` body with the real
rule-dense reward. Until then, return is a reward-plumbing smoke signal and is
not a task-quality metric.

The OpenVLA policy predicts eight 10 Hz Euler-delta actions per query. Each pair
is composed on SE(3) into one 5 Hz rotvec delta for the supplied Ctrl-World
checkpoint. World-model camera order is `pool,pool1,paper`; policy input order
is `paper,pool,pool1`, matching the corresponding training datasets.
