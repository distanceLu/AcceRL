# Minimal Distributed Model-Free GIPO

This repository contains a minimal, standalone, and runnable distributed Gaussian Importance sampling Policy Optimization (GIPO) framework. It is extracted from the full `ds_libero_gipo_discrete.py` pipeline. 

This minimal version preserves the core **Ray Actor topology**, the **GIPO mathematical logic**, and the **DeepSpeed/ZeRO training paths**, while stripping away heavy dependencies like OpenVLA and LIBERO. It utilizes the following lightweight substitutes:
- `fake_env.py`
- `ds_com.py`
- `fake_models.py`
- `main_ray_gipo_ds_standalone.py`
- `README.md`
- `requirements.txt`
- `requirements.txt`

## Architecture Overview

The system retains the core microservice Actor roles from the original architecture:

* **`StatsActor`**: Aggregates training and evaluation statistics, including episode returns, success rates, average episode lengths, and inference latencies.
* **`ReplayBufferActor`**: Stores sampled data as complete trajectories. The `RolloutWorkerActor` writes to this buffer, and the `TrainerActor` samples from it.
* **`RolloutWorkerActor`**: Manages multiple `FakeEnv` instances. It requests actions from the `InferenceActor`, steps the environment, packages the transitions into trajectories, and pushes them to the `ReplayBufferActor`.
* **`EvaluationWorkerActor`**: Operates similarly to the rollout worker but strictly uses a deterministic policy (argmax) for evaluation. It reports directly to the `StatsActor` and does not write to the replay buffer.
* **`InferenceActor`**: Hosts the `FakeActorCritic`. It receives observations and outputs discrete action tokens, continuous environment actions (`action_env`), behavior policy logits, and value estimates.
* **`TrainerActor`**: Samples trajectories from the `ReplayBufferActor`. It computes Generalized Advantage Estimation (GAE), normalized advantages, and value targets. It then executes the PPO/GIPO optimization (clipped surrogate objective, value loss, entropy regularization, and KL penalty) and synchronizes the updated weights to all `InferenceActors`.

### Data Flow Execution
1. The `RolloutWorkerActor` extracts an observation from the `FakeEnv`.
2. The observation is dispatched to the `InferenceActor`.
3. The `InferenceActor` returns the action, logits, and value estimate.
4. The `RolloutWorkerActor` executes the action in the environment and accumulates the transitions into a `Trajectory`.
5. The completed `Trajectory` is written to the `ReplayBufferActor`.
6. The `TrainerActor` samples trajectories from the `ReplayBufferActor` and performs a GIPO/PPO gradient update.
7. The main thread synchronizes the `TrainerActor`'s latest weights to all `InferenceActors`.
8. Concurrently, the `EvaluationWorkerActor` runs continuous evaluations and logs the results to the `StatsActor`.

##  Local Quick Start

### Dependencies
It is recommended to install the minimal dependencies first:
```bash
pip install ray torch numpy tensorboard
```

### Execution
Navigate to the directory and run the default script:
```bash
cd minimal_modelfree_GIPO
python main_ray_gipo.py
```

**Lightweight Quick Test:**
```bash
python main_ray_gipo_ds_standalone.py \
  --cuda-visible-devices "0,1" \
  --train-iters 5 \
  --num-rollout-workers 1 \
  --num-eval-workers 1 \
  --train-batch-size 16 \
  --accumulation-steps 2
```

**Hardware Allocation:**
The framework runs on the CPU by default (`--trainer-num-gpus 0` and `--inference-num-gpus 0`). If you wish to allocate GPUs to the trainer or inference actors, adjust the flags accordingly:
```bash
python main_ray_gipo_ds_standalone.py \
  --cuda-visible-devices "0,1" \
  --trainer-num-gpus 1 \
  --inference-num-gpus 1 \
  --num-inference-actors 1
```

Upon completion, the framework will save the model weights to the current directory as `minimal_ray_gipo_ckpt.pt`.

##  Key Retained Features

This minimal version isolates the most critical algorithmic components of the PPO/GIPO pipeline for research and debugging:
* Trajectory-based experience storage.
* Bootstrap value computation for truncated trajectories.
* Generalized Advantage Estimation (`_compute_gae`).
* Global advantage normalization.
* PPO ratio and clipped surrogate objective (with GIPO soft-clipping).
* Value loss, entropy regularization, and KL penalty.
* Physical separation of the Trainer and Inference modules.

##  Extension Guide

### Integrating a Real Environment
To replace the fake environment with a real physical simulator:
1. Replace `FakeEnv` in `fake_env.py` with your real environment wrapper.
2. Ensure the standard RL interfaces are maintained:
   * `reset(seed)` -> `(obs, info)`
   * `step(action)` -> `(next_obs, reward, terminated, truncated, info)`
3. Ensure the observation format is compatible with the model inputs.
4. If your action space deviates from `(NUM_ACTIONS_CHUNK, ACTION_DIM)`, update the output shapes in `fake_model.py` and the action execution logic in the `RolloutWorkerActor`.

### Integrating a Real Neural Network
To replace the fake models with a real VLA backbone (e.g., OpenVLA):
1. Update `forward(obs_batch)` to return `action_logits` and `values`.
2. Update `post_process(...)` to return discrete `action_tokens` and the continuous `action_env` required by the simulator.
3. Update `prepare_inputs_batch(...)` to correctly collate and batch a list of observations.

As long as these three interfaces are maintained, the distributed Actor framework and GIPO training logic in `main_ray_gipo.py` will work seamlessly without modification.

##  DeepSpeed & TensorBoard Integration

The current version enables **DeepSpeed** by default, utilizing `deepspeed.initialize` (including `ds_config` and ZeRO optimizations):

```bash
python main_ray_gipo_ds_standalone.py \
  --cuda-visible-devices "0,1" \
  --train-iters 5 \
  --train-batch-size 16 \
  --accumulation-steps 2
```
* **ZeRO Stages**: You can switch ZeRO stages using `--zero-stage {0,1,2,3}` or load external configurations via `--ds-config-json /path/to/ds_config.json`. 
* **Pure PyTorch**: To run a pure PyTorch baseline experiment, append the `--disable-deepspeed` flag.

### Standalone Communication Version
We also provide `main_ray_gipo_ds_standalone.py`. This version preserves the broadcast handshake and communication skeleton between `TrainerActorCom` and `InferenceActorCom`, but uses fake environments and models to isolate and test network overhead.

```bash
python main_ray_gipo_ds_standalone.py \
  --train-iters 5 \
  --num-rollout-workers 1 \
  --num-inference-actors 1
```

**TensorBoard Logging**: 
Logs are automatically saved to `runs/minimal_ray_gipo`. To visualize training metrics in real-time, run:
```bash
tensorboard --logdir runs/minimal_ray_gipo --port 6006
```

##  Limitations & Scope
* This minimal version defaults to supporting exactly **1** `TrainerActor`.
* Heavy foundation models and complex simulators have been intentionally removed.
* **Primary Goal**: This repository is designed to help researchers study the Actor topology, the rollout-to-replay data flow, and the underlying PPO/GIPO mathematics on a local, offline machine.

If you plan to scale this back to a production system, we recommend replacing the environment first, followed by the policy model, and finally introducing multi-node trainers (DDP).
