"""Run the real parser, host-specific checks and one simulator reset on CPU."""
import hashlib
import json
import os
from pathlib import Path
import sys

from launch import ENGINE, RECIPE, ROOT, expand, settings

import train
import ctrl_world.config
import torch
from libero.libero import benchmark
from rl import actor_critic_model_discrete
from rl.libero_env import LiberoEnvWrapper
from initial_state_env import reset_with_initial_state
from policy_joint_batch_mode import verify_installed_sources
from cost_scheduler_runtime import load_settings


def main():
    assert Path(train.__file__).resolve()==ENGINE/"train.py"
    assert Path(ctrl_world.config.__file__).resolve()==ROOT/"weaver_run/vendor/ctrl/ctrl_world/config.py"
    assert Path(actor_critic_model_discrete.__file__).resolve()==ROOT/"rl/actor_critic_model_discrete.py"
    assert hashlib.sha256(Path(benchmark.__file__).read_bytes()).hexdigest()==json.loads((RECIPE/"reference.json").read_bytes())["libero_benchmark_sha256"], "LIBERO differs. Run ./run.sh setup or use the pinned LIBERO checkout and patch."
    verify_installed_sources()
    before=os.environ["CUDA_VISIBLE_DEVICES"]
    args=train.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=before
    work=Path(os.environ["ACCERL_EXPERIMENT_ROOT"])
    expected=expand(json.loads((RECIPE/"parsed-args.json").read_bytes()),
                    dict(settings(),ids=os.environ["ACCERL_ALLOCATED_GPU_IDS"],checkpoints=str(work/"checkpoints"),
                         world=str(work/"world.json"),pipeline=os.environ["ACCERL_PIPELINE_CONFIG"],
                         agent=str(ROOT/"envs/config/agent.yaml"),trainer=str(ROOT/"envs/config/trainer.yaml")))
    diff={k:[expected.get(k),vars(args).get(k)] for k in expected.keys()|vars(args).keys() if expected.get(k)!=vars(args).get(k)}
    if diff:
        raise ValueError("Command differs from the observed training run: "+json.dumps(diff))
    assert load_settings()["training_gangs"]==2
    env=LiberoEnvWrapper(benchmark_name="libero_spatial",task_id=0,image_size=224,render_mode="rgb_array")
    try:
        assert len(env.initial_states)==50
        obs,info=reset_with_initial_state(env,41,1997714015,220)
        assert info["initial_state_hash"]=="059790d8e9594cc2ca835e58010a4eb0ce4130ae459511df4162e5dcbc52f7f3"
        assert obs["full_image"].shape==obs["wrist_image"].shape==(224,224,3)
    finally:
        env.close()
    assert not torch.cuda.is_initialized()
    print(json.dumps({"parsed_arguments_matched":len(expected),"libero_initial_states":50,"cuda_initialized":False}))


if __name__=="__main__":
    main()
