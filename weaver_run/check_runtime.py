"""Run the real parser, host-specific checks and one simulator reset on CPU."""
import hashlib
import json
import os
from pathlib import Path
import sys

from launch import (ENGINE, RECIPE, ROOT, expand, settings,
                    fixed_validation_options, fixed_validation_overrides, batch_autotune_options)

import train
import torch
from libero.libero import benchmark
from rl import actor_critic_model_discrete
from rl.libero_env import LiberoEnvWrapper
from initial_state_env import reset_with_initial_state
from policy_joint_batch_mode import verify_installed_sources
from cost_scheduler_runtime import load_settings
from evaluation_threshold import resolve_online_evaluation


def main():
    assert Path(train.__file__).resolve()==ENGINE/"train.py"
    assert Path(actor_critic_model_discrete.__file__).resolve()==ROOT/"rl/actor_critic_model_discrete.py"
    assert hashlib.sha256(Path(benchmark.__file__).read_bytes()).hexdigest()==json.loads((RECIPE/"reference.json").read_bytes())["libero_benchmark_sha256"], "LIBERO differs. Run ./run.sh setup or use the pinned LIBERO checkout and patch."
    verify_installed_sources()
    before=os.environ["CUDA_VISIBLE_DEVICES"]
    args=train.parse_args()
    if not args.model_free:
        import ctrl_world.config
        assert Path(ctrl_world.config.__file__).resolve()==ROOT/"weaver_run/vendor/ctrl/ctrl_world/config.py"
    os.environ["CUDA_VISIBLE_DEVICES"]=before
    work=Path(os.environ["ACCERL_EXPERIMENT_ROOT"])
    expected=expand(json.loads((RECIPE/"parsed-args.json").read_bytes()),
                    dict(settings(),ids=os.environ["ACCERL_ALLOCATED_GPU_IDS"],checkpoints=str(work/"checkpoints"),
                         world=str(work/"world.json"),pipeline=os.environ["ACCERL_PIPELINE_CONFIG"],
                         agent=str(ROOT/"envs/config/agent.yaml"),trainer=str(ROOT/"envs/config/trainer.yaml")))
    expected['model_free']=args.model_free
    if args.model_free:
        expected.update(num_rollout_workers=4,num_ctrl_inference_actors=0,num_reward_inference_actors=0,
                        ctrl_eval_interval=0,world_backend_config=None)
    count=int(os.environ.get('ACCERL_NUM_EVAL_WORKERS',0 if args.model_free else 3))
    tuning=os.environ.get('ACCERL_BATCH_AUTOTUNE','0')
    if tuning not in ('0','1'):
        raise ValueError('ACCERL_BATCH_AUTOTUNE must be 0 or 1')
    expected['batch_autotune']=batch_autotune_options(args.model_free,count,tuning=='1')
    if args.model_free or count!=3:
        expected.update(num_eval_workers=count,eval_inference_replicas=','.join(map(str,range(count))))
    if args.model_free:
        expected['exp_name']=f'OpenVLA_ModelFree_Real4_Imag0_Eval{count}_Budget1M_Manual_gpu47_fresh'
        validation = fixed_validation_options(True, count, fixed_validation_overrides(args))
        expected.update({'fixed_validation_' + key: value for key, value in validation.items()})
    diff={k:[expected.get(k),vars(args).get(k)] for k in expected.keys()|vars(args).keys() if expected.get(k)!=vars(args).get(k)}
    if diff:
        raise ValueError("Command differs from the observed training run: "+json.dumps(diff))
    online_eval = resolve_online_evaluation(args.num_eval_workers, args.num_inference_actors,
        args.eval_inference_replicas, os.environ.get('ACCERL_EVAL_MIN_EPISODES'))
    if args.model_free:
        assert load_settings()['enabled'] is False
    else:
        assert load_settings()["training_gangs"]==1
    env=LiberoEnvWrapper(benchmark_name="libero_spatial",task_id=0,image_size=224,render_mode="rgb_array")
    try:
        assert len(env.initial_states)==50
        obs,info=reset_with_initial_state(env,41,1997714015,220)
        assert info["initial_state_hash"]=="059790d8e9594cc2ca835e58010a4eb0ce4130ae459511df4162e5dcbc52f7f3"
        assert obs["full_image"].shape==obs["wrist_image"].shape==(224,224,3)
    finally:
        env.close()
    assert not torch.cuda.is_initialized()
    print(json.dumps({"parsed_arguments_matched":len(expected),"libero_initial_states":50,
                      "cuda_initialized":False,"online_evaluation":online_eval}))


if __name__=="__main__":
    main()
