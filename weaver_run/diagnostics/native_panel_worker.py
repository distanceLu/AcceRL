"""Private native-Torch child of a queue-admitted frozen diagnostic. No training."""
import os,sys,json,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'weaver_run/engine'))
from weaver_subprocess import send_frame,receive_frame,cpu_wire

def score_real_episode(b,ep):
    """Actual observations only; each endpoint once, no diffusion or noise draw."""
    import torch
    import numpy as np
    from .panel_common import validate_episode
    from .rm_probe import rm_prefix,real_scoring_blocks
    observed=validate_episode(ep)
    n,success,targets=rm_prefix(observed,ep['success'])
    incoming=np.concatenate([np.zeros((1,7),np.float32),ep['actions']],axis=0)
    rows=[]
    for context,frames in real_scoring_blocks(n):
        idx=np.asarray(context)
        with torch.inference_mode(),b._autocast():
            states=b._normal(b._tensor(ep['states'][idx]),'state').unsqueeze(0)
            actions=b._normal(b._tensor(incoming[idx]),'actions').unsqueeze(0)
            images=b._selected_images([dict(observations=ep['frames'])],idx)
            enc=b._encoded(images,states)
            latent={key:enc[key+'_features'] for key in b.image_keys};latent['states']=enc['states']
            scores=b.upstream_model.rm(latent,actions,b._task_features([ep['instruction']])).float()[0,-len(frames):]
        values=scores.detach().cpu().numpy().reshape(-1)
        if len(values)!=len(frames) or not np.isfinite(values).all() or np.any((values<0)|(values>1)):
            raise ValueError('Invalid native RM outputs')
        for i,value in zip(frames,values):
            rows.append(dict(episode_id=ep['episode_id'],initial_state_id=ep['initial_state_id'],
                frame_index=i,episode_length=n,success=success,target=targets[i-1],prediction=float(value)))
    return rows

def score_episode(b,ep,cfg,version):
    import torch
    import numpy as np
    from .panel_common import context,seed,discounted_targets,validate_episode
    from .learning_metrics import paired_wm_metrics,paired_rm_metrics
    n=validate_episode(ep);rows=[]
    for anchor in (0,31):
        max_h=min(32,(n-anchor)//8*8)
        if max_h<8:continue
        for noise in (19,47):
            st=b.init_rollout(context(ep,anchor,cfg),version)
            pred_images=[];pred_states=[];rm_gen=[];rm_real=[]
            for off in range(0,max_h,8):
                torch.manual_seed(seed(ep['initial_state_id'],anchor,noise,off))
                output=b.predict_chunk(st,ep['actions'][anchor+off:anchor+off+8],version,['all_observations'])
                st=output['next_state']
                pred_images.append(np.asarray(cpu_wire(output['observations']))[0])
                pred_states.append(np.asarray(cpu_wire(output['predicted_states']))[0])
                rm_gen.extend(np.asarray(cpu_wire(output['reward_predictions']))[0].reshape(-1).tolist())
                end=anchor+off+8
                # Actual observations and incoming actions, two history + eight future frames.
                idx=np.maximum(np.arange(end-9,end+1),0)
                incoming=np.concatenate([np.zeros((1,7),np.float32),ep['actions']],axis=0)
                states=b._normal(b._tensor(ep['states'][idx]),'state').unsqueeze(0)
                actions=b._normal(b._tensor(incoming[idx]),'actions').unsqueeze(0)
                with torch.no_grad(),b._autocast():
                    # Native encoder expects [0,1]; reuse its exact byte conversion/resize.
                    images=b._selected_images([dict(observations=ep['frames'])],idx)
                    enc=b._encoded(images,states)
                    latent={key:enc[key+'_features'] for key in b.image_keys};latent['states']=enc['states']
                    scores=b.upstream_model.rm(latent,actions,b._task_features([ep['instruction']])).float()[0,-8:]
                rm_real.extend(scores.detach().cpu().numpy().reshape(-1).tolist())
                h=off+8
                if h not in (8,16,32):continue
                ids=np.arange(anchor+1,anchor+h+1)
                truth=b._selected_images([dict(observations=ep['frames'])],ids)[0].detach().cpu().numpy()
                pi=np.concatenate(pred_images);ps=np.concatenate(pred_states)
                targets=discounted_targets(ep,ids);valid=np.ones(h,bool);eids=[ep['episode_id']]*h
                rows.append(dict(episode_id=ep['episode_id'],initial_state_id=ep['initial_state_id'],success=ep['success'],
                    anchor=anchor,noise=noise,horizon=h,world_version=version,
                    wm=paired_wm_metrics((pi+1)/2,truth,ps,ep['states'][ids],valid),
                    rm_real=paired_rm_metrics(rm_real,targets,valid,eids),
                    rm_generated=paired_rm_metrics(rm_gen,targets,valid,eids),
                    rm_real_predictions=list(rm_real),rm_generated_predictions=list(rm_gen),targets=targets.tolist(),
                    predicted_states=ps.tolist(),measured_states=ep['states'][ids].tolist()))
    return rows


def main():
    wire=sys.stdout.buffer;sys.stdout=sys.stderr
    from weaver_gpu_worker import arm_parent_guard
    arm_parent_guard()
    import torch
    import numpy as np
    from .panel_common import context,seed,discounted_targets,validate_episode
    from .learning_metrics import paired_wm_metrics,paired_rm_metrics
    # Upstream optional TensorFlow must never reserve a diagnostic GPU.
    try:
        import tensorflow as tf
        tf.config.set_visible_devices([],'GPU')
    except ImportError:pass
    from weaver_online_gpu_worker import load_backend
    b=None;cfg=None;state=None;version=None
    send_frame(wire,dict(ready=True,pid=os.getpid()))
    while True:
        req=receive_frame(sys.stdin.buffer)
        op=req['op']
        if op=='close':send_frame(wire,dict(closed=True));return
        try:
            if op=='load':
                state=None
                cfg=json.loads(Path(req['model']['config']).read_text())
                if b is None:b,_,receipt=load_backend(cfg,role='inference',rank=0)
                payload=torch.load(req['model']['world'],map_location='cpu',mmap=True,weights_only=True)
                m=payload['manifest']
                if (m['run_id'],m['policy_step'],m['model_version'])!=(req['model']['run_id'],req['model']['policy_step'],req['model']['world_step']):
                    raise ValueError('Frozen World checkpoint identity differs')
                b.upstream_model.load_state_dict(payload['model'],strict=True)
                b._task_cache.clear();b._feature_cache=b._encoder_guard=b._feature_namespace=b._task_guard=None
                b.published_version=None;version=m['model_version'];b.set_published_version(version)
                b.upstream_model.eval();b.upstream_model.requires_grad_(False)
                send_frame(wire,dict(manifest=m));del payload
            elif op=='init':
                state=b.init_rollout(context(req['episode'],req['anchor'],cfg),version)
                send_frame(wire,dict(phi=float(state['initial_reward_score'])))
            elif op=='step':
                if state is None:raise ValueError('No initialized frozen World state')
                torch.manual_seed(req['seed'])
                result=b.predict_chunk(state,req['actions'],version,['all_observations'])
                state=result.pop('next_state')
                result={k:cpu_wire(v) for k,v in result.items()}
                send_frame(wire,result)
            elif op=='score':
                rows=score_episode(b,req['episode'],cfg,version)
                send_frame(wire,dict(rows=rows))
            elif op=='score_real':
                if b is None:raise ValueError('No frozen RM checkpoint loaded')
                send_frame(wire,dict(rows=score_real_episode(b,req['episode'])))
            else:raise ValueError('Unknown diagnostic operation')
        except BaseException:
            send_frame(wire,dict(error=traceback.format_exc()));raise

if __name__=='__main__':main()
