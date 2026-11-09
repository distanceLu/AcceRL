"""Fresh fixed-initial-state collection and frozen Policy gradient measurement."""
import json,os,pickle,random,subprocess,sys,time
from pathlib import Path
import numpy as np
from .panel_common import save,seed,validate_episode,flatten_segments


def dump(path,data):
    with Path(path).open('xb') as f:pickle.dump(data,f,protocol=5)

def load(path):
    # Private files written by this job, never arbitrary downloaded pickle inputs.
    with Path(path).open('rb') as f:return pickle.load(f)

def forward(model,cfg,dtype,obs,instruction,draw=True):
    import torch
    from rl.utils import prepare_one_obs
    from weaver_subprocess import cpu_wire
    inp=prepare_one_obs(cfg,model.processor,obs,instruction,dtype)
    with torch.no_grad():
        logits,value=model(model.prepare_inputs_batch([inp]))
        sample=None
        if draw:
            _,tokens,actions=model.post_process(logits,[False])
            sample=dict(tokens=tokens.detach().cpu().numpy().reshape(8,7),
                actions=model.vla._unnormalize_actions(actions[0],cfg.unnorm_key).astype(np.float32))
    return dict(input=cpu_wire(inp),logits=logits.detach().float().cpu().numpy().reshape(8,7,-1),
                value=float(value.detach().reshape(-1)[0].cpu()),sample=sample)

def collect(model,cfg,dtype,plan,key,out):
    import torch
    from rl.libero_env import LiberoEnvWrapper
    from initial_state_env import reset_with_initial_state,initial_state_pool_metadata
    from episode_frame_store import encode_observations
    records=[];segments=[]
    for sid in plan['initial_states']:
        env=LiberoEnvWrapper(benchmark_name='libero_spatial',task_id=0,image_size=224,render_mode='rgb_array')
        try:
            pool=initial_state_pool_metadata(env)
            if pool['count']!=50 or pool['reset_steps']!=10:raise ValueError('Unexpected fixed task/state pool')
            obs,info=reset_with_initial_state(env,sid,seed('env',sid),220)
            torch.manual_seed(seed('policy',sid))
            observations=[obs];actions=[];chunk_inputs=[];logits=[];tokens=[];masks=[];rewards=[]
            done=False;success=False
            while not done:
                row=forward(model,cfg,dtype,obs,env.task_description)
                chunk_inputs.append(row['input']);logits.append(row['logits']);tokens.append(row['sample']['tokens'])
                executed=0;reward=0.
                for act in row['sample']['actions']:
                    obs,r,term,trunc,end_info=env.step(act)
                    actions.append(act.copy());observations.append(obs);executed+=1;reward+=float(r)
                    if term or trunc:
                        success=bool(end_info.get('is_success',False));done=True;break
                mask=np.zeros((8,7),bool);mask[:executed]=True;masks.append(mask);rewards.append(reward)
                if len(actions)>220:raise ValueError('Episode exceeded fixed horizon')
            eid=key+':state'+str(sid)
            ep=dict(episode_id=eid,initial_state_id=sid,initial_state_hash=info['initial_state_hash'],
                success=success,instruction=env.task_description,frames=encode_observations(observations),
                states=np.stack([o['state'] for o in observations]).astype(np.float32),
                actions=np.stack(actions),state_pool=pool)
            validate_episode(ep)
            seg=dict(episode_id=eid,source='real',inputs=chunk_inputs,old_logits=np.stack(logits),
                tokens=np.stack(tokens),masks=np.stack(masks),rewards=rewards,
                endpoint=None,terminal=True,full_episode=True)
            dump(out/f'{key}-state{sid}.pkl',ep)
            records.append(ep);segments.append(seg)
            save(out/'progress.json',dict(stage='real_collection',policy=key,episodes=len(records),target=16))
        finally:env.close()
    return records,segments

class Native:
    def __init__(self,out,plan,repo):
        from weaver_subprocess import process_identity,receive_frame
        from .panel_common import GPUS
        parent=process_identity(os.getpid())
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPUS[1],ACCERL_EXPECTED_GPU_UUID=GPUS[1],ACCERL_USE_MPS='1',
            ACCERL_PHYSICAL_GPU='5',WEAVER_GPU_WORKER_ID='frozen-learning-diagnostic',
            WEAVER_PARENT_PID=str(parent['pid']),WEAVER_PARENT_UID=str(parent['uid']),WEAVER_PARENT_BIRTH=str(parent['birth']))
        self.log=(out/'native.log').open('ab')
        self.proc=subprocess.Popen([plan['native_python'],'-u','-m','weaver_run.diagnostics.native_panel_worker'],
            cwd=repo,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log)
        try:
            reply=receive_frame(self.proc.stdout)
            if not reply.get('ready'):raise RuntimeError('Native worker did not initialize')
        except BaseException:
            self.close();raise
    def call(self,**request):
        from weaver_subprocess import send_frame,receive_frame
        send_frame(self.proc.stdin,request);reply=receive_frame(self.proc.stdout)
        if 'error' in reply:raise RuntimeError(reply['error'])
        return reply
    def close(self):
        # Only the directly owned child handle. No RPC wait after a job deadline.
        try:
            if self.proc.poll() is None:
                self.proc.terminate()
                try:self.proc.wait(timeout=15)
                except subprocess.TimeoutExpired:self.proc.kill();self.proc.wait(timeout=15)
        finally:
            self.proc.stdin.close();self.proc.stdout.close()
            self.log.close()

def imagined(model,cfg,dtype,episodes,ntraj,native,key,out):
    import torch
    from envs.utils import tensor_to_image
    from fresh_seed_scheduler import context_seed_starts,seed_bucket,FreshSeedScheduler
    pending=[];byid={e['episode_id']:e for e in episodes}
    # Same six-stratum rule on this finite independent panel. No success oracle from WM.
    for ep in episodes:
        for start in context_seed_starts(len(ep['frames']),32,16,include_early=True):
            a=start+31;pending.append((ep['episode_id'],a,seed_bucket(ep['success'],a,len(ep['frames']))))
    if len(pending)<ntraj:raise ValueError('Insufficient distinct seed contexts; no duplicate fallback')
    scheduler=FreshSeedScheduler(seed=20261109);result=[]
    for j in range(ntraj):
        chosen=scheduler.choose(pending);pending.remove(chosen);eid,anchor,bucket=chosen;ep=byid[eid]
        phi=native.call(op='init',episode=ep,anchor=anchor)['phi']
        # Production seed input uses float-normalized real views, inverted by tensor_to_image.
        views=torch.tensor(ep['frames'][anchor],dtype=torch.float32)/255*2-1
        obs=dict(full_image=tensor_to_image(views[0]),wrist_image=tensor_to_image(views[1]))
        inputs=[];tokens=[];logits=[];rewards=[]
        torch.manual_seed(seed('imag-policy',ep['initial_state_id'],anchor))
        for step in range(4):
            row=forward(model,cfg,dtype,obs,ep['instruction'])
            inputs.append(row['input']);tokens.append(row['sample']['tokens']);logits.append(row['logits'])
            pred=native.call(op='step',actions=row['sample']['actions'],seed=seed('imag-world',ep['initial_state_id'],anchor,step))
            scores=np.asarray(pred['reward_predictions']).reshape(-1)
            rewards.append(float(scores[-1])-phi);phi=float(scores[-1])
            views=torch.from_numpy(np.asarray(pred['observations'])[0,-1])
            obs=dict(full_image=tensor_to_image(views[0]),wrist_image=tensor_to_image(views[1]))
        endpoint=forward(model,cfg,dtype,obs,ep['instruction'],draw=False)['input']
        result.append(dict(episode_id=eid,source='imagined',inputs=inputs,old_logits=np.stack(logits),
            tokens=np.stack(tokens),masks=np.ones((4,8,7),bool),rewards=rewards,endpoint=endpoint,
            terminal=False,seed_anchor=anchor,seed_bucket=bucket))
        save(out/'progress.json',dict(stage='imagination',policy=key,trajectories=j+1,target=ntraj))
    return result

def restore_input(inp,dtype):
    import torch
    return {k:(torch.from_numpy(v).to(dtype=dtype) if k=='pixel_values' else torch.from_numpy(v))
            if isinstance(v,np.ndarray) else v for k,v in inp.items()}

def gradients(model,dtype,segments,selected,ema,args,key,out):
    import torch
    from fresh_policy_replay import generalized_advantages
    from .learning_metrics import source_losses,loss_gradients,gradient_geometry
    params=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    if any(isinstance(m,torch.nn.Dropout) and m.p for m in model.modules()):
        raise ValueError('Frozen diagnostic requires the actual zero-dropout recipe')
    if hasattr(model.vla,'gradient_checkpointing_disable'):model.vla.gradient_checkpointing_disable()
    model.eval()
    # Complete suffixes; no truncated-chunk approximation. Fixed weights and fixed EMA.
    with torch.no_grad():
        for s in segments:
            inputs=s['inputs']+([] if s['terminal'] else [s['endpoint']]);v=[]
            for off in range(0,len(inputs),2):
                batch=model.prepare_inputs_batch([restore_input(x,dtype) for x in inputs[off:off+2]])
                _,value=model(batch);v.extend(value.detach().float().cpu().reshape(-1).tolist())
            if s['terminal']:v.append(0.)
            n=len(s['rewards']);terminal=[False]*(n-1)+[s['terminal']];truncated=[False]*(n-1)+[not s['terminal']]
            adv,target=generalized_advantages(s['rewards'],v,terminal,truncated,args['gamma'],args['lambda_'])
            s.update(adv=(adv-ema['mean'])/ema['scale'],targets=target,prepass_values=np.array(v[:-1]))
    names=[src+'_'+term for src in ('real','imagined') for term in ('actor','critic','regularizer')]
    sums={name:{n:torch.zeros_like(p,dtype=torch.float32) for n,p in params} for name in names}
    micro=int(args['train_batch_size'])
    if micro<1 or len(selected)%micro:raise ValueError('Diagnostic batch must fit saved micro size')
    max_value_delta=0.;micro_count=len(selected)//micro;loss_sum={n:0. for n in names}
    direct_error=None
    for off in range(0,len(selected),micro):
        rows=[(segments[j],i) for j,i in selected[off:off+micro]]
        batch=model.prepare_inputs_batch([restore_input(s['inputs'][i],dtype) for s,i in rows])
        logits,values=model(batch)
        ref=values.new_tensor([s['prepass_values'][i] for s,i in rows])
        max_value_delta=max(max_value_delta,float((values.reshape(-1)-ref).detach().abs().max().cpu()))
        tensor=lambda k,dt:torch.as_tensor(np.stack([s[k][i] for s,i in rows]),device=values.device,dtype=dt)
        actions=tensor('tokens',torch.long);old=tensor('old_logits',logits.dtype)
        adv=tensor('adv',torch.float32);targets=tensor('targets',torch.float32);masks=tensor('masks',torch.bool)
        token_weights=torch.ones(micro,device=values.device)/masks.sum()
        sources=torch.tensor([0 if s['source']=='real' else 1 for s,i in rows],device=values.device)
        cfg={k:args[k] for k in ('vf_coef','ent_coef','kl_coef','clip_mode','clip_eps')}
        split=source_losses(logits,values,actions,old,adv,targets,masks,token_weights,sources,**cfg)
        first_grads=[]
        for index,name in enumerate(names):
            src,term=name.split('_',1);loss=split[src][term]/micro_count
            loss_sum[name]+=float(loss.detach().cpu())
            grads=loss_gradients(loss,params,retain_graph=index<len(names)-1 or off==0,
                storage_device=values.device,storage_dtype=torch.float32)
            for n,g in grads.items():sums[name][n].add_(g)
            if off==0:first_grads.append(grads)
        if off==0:
            from jit_gae import masked_policy_loss
            joint=masked_policy_loss(logits,values,actions,old,adv,targets,masks,token_weights,**cfg)[0]/micro_count
            direct=loss_gradients(joint,params,storage_device=values.device,storage_dtype=torch.float32)
            error=base=0.
            for n in direct:
                combined=sum(g[n] for g in first_grads)
                error+=float((combined-direct[n]).double().square().sum().cpu())
                base+=float(direct[n].double().square().sum().cpu())
            direct_error=(error/max(base,1e-30))**.5
            if direct_error>.05:raise ValueError('BF16 source split vs direct gradient discrepancy exceeds diagnostic limit')
            del direct,first_grads
        del grads,split,logits,values,batch
        if off%24==0:save(out/'progress.json',dict(stage='gradients',policy=key,chunks=off+micro,target=len(selected)))
    result=dict(policy=key,chunks=len(selected),source_chunks={src:sum(segments[j]['source']==src for j,i in selected) for src in ('real','imagined')},
        ema=ema,max_value_prepass_forward_abs_delta=max_value_delta,first_micro_gradient_relative_error=direct_error,
        micro_batch_size=micro,normalization='mean over saved-size micros; serial averaged DP/GAS objective on this panel, not a live distributed gradient capture',losses=loss_sum,geometry={})
    # Copy accumulated vectors once; do not transfer a whole gradient after every micro.
    sums={name:{n:g.cpu() for n,g in values.items()} for name,values in sums.items()}
    for term in ('actor','critic','regularizer','total'):
        sides={src:{n:(sum(sums[src+'_'+t][n] for t in ('actor','critic','regularizer')) if term=='total' else sums[src+'_'+term][n]) for n,p in params} for src in ('real','imagined')}
        result['geometry'][term]={}
        for group,predicate in (('all',lambda n:True),('policy',lambda n:n.startswith('vla.')),('value',lambda n:not n.startswith('vla.'))):
            a={n:g for n,g in sides['real'].items() if predicate(n)};b={n:g for n,g in sides['imagined'].items() if predicate(n)}
            if a:result['geometry'][term][group]=gradient_geometry(a,b)
    save(out/f'{key}-gradient-results.json',result)
    torch.save(sums,out/f'{key}-source-gradients.pt')
    return result
