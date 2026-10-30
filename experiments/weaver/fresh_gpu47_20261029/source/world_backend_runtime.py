"""CPU-only orchestration for the isolated native WEAVER training owner.

World tensors and optimizer state never enter the controller's Torch runtime.
The final monolith deliberately contains Policy/Reward only; native World
weights retain their own strict WEAVER-env audit and artifact identity.
"""
from pathlib import Path
import json
import re


def isolated_online_weaver(config):
    selected = config.backend == 'weaver' and config.world_trainable
    if selected and config.backend_options.get('transport') != 'isolated_online_ddp':
        raise ValueError('Online WEAVER requires its dedicated-interpreter native DDP bridge')
    return selected


def validate_online_runtime_arguments(config, args):
    if not isolated_online_weaver(config):return
    expected={'num_trainer_gpus':4,'ctrl_batch_size':4,'ctrl_accumulation_steps':64,
              'ctrl_train_interval':5}
    if config.reward_provider=='external_vlm':expected['reward_train_interval']=5
    else:expected['num_reward_inference_actors']=0
    for key,value in expected.items():
        actual=getattr(args,key,None)
        if type(actual) is not int or actual!=value:
            raise ValueError('Online WEAVER runtime requires %s=%s before actor construction' % (key,value))


def prune_online_world_snapshots(checkpoint_dir, descriptor, keep_recent=2):
    """Prune only exact native filenames after the referring manifest commits.

    The initial file and current manifest's artifact are always retained. Any
    unknown file, symlink, mismatch or absent committed manifest is untouched.
    This does not delete old snapshots before the new P/R save succeeds.
    """
    root=Path(checkpoint_dir).resolve()
    folder=root/'checkpoints_world'
    result={'pruned':[],'retained':[],'status':'not_committed_or_missing'}
    manifest=root/'manifest.json'
    if not manifest.is_file() or not folder.is_dir() or folder.is_symlink():return result
    saved=json.loads(manifest.read_text(encoding='utf-8')).get('metadata',{}).get('online_world')
    if saved != descriptor:return result
    current=Path(descriptor['weights_file'])
    if (not current.is_absolute() or current.is_symlink() or current.resolve().parent!=folder
            or folder.resolve()!=folder):
        raise ValueError('Native retention artifact must be a real file within checkpoints_world')
    pattern=re.compile(r'^world_p(\d{8})_w(\d{8})_(initial|periodic|final|final_before_first_update)\.pt$')
    files=[]
    for file in folder.iterdir():
        match=pattern.fullmatch(file.name)
        if match and file.is_file() and not file.is_symlink() and file.resolve().parent==folder:
            files.append((int(match[1]),int(match[2]),match[3],file))
    recent={row[3] for row in sorted(files,key=lambda r:(r[0],r[1],r[2]),reverse=True)[:keep_recent]}
    for policy_step,world_step,reason,file in files:
        if reason=='initial' or file==current or file in recent:
            result['retained'].append(file.name)
        else:
            # Recheck the absolute target immediately before the one-file unlink.
            if file.is_symlink() or file.resolve().parent!=folder:raise ValueError('Native retention target changed')
            file.unlink()
            result['pruned'].append(file.name)
    result['status']='committed_native_retention_applied'
    return result


def save_online_split_snapshot(training_groups, config, checkpoint_dir, reason, metadata,
                               *, ray_get, save_split_weights):
    """Save P/R in their native runtime and reference the separate World file."""
    if not isolated_online_weaver(config):
        raise ValueError('Separate native World snapshots require online WEAVER')
    integrated=config.reward_semantics=='delta_native_score'
    kinds = ('policy',) if integrated else ('policy','reward')
    components = ray_get([training_groups[kind][0].export_snapshot_component.remote(kind)
                          for kind in kinds])
    boundaries = {component['policy_step'] for component in components}
    if len(boundaries) != 1:
        raise RuntimeError('Policy/Reward snapshot boundaries differ before native World save')
    policy_step = boundaries.pop()
    descriptor = ray_get(training_groups['ctrl'][0].export_native_snapshot.remote(
        str(Path(checkpoint_dir) / 'checkpoints_world'), reason, policy_step))
    if (not isinstance(descriptor, dict) or descriptor.get('protocol') != (
            'weaver_online_joint_weights_v1' if integrated else 'weaver_online_weights_v1')
            or descriptor.get('run_id') != metadata.get('run_id')
            or descriptor.get('world_size') != 4 or descriptor.get('policy_step') != policy_step
            or descriptor.get('successful_updates') != policy_step // 5
            or descriptor.get('model_version') != policy_step // 5
            or descriptor.get('tensor_count') != 866
            or descriptor.get('source_optimizer_steps') != 3200
            or descriptor.get('source_bundle_sha256') != config.backend_options['initial_checkpoint_sha256']
            or descriptor.get('weights_only') is not True):
        raise RuntimeError('Native WEAVER artifact differs from its common completed Policy boundary')
    if integrated and (descriptor.get('trained_parameter_families')!=['wm','rm']
            or type(descriptor.get('rm_updates')) is not int
            or not 0<=descriptor['rm_updates']<=policy_step//5):
        raise RuntimeError('Joint native reward family/update audit differs')
    # Retain the exact native descriptor rather than renaming audit fields.
    payload_metadata = dict(metadata, online_world=descriptor)
    result=save_split_weights(dict(zip(kinds, components)), checkpoint_dir, reason=reason,
                             metadata=payload_metadata, active_kinds=kinds)
    retention=prune_online_world_snapshots(checkpoint_dir,descriptor)
    return dict(result,native_retention=retention) if isinstance(result,dict) else result
