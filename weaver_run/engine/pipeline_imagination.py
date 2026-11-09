"""One bounded imagination seed, preserving the original transition/GAE path."""
from __future__ import annotations

def imagine_ctrl_seed(self, job):
    import numpy as np
    import torch
    from capture_hooks import profile_rpc, lineage_begin, lineage_end, lineage_frame_parent
    from prismatic.vla.constants import ACTION_DIM
    source_episode_id, i = job['episode_id'], job['start']
    task_description, source_frame_groups = job['instruction'], job['frame_groups']
    self.capture_episode_id = source_episode_id
    self.local_buffer.clear()
    # Per start: add_imagine_reward() below accumulates this
    # count, so carrying it across starts over-counts.
    num_imagine_samples = 0
    seed_parents = lineage_frame_parent(source_frame_groups, i + self.num_step_cond - 1)
    lineage_begin(self, "imagined_seed", seed_parents,
                  episode_id=source_episode_id, observation_start=i,
                  observation_end_exclusive=i + self.num_step_cond)
    obs_tensor = torch.from_numpy(job['frames'].copy()).float() / 255.0 * 2.0 - 1.0
    action_window = torch.from_numpy(job['actions'].copy()).float()
    last_succ_prob = self.predict_rew_end(obs_tensor[-1], task_description)[0]

    latent_history, current_latent = profile_rpc(self, 'ctrl_infer', self.ctrl_world_infer, 'init_latent_state', obs_tensor)
    prefix_group = lineage_end(self)

    end = False
    for j in range(self.num_policy_chunks):
        lineage_begin(self, "imagined_transition", (prefix_group,),
                      episode_id=source_episode_id, seed_start=i, transition_index=j)
        inputs_t = self.obs2inp(obs_tensor[-1], task_description)
        act_norm, action_env, action_token, logits, value = profile_rpc(self, 'policy_infer', self.infer, 'request', inputs_t, deterministic=False, _capture_context={'workflow': 'imagined_action'})
        num_imagine_samples += 1
        chunk_reward = 0.0
        action_env = torch.from_numpy(
            np.asarray(action_env, dtype=np.float32)
        )
        stride = self.ctrl_world_infer_stride
        for group_start in range(0, len(action_env), stride):
            future_actions = action_env[group_start:group_start + stride]
            valid_count = len(future_actions)
            if valid_count < stride:
                pad_value = future_actions[-1:] if valid_count else torch.zeros(1, ACTION_DIM)
                future_actions = torch.cat(
                    [future_actions, pad_value.repeat(stride - valid_count, 1)],
                    dim=0,
                )
            action_condition = torch.cat(
                [
                    action_window[-(self.ctrl_world_num_history + 1):],
                    future_actions,
                ],
                dim=0,
            )
            future_obs, future_latents = self.predict_next_chunk(
                current_latent,
                latent_history,
                action_condition,
                task_description,
            )
            for k in range(valid_count):
                nxt = future_obs[k]
                next_latent = future_latents[k]
                latent_history = torch.cat(
                    [latent_history[1:], current_latent.unsqueeze(0)], dim=0
                )
                current_latent = next_latent
                obs_tensor = torch.roll(obs_tensor, -1, dims=0)
                obs_tensor[-1] = nxt
                action_window = torch.cat(
                    [
                        action_window[1:],
                        future_actions[k:k + 1],
                    ],
                    dim=0,
                )
                succ_prob, end = self.predict_rew_end(nxt, task_description)
                rew = succ_prob - last_succ_prob
                last_succ_prob = succ_prob
                chunk_reward += rew * self.reward_scale
                if end:
                    break
            if end:
                break
        prefix_group = lineage_end(self)
        capture_versions = dict(getattr(self,"capture_last_versions",{}))
        capture_versions["lineage_group_id"] = prefix_group
        self.local_buffer.append((inputs_t, action_token, chunk_reward, logits, value, capture_versions))
        if end:
            break
    if self.local_buffer:
        self.capture_bootstrap_lineage = ()
        if end:
            self._process_traj(self.local_buffer, 0.0)
        else:
            inputs_t = self.obs2inp(obs_tensor[-1], task_description)
            lineage_begin(self, "imagined_bootstrap", (prefix_group,),
                          episode_id=source_episode_id, seed_start=i)
            _, _, _, _, bootstrap_val = profile_rpc(self, 'policy_infer', self.infer, 'request', inputs_t, deterministic=False, _capture_context={'workflow': 'imagined_bootstrap'})
            self.capture_bootstrap_lineage = (lineage_end(self),)
            self._process_traj(self.local_buffer, bootstrap_val)
        imagine_rewards = [exp[2] for exp in self.local_buffer]
        avg_imagine_reward = sum(imagine_rewards) / len(imagine_rewards)
        self.stats_actor.add_imagine_reward.remote(avg_imagine_reward, self.wid, num_imagine_samples)
    self.local_buffer.clear()


def imagine_weaver_seed(worker, job):
    """Release the exact latest resident capability on every seed exit."""
    import sys
    from capture_hooks import profile_rpc
    state_owner = [None]
    try:
        return _imagine_weaver_seed_owned(worker, job, state_owner)
    finally:
        primary_error = sys.exc_info()[1]
        state = state_owner[0]
        if isinstance(state, dict) and '_resident' in state:
            try:
                profile_rpc(worker, 'ctrl_infer', worker.ctrl_world_infer, 'release_rollout', state,
                    _capture_context={'workflow': 'imagined_seed_release'})
            except Exception as error:
                # Publication legitimately clears an old version's registry.
                if 'Stale WEAVER rollout/cache version' not in str(error):
                    if primary_error is None:
                        raise
                    if hasattr(primary_error, 'add_note'):
                        primary_error.add_note('Resident release also failed: ' + str(error))


def _imagine_weaver_seed_owned(worker, job, state_owner):
    import visualization as visual_preview
    import numpy as np
    import torch
    from capture_hooks import profile_rpc, lineage_begin, lineage_end, lineage_frame_parent
    from world_backends.rollout import consume_predicted_chunk
    config = worker.world_config
    context_frames = worker.num_step_cond
    stride = worker.ctrl_world_infer_stride
    integrated = config.reward_provider == 'world_head'
    native_score=config.reward_semantics=='delta_native_score'
    if job.get('real_anchor_terminal'):
        worker.local_buffer.clear()
        return
    task_description, episode_id = job['instruction'], job['episode_id']
    start, frame_groups = job['start'], job['frame_groups']
    frame_start = int(job.get('frame_start', max(start, 0)))
    seed_anchor = int(job.get('seed_anchor', start + context_frames - 1))
    worker.capture_episode_id = episode_id
    worker.local_buffer.clear()
    num_imagine_samples = 0
    seed_parents = lineage_frame_parent(frame_groups, seed_anchor)
    lineage_begin(worker, "imagined_seed", seed_parents, episode_id=episode_id,
                  observation_start=frame_start, observation_end_exclusive=seed_anchor + 1,
                  context_start=start, context_padding_count=max(0,-start), seed_anchor=seed_anchor,
                  backend="weaver", reward_provider=config.reward_provider)
    cpu_admission = config.backend_options.get('optimizations', {}).get('validate_on_cpu')
    observations = torch.from_numpy((job['frames'][-1:] if cpu_admission else job['frames']).copy()).float() / 255.0 * 2.0 - 1.0
    action_mask = [True] * context_frames
    if start == 0:
        action_mask[0] = False
    action_mask = job.get('action_mask', action_mask)
    context = {"obs": observations, "states": job['states'], "actions": job['actions'],
               "action_mask": action_mask, "valid_mask": job.get('valid_mask', [True] * context_frames),
               "dt": job.get('dt', [config.schema['dt_seconds'] if flag else 0.0 for flag in action_mask]),
               "schema": dict(config.schema), "instruction": task_description,
               "episode_id": episode_id, "sample_id": f"{episode_id}:seed{frame_start}",
               "request_lineage": {"episode_id": episode_id, "seed_start": frame_start,
                                   "context_start":start,"seed_anchor":seed_anchor,
                                   "parent_group_ids": list(seed_parents)}}
    for name in ('frame_indices', 'context_padding'):
        if name in job:
            context[name] = job[name]
    if cpu_admission:
        # Canonical bytes cross the private CPU protocol; normalize only the
        # selected images in the native worker after whole-context admission.
        context['obs'] = job['frames']
        context['frame_start'] = frame_start
    state = profile_rpc(worker, "ctrl_infer", worker.ctrl_world_infer, "init_rollout", context,
                        _capture_context={"workflow": "imagined_seed", "world_outputs": ["seed_state", "reward_score"] if native_score else ["seed_state", "success"] if integrated else ["seed_state"]})
    state_owner[0] = state
    last_observation = observations[-1]
    last_probability = (float(state['initial_reward_score']) if native_score else float(state["initial_success_probability"]) if integrated
                        else worker.predict_rew_end(last_observation, task_description)[0])
    prefix_group = lineage_end(worker)
    end, stale = False, False
    preview = visual_preview.begin('imagined', worker.wid, f'{episode_id}:anchor{seed_anchor}',
        task=task_description, deterministic=False, source_episode_id=episode_id,
        seed_anchor=seed_anchor, reward_semantics=config.reward_semantics,
        sparse_decision_frames=True, reward_is_real_success=False,
        preview_fps=1.0, playback_note='Sparse decision snapshots at 1 fps; see exact action indices')
    preview.add(last_observation, 0, normalized=True, rm_score=float(last_probability))
    preview_actions, preview_world_versions, preview_policy_versions = 0, set(), set()
    for transition in range(worker.num_policy_chunks):
        lineage_begin(worker, "imagined_transition", (prefix_group,), episode_id=episode_id,
                      seed_start=frame_start, seed_anchor=seed_anchor,
                      context_start=start, transition_index=transition, backend="weaver")
        inputs = worker.obs2inp(last_observation, task_description)
        _, actions, action_token, logits, value = profile_rpc(
            worker, "policy_infer", worker.infer, "request", inputs, deterministic=False,
            _capture_context={"workflow": "imagined_action"})
        num_imagine_samples += 1
        actions = torch.from_numpy(np.asarray(actions, dtype=np.float32))
        preview_policy_version = getattr(worker, 'capture_last_versions', {}).get('policy')
        if preview_policy_version is not None:
            preview_policy_versions.add(int(preview_policy_version))
        chunk_reward = 0.
        phi_before = float(last_probability)
        served_version = state["version"]
        for offset in range(0, len(actions), stride):
            future_actions = actions[offset:offset + stride]
            try:
                result = profile_rpc(worker, "ctrl_infer", worker.ctrl_world_infer, "request", state,
                                     future_actions, task_description,
                                     _capture_context={"workflow": "imagined_generation",
                                                       "world_outputs": ["observations", "reward_score"] if native_score else ["observations", "success", "terminal"] if integrated else ["observations"]})
            except Exception as error:
                # Publication can invalidate an in-flight imagined seed.
                # Discard its unfinished targets; never mix two bundles.
                if "Stale WEAVER rollout/cache version" not in str(error):
                    raise
                stale = True
                break
            def external_classifier(obs):
                probability, terminal = worker.predict_rew_end(obs, task_description)
                if config.terminal_rule == "success_probability":
                    terminal = probability >= config.terminal_threshold
                return probability, terminal
            external = None if integrated else external_classifier
            reward, last_probability, end, last_observation, _, consumed = consume_predicted_chunk(
                result, last_probability, worker.reward_scale, external,config.reward_semantics)
            chunk_reward += reward
            state = result["next_state"]
            state_owner[0] = state
            served_version = result["served_version"]
            preview_actions += consumed
            preview_world_versions.add(served_version)
            preview.add(last_observation, preview_actions, normalized=True, force=True,
                policy_version=preview_policy_version, world_version=served_version,
                rm_score=float(last_probability), reward=float(reward))
            if end:
                break
        prefix_group = lineage_end(worker)
        if stale:
            worker.local_buffer.clear()
            break
        versions = dict(getattr(worker, "capture_last_versions", {}))
        versions.update({"lineage_group_id": prefix_group, "world_bundle_version": served_version})
        if integrated:
            # An output dependency on the same world RPC, never a
            # fictional Reward RPC, optimizer step, or publication.
            versions["world_reward_bundle_version"] = served_version
        versions.update(phi_before=phi_before,phi_after=float(last_probability),seed_anchor=seed_anchor)
        worker.local_buffer.append((inputs, action_token, chunk_reward, logits, value, versions))
        if end:
            break
    if stale or not worker.local_buffer:
        preview.finish(discarded=True, discard_reason='stale_version' if stale else 'empty',
            action_steps=preview_actions, policy_versions=sorted(preview_policy_versions),
            world_versions=sorted(preview_world_versions))
        return
    worker.capture_bootstrap_lineage = ()
    if end:
        endpoint_inputs = worker.obs2inp(last_observation, task_description)
        worker._process_traj(worker.local_buffer, 0.0, bootstrap_obs=endpoint_inputs,
            terminated=True,truncated=False,seed_anchor=seed_anchor)
    else:
        inputs = worker.obs2inp(last_observation, task_description)
        lineage_begin(worker, "imagined_bootstrap", (prefix_group,), episode_id=episode_id,
                      seed_start=frame_start, seed_anchor=seed_anchor, context_start=start)
        _, _, _, _, bootstrap = profile_rpc(worker, "policy_infer", worker.infer, "request", inputs,
                                            deterministic=False, _capture_context={"workflow": "imagined_bootstrap"})
        worker.capture_bootstrap_lineage = (lineage_end(worker),)
        worker._process_traj(worker.local_buffer, bootstrap, bootstrap_obs=inputs,
            terminated=False,truncated=True,seed_anchor=seed_anchor)
    mean_reward = sum(row[2] for row in worker.local_buffer) / len(worker.local_buffer)
    worker.stats_actor.add_imagine_reward.remote(mean_reward, worker.wid, num_imagine_samples)
    preview.finish(discarded=False, predicted_terminal=bool(end), action_steps=preview_actions,
        policy_versions=sorted(preview_policy_versions), world_versions=sorted(preview_world_versions),
        predicted_mean_chunk_reward=float(mean_reward))
    worker.local_buffer.clear()
