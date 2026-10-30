"""WEAVER imagined rollouts using the existing policy, GAE and lineage methods.

Model history stays inside the world backend. Imports of tensor and capture
dependencies occur only when the live rollout entry point is called.
"""
from __future__ import annotations

import math
import random
from contextlib import contextmanager
import sys
from typing import Mapping

from .contracts import WorldBackendConfig


@contextmanager
def _resident_seed_lifetime(worker, rpc):
    """Release the latest capability on every seed exit, including continue."""
    latest=[None]
    try:
        yield latest
    finally:
        state=latest[0]
        if isinstance(state,Mapping) and '_resident' in state:
            primary=sys.exc_info()[1]
            try:
                rpc(worker,'ctrl_infer',worker.ctrl_world_infer,'release_rollout',state,
                    _capture_context={'workflow':'imagined_seed_release'})
            except Exception as error:
                if 'Stale WEAVER rollout/cache version' not in str(error):
                    if primary is None:
                        raise
                    if hasattr(primary,'add_note'):
                        primary.add_note('Resident release also failed: '+str(error))


def real_context_payload(observations, raw_observations, transition_actions, start,
                         schema, instruction, episode_id, request_lineage):
    """Pack measured incoming actions/states, including the masked reset anchor."""
    count = len(observations)
    if start < 0 or count < 1 or start + count > len(raw_observations):
        raise ValueError("Real context range falls outside the episode")
    states, actions, masks, intervals = [], [], [], []
    for frame in range(start, start + count):
        state = raw_observations[frame].get("state")
        if state is None or len(state) != schema["state_dim"]:
            raise ValueError("WEAVER context requires measured state matching the declared state schema")
        states.append([float(value) for value in state])
        action = ([0.] * schema["action_dim"] if frame == 0 else transition_actions[frame - 1])
        if len(action) != schema["action_dim"]:
            raise ValueError("WEAVER real action dimension differs from its schema")
        actions.append([float(value) for value in action])
        masks.append(frame != 0)
        intervals.append(schema["dt_seconds"] if frame != 0 else 0.)
    if any(not math.isfinite(v) for vector in states + actions for v in vector):
        raise ValueError("Real context states/actions must be finite")
    return {"obs": observations, "states": states, "actions": actions, "action_mask": masks,
            "valid_mask": [True] * count, "dt": intervals, "schema": dict(schema),
            "instruction": instruction, "episode_id": str(episode_id),
            "sample_id": "%s:seed%d" % (episode_id, start), "request_lineage": dict(request_lineage)}


def consume_predicted_chunk(result, initial_probability, reward_scale=1.0, external_reward=None,
                            reward_semantics='delta_success'):
    """Apply the legacy potential difference and stop at the first terminal.

    The returned state/frame correspond to the last *consumed* prediction. A
    backend may have generated a longer chunk; later predictions receive no
    reward and are never turned into policy experience after terminal.
    """
    if not isinstance(result, Mapping) or "observations" not in result or "predicted_states" not in result:
        raise ValueError("World rollout result is missing aligned observations/state")
    observations, states = result["observations"], result["predicted_states"]
    native_score=reward_semantics=='delta_native_score'
    from .sparse_output import sparse_last_geometry
    count=len(states)
    sparse=sparse_last_geometry(result,count,allowed=native_score and external_reward is None)
    if count < 1 or len(observations)!=(1 if sparse else count):
        raise ValueError("World prediction observations/state are not frame aligned")
    if sparse:
        mask,intervals=result.get('valid_mask'),result.get('dt')
        if (mask is None or len(mask)!=count or any(type(v.item() if hasattr(v,'item') else v) is not bool or not bool(v) for v in mask)
                or intervals is None or len(intervals)!=count
                or any(not math.isfinite(float(v)) or float(v)<=0 for v in intervals)):
            raise ValueError('Sparse native output requires full valid masks and dt')
    if native_score:
        if external_reward is not None or result.get('reward_semantics')!='delta_native_score':
            raise ValueError('Native reward score cannot call the removed external Reward or use legacy success outputs')
        probabilities=result.get('reward_predictions')
        terminals=result.get('terminal_predictions')
        if (probabilities is None or terminals is None or len(probabilities)!=count or len(terminals)!=count
                or any(bool(t) for t in terminals) or result.get('reward_calibrated') is not False):
            raise ValueError('Uncalibrated native reward requires bounded scores and horizon-only terminals')
        if sparse and any(type(t.item() if hasattr(t,'item') else t) is not bool for t in terminals):
            raise ValueError('Sparse native terminals must contain actual booleans')
        if 'success_probabilities' in result:
            raise ValueError('Native reward score is not a calibrated success probability')
    elif external_reward is None:
        probabilities, terminals = result.get("success_probabilities"), result.get("terminal_predictions")
        if probabilities is None or terminals is None or len(probabilities) != count or len(terminals) != count:
            raise ValueError("Integrated world rollout needs frame-aligned success and terminal outputs")
    previous = float(initial_probability)
    if not math.isfinite(previous) or not 0 <= previous <= 1 or not math.isfinite(float(reward_scale)):
        raise ValueError("Invalid initial success probability or reward scale")
    total = 0.
    for index in range(count):
        observation=observations[-1] if sparse else observations[index]
        probability, terminal = (external_reward(observation) if external_reward is not None
                                 else (probabilities[index], terminals[index]))
        probability = float(probability)
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Reward potential must be a finite bounded score/probability")
        total += (probability - previous) * reward_scale
        previous = probability
        if bool(terminal):
            return total, previous, True, observation, states[index], index + 1
    return total, previous, False, observations[-1], states[-1], count


def run_weaver_rollouts(worker):
    """Run the worker's WEAVER branch; retain its policy samples and GAE routine."""
    import numpy as np
    import torch
    from capture_hooks import profile_rpc, lineage_begin, lineage_end, lineage_frame_parent

    config = worker.world_config
    if not isinstance(config, WorldBackendConfig):
        config = WorldBackendConfig.from_mapping(config)
    if config.backend != "weaver":
        raise ValueError("run_weaver_rollouts requires backend='weaver'")
    geometry = profile_rpc(worker, "ctrl_infer", worker.ctrl_world_infer, "get_context_geometry")
    context_frames = max(worker.num_step_cond, int(geometry["required_context_frames"]))
    stride = int(geometry["horizon"])
    if stride < 1:
        raise ValueError("World backend returned an invalid prediction horizon")
    integrated = config.reward_provider == "world_head"
    native_score=config.reward_semantics=='delta_native_score'
    imagine_step = 0
    while True:
        if imagine_step % worker.real_traj_collect_interval == 0:
            worker.get_one_episode()
        imagine_step += 1
        episode = random.choice(worker.episodes)
        obs_list, reward_list, done_list, action_env_list, task_description, episode_id, frame_groups = episode
        worker.capture_episode_id = episode_id
        for start in range(len(obs_list) - context_frames):
            anchor=start+context_frames-1
            if raw_observations_terminal(obs_list[anchor]):
                continue
            worker.local_buffer.clear()
            num_imagine_samples = 0
            seed_parents = lineage_frame_parent(frame_groups, start + context_frames - 1)
            lineage_begin(worker, "imagined_seed", seed_parents, episode_id=episode_id,
                          observation_start=start, observation_end_exclusive=start + context_frames,
                          backend="weaver", reward_provider=config.reward_provider)
            observations = torch.stack([worker._obs_to_views(obs) for obs in obs_list[start:start + context_frames]])
            context = real_context_payload(observations, obs_list, action_env_list, start, config.schema,
                                           task_description, episode_id,
                                           {"episode_id": episode_id, "seed_start": start, "parent_group_ids": list(seed_parents)})
            with _resident_seed_lifetime(worker,profile_rpc) as resident_state:
                state = profile_rpc(worker, "ctrl_infer", worker.ctrl_world_infer, "init_rollout", context,
                                    _capture_context={"workflow": "imagined_seed", "world_outputs": ["seed_state", "reward_score"] if native_score else ["seed_state", "success"] if integrated else ["seed_state"]})
                resident_state[0] = state
                last_observation = observations[-1]
                last_probability = (float(state['initial_reward_score']) if native_score else float(state["initial_success_probability"]) if integrated
                                    else worker.predict_rew_end(last_observation, task_description)[0])
                prefix_group = lineage_end(worker)
                end, stale = False, False
                for transition in range(worker.num_policy_chunks):
                    lineage_begin(worker, "imagined_transition", (prefix_group,), episode_id=episode_id,
                                  seed_start=start, transition_index=transition, backend="weaver")
                    inputs = worker.obs2inp(last_observation, task_description)
                    _, actions, action_token, logits, value = profile_rpc(
                        worker, "policy_infer", worker.infer, "request", inputs, deterministic=False,
                        _capture_context={"workflow": "imagined_action"})
                    num_imagine_samples += 1
                    actions = torch.from_numpy(np.asarray(actions, dtype=np.float32))
                    chunk_reward = 0.
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
                        reward, last_probability, end, last_observation, _, _ = consume_predicted_chunk(
                            result, last_probability, worker.reward_scale, external,config.reward_semantics)
                        chunk_reward += reward
                        state = result["next_state"]
                        resident_state[0] = state
                        served_version = result["served_version"]
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
                    worker.local_buffer.append((inputs, action_token, chunk_reward, logits, value, versions))
                    if end:
                        break
                if stale or not worker.local_buffer:
                    continue
                worker.capture_bootstrap_lineage = ()
                if end:
                    worker._process_traj(worker.local_buffer, 0.0)
                else:
                    inputs = worker.obs2inp(last_observation, task_description)
                    lineage_begin(worker, "imagined_bootstrap", (prefix_group,), episode_id=episode_id, seed_start=start)
                    _, _, _, _, bootstrap = profile_rpc(worker, "policy_infer", worker.infer, "request", inputs,
                                                        deterministic=False, _capture_context={"workflow": "imagined_bootstrap"})
                    worker.capture_bootstrap_lineage = (lineage_end(worker),)
                    worker._process_traj(worker.local_buffer, bootstrap)
                mean_reward = sum(row[2] for row in worker.local_buffer) / len(worker.local_buffer)
                worker.stats_actor.add_imagine_reward.remote(mean_reward, worker.wid, num_imagine_samples)
                worker.local_buffer.clear()


def raw_observations_terminal(observation):
    """Only explicit real labels can reject a seed; a model score cannot."""
    return bool(observation.get('terminated',False) or observation.get('truncated',False)
                or observation.get('is_success',False))
