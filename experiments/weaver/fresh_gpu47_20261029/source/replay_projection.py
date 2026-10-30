"""Materialize only a consumer's shard, preserving the common Replay draw.

Canonical images remain uint8 until preparation in the owning trainer. Float
inputs retain the legacy contract (including WEAVER's post-normalization resize).
This module does not import torch, Ray, or initialize a model/device.
"""
from __future__ import annotations

import numpy as np


def native_reward_cohort(records, specification=None):
    """Count only real labels from the shared1024 draw; never touch pixels."""
    if len(records) != 1024:
        raise ValueError('Native reward balancing requires the complete global1024 cohort')
    if (specification or {}).get('mode') == 'libero_discounted_success_value_v1':
        from fresh_reward_targets import dense_reward_cohort
        if any(sample.length != 40 for sample in records):
            raise ValueError('Native dense reward requires measured40-frame simulator windows')
        return dense_reward_cohort(records, specification)
    positives=0
    for sample in records:
        episode=sample.episode
        if episode.metadata.get('label_source') != 'libero_env_step' or sample.length != 40:
            raise ValueError('Native reward labels must come from measured40-frame simulator windows')
        labels=np.asarray(episode.success[sample.start+30:sample.start+40])
        if labels.dtype != np.bool_ or labels.shape != (10,):
            raise ValueError('Native reward selected ten labels must be actual boolean environment success')
        positives+=int(np.count_nonzero(labels))
    negatives=10240-positives
    active=positives>0 and negatives>0
    return {'protocol':'weaver_reward_cohort_v1','window_count':1024,'label_frames_per_window':10,
        'positive_count':positives,'negative_count':negatives,
        'positive_weight':float(negatives/positives) if active else 0.,'negative_weight':1.,
        'rm_active':active,'label_source':'libero_env_step','frame_slice':[30,40]}


def normalized_observations(observations):
    """Match image_to_tensor's float32 /255, *2, -1 operation order exactly."""
    source = np.asarray(observations)
    if source.dtype == np.uint8:
        result = source.astype(np.float32)
        result /= np.float32(255.0)
        result *= np.float32(2.0)
        result -= np.float32(1.0)
        return result
    if source.dtype != np.float32:
        raise ValueError("Replay images must be canonical uint8 or normalized float32")
    return source


def reward_last_primary(observations):
    """Accept compact [B,C,H,W] or legacy [B,T,V,C,H,W] Reward inputs."""
    source = np.asarray(observations)
    if source.ndim == 6:
        source = source[:, -1, 0]
    elif source.ndim != 4:
        raise ValueError("Reward observations must be projected images or full WM windows")
    if source.shape[1] != 3:
        raise ValueError("Reward primary images must contain three channels")
    # Do not simply hand canonical bytes to the VLM: the original float32
    # normalize/inverse/truncation roundtrip loses one LSB for some pixel values.
    return normalized_observations(source)


def materialize_window_shard(records, stats, rank, local_size, world_size, consumer):
    """Project one pinned ticket inside Replay, before any pixel RPC payload.

    `records` is a single global draw of lightweight episode window references.
    Call this for both consumer groups against the same draw and capture_meta.
    Reward never accesses world observations, actions, or backend targets.
    """
    from training_topology import shard_metadata

    if consumer not in ("ctrl", "reward", "combined"):
        raise ValueError("Unknown WM projection consumer")
    if (isinstance(world_size, bool) or not isinstance(world_size, int) or world_size < 1
            or isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < world_size
            or isinstance(local_size, bool) or not isinstance(local_size, int) or local_size < 1
            or len(records) != local_size * world_size):
        raise ValueError("Invalid window ticket shard dimensions")
    start, end = rank * local_size, (rank + 1) * local_size
    samples = records[start:end]
    metadata = {key: value for key, value in stats.items() if key != "backend_samples"}
    metadata["capture_meta"] = shard_metadata(stats["capture_meta"], start, end, rank, world_size)
    if metadata["capture_meta"]["sample_ids"] != [sample.sample_id for sample in samples]:
        raise ValueError("Pinned window identities differ from the reserved global draw")
    metadata["sample_stats_scope"] = "global_batch"
    if consumer == "reward":
        observations = np.stack([sample.reward_primary for sample in samples])
        if observations.ndim != 4 or observations.shape[1] != 3:
            raise ValueError("Reward projection must contain only last primary images")
        actions = np.empty((local_size, 0), dtype=np.float32)
        metadata["observation_projection"] = "last_primary"
    else:
        observations = np.stack([sample.world_observations for sample in samples])
        if observations.ndim != 6:
            raise ValueError("World projection must contain complete multi-view windows")
        actions = np.stack([sample.act for sample in samples])
        payloads = [getattr(sample, "backend_payload", {}) for sample in samples]
        if any(not isinstance(payload, dict) for payload in payloads):
            raise ValueError("WM backend payloads must be dictionaries")
        if any(payloads):
            metadata["backend_samples"] = payloads
        metadata["observation_projection"] = "full_window"
    if observations.dtype not in (np.dtype("uint8"), np.dtype("float32")):
        raise ValueError("Projected images must use the canonical or legacy image contract")
    rewards = np.asarray([sample.rew for sample in samples], dtype=np.float32)
    instructions = [sample.instruction for sample in samples]
    return observations, actions, rewards, instructions, metadata


def describe_window_shard(records, stats, rank, local_size, world_size, consumer="ctrl"):
    """Describe a pinned rank shard; never access or materialize pixel properties."""
    from training_topology import shard_metadata
    if consumer != "ctrl" or not 0 <= rank < world_size or len(records) != local_size * world_size:
        raise ValueError("Only a valid World rank can reserve a streaming WM shard")
    start, end = rank * local_size, (rank + 1) * local_size
    metadata = shard_metadata(stats["capture_meta"], start, end, rank, world_size)
    if metadata["sample_ids"] != [sample.sample_id for sample in records[start:end]]:
        raise ValueError("World stream identities differ from the pinned reservation")
    return {"capture_meta": metadata, "local_size": local_size, "consumer": consumer,
            "unique_episodes": stats.get('unique_episodes', 0),
            "fresh_episode_ratio": stats.get('fresh_episode_ratio', 0.),
            "meaning": "reserved_identifiers_not_training_consumption"}


def materialize_window_microshard(records, stats, rank, local_size, world_size, start, end):
    """Project exactly one World micro from its original common WM reservation."""
    from training_topology import shard_metadata
    if (any(isinstance(v, bool) or not isinstance(v, int) for v in (rank, local_size, world_size, start, end))
            or not 0 <= rank < world_size or local_size < 1
            or len(records) != local_size * world_size or not 0 <= start < end <= local_size):
        raise ValueError("Invalid streamed World micro range")
    first, last = rank * local_size + start, rank * local_size + end
    samples = records[first:last]
    metadata = {key: value for key, value in stats.items() if key != "backend_samples"}
    metadata["capture_meta"] = shard_metadata(stats["capture_meta"], first, last, rank, world_size)
    if metadata["capture_meta"]["sample_ids"] != [sample.sample_id for sample in samples]:
        raise ValueError("World micro differs from the common reserved sample slice")
    payloads = [sample.backend_payload for sample in samples]
    if not all(isinstance(payload, dict) and payload for payload in payloads):
        raise ValueError("Online WEAVER micro requires complete real state/action/frame metadata")
    metadata.update(backend_samples=payloads, sample_stats_scope="global_batch",
                    observation_projection="world_micro_full_window", micro_start=start, micro_end=end)
    observations = np.stack([sample.world_observations for sample in samples])
    if observations.dtype != np.uint8 or observations.ndim != 6 or observations.shape[1:4] != (40, 2, 3):
        raise ValueError("Online WEAVER micro must retain canonical uint8 [B,40,2,3,H,W]")
    actions = np.stack([sample.act for sample in samples])
    if actions.shape != (end - start, 40, 7):
        raise ValueError("Online WEAVER micro requires forty incoming LIBERO seven-axis controls")
    return observations, actions, np.asarray([s.rew for s in samples], dtype=np.float32), [s.instruction for s in samples], metadata
