"""Publish immutable rollout images once and fetch only a rank's JIT suffixes."""
from dataclasses import dataclass, replace

import torch


@dataclass(frozen=True)
class StoredPixels:
    ref: object
    dtype: torch.dtype


def stage_segment_pixels(segment, put):
    staged = []
    for observation in segment.observations:
        pixels = observation['pixel_values']
        if pixels.device.type != 'cpu':
            raise ValueError('Staged JIT replay requires CPU rollout pixels')
        # NumPy travels through Ray's shared-memory buffers. Reinterpret the
        # bytes so bfloat16 keeps its exact bits through NumPy on older Torch.
        buffer = pixels.detach().contiguous().view(torch.uint8).numpy()
        staged.append(dict(observation, pixel_values=StoredPixels(put(buffer), pixels.dtype)))
    return replace(segment, observations=tuple(staged))


def resolve_shard_pixels(shard, get):
    selected = shard[0]
    contexts = {}
    starts = {}
    for obs in selected:
        if isinstance(obs.get('pixel_values'), StoredPixels):
            context = obs['_jit_segment']
            key = context['segment_id']
            contexts[key] = context
            starts[key] = min(starts.get(key, obs['_jit_index']), obs['_jit_index'])
    if not contexts:
        return shard, 0

    # materialize() attached a global-cohort suffix. Only this rank's first
    # selected position through its learning boundary needs physical pixels.
    needed = {}
    for key, context in contexts.items():
        stop = context['count'] + int(not context['terminated'][-1])
        for index in range(starts[key], stop):
            image = context['observations'][index]['pixel_values']
            needed[image.ref] = image.dtype
    buffers = get(list(needed))
    tensors = {ref: torch.from_numpy(buffer).view(dtype)
               for (ref, dtype), buffer in zip(needed.items(), buffers)}

    resolved = {}
    for key, context in contexts.items():
        observations = [None] * (context['count'] + 1)
        stop = context['count'] + int(not context['terminated'][-1])
        for index in range(starts[key], stop):
            obs = context['observations'][index]
            observations[index] = dict(obs, pixel_values=tensors[obs['pixel_values'].ref])
        resolved[key] = dict(context, observations=tuple(observations))

    observations = []
    for obs in selected:
        context = resolved.get(obs['_jit_segment']['segment_id'])
        if context is not None:
            pixels = context['observations'][obs['_jit_index']]['pixel_values']
            obs = dict(obs, pixel_values=pixels, _jit_segment=context)
        observations.append(obs)
    return (observations, *shard[1:]), len(needed)
