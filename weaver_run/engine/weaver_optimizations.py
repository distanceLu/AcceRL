"""Explicit, bounded, opt-in WEAVER execution optimizations.

No optimizer, reward target, timestep schedule, NFE, RNG seed or replay budget
is changed by these options. An absent options mapping retains the legacy path.
"""
DEFAULTS = {
    'selected_frame_encoding': False, 'kv_cache': False, 'decode_mode': 'all',
    'feature_cache_max_bytes': 0, 'resident_rollout_state': False,
    'resident_max_sessions': 8, 'resident_max_bytes': 64 * 1024 * 1024,
    'validate_on_cpu': False, 'task_cache_entries': 0, 'safety_checks': 'legacy',
}


def optimization_options(config):
    raw = config.get('optimizations', {})
    if not isinstance(raw, dict) or set(raw) - set(DEFAULTS):
        raise ValueError('Unknown WEAVER optimization options')
    options = {**DEFAULTS, **raw}
    for key in ('selected_frame_encoding', 'kv_cache', 'resident_rollout_state', 'validate_on_cpu'):
        if type(options[key]) is not bool:
            raise ValueError(key + ' must be an explicit boolean')
    for key, low, high in [('feature_cache_max_bytes', 0, 256 * 1024 * 1024),
                           ('resident_max_bytes', 1, 256 * 1024 * 1024),
                           ('resident_max_sessions', 1, 32), ('task_cache_entries', 0, 256)]:
        if type(options[key]) is not int or not low <= options[key] <= high:
            raise ValueError(key + ' exceeds its finite cache bound')
    if options['decode_mode'] not in ('all', 'last') or options['safety_checks'] not in ('legacy', 'batched'):
        raise ValueError('Invalid WEAVER decode/safety execution mode')
    if options['feature_cache_max_bytes'] and not options['validate_on_cpu']:
        raise ValueError('Frozen features need canonical CPU validation')
    if options['selected_frame_encoding'] and not options['validate_on_cpu']:
        raise ValueError('Select images only after validating the full CPU window')
    if options['decode_mode'] == 'last' and any(config.get(k) != v for k, v in (
            ('reward_provider', 'world_head'), ('reward_semantics', 'delta_native_score'),
            ('terminal_rule', 'horizon_only'))):
        raise ValueError('Last-only images require native bounded scores and horizon-only termination')
    return options
