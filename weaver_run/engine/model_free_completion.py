"""CPU-only evidence checks shared by the training and queue supervisors."""

ROLLOUT_TARGET_REASON = 'rollout_success_target_reached'
LEGACY_VALIDATION_REASONS = ('fixed_validation_plateau', 'fixed_validation_target_reached')
AUTOMATIC_REASONS = ('real_budget_exhausted', ROLLOUT_TARGET_REASON, *LEGACY_VALIDATION_REASONS)


def rollout_target_config():
    from model_free_success import WINDOW_SIZE, TARGET_SUCCESSES
    return {'window_size': WINDOW_SIZE, 'window_successes': TARGET_SUCCESSES}


def validate_automatic_completion(details, state, stop_flag=None):
    def require(condition, message):
        if not condition:
            raise RuntimeError('Model-free automatic completion: '+message)

    def natural(value):
        return type(value) is int and value >= 0

    reason=details.get('stop_reason')
    require(state.get('model_free') is True and details.get('model_free') is True,
            'mode evidence missing')
    require(not stop_flag and not state.get('error'), 'stop/fault evidence takes precedence')
    require(reason in AUTOMATIC_REASONS, 'unsupported stop reason')
    budget=details.get('budget_status') or {}
    fields=('max_env_steps','consumed_env_steps','available_env_steps','reset_steps',
            'reserved_env_steps','active_real','episodes_inflight',
            'conservative_unknown_env_steps','episodes_cancelled')
    require(all(natural(budget.get(key)) for key in fields), 'incomplete budget counters')
    require(budget.get('initialized') is True and not budget.get('failure'), 'budget not healthy')
    require(all(budget[key] == 0 for key in fields[4:]), 'real collection not cleanly drained')
    require(budget['max_env_steps'] > 0
            and budget['max_env_steps'] == state.get('pipeline_config',{}).get('real_env_step_budget')
            and budget['consumed_env_steps']+budget['available_env_steps'] == budget['max_env_steps'],
            'budget does not match frozen capacity')
    receipt = {'stop_reason': reason}
    if ('model_free_success_target' in state or 'model_free_success' in details
            or reason == ROLLOUT_TARGET_REASON):
        from model_free_success import validate_model_free_success
        require(state.get('model_free_success_target') == rollout_target_config(),
                'frozen rollout target differs')
        require(reason not in LEGACY_VALIDATION_REASONS, 'fixed validation is diagnostic')
        try:
            snapshot = validate_model_free_success(details.get('model_free_success'))
        except ValueError as exc:
            raise RuntimeError('Model-free automatic completion: invalid rollout statistics') from exc
        require(snapshot['run_started_ns'] == state.get('started_ns') == details.get('run_started_ns'),
                'rollout statistics belong to a different run')
        require(snapshot['total_env_steps'] + snapshot['total_episodes_processed'] * budget['reset_steps']
                <= budget['consumed_env_steps'], 'rollout actions and resets exceed consumed budget')
        require(('99' in snapshot['crossings']) == (reason == ROLLOUT_TARGET_REASON),
                'rollout crossing disagrees with stop decision')
        require(natural(details.get('global_step')), 'missing final Policy step')
        receipt['model_free_success'] = snapshot
    if reason == 'real_budget_exhausted':
        require(budget.get('budget_exhausted') is True
                and budget['available_env_steps'] <= budget['reset_steps'], 'budget is not exhausted')
    elif reason in LEGACY_VALIDATION_REASONS:
        target = reason == 'fixed_validation_target_reached'
        summary=details.get('fixed_validation') or {}
        config=summary.get('config') or {}
        require(config == state.get('fixed_validation_config'), 'validation configuration changed')
        require(all(natural(config.get(key)) for key in
                    ('interval_env_steps','patience','min_env_steps','min_delta_successes'))
                and config['interval_env_steps'] > 0 and config['patience'] > 0
                and 1 <= config['min_delta_successes'] <= 50, 'invalid plateau configuration')
        require(summary.get('enabled') is True and summary.get('case_count') == 50
                and summary.get('target_reached' if target else 'plateau_reached') is True,
                'target decision missing' if target else 'plateau decision missing')
        require(all(natural(summary.get(key)) for key in
                    ('completed_validations','post_baseline_validations','stale_count','best_successes','best_policy_step')),
                'incomplete validation counters')
        require(summary['completed_validations'] == summary['post_baseline_validations']+1,
                'validation counts do not match baseline')
        if not target:
            require(config['patience'] <= summary['stale_count'] <= summary['post_baseline_validations'],
                    'patience was not reached')
        baseline,last=summary.get('baseline') or {},summary.get('last') or {}
        for row in (baseline,last):
            require(all(natural(row.get(key)) for key in ('successes','cases','policy_step','consumed_env_steps'))
                    and row['cases'] == 50 and row['successes'] <= 50, 'incomplete fixed validation result')
        require(baseline['policy_step'] == 0 and baseline['consumed_env_steps'] <= last['consumed_env_steps'],
                'initial Policy baseline missing')
        require(natural(details.get('global_step')) and last['policy_step'] == details['global_step']
                and last['consumed_env_steps'] <= budget['consumed_env_steps'],
                'validation does not cover the final Policy/budget')
        if target:
            require(last['successes'] == 50, 'target requires all 50 completed cases to succeed')
        else:
            require(last['consumed_env_steps'] >= config['min_env_steps'],
                    'plateau does not cover the minimum budget')
            require(baseline['successes'] <= summary['best_successes'] <= 50
                    and summary['best_policy_step'] <= last['policy_step']
                    and last['successes'] < summary['best_successes']+config['min_delta_successes'],
                    'best validation result conflicts with plateau')
        from model_free_control import ModelFreeControl, ModelFreeControlConfig
        history=summary.get('validations')
        require(isinstance(history,list) and len(history) == summary['completed_validations'],
                'validation history missing')
        replay=ModelFreeControl(ModelFreeControlConfig(**config))
        try:
            for row in history:
                if replay.target_reached:
                    raise ValueError('Historical validation continued after its terminal full score')
                replay.observe(**row)
        except (TypeError,ValueError) as exc:
            raise RuntimeError('Model-free automatic completion: invalid validation history') from exc
        require(replay.summary() == summary, 'validation history disagrees with stop decision')
    return receipt
