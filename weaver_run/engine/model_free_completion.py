"""CPU-only evidence checks shared by the training and queue supervisors."""


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
    require(reason in ('real_budget_exhausted','fixed_validation_plateau'), 'unsupported stop reason')
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
    if reason == 'real_budget_exhausted':
        require(budget.get('budget_exhausted') is True
                and budget['available_env_steps'] <= budget['reset_steps'], 'budget is not exhausted')
    else:
        summary=details.get('fixed_validation') or {}
        config=summary.get('config') or {}
        require(config == state.get('fixed_validation_config'), 'validation configuration changed')
        require(all(natural(config.get(key)) for key in
                    ('interval_env_steps','patience','min_env_steps','min_delta_successes'))
                and config['interval_env_steps'] > 0 and config['patience'] > 0
                and 1 <= config['min_delta_successes'] <= 50, 'invalid plateau configuration')
        require(summary.get('enabled') is True and summary.get('plateau_reached') is True
                and summary.get('case_count') == 50, 'plateau decision missing')
        require(all(natural(summary.get(key)) for key in
                    ('completed_validations','post_baseline_validations','stale_count','best_successes','best_policy_step')),
                'incomplete validation counters')
        require(summary['completed_validations'] == summary['post_baseline_validations']+1
                and config['patience'] <= summary['stale_count'] <= summary['post_baseline_validations'],
                'patience was not reached')
        baseline,last=summary.get('baseline') or {},summary.get('last') or {}
        for row in (baseline,last):
            require(all(natural(row.get(key)) for key in ('successes','cases','policy_step','consumed_env_steps'))
                    and row['cases'] == 50 and row['successes'] <= 50, 'incomplete fixed validation result')
        require(baseline['policy_step'] == 0 and baseline['consumed_env_steps'] <= last['consumed_env_steps'],
                'initial Policy baseline missing')
        require(last['policy_step'] == details.get('global_step')
                and config['min_env_steps'] <= last['consumed_env_steps'] <= budget['consumed_env_steps'],
                'plateau does not cover the final Policy/minimum budget')
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
                replay.observe(**row)
        except (TypeError,ValueError) as exc:
            raise RuntimeError('Model-free automatic completion: invalid validation history') from exc
        require(replay.summary() == summary, 'validation history disagrees with plateau decision')
    return {'stop_reason':reason}
