"""Explicit Policy-only mode; no model imports or World asset inspection."""
import json
from pathlib import Path

from pipeline_budget import PipelineBudgetConfig


def configure_training_mode(args):
    enabled = bool(getattr(args, 'model_free', False))
    if getattr(args, 'num_eval_workers', None) is None:
        args.num_eval_workers = 0 if enabled else 10
    if not enabled:
        return False
    if (args.num_trainer_gpus, args.num_inference_actors) != (4, 4):
        raise ValueError('Model-free mode requires four Policy trainer ranks and four inference replicas')
    if args.gpu_layout != 'four_rank_split_train':
        raise ValueError('Model-free mode requires the concurrent four-card split layout')
    args.num_rollout_workers = 4
    args.num_ctrl_inference_actors = args.num_reward_inference_actors = 0
    args.ctrl_eval_interval = 0
    if args.num_eval_workers == 0:
        args.eval_inference_replicas = ''
    elif not getattr(args, 'eval_inference_replicas', None):
        args.eval_inference_replicas = None
    return True


def pipeline_configuration(args):
    source = getattr(args, 'pipeline_config', None)
    if not getattr(args, 'model_free', False):
        return PipelineBudgetConfig.from_json(source)
    if isinstance(source, PipelineBudgetConfig):
        values = source.to_dict()
    elif isinstance(source, dict):
        values = dict(source)
    elif source:
        values = json.loads(Path(source).read_text(encoding='utf8'))
    else:
        values = {}
    values.update(model_free=True, real_workers=4, max_real_in_flight=4,
                  imagination_workers=0, max_imagine_in_flight=0,
                  inventory_refill_enabled=False, seed_reserve_per_bucket=0)
    return PipelineBudgetConfig.from_json(values)


def metric_enabled(tag, *, model_free=False, eval_workers=1):
    """An absent service has no measured zero: omit its metrics entirely."""
    if eval_workers == 0 and (tag.startswith('Eval/') or tag.startswith('System/Eval_')):
        return False
    if model_free and (tag.startswith(('System/WM_', 'Loss/World', 'Loss/Ctrl', 'Loss/Reward', 'WorldModel/',
                                      'Train/Learning_Rate/Ctrl', 'Train/Learning_Rate/Reward'))
                      or 'Imagine' in tag or tag in ('Performance/ctrl_train_time', 'Performance/reward_train_time')):
        return False
    return True
