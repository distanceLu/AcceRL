"""First 98% and 99% windows of completed, uncensored real training episodes.

StatsActor serializes observations from every worker and task. total_env_steps
counts executed actions in those episodes only, excluding reset/settling steps.
Timestamps are StatsActor observation times; elapsed_seconds starts at run launch.
"""
from collections import deque
from copy import deepcopy
import math


WINDOW_SIZE = 1000
TARGET_SUCCESSES = 990
_MILESTONES = (('98', 980), ('99', TARGET_SUCCESSES))


class ModelFreeSuccess:
    def __init__(self, run_started_ns):
        if type(run_started_ns) is not int or run_started_ns < 0:
            raise ValueError('run_started_ns must be a nonnegative integer')
        self.run_started_ns = run_started_ns
        self.observed_ns = run_started_ns
        self.outcomes = deque(maxlen=WINDOW_SIZE)
        self.task_ids = deque(maxlen=WINDOW_SIZE)
        self.successes = 0
        self.episodes = 0
        self.actions = 0
        self.crossings = {}

    def observe(self, success, actions, observed_ns, task_id=None):
        if success not in (0, 1) or type(actions) is not int or actions < 1:
            raise ValueError('Expected a binary real episode outcome and a positive action count')
        if type(observed_ns) is not int or observed_ns < self.observed_ns:
            raise ValueError('Episode observation clock moved backwards')
        if task_id is not None and (type(task_id) is not int or not 0 <= task_id < 10):
            raise ValueError('Spatial task ID must be in 0..9')
        if self.episodes and (task_id is not None) != bool(self.task_ids):
            raise ValueError('The task coverage of the rollout stream changed')
        if len(self.outcomes) == WINDOW_SIZE:
            self.successes -= self.outcomes[0]
        self.outcomes.append(int(success))
        if task_id is not None:
            self.task_ids.append(task_id)
        self.successes += self.outcomes[-1]
        self.episodes += 1
        self.actions += actions
        self.observed_ns = observed_ns
        # 逐回合锁定首次达标，两次日志之间短暂越线也能记住。
        if len(self.outcomes) == WINDOW_SIZE:
            for key, minimum in _MILESTONES:
                if key not in self.crossings and self.successes >= minimum:
                    self.crossings[key] = self._row()

    def _row(self):
        count = len(self.outcomes)
        result = dict(window_size=WINDOW_SIZE, window_episodes=count, window_successes=self.successes,
                    success_rate=self.successes / count if count else 0.,
                    total_episodes_processed=self.episodes, total_env_steps=self.actions,
                    observed_ns=self.observed_ns, run_started_ns=self.run_started_ns,
                    elapsed_seconds=(self.observed_ns - self.run_started_ns) / 1e9)
        if self.task_ids:
            tasks = {str(i): dict(episodes=0, successes=0) for i in range(10)}
            for task_id, outcome in zip(self.task_ids, self.outcomes, strict=True):
                task = tasks[str(task_id)]
                task['episodes'] += 1
                task['successes'] += outcome
            result['window_by_task'] = tasks
        return result

    def snapshot(self):
        return dict(self._row(), crossings=deepcopy(self.crossings))


def _validate_row(row):
    integer_fields = ('window_size', 'window_episodes', 'window_successes', 'total_episodes_processed',
                      'total_env_steps', 'observed_ns', 'run_started_ns')
    if not isinstance(row, dict) or any(type(row.get(key)) is not int or row[key] < 0
                                        for key in integer_fields):
        raise ValueError('Invalid model-free success counters or timestamps')
    episodes = row['window_episodes']
    successes = row['window_successes']
    total_episodes = row['total_episodes_processed']
    if (row['window_size'] != WINDOW_SIZE or episodes != min(WINDOW_SIZE, total_episodes)
            or successes > episodes or row['total_env_steps'] < total_episodes
            or (episodes == 0 and row['total_env_steps'] != 0)
            or row['observed_ns'] < row['run_started_ns']):
        raise ValueError('Inconsistent model-free success counters or timestamps')
    if 'window_by_task' in row:
        tasks = row['window_by_task']
        if (not isinstance(tasks, dict) or set(tasks) != {str(i) for i in range(10)}
                or any(not isinstance(v, dict) or type(v.get('episodes')) is not int
                       or type(v.get('successes')) is not int or not 0 <= v['successes'] <= v['episodes']
                       for v in tasks.values())
                or sum(v['episodes'] for v in tasks.values()) != episodes
                or sum(v['successes'] for v in tasks.values()) != successes):
            raise ValueError('Per-task episode counts disagree with the pooled rollout window')
    for key, expected in (('success_rate', successes / episodes if episodes else 0.),
                          ('elapsed_seconds', (row['observed_ns'] - row['run_started_ns']) / 1e9)):
        if type(row.get(key)) not in (int, float) or not math.isfinite(row[key]) or row[key] != expected:
            raise ValueError('Model-free success evidence disagrees with ' + key)


def _validate_order(earlier, later):
    episode_delta = later['total_episodes_processed'] - earlier['total_episodes_processed']
    action_delta = later['total_env_steps'] - earlier['total_env_steps']
    success_delta = later['window_successes'] - earlier['window_successes']
    if (earlier['run_started_ns'] != later['run_started_ns'] or earlier['observed_ns'] > later['observed_ns']
            or action_delta < episode_delta or abs(success_delta) > episode_delta
            or (episode_delta == 0 and (action_delta != 0 or earlier['observed_ns'] != later['observed_ns']))):
        raise ValueError('Inconsistent ordering of model-free success observations')


def validate_model_free_success(snapshot):
    """Verify the latest window and first crossing records before accepting a stop."""
    _validate_row(snapshot)
    crossings = snapshot.get('crossings')
    if (not isinstance(crossings, dict) or crossings.keys() - {'98', '99'}
            or ('99' in crossings and '98' not in crossings)):
        raise ValueError('Invalid model-free success crossing keys')
    previous = None
    for key, minimum in _MILESTONES:
        if key not in crossings:
            if snapshot['window_episodes'] == WINDOW_SIZE and snapshot['window_successes'] >= minimum:
                raise ValueError('Missing first model-free success crossing')
            continue
        row = crossings[key]
        _validate_row(row)
        if (row['window_episodes'] != WINDOW_SIZE or row['window_successes'] < minimum
                or (row['total_episodes_processed'] > WINDOW_SIZE and row['window_successes'] != minimum)):
            raise ValueError('Invalid first model-free success crossing')
        if previous is not None:
            _validate_order(previous, row)
        _validate_order(row, snapshot)
        previous = row
    first = crossings.get('98')
    if first and first['window_successes'] >= TARGET_SUCCESSES and crossings.get('99') != first:
        raise ValueError('Simultaneous model-free crossings require identical evidence')
    return snapshot
