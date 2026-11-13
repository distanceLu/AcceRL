"""CPU-only, frozen real-observation RM report. No inference or training imports.

The caller freezes checkpoints and collects independent real episodes. This
module cannot establish training-set independence from scalar rows alone.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
from numbers import Integral, Real

import numpy as np

_FIELDS = ("model", "episode_id", "initial_state_id", "frame_index", "episode_length",
           "success", "target", "prediction")
_WEIGHTINGS = ("frame_weighted", "equal_episode")


def _id(value, name):
    if isinstance(value, Integral) and not isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, str) and value.strip():
        return value
    raise ValueError(f"{name} must be a nonempty string or integer")


def _order(value):
    return (0 if isinstance(value, int) else 1, value)


def _ids(values, name):
    if values is None or isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be an explicit nonempty collection")
    try:
        result = [_id(v, name) for v in values]
    except TypeError as exc:
        raise ValueError(f"{name} must be an explicit collection") from exc
    if not result or len(set(result)) != len(result):
        raise ValueError(f"{name} must be nonempty without duplicate IDs")
    return set(result)


def _score(value, field, valid):
    if value is None and not valid:
        return None
    if not isinstance(value, Real) or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{field} must be a numeric scalar")
    value = float(value)
    if not np.isfinite(value):
        if valid:
            raise ValueError(f"Nonfinite valid {field}")
        return None  # Explicitly masked nonfinite values become JSON null.
    if valid and not 0. <= value <= 1.:
        raise ValueError(f"Valid {field} must be in [0,1]")
    return value


def _normalize(rows, valid):
    try:
        rows = list(rows)
    except TypeError as exc:
        raise ValueError("rows must be a nonempty iterable of mappings") from exc
    if not rows:
        raise ValueError("No rows supplied")
    mask = np.ones(len(rows), dtype=bool) if valid is None else np.asarray(valid)
    if mask.dtype != bool or mask.shape != (len(rows),) or not mask.any():
        raise ValueError("valid mask must be aligned, one-dimensional boolean with support")
    normalized, episodes, panels = [], {}, defaultdict(dict)
    for i, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"Row {i} must be a mapping")
        missing = set(_FIELDS) - row.keys()
        if missing:
            raise ValueError(f"Row {i} missing fields: {sorted(missing)}")
        model = row["model"]
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a nonempty string")
        eid = _id(row["episode_id"], "episode_id")
        state = _id(row["initial_state_id"], "initial_state_id")
        for field in ("frame_index", "episode_length"):
            if not isinstance(row[field], Integral) or isinstance(row[field], (bool, np.bool_)):
                raise ValueError(f"{field} must be an integer")
        frame, length = int(row["frame_index"]), int(row["episode_length"])
        if length < 1 or not 0 <= frame <= length:
            raise ValueError("frame_index must be in [0, episode_length], episode_length >= 1")
        if not isinstance(row["success"], (bool, np.bool_)):
            raise ValueError("success must be boolean")
        policy = row.get("policy_panel")
        if policy is not None and (not isinstance(policy, str) or not policy.strip()):
            raise ValueError("policy_panel must be a nonempty string when supplied")
        metadata = (state, length, bool(row["success"]), policy)
        if eid in episodes and episodes[eid] != metadata:
            raise ValueError(f"Inconsistent episode metadata for {eid!r}")
        episodes[eid] = metadata
        key = (eid, frame)
        if key in panels[model]:
            raise ValueError(f"duplicate (model, episode_id, frame_index): {(model, *key)!r}")
        r = dict(model=model, episode_id=eid, initial_state_id=state, frame_index=frame,
                 episode_length=length, success=bool(row["success"]), valid=bool(mask[i]),
                 target=_score(row["target"], "target", mask[i]),
                 prediction=_score(row["prediction"], "prediction", mask[i]))
        if policy is not None:
            r["policy_panel"] = policy
        normalized.append(r)
        panels[model][key] = r
    models = sorted(panels)
    reference = panels[models[0]]
    for model in models[1:]:
        if panels[model].keys() != reference.keys():
            raise ValueError(f"Model {model!r} has a different (episode_id, frame_index) panel")
        for key, r in panels[model].items():
            expected = reference[key]
            if r["valid"] != expected["valid"]:
                raise ValueError(f"Different valid mask across models at {key!r}")
            if r["valid"] and r["target"] != expected["target"]:
                raise ValueError(f"Different target across models at {key!r}")
    supported = {r["episode_id"] for r in reference.values() if r["valid"]}
    if supported != set(episodes):
        raise ValueError("Every requested episode needs valid frame support; cannot silently drop episodes")
    normalized.sort(key=lambda r: (r["model"], _order(r["episode_id"]), r["frame_index"]))
    return normalized, episodes, models


def _partition(episodes, calibration_ids, test_ids, calibration_states, test_states):
    state_mode = calibration_states is not None or test_states is not None
    if state_mode:
        if calibration_ids is not None or test_ids is not None:
            raise ValueError("Choose episode IDs or initial-state IDs, not both")
        cs = _ids(calibration_states, "calibration_initial_states")
        ts = _ids(test_states, "test_initial_states")
        if cs & ts:
            raise ValueError("Calibration and test initial-state sets must be disjoint")
        if cs | ts != {v[0] for v in episodes.values()}:
            raise ValueError("Initial-state partitions must cover exactly the supplied panel")
        cal = {eid for eid, v in episodes.items() if v[0] in cs}
        test = set(episodes) - cal
    else:
        cal, test = _ids(calibration_ids, "calibration_ids"), _ids(test_ids, "test_ids")
        if cal & test:
            raise ValueError("Calibration and test episode sets must be disjoint")
        if cal | test != set(episodes):
            raise ValueError("Episode partitions must cover exactly the supplied panel")
        cs, ts = ({episodes[eid][0] for eid in ids} for ids in (cal, test))
        if cs & ts:
            raise ValueError("Calibration/test initial-state overlap leaks related episodes")
    by_state = defaultdict(list)
    for eid, metadata in episodes.items():
        by_state[metadata[0]].append(eid)
    manifest = dict(calibration_ids=sorted(cal, key=_order), test_ids=sorted(test, key=_order),
                    calibration_initial_states=sorted(cs, key=_order), test_initial_states=sorted(ts, key=_order),
                    episodes_by_initial_state=[dict(initial_state_id=s, episode_ids=sorted(by_state[s], key=_order),
                                                    split="calibration" if s in cs else "test")
                                               for s in sorted(by_state, key=_order)])
    return cal, test, manifest


def _support(rows):
    return dict(frames=len(rows), episodes=len({r["episode_id"] for r in rows}),
                initial_states=len({r["initial_state_id"] for r in rows}))


def _weights(rows, weighting):
    if weighting == "frame_weighted":
        return np.full(len(rows), 1. / len(rows))
    counts = Counter(r["episode_id"] for r in rows)
    return np.array([1. / (len(counts) * counts[r["episode_id"]]) for r in rows])


def _mean(values, weights):
    # Shift before averaging so a flat decimal trace stays exactly flat.
    values = np.asarray(values, dtype=np.float64)
    return float(values[0] + np.dot(values - values[0], weights) / weights.sum())


def _errors(rows, constants):
    if not rows:
        return {w: dict(mse=None, bias=None, constant_mse=None, skill_vs_constant=None) for w in _WEIGHTINGS}
    y = np.array([r["target"] for r in rows])
    error = np.array([r["prediction"] for r in rows]) - y
    result = {}
    for weighting in _WEIGHTINGS:
        weights = _weights(rows, weighting)
        mse = _mean(error ** 2, weights)
        constant_mse = _mean((constants[weighting] - y) ** 2, weights)
        result[weighting] = dict(mse=mse, bias=_mean(error, weights), constant_mse=constant_mse,
                                 skill_vs_constant=1. - mse / constant_mse if constant_mse > 0. else None)
    return result


def _distribution(rows):
    p = np.array([r["prediction"] for r in rows])
    order = np.argsort(p, kind="stable")
    result = {}
    for weighting in _WEIGHTINGS:
        weights = _weights(rows, weighting)
        mean = _mean(p, weights)
        cdf = np.cumsum(weights[order])
        cdf /= cdf[-1]
        quantiles = {name: float(p[order[min(int(np.searchsorted(cdf, q, side="left")), len(p) - 1)]])
                     for name, q in (("p05", .05), ("p25", .25), ("p50", .5), ("p75", .75), ("p95", .95))}
        result[weighting] = dict(mean=mean, std=float(np.sqrt(_mean((p - mean) ** 2, weights))), quantiles=quantiles)
    return result


def _phase(row):
    # Right-closed thirds of the evaluated action horizon: 1..50,51..100,101..150.
    f, length = row["frame_index"], row["episode_length"]
    return "early" if 3 * f <= length else "middle" if 3 * f <= 2 * length else "late"


def _groups(rows, constants):
    result = []
    for outcome in ("all", "success", "failure"):
        for phase in ("all", "early", "middle", "late"):
            if outcome == phase == "all":
                continue  # The full panel already appears in headline.
            selected = [r for r in rows if (outcome == "all" or r["success"] == (outcome == "success"))
                        and (phase == "all" or _phase(r) == phase)]
            result.append(dict(outcome=outcome, phase=phase, support=_support(selected),
                               status="ok" if selected else "no_support",
                               reason=None if selected else "No valid held-out frames in this group",
                               **_errors(selected, constants)))
    return result


def _discrimination(rows):
    by_episode = defaultdict(list)
    for row in rows:
        by_episode[row["episode_id"]].append(row)
    positive, negative, pos_states, neg_states = [], [], set(), set()
    for trace in by_episode.values():
        scores, states = (positive, pos_states) if trace[0]["success"] else (negative, neg_states)
        scores.append(_mean([r["prediction"] for r in trace], np.ones(len(trace))))
        states.add(trace[0]["initial_state_id"])
    result = dict(status="ok", reason=None, successful_episodes=len(positive), failed_episodes=len(negative),
                  successful_initial_states=len(pos_states), failed_initial_states=len(neg_states),
                  episode_mean_auc=None, episode_mean_score_gap=None)
    if not positive or not negative:
        result.update(status="no_support", reason="Both success and failure episodes are required")
        return result
    p, n = np.array(positive)[:, None], np.array(negative)[None, :]
    result.update(episode_mean_auc=float(np.mean((p > n) + .5 * (p == n))),
                  episode_mean_score_gap=_mean(positive, np.ones(len(positive)))
                  - _mean(negative, np.ones(len(negative))))
    if min(len(pos_states), len(neg_states)) < 2:
        result.update(status="limited_support", reason="Fewer than two initial states in a success/failure class")
    return result


def _progression(rows):
    by_episode = defaultdict(list)
    for row in rows:
        if row["success"]:
            by_episode[row["episode_id"]].append(row)
    selected, squared_error, target_squared, by_trace = [], [], [], []
    for eid, trace in by_episode.items():
        if len(trace) < 2:
            continue
        y = np.array([r["target"] for r in trace])
        p = np.array([r["prediction"] for r in trace])
        yc, pc = y - y[0], p - p[0]
        yc -= yc.mean()
        pc -= pc.mean()
        err2, y2 = (pc - yc) ** 2, yc ** 2
        selected.extend(trace)
        squared_error.extend(err2)
        target_squared.extend(y2)
        by_trace.append(dict(episode_id=eid, frames=len(trace), centered_mse=float(err2.mean()),
                             centered_target_variance=float(y2.mean())))
    result = dict(status="ok" if selected else "no_support", support=_support(selected), by_episode=by_trace,
                  excluded_single_frame_episodes=sum(len(trace) == 1 for trace in by_episode.values()),
                  reason=None if selected else "Need successful episodes with at least two valid frames")
    for weighting in _WEIGHTINGS:
        mse = variance = skill = None
        if selected:
            weights = _weights(selected, weighting)
            mse, variance = _mean(squared_error, weights), _mean(target_squared, weights)
            skill = 1. - mse / variance if variance > 0. else None
            if variance == 0.:
                result.update(status="no_support", reason="No within-episode target variation for progression skill")
        result[weighting] = dict(centered_mse=mse, centered_target_variance=variance, skill_vs_flat=skill)
    return result


def rm_state_report(rows, calibration_ids=None, test_ids=None, *, calibration_initial_states=None,
                    test_initial_states=None, valid=None, include_oracle=False):
    """Return a JSON-safe report on a fixed real-observation panel, without I/O.

    Each row needs model (str), episode_id/initial_state_id (str or int),
    frame_index (0..episode_length), episode_length (positive action count),
    success (bool), target and prediction (finite [0,1] scalars). Optional
    policy_panel is retained. For a 150-action prefix, supply min(full_N,150)
    as episode_length and the observed prefix success/discounted target.

    Supply BOTH calibration_initial_states and test_initial_states (preferred),
    OR BOTH episode-ID collections. Partitions must be nonempty, disjoint,
    exhaustive, and initial-state disjoint even in episode-ID mode. Use the
    fixed predeclared states, e.g. STATES[::2] and STATES[1::2], for all models.
    Never choose a split after inspecting predictions or held-out outcomes.

    valid is an optional strictly boolean 1-D mask aligned with rows. Models
    must cover exactly the same keys, masks, targets and episode metadata,
    including calibration; duplicate keys fail. Each episode must retain a
    valid frame. Masked numeric NaN/Inf are retained as null in raw_rows.

    Each constant minimizes calibration MSE for its own weighting. Equal-
    episode weighting gives each episode equal total mass, including within
    groups (only episodes represented in that group). All groups use the SAME
    calibration constants. Bias is prediction minus target. Population std
    and inverse-empirical-CDF quantiles use the corresponding frame weights.

    AUC ranks episode-mean scores, counts ties as 1/2, and is descriptive.
    Progression compares episode-demeaned predictions and targets on successful
    traces with >=2 observed frames. No frame-independent significance or CIs
    are inferred. Oracle output, when requested, is only the held-out optimal
    CONSTANT-class error; state-dependent predictors may beat it.
    """
    if not isinstance(include_oracle, (bool, np.bool_)):
        raise ValueError("include_oracle must be boolean")
    normalized, episodes, models = _normalize(rows, valid)
    cal, test, split = _partition(episodes, calibration_ids, test_ids,
                                  calibration_initial_states, test_initial_states)
    for row in normalized:
        row["split"] = "calibration" if row["episode_id"] in cal else "test"
    reference = [r for r in normalized if r["model"] == models[0]]
    calibration = [r for r in reference if r["valid"] and r["episode_id"] in cal]
    held_out = [r for r in reference if r["valid"] and r["episode_id"] in test]
    constants = {w: _mean([r["target"] for r in calibration], _weights(calibration, w)) for w in _WEIGHTINGS}
    split.update(calibration_support=_support(calibration), test_support=_support(held_out),
                 panel_pairs=len(reference), masked_pairs=sum(not r["valid"] for r in reference))
    result = dict(schema_version=1, split=split, constant_reference=dict(fit_split="calibration", **constants),
                  models={}, raw_rows=normalized, notes=[
                      "Caller supplies frozen-checkpoint predictions on fresh independent real observations; "
                      "scalar rows cannot prove no overlap with training replay.",
                      "Continuous discounted-success targets and RM scores are not calibrated binary probabilities.",
                      "AUC uses episode-mean scores; frames are correlated, and policies sharing an initial state "
                      "are related. No independence-based significance or confidence intervals are reported.",
                      "Progression removes each successful episode's mean; flat predictions are the centered reference.",
                      "Groups use right-closed thirds of the supplied evaluated action horizon; missing groups "
                      "have no_support and null metrics. Single-class state support is reported explicitly.",
                  ])
    for model in models:
        selected = [r for r in normalized if r["model"] == model and r["split"] == "test" and r["valid"]]
        result["models"][model] = dict(support=_support(selected), headline=_errors(selected, constants),
                                       prediction_distribution=_distribution(selected),
                                       groups=_groups(selected, constants),
                                       success_failure=_discrimination(selected), progression=_progression(selected))
    if include_oracle:
        y = np.array([r["target"] for r in held_out])
        oracle = dict(deployable=False, scope="constant_class_lower_bound", fit_split="test")
        for weighting in _WEIGHTINGS:
            weights = _weights(held_out, weighting)
            value = _mean(y, weights)
            oracle[weighting] = dict(value=value, mse=_mean((value - y) ** 2, weights))
        result["held_out_oracle"] = oracle
    return result
