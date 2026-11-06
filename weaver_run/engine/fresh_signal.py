"""Finite, small per-sample provenance records without images or model tensors."""
from __future__ import annotations

import json
import math


SIGNAL_PROTOCOL = 'weaver_fresh_policy_signal_v1'


def _primitive(value, depth=0):
    if depth > 5:
        raise ValueError('FRESH signal metadata is too deeply nested')
    if value is None or type(value) in (str, int, bool):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError('FRESH signal metadata must be finite')
        return value
    if isinstance(value, dict) and len(value) <= 128 and all(type(k) is str for k in value):
        return {k: _primitive(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (tuple, list)) and len(value) <= 128:
        return [_primitive(v, depth + 1) for v in value]
    raise ValueError('FRESH telemetry accepts bounded primitive metadata only')


def record_signal(writer, *, phase, source, sample_id, **details):
    if phase not in ('generated', 'refreshed', 'consumed'):
        raise ValueError('Unknown FRESH signal phase')
    if source not in ('real', 'imagined'):
        raise ValueError('Unknown FRESH data source')
    if type(sample_id) is not str or not sample_id:
        raise ValueError('FRESH signal requires an actual sample identity')
    record = _primitive(dict(details, protocol=SIGNAL_PROTOCOL, phase=phase,
                             source=source, sample_id=sample_id))
    if len(json.dumps(record, allow_nan=False).encode('utf8')) > 32768:
        raise ValueError('FRESH signal telemetry exceeds its bounded record size')
    if writer is not None:
        writer.event('fresh_policy_signal', details=record)
    return record
