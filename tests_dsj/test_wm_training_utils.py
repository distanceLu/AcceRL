import numpy as np
import pytest
import ast
from pathlib import Path

from rl.wm_training_utils import (
    cosine_warmup_lr,
    frame_align_transition_actions,
    stable_episode_fraction,
)


def test_frame_align_transition_actions_matches_offline_convention():
    transitions = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    aligned = frame_align_transition_actions(transitions, num_observations=3)
    np.testing.assert_array_equal(aligned[0], np.zeros(2, dtype=np.float32))
    np.testing.assert_array_equal(aligned[1:], transitions)


def test_frame_align_transition_actions_rejects_length_mismatch():
    with pytest.raises(ValueError, match="Expected 2 transition actions"):
        frame_align_transition_actions(np.zeros((1, 7), np.float32), 3)


def test_episode_split_fraction_is_stable_and_bounded():
    value = stable_episode_fraction("worker3:seed123")
    assert value == stable_episode_fraction("worker3:seed123")
    assert 0.0 <= value < 1.0


def test_cosine_warmup_uses_optimizer_update_horizon():
    peak = 1e-6
    assert cosine_warmup_lr(0, peak, warmup_steps=100, total_steps=1500) == 0.0
    assert cosine_warmup_lr(50, peak, warmup_steps=100, total_steps=1500) == pytest.approx(5e-7)
    assert cosine_warmup_lr(100, peak, warmup_steps=100, total_steps=1500) == pytest.approx(peak)
    assert cosine_warmup_lr(1500, peak, warmup_steps=100, total_steps=1500) == pytest.approx(0.0)


def test_ctrl_ema_state_is_initialized_on_ctrl_actor_only():
    source_path = Path(__file__).with_name("ds_wm_discrete_ctrl_train_wm.py")
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    assignments = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        attrs = set()
        for child in ast.walk(node):
            if isinstance(child, (ast.Assign, ast.AnnAssign)):
                targets = child.targets if isinstance(child, ast.Assign) else [child.target]
                for target in targets:
                    if (
                        isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "self"
                    ):
                        attrs.add(target.attr)
        assignments[node.name] = attrs
    assert {"ema_decay", "ema_initialized", "ema_updates"} <= assignments[
        "CtrlWorldInferenceActor"
    ]
    assert "ema_initialized" not in assignments["RewardInferenceActor"]
