"""Bind the training driver and its local Ray workers to the same checkout."""
import os
from pathlib import Path


ENGINE = Path(__file__).resolve().parent
CHECKOUT = ENGINE.parent.parent


def training_pythonpath(inherited):
    paths = [str(ENGINE), str(CHECKOUT), *inherited.split(os.pathsep)]
    return os.pathsep.join(dict.fromkeys(path for path in paths if path))


def verify_policy_imports(utils, model, *, compile_actions):
    paths = {}
    for module in (utils, model):
        actual = Path(module.__file__).resolve()
        expected = CHECKOUT.joinpath(*module.__name__.split('.')).with_suffix('.py')
        if actual != expected:
            raise RuntimeError(f'Policy source mismatch for {module.__name__}: expected {expected}, imported {actual}')
        paths[module.__name__] = str(actual)
    if compile_actions:
        from inspect import signature
        for owner, name in ((utils, 'PromptTokenCache'), (model.ActorCritic, 'real_action_vision'),
                            (model.ActorCritic, 'real_action_from_patches')):
            if not callable(getattr(owner, name, None)):
                raise RuntimeError(f'Compiled action inference requires {name} in {paths}')
        if 'prompt_cache' not in signature(utils.prepare_one_obs).parameters:
            raise RuntimeError(f'Compiled action inference requires prepare_one_obs(prompt_cache=...) in {paths}')
    return paths
