"""Resolve policy modules in a fresh worker interpreter with a shadowing legacy package."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from training_topology import ray_init_options, SPLIT_LAYOUT


class PolicySourceTests(unittest.TestCase):
    def test_worker_uses_frozen_package_even_if_legacy_repo_is_inherited(self):
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as temp:
            legacy = Path(temp)/'rl'
            legacy.mkdir()
            (legacy/'__init__.py').touch()
            (legacy/'utils.py').touch()
            (legacy/'actor_critic_model_discrete.py').touch()
            with patch.dict(os.environ, PYTHONPATH=temp):
                env = ray_init_options([], SPLIT_LAYOUT, ('policy',))['runtime_env']['env_vars']
            code = ("from importlib.util import find_spec; "
                    "print(find_spec('rl.utils').origin); print(find_spec('rl.actor_critic_model_discrete').origin)")
            # Ray can put its driver's working directory first. A packaged
            # worker launched elsewhere must also resolve from its explicit env.
            with tempfile.TemporaryDirectory() as cwd:
                result = subprocess.check_output([sys.executable,'-c',code], cwd=cwd,
                                                  env={**os.environ,**env},text=True)
            self.assertEqual([Path(p) for p in result.splitlines()],
                             [root/'rl/utils.py',root/'rl/actor_critic_model_discrete.py'])
