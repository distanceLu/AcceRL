import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import git_version


class SourceRevisionTests(unittest.TestCase):
    def test_only_a_full_commit_on_the_verified_branch_can_be_replayed(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {'GIT_CONFIG_COUNT':'0'}):
            repo = Path(folder)
            def git(*args):
                return subprocess.check_output(['git', '-C', folder, '-c', 'user.name=test',
                    '-c', 'user.email=test@invalid', *args], text=True).strip()
            git('init', '-q')
            git('commit', '--allow-empty', '-qm', 'base')
            base = git('rev-parse', 'HEAD')
            git('commit', '--allow-empty', '-qm', 'production')
            head = git('rev-parse', 'HEAD')
            git('checkout', '-q', '--detach', base)
            git('commit', '--allow-empty', '-qm', 'unpublished')
            other = git('rev-parse', 'HEAD')
            self.assertEqual(git_version.source_revision(repo, head, None), head)
            self.assertEqual(git_version.source_revision(repo, head, base), base)
            for requested in ('HEAD~1', head[:7], other, git('rev-parse', head+'^{tree}')):
                with self.assertRaises(ValueError):
                    git_version.source_revision(repo, head, requested)
