from pathlib import Path
import os
import tempfile
import unittest
from inspect_world_backend import inspect, interpreter_entry


class InspectBackend(unittest.TestCase):
    def test_interpreter_entry_normalizes_path_without_resolving_venv(self):
        root = Path(tempfile.gettempdir()) / 'venv_identity_fixture'
        self.assertEqual(interpreter_entry(root / 'a' / '..' / 'a' / 'python'),
                         interpreter_entry(root / 'a' / 'python'))
        self.assertNotEqual(interpreter_entry(root / 'weaver' / 'bin' / 'python'),
                            interpreter_entry(root / 'merged' / 'bin' / 'python'))

    def test_two_venv_entries_sharing_one_binary_remain_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / 'base_python'
            binary.write_text('CPU path identity fixture only')
            entries = [root / name / 'bin' / 'python' for name in ('weaver', 'merged')]
            try:
                for entry in entries:
                    entry.parent.mkdir(parents=True)
                    entry.symlink_to(binary)
            except (OSError, NotImplementedError) as error:
                self.skipTest('Symbolic-link creation is unavailable: ' + str(error))
            self.assertEqual(entries[0].resolve(), entries[1].resolve())
            self.assertNotEqual(interpreter_entry(entries[0]), interpreter_entry(entries[1]))

    def test_ctrl_default_admitted_without_loading_models(self):
        result = inspect(Path(__file__).parent / "examples/ctrl_external.json")
        self.assertTrue(result["configuration_admitted"])
        self.assertEqual(result["service_plan"]["expected_gpu_model_processes"], 24)
        self.assertFalse(result["real_model_loaded"])

    def test_weaver_templates_are_honestly_blocked(self):
        for name, count in (("weaver_external.template.json", 24), ("weaver_integrated.template.json", 16)):
            result = inspect(Path(__file__).parent / "examples" / name)
            self.assertFalse(result["configuration_admitted"])
            self.assertEqual(result["status"], "blocked")
            self.assertEqual(result["service_plan"]["expected_gpu_model_processes"], count)
            self.assertTrue(result["errors"])


if __name__ == "__main__":
    unittest.main()
