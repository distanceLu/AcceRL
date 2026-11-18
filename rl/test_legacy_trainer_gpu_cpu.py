"""Exercise the legacy trainer startup without Ray, CUDA, or model weights."""
import ast
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class StopAtDistributedInit(Exception):
    """Stop before the trainer opens the process group or loads model weights."""


class LegacyTrainerGpuTests(unittest.TestCase):
    uuid = "GPU-11111111-2222-3333-4444-555555555555"

    def check_setup(self, mask, cuda_uuid, expected_index, visible_count=1):
        source = Path(__file__).with_name("ds_libero_ppo_discrete.py").read_text(encoding="utf-8")
        trainer = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "TrainerActor")
        setup = next(n for n in trainer.body if n.name == "setup_deepspeed_group")
        accelerator = SimpleNamespace(_get_nvml_gpu_id=lambda local: int(mask.split(",")[local]))
        cuda = SimpleNamespace(device_count=lambda: visible_count,
                               get_device_properties=lambda local: SimpleNamespace(uuid=cuda_uuid))

        def distributed_init(dist_backend):
            self.assertEqual(accelerator._get_nvml_gpu_id(0), expected_index)
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], mask)
            raise StopAtDistributedInit()

        modules = {"deepspeed.accelerator": SimpleNamespace(get_accelerator=lambda: accelerator)}
        namespace = dict(os=os, torch=SimpleNamespace(cuda=cuda),
                         deepspeed=SimpleNamespace(init_distributed=distributed_init))
        exec(compile(ast.Module(body=[setup], type_ignores=[]), "<trainer-setup>", "exec"), namespace)
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": mask}), patch.dict("sys.modules", modules):
            namespace["setup_deepspeed_group"](SimpleNamespace(rank=2, world_size=3), "127.0.0.1", "9099")

    def test_uuid_uses_verified_gpu_index_before_distributed_init(self):
        def query(command, **kwargs):
            self.assertIn("--id=" + self.uuid, command)
            self.assertIn("--query-gpu=index", command)
            return "6\n"

        with patch("subprocess.check_output", side_effect=query), self.assertRaises(StopAtDistributedInit):
            self.check_setup(self.uuid, self.uuid[4:], 6)

    def test_uuid_mismatch_or_multiple_visible_devices_rejected(self):
        with patch("subprocess.check_output", side_effect=AssertionError("unexpected NVML query")):
            for cuda_uuid, count in (("aaaaaaaa-2222-3333-4444-555555555555", 1), (self.uuid[4:], 2)):
                with self.subTest(cuda_uuid=cuda_uuid, count=count), self.assertRaises(RuntimeError):
                    self.check_setup(self.uuid, cuda_uuid, 6, count)

    def test_numeric_mask_preserves_original_mapper(self):
        with patch("subprocess.check_output", side_effect=AssertionError("unexpected NVML query")):
            with self.assertRaises(StopAtDistributedInit):
                self.check_setup("7", None, 7)


if __name__ == "__main__":
    unittest.main()
