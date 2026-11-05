import json
from pathlib import Path
import tempfile
import unittest

from checkpoint_evaluation import preserve_initial_checkpoint


class InitialCheckpointTests(unittest.TestCase):
    def test_later_atomic_publication_preserves_initial_bytes_and_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / 'checkpoints'
            source.mkdir()
            (source / 'latest_weights.pt').write_bytes(b'initial trained recipe weights')
            manifest = {'reason': 'initial', 'steps': {'policy': 0}}
            (source / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            saved = preserve_initial_checkpoint(source)
            for name, data in (('latest_weights.pt', b'final weights'), ('manifest.json', b'{"reason":"final"}')):
                temporary_file = source / (name + '.tmp')
                temporary_file.write_bytes(data)
                temporary_file.replace(source / name)
            self.assertEqual(Path(saved['initial_checkpoint']).read_bytes(), b'initial trained recipe weights')
            self.assertEqual(json.loads(Path(saved['manifest']).read_text()), manifest)

    def test_rejects_noninitial_baseline_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / 'checkpoints'
            source.mkdir()
            (source / 'latest_weights.pt').write_bytes(b'x')
            path = source / 'manifest.json'
            path.write_text(json.dumps({'reason': 'final', 'steps': {'policy': 10}}))
            with self.assertRaises(ValueError):
                preserve_initial_checkpoint(source)
            path.write_text(json.dumps({'reason': 'initial', 'steps': {'policy': 0}}))
            preserve_initial_checkpoint(source)
            with self.assertRaises(FileExistsError):
                preserve_initial_checkpoint(source)


if __name__ == '__main__':
    unittest.main()
