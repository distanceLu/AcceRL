"""Preserve the initial immutable snapshot for later paired policy evaluation."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil


def preserve_initial_checkpoint(checkpoint_dir):
    source = Path(checkpoint_dir)
    manifest = json.loads((source / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('reason') != 'initial' or manifest.get('steps', {}).get('policy') != 0:
        raise ValueError('The baseline must be the untrained initial published snapshot')
    target = source.parent / 'checkpoints_initial'
    target.mkdir(exist_ok=False)
    # Publication replaces files atomically. A hardlink therefore keeps the
    # initial inode even when subsequent updates replace latest_weights.pt.
    for name in ('latest_weights.pt', 'manifest.json'):
        try:
            os.link(source / name, target / name)
        except OSError:
            shutil.copyfile(source / name, target / name)
        with (target / name).open('r+b') as handle:
            os.fsync(handle.fileno())
    return {'initial_checkpoint': str(target / 'latest_weights.pt'),
            'manifest': str(target / 'manifest.json')}
