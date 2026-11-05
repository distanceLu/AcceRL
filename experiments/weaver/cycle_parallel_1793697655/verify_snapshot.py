"""CPU-only integrity check. Does not import training or launch subprocesses."""
import ast
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
publication = json.loads((root / 'publication_manifest.json').read_text(encoding='utf-8'))
for name, expected in publication['files'].items():
    path = root / name
    assert path.resolve().is_relative_to(root.resolve())
    assert sha(path) == expected, name
source = json.loads((root / 'source/source_manifest.json').read_text(encoding='utf-8'))
assert len(source['files']) == 267
for name, expected in source['files'].items():
    assert sha(root / 'source' / name) == expected, name
digest = hashlib.sha256(json.dumps(source['files'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
assert digest == source['source_sha256'] == publication['source_sha256']
identity = json.loads((root / 'run_identity.json').read_text(encoding='utf-8'))
assert sha(root / 'source/source_manifest.json') == identity['source_manifest_sha256']
assert identity['eval_workers_at_runtime'] == 10
assert '--num-eval-workers 10' in (root / 'source/run.sh').read_text(encoding='utf-8')
parsed = 0
for name in publication['files']:
    if name.endswith('.py'):
        ast.parse((root / name).read_text(encoding='utf-8'), filename=name)
        parsed += 1
print(json.dumps({'passed': True, 'source_files': len(source['files']), 'publication_files_verified': len(publication['files']), 'python_ast_parsed': parsed, 'source_sha256': digest, 'models_imported': False, 'gpu_work_launched': False}))
