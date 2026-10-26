"""Verify the archived IPC baseline without importing its runtime or using GPUs."""
from pathlib import Path
import ast
import hashlib
import json


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / 'source_manifest_v3.json').read_text(encoding='utf8'))
    if len(manifest) != 57:
        raise ValueError('Expected exactly 57 files in the historical source snapshot')
    source = (root / 'source_v3').resolve()
    syntax_checked = 0
    for name, expected in manifest.items():
        path = (source / name).resolve()
        if not path.is_relative_to(source):
            raise ValueError('Manifest path leaves source_v3')
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError('Frozen source mismatch: ' + name)
        if path.suffix == '.py':
            ast.parse(raw, filename=name)
            syntax_checked += 1
    support = root / 'host_support'
    support_manifest = json.loads((support / 'support_manifest.json').read_text(encoding='utf8'))
    for name, record in support_manifest['files'].items():
        path = (support / name).resolve()
        if not path.is_relative_to(support.resolve()):
            raise ValueError('Support manifest path leaves host_support')
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != record['sha256']:
            raise ValueError('Archived CPU support mismatch: ' + name)
        ast.parse(raw, filename=name)
    final = json.loads((root / 'evidence' / 'final_stop_audit_v3_summary.json').read_text(encoding='utf8'))
    if not final['passed'] or final['run_id'] != 'ipcpwmanual_1022_161523_17963ef5':
        raise ValueError('Final audit does not describe this baseline')
    if final['steps'] != {'policy':132,'ctrl':26,'reward':26}:
        raise ValueError('Final successful update counts differ')
    print(json.dumps({'passed':True,'frozen_source_files':len(manifest),
                      'python_syntax_checked':syntax_checked,'cpu_support_files':len(support_manifest['files']),
                      'runtime_imports':False,'gpu_work':False}, indent=2))


if __name__ == '__main__':
    main()
