"""Read-only archive checks. No training imports, RPCs or CUDA initialization."""
from pathlib import Path
import ast
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = ROOT / "source"
    manifest_path = source / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert digest(manifest_path) == "8fc7cad70ede2a08eb8615bd5faa106e43cac1569c5c9869cb9524a9fe4ba605"
    actual = {p.relative_to(source).as_posix() for p in source.rglob("*") if p.is_file()}
    assert actual == set(manifest["files"]) | {"source_manifest.json"}
    assert len(manifest["files"]) == 187
    for name, expected in manifest["files"].items():
        assert digest(source / name) == expected, name
    canonical = json.dumps(manifest["files"], sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(canonical).hexdigest() == manifest["source_sha256"]
    support = json.loads((ROOT / "runtime_provenance.json").read_text(encoding="utf-8"))
    for name, record in support["files"].items():
        assert digest(ROOT / name) == record["sha256"], name
        assert (ROOT / name).stat().st_size == record["bytes"], name
    count = 0
    for file in ROOT.rglob("*.py"):
        ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
        count += 1
    forbidden = {".pt", ".pth", ".ckpt", ".sqlite", ".db", ".pkl", ".safetensors", ".pyc"}
    secrets = re.compile(rb"-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|AKIA[0-9A-Z]{16}")
    for file in ROOT.rglob("*"):
        if not file.is_file():
            continue
        assert file.suffix not in forbidden, str(file)
        assert not file.name.startswith("events.out.tfevents"), str(file)
        assert not secrets.search(file.read_bytes()), str(file)
    publication = ROOT / "publication_manifest.json"
    if publication.exists():
        records = json.loads(publication.read_text(encoding="utf-8"))["files"]
        all_files = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*") if p.is_file()}
        assert all_files == set(records) | {"publication_manifest.json"}
        for name, expected in records.items():
            assert digest(ROOT / name) == expected, name
    print(json.dumps({"source_files": 187, "support_files": len(support["files"]),
                      "python_files_parsed": count, "source_sha256": manifest["source_sha256"],
                      "publication_manifest_verified": publication.exists(),
                      "training_or_gpu_calls": False, "passed": True}))


if __name__ == "__main__":
    main()
