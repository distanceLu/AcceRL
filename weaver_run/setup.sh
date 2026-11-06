#!/usr/bin/env bash
# Prepare environments owned by this checkout, using the observed Linux/Python 3.11 builds.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$root"
umask 077
export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES="" PIP_DISABLE_PIP_VERSION_CHECK=1
unset PYTHONPATH
mkdir -p .weaver-deps
exec 8>.weaver-deps/setup.lock
flock 8
py="${WEAVER_SETUP_PYTHON:-python3.11}"
"$py" -c 'import sys; assert sys.version_info[:2]==(3,11), "Python 3.11 required"'
for role in control native; do
    dir="$root/.venv-$role"
    if [[ -e "$dir" && ! -f ".weaver-deps/owned-$role" ]]; then
        echo "Existing $dir is unmanaged; specify WEAVER_CONTROL_PYTHON and WEAVER_NATIVE_PYTHON instead." >&2
        exit 2
    fi
    touch ".weaver-deps/owned-$role"
    [[ -x "$dir/bin/python" ]] || "$py" -m venv "$dir"
done
c="$root/.venv-control/bin/python"
n="$root/.venv-native/bin/python"
"$c" -m pip install torch==2.12.0 torchvision==0.27.0 --extra-index-url "${WEAVER_CONTROL_TORCH_INDEX:-https://download.pytorch.org/whl/cu130}"
"$n" -m pip install torch==2.7.0+cu126 torchvision==0.22.0+cu126 --extra-index-url "${WEAVER_NATIVE_TORCH_INDEX:-https://download.pytorch.org/whl/cu126}"
# The reference environment contains intentional dependency-version overrides.
# Install the recorded transitive set verbatim; leave Torch's CUDA wheels intact.
"$c" -m pip install --no-deps -r weaver_run/recipe/control-requirements.txt
"$n" -m pip install --no-deps -r weaver_run/recipe/native-requirements.txt
"$c" -m pip install --no-deps "git+https://github.com/moojink/transformers-openvla-oft.git@bc339d9ad707454c0c115970db43c260067c61ab" \
    "git+https://github.com/moojink/dlimp_openvla.git@040105d256bd28866cc6620621a3d5f7b6b91b46"
"$c" -m pip install --no-deps --no-build-isolation deepspeed==0.19.0 flash-attn==2.5.5
"$c" - <<'PY'
import hashlib
import importlib.metadata as md
from pathlib import Path
import shutil
import sys

root = Path.cwd()
env = root / ".venv-control"
assert Path(sys.prefix) == env and (root / ".weaver-deps/owned-control").is_file()
ds = md.distribution("deepspeed")
assert ds.version == "0.19.0"
for name, digest in {
    "runtime/engine.py": "310443345985c10ef9b5e8eea4176bb03d981e26fe3671054b331138c7ababdc",
    "runtime/zero/stage_1_and_2.py": "478209eee30499ed06aaa199f8aabe760cc176853e251cf929a0ce1333d68b9f",
}.items():
    source = root / "weaver_run/vendor/deepspeed" / name
    target = Path(ds.locate_file("deepspeed/" + name)).resolve()
    assert target.is_relative_to(env) and hashlib.sha256(source.read_bytes()).hexdigest() == digest
    if target.read_bytes() != source.read_bytes():
        shutil.copyfile(source, target)
PY
lib=".weaver-deps/LIBERO"
if [[ ! -d "$lib/.git" ]]; then
    staging="$(mktemp -d "$root/.weaver-deps/libero-clone-XXXXXX")"
    git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "$staging"
    git -C "$staging" checkout --detach 8f1084e3132a39270c3a13ebe37270a43ece2a01
    mv -T "$staging" "$lib"
fi
test "$(git -C "$lib" rev-parse HEAD)" = 8f1084e3132a39270c3a13ebe37270a43ece2a01
if ! git -C "$lib" apply --reverse --check "$root/weaver_run/libero-modern-torch.patch" 2>/dev/null; then
    git -C "$lib" apply "$root/weaver_run/libero-modern-torch.patch"
fi
"$c" -m pip install --no-deps -e "$lib"
touch .weaver-deps/ready
echo "Environments ready. Run ./run.sh check with your local assets."
