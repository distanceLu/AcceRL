"""In-place torch.compile for OpenVLA policy trainer blocks.

Compiles Llama decoder layers, ViT vision blocks, and the multimodal projector
in place via ``nn.Module.compile`` so parameter names, buffer names, DeepSpeed
ZeRO-2 gradient hooks, and CUDA IPC weight inventories remain byte-identical.
Variable-batch ``torch.no_grad`` context passes run under ``force_eager`` so
compiled graphs stay specialized for fixed training micro-batches.
"""
from __future__ import annotations

from contextlib import nullcontext
import os
from pathlib import Path
import tempfile

import torch


_SITE_CACHE_ROOT = Path("/mnt/data/lcx1/dong/WEAVER/.torch_cache")


def compile_trainer_enabled(env=None) -> bool:
    environ = os.environ if env is None else env
    explicit = environ.get("ACCERL_COMPILE_TRAINER")
    if explicit is not None:
        return explicit == "1"
    return environ.get("ACCERL_FAST_JIT_BOUNDARIES") == "1"


def configure_compile_cache(env=None, *, rank=None) -> dict:
    """Place Inductor and Triton caches on the data disk, never /home."""
    environ = os.environ if env is None else env
    if environ.get("TORCHINDUCTOR_CACHE_DIR") and environ.get("TRITON_CACHE_DIR"):
        cache_root = Path(environ["TORCHINDUCTOR_CACHE_DIR"]).parent
    elif _SITE_CACHE_ROOT.parent.is_dir():
        cache_root = _SITE_CACHE_ROOT
    elif environ.get("ACCERL_EXPERIMENT_ROOT"):
        cache_root = Path(environ["ACCERL_EXPERIMENT_ROOT"]).parent / ".torch_cache"
    else:
        cache_root = Path(tempfile.gettempdir()) / "accerl_torch_cache"

    scoped_root = cache_root / f"rank_{rank}" if rank is not None else cache_root
    inductor_dir = Path(environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(scoped_root / "inductor")))
    triton_dir = Path(environ.setdefault("TRITON_CACHE_DIR", str(scoped_root / "triton")))
    environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
    for directory in (inductor_dir, triton_dir):
        directory.mkdir(parents=True, exist_ok=True)
    return {
        "inductor_cache_dir": str(inductor_dir),
        "triton_cache_dir": str(triton_dir),
        "fx_graph_cache": environ.get("TORCHINDUCTOR_FX_GRAPH_CACHE", ""),
    }


def eager_inference_context():
    """Avoid recompiling trainer blocks on variable-sized no_grad context calls."""
    compiler = getattr(torch, "compiler", None)
    set_stance = getattr(compiler, "set_stance", None)
    if callable(set_stance):
        return set_stance("force_eager")
    return nullcontext()


def _unwrap_vla(base_model):
    vla = getattr(base_model, "vla", None)
    if vla is None:
        return None
    if hasattr(vla, "get_base_model") and callable(vla.get_base_model):
        try:
            inner = vla.get_base_model()
            if inner is not None:
                return inner
        except Exception:
            pass
    return vla


def _snapshot_inventory(module):
    params = tuple((name, tuple(p.shape), str(p.dtype), bool(p.requires_grad))
                   for name, p in module.named_parameters(recurse=True))
    buffers = tuple((name, tuple(b.shape), str(b.dtype))
                    for name, b in module.named_buffers(recurse=True))
    return params, buffers


def compile_policy_trainer(base_model, *, enabled=None, backend="inductor", mode="default", dynamic=False, rank=None):
    """Compile OpenVLA transformer and vision blocks in place without altering inventory."""
    active = compile_trainer_enabled() if enabled is None else bool(enabled)
    receipt = {
        "enabled": active,
        "compiled": False,
        "backend": backend,
        "mode": mode,
        "dynamic": dynamic,
        "llama_layers": 0,
        "vision_blocks": 0,
        "projector_compiled": False,
        "tf32_enabled": False,
        "reason": None,
    }
    if not active:
        receipt["reason"] = "disabled"
        return receipt
    if not hasattr(torch.nn.Module, "compile"):
        receipt["reason"] = "module_compile_unavailable"
        return receipt

    vla = _unwrap_vla(base_model)
    if vla is None:
        receipt["reason"] = "missing_vla"
        return receipt

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        receipt["tf32_enabled"] = True

    receipt.update(configure_compile_cache(rank=rank))
    try:
        import torch._dynamo as dynamo
        if getattr(dynamo.config, "cache_size_limit", 8) < 32:
            dynamo.config.cache_size_limit = 32
        if getattr(dynamo.config, "accumulated_cache_size_limit", 256) < 512:
            dynamo.config.accumulated_cache_size_limit = 512
    except Exception:
        pass

    before = _snapshot_inventory(base_model)
    compile_kwargs = dict(backend=backend, mode=mode, dynamic=dynamic)

    lm = getattr(vla, "language_model", None)
    lm_core = getattr(lm, "model", None) if lm is not None else None
    layers = getattr(lm_core, "layers", None) if lm_core is not None else None
    if layers is not None:
        for layer in layers:
            layer.compile(**compile_kwargs)
            receipt["llama_layers"] += 1

    vision = getattr(vla, "vision_backbone", None)
    if vision is not None:
        for tower_name in ("featurizer", "fused_featurizer"):
            tower = getattr(vision, tower_name, None)
            blocks = getattr(tower, "blocks", None) if tower is not None else None
            if blocks is None:
                continue
            for block in blocks:
                block.compile(**compile_kwargs)
                receipt["vision_blocks"] += 1

    projector = getattr(vla, "projector", None)
    if projector is not None and isinstance(projector, torch.nn.Module):
        projector.compile(**compile_kwargs)
        receipt["projector_compiled"] = True

    after = _snapshot_inventory(base_model)
    if before != after:
        raise RuntimeError("In-place trainer compilation altered model parameter or buffer inventory")

    receipt["compiled"] = bool(receipt["llama_layers"] or receipt["vision_blocks"] or receipt["projector_compiled"])
    if not receipt["compiled"]:
        receipt["reason"] = "no_compilable_blocks_found"
    return receipt
