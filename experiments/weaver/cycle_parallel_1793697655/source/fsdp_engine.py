"""FSDP2 adapter for the existing Policy and Reward microstep contracts.

No loss, LR, batch, or Replay changes. Every backward reduces FP32 gradient
shards; accumulation never retains full replicated gradients between micros.
"""
import os

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.tensor import DTensor


class PolicyFSDPEngine(torch.nn.Module):
    backend = "fsdp2_full_shard"

    def __init__(self, module, *, micro_batch, accumulation_steps, policy_lr=None,
                 value_lr=None, kind="policy", learning_rate=None):
        super().__init__()
        if kind not in {"policy", "reward"}:
            raise ValueError("This adapter supports Policy and Reward owners only")
        self.kind = kind
        if dist.get_world_size() != 4:
            raise ValueError("FSDP requires an independent four-rank group")
        self.original_dtypes = {n: p.dtype for n, p in module.named_parameters()}
        self.trainable_names = [n for n, p in module.named_parameters() if p.requires_grad]
        # DeepSpeed BF16 Adam maintains FP32 master parameters. Keep the same
        # precision for the optimizer; FSDP casts all-gathered forward params.
        with torch.no_grad():
            for p in module.parameters():
                if p.requires_grad:
                    p.data = p.data.float()
            for value in list(module.parameters()) + list(module.buffers()):
                dist.broadcast(value.detach(), src=0)
        mesh = init_device_mesh("cuda", (4,))
        # HF 4.x DynamicCache inherits a fieldless dataclass. FSDP's generic
        # input-casting traversal reconstructs it empty at each decoder layer.
        # The original VLA forward already owns BF16 autocast; retain its input
        # objects and cache identity while still casting gathered parameters.
        policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                                      cast_forward_inputs=False)
        self.shard_units = []
        # Independent decoder/vision blocks bound transient all-gather memory.
        for name, child in list(module.named_modules()):
            if type(child).__name__ in {"LlamaDecoderLayer", "Block"}:
                fully_shard(child, mesh=mesh, mp_policy=policy, reshard_after_forward=True)
                self.shard_units.append(name)
        if not self.shard_units:
            raise RuntimeError("No known transformer blocks found for FSDP")
        fully_shard(module, mesh=mesh, mp_policy=policy, reshard_after_forward=True)
        self.module = module
        if kind == "policy":
            groups = [{"params": g["params"], "name": g["name"],
                       "lr": policy_lr if g["name"] == "policy" else value_lr}
                      for g in module.get_parameter_groups()]
        else:
            if learning_rate is None:
                raise ValueError("Reward FSDP needs its unchanged learning rate")
            groups = [{"params": [p for p in module.parameters() if p.requires_grad],
                       "lr": learning_rate}]
        # Match DeepSpeed FusedAdam defaults (not torch AdamW's 0.01 decay).
        self.optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8,
                                           weight_decay=0.0, foreach=False)
        self._micro_batch = micro_batch
        self._gas = accumulation_steps
        self._micros = 0
        self.global_steps = 0
        self.skipped_steps = 0
        self._last_grad_norm = None
        self._cpu_export = None
        self._cpu_full_state = None
        self.optimizer.zero_grad(set_to_none=True)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def train_batch_size(self):
        return self._micro_batch * self._gas * dist.get_world_size()

    def is_gradient_accumulation_boundary(self):
        return (self._micros + 1) % self._gas == 0

    def backward(self, loss):
        self.module.set_requires_gradient_sync(True)
        (loss / self._gas).backward()

    def get_global_grad_norm(self):
        return self._last_grad_norm

    @torch.no_grad()
    def step(self):
        self._micros += 1
        if self._micros % self._gas:
            return
        square = torch.zeros((), dtype=torch.float64, device="cuda")
        for p in self.module.parameters():
            if p.grad is not None:
                g = p.grad.to_local() if isinstance(p.grad, DTensor) else p.grad
                square += g.double().square().sum()
        dist.all_reduce(square)
        norm = square.sqrt()
        if not torch.isfinite(norm).item():
            # A true numerical error must stop the run, not pretend to update.
            raise FloatingPointError("Non-finite synchronized FSDP gradient")
        self._last_grad_norm = float(norm.item())
        scale = min(1.0, 1.0 / (self._last_grad_norm + 1e-6))
        for p in self.module.parameters():
            if p.grad is not None:
                p.grad.mul_(scale)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.global_steps += 1
        self._cpu_export = None
        self._cpu_full_state = None

    @torch.no_grad()
    def export_cpu_weights(self, kind, version):
        from training_topology import _selected_weights, _weight_hash
        if kind != self.kind or version != self.global_steps:
            raise ValueError("FSDP weight export does not match the successful update boundary")
        tensors = {}
        for name, value in _selected_weights(self.module, kind):
            # All four ranks enter this method, including at version zero.
            full = value.detach().full_tensor() if isinstance(value, DTensor) else value.detach()
            tensors[name] = full.to(device="cpu", dtype=self.original_dtypes.get(name, full.dtype),
                                    copy=True).contiguous()
            del full
        payload = {"kind": kind, "version": version, "tensors": tensors,
                   "hash": _weight_hash(sorted(tensors.items()), kind),
                   "bytes": sum(t.numel() * t.element_size() for t in tensors.values()),
                   "tensor_count": len(tensors)}
        if kind == "reward":
            # The original Reward checkpoint is full-state. Frozen parameters
            # stay out of the inference delta contract, but all ranks gather
            # them here so rank 0 never starts a checkpoint-only collective.
            state = dict(tensors)
            for name, value in self.module.state_dict().items():
                if name not in state:
                    full = value.detach().full_tensor() if isinstance(value, DTensor) else value.detach()
                    state[name] = full.to(device="cpu", dtype=self.original_dtypes.get(name, full.dtype),
                                          copy=True).contiguous()
                    del full
            self._cpu_full_state = state
        self._cpu_export = payload
        return payload

    def snapshot_weights(self, version):
        if self._cpu_export is None or self._cpu_export["version"] != version:
            raise RuntimeError("FSDP checkpoint needs the completed all-rank CPU export first")
        # Inference broadcasts include transient rotary caches; checkpoints
        # must contain only state_dict keys, matching the original format.
        # FSDP2 state_dict is sharded and does not gather parameters here.
        state_keys = self.module.state_dict().keys()
        values = self._cpu_full_state if self.kind == "reward" else self._cpu_export["tensors"]
        if values is None:
            raise RuntimeError("Reward checkpoint needs a completed full-state CPU export")
        saved = {n: t for n, t in values.items() if n in state_keys}
        if not set(self.trainable_names) <= set(saved):
            raise RuntimeError("FSDP snapshot is missing trainable parameters")
        if self.kind == "reward" and set(saved) != set(state_keys):
            raise RuntimeError("Reward full-state snapshot is missing frozen state")
        return {"state_dict": saved,
                "trainable_names": sorted(self.trainable_names), "full_state": self.kind == "reward"}

    def shard_report(self):
        params = list(self.module.parameters())
        return {"backend": self.backend, "model": self.kind, "units": self.shard_units,
                "global_parameter_elements": sum(p.numel() for p in params),
                "local_parameter_elements": sum(p.to_local().numel() if isinstance(p, DTensor)
                                                else p.numel() for p in params),
                "reshard_after_forward": True, "gradient_sync_every_microstep": True,
                "optimizer_master_dtype": "float32", "weight_decay": 0.0}
