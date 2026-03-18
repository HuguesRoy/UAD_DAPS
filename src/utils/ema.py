import copy
import torch
import torch.nn as nn
from typing import Dict, Optional, Any


def copy_mod(mod: nn.Module) -> nn.Module:
    """Recursively copy a module (shallow for params, deep for children)."""
    new_mod = copy.copy(mod)
    new_mod._parameters = copy.copy(mod._parameters)
    new_mod._buffers = copy.copy(mod._buffers)
    new_mod._modules = {n: copy_mod(c) for n, c in mod._modules.items()}
    return new_mod


class EMA:
    """
    Exponential Moving Average for any nn.Module.
    Tracks both parameters and (optionally) buffers (e.g. BatchNorm stats).
    Compatible with PyTorch, MONAI, Lightning, etc.
    """

    def __init__(self, mu: float = 0.999, update_buffers: bool = True):
        self.mu = mu
        self.update_buffers = update_buffers
        self.shadow: Dict[str, torch.Tensor] = {}

    def register(self, module: nn.Module):
        """Initialize shadow weights (and buffers if enabled)."""
        device = torch.device("cpu")
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.detach().clone().to(device)

        if self.update_buffers:
            for name, buf in module.named_buffers():
                self.shadow[f"buffer:{name}"] = buf.data.clone()

    def update(self, module: nn.Module):
        """Update shadow weights using EMA formula."""
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.mu).add_(
                    param.data.detach().cpu(), alpha=1.0 - self.mu
                )

        if self.update_buffers:
            for name, buf in module.named_buffers():
                self.shadow[f"buffer:{name}"].data.copy_(buf.data)

    def apply_to(self, module: nn.Module):
        """Copy shadow weights into the given module (in-place)."""
        for name, param in module.named_parameters():
            if param.requires_grad and name in self.shadow:
                param.data.copy_(self.shadow[name].data)

        if self.update_buffers:
            for name, buf in module.named_buffers():
                key = f"buffer:{name}"
                if key in self.shadow:
                    buf.data.copy_(self.shadow[key].data)

    def copy_model(self, module: nn.Module) -> nn.Module:
        """Return a *copy* of the module with EMA weights applied."""
        module_copy = copy_mod(module)
        self.apply_to(module_copy)
        return module_copy

    def state_dict(self) -> Dict[str,Any]:
        """Return a serializable state dictionary."""
        return {
            "mu": self.mu,
            "update_buffers": self.update_buffers,
            "shadow": {k: v.cpu() for k, v in self.shadow.items()},
        }

    def load_state_dict(self, state_dict: Dict[str,Any]):
        """Load the EMA state dictionary."""
        # Handle legacy case where only shadow weights were saved
        if all(isinstance(v, torch.Tensor) for v in state_dict.values()):
            self.shadow = {k: v.cpu() for k, v in state_dict.items()}
            return

        # Modern case: full EMA state dict
        self.mu = state_dict.get("mu", self.mu)
        self.update_buffers = state_dict.get("update_buffers", self.update_buffers)
        shadow = state_dict.get("shadow", {})
        self.shadow = {k: v.cpu() for k, v in shadow.items()}


class EMAHelper:
    def __init__(self, ema, module):
        self.ema = ema
        self.module = module
        self.backup = {}
    def __enter__(self):
        for n, p in self.module.named_parameters():
            if p.requires_grad:
                self.backup[n] = p.data.clone()
                p.data.copy_(self.ema.shadow[n])
    def __exit__(self, *args):
        for n, p in self.module.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.backup[n])
