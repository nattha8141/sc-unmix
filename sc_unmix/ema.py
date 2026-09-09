"""Small, checkpointable exponential moving average for model weights."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterator

import torch
from torch import nn


class ExponentialMovingAverage:
    """Track a warm-started EMA without adding inference parameters.

    The warm-up decay prevents an initially random model from dominating the
    average during the first optimizer updates.  Shadows stay on the model's
    device; for this 3.45 M-parameter network the extra L4 memory is small.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must be in [0,1).")
        self.decay = float(decay)
        self.num_updates = 0
        self.shadow = {
            name: tensor.detach().clone() for name, tensor in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.num_updates += 1
        decay = min(self.decay, (1.0 + self.num_updates) / (10.0 + self.num_updates))
        current = model.state_dict()
        if current.keys() != self.shadow.keys():
            raise RuntimeError("Model state changed after EMA initialization.")
        for name, value in current.items():
            shadow = self.shadow[name]
            value = value.detach()
            if torch.is_floating_point(shadow):
                shadow.lerp_(value.to(dtype=shadow.dtype), 1.0 - decay)
            else:
                shadow.copy_(value)

    def state_dict(self) -> Dict[str, object]:
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "shadow": {
                name: tensor.detach().clone() for name, tensor in self.shadow.items()
            },
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        self.decay = float(state["decay"])
        self.num_updates = int(state["num_updates"])
        incoming = state["shadow"]
        if not isinstance(incoming, dict) or incoming.keys() != self.shadow.keys():
            raise ValueError("EMA checkpoint does not match model state.")
        for name, value in incoming.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"EMA entry {name!r} is not a tensor.")
            self.shadow[name].copy_(value.to(self.shadow[name].device))

    def averaged_model_state(self) -> Dict[str, torch.Tensor]:
        return {
            name: tensor.detach().cpu().clone() for name, tensor in self.shadow.items()
        }

    @contextmanager
    def average_parameters(self, model: nn.Module) -> Iterator[None]:
        original = {
            name: tensor.detach().clone() for name, tensor in model.state_dict().items()
        }
        model.load_state_dict(self.shadow, strict=True)
        try:
            yield
        finally:
            model.load_state_dict(original, strict=True)


__all__ = ["ExponentialMovingAverage"]
