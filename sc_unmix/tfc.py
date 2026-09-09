"""Residual local time-frequency refinement."""

from __future__ import annotations

import torch
from torch import nn


def _group_count(channels: int, preferred: int = 8) -> int:
    """Return a GroupNorm group count that divides ``channels``."""

    channels = int(channels)
    for groups in range(min(int(preferred), channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class TFCBlock(nn.Module):
    """Residual local refinement over the frequency and time axes.

    The input and output are both ``[B, C, F, T]``.  Each same-width 2-D
    convolution sees a local frequency/time neighbourhood; IDPM supplies the
    longer-range context after this block.  There is deliberately no TDF or
    full-band linear layer in this block.
    """

    def __init__(self, channels: int, layers: int = 3, kernel_size: int = 3) -> None:
        super().__init__()
        self.channels = int(channels)
        self.num_layers = int(layers)
        self.kernel_size = int(kernel_size)
        if self.channels < 1:
            raise ValueError("channels must be positive")
        if self.num_layers < 1 or self.kernel_size < 1 or self.kernel_size % 2 == 0:
            raise ValueError("layers must be positive and kernel_size must be odd")

        self.net = nn.ModuleList()
        for _ in range(self.num_layers):
            self.net.append(
                nn.Sequential(
                    nn.Conv2d(
                        self.channels,
                        self.channels,
                        kernel_size=self.kernel_size,
                        padding=self.kernel_size // 2,
                        bias=False,
                    ),
                    nn.GroupNorm(_group_count(self.channels), self.channels),
                    nn.SiLU(),
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply local residual refinement without changing the shape."""

        if x.ndim != 4 or x.shape[1] != self.channels:
            raise RuntimeError(
                f"Expected [B,{self.channels},F,T], got {tuple(x.shape)}"
            )
        for layer in self.net:
            x = x + layer(x)
        return x


__all__ = ["TFCBlock"]
