from __future__ import annotations

import torch
from torch import nn


def _group_count(channels, preferred=8):
    for groups in range(min(preferred, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class DualAxisGRU(nn.Module):
    """One recurrent unit that consumes either the time or frequency axis.

    Input convention is ``[B, outer, sequence, features]``. The recurrent
    output must be projected back to ``features`` and returned with the same
    shape so that a residual connection is possible.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.bidirectional = bool(bidirectional)
        self.norm = nn.GroupNorm(_group_count(self.feature_dim), self.feature_dim)
        self.gru = nn.GRU(
            input_size=self.feature_dim,
            hidden_size=self.hidden_dim,
            batch_first=True,
            bidirectional=self.bidirectional,
        )
        direction_factor = 2 if self.bidirectional else 1
        self.projection = nn.Linear(
            self.hidden_dim * direction_factor, self.feature_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process one axis and preserve ``[B,outer,sequence,features]``."""

        batch, outer, sequence, features = x.shape
        if features != self.feature_dim:
            raise RuntimeError("DualAxisGRU received the wrong feature width")

        flat = x.reshape(batch * outer, sequence, features)
        normalized = self.norm(flat.transpose(1, 2)).transpose(1, 2)
        recurrent, _ = self.gru(normalized)
        recurrent = self.projection(recurrent)
        recurrent = recurrent.reshape(batch, outer, sequence, features)

        return x + recurrent


class IDPM(nn.Module):
    """Interleaved dual-path module with optional channel heads.

    The intended sequence is: split channels into heads, run a time-axis
    recurrent unit, transpose the two sequence axes, run a frequency-axis
    recurrent unit, and repeat. Finally merge the heads back to the original
    channel count. This module replaces SCNet's ``SeparationNet`` while
    retaining its single input/output feature-map contract.

    Input and output shape:
        ``[B, C, F, T]``.
    """

    def __init__(
        self,
        channels: int,
        heads: int = 2,
        repeats: int = 2,
        hidden_multiplier: float = 2.0,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.heads = int(heads)
        self.num_repeats = int(repeats)

        if self.channels < 1 or self.heads < 1:
            raise ValueError("channels and heads must be positive")
        if self.channels % self.heads:
            raise ValueError("channels must be divisible by heads")
        if self.num_repeats < 1:
            raise ValueError("repeats must be positive")

        self.feature_dim = self.channels // self.heads
        self.hidden_multiplier = float(hidden_multiplier)

        hidden_dim = max(
            int(round(self.feature_dim * self.hidden_multiplier)),
            1,
        )

        self.repeats = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "time": DualAxisGRU(self.feature_dim, hidden_dim),
                        "frequency": DualAxisGRU(self.feature_dim, hidden_dim),
                    }
                )
                for _ in range(self.num_repeats)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run interleaved time/frequency context and preserve input shape."""
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise RuntimeError("IDPM expected [B,C,F,T] with the configured C")

        batch, channels, frequency, frames = x.shape
        y = x.reshape(batch, self.heads, self.feature_dim, frequency, frames).reshape(
            batch * self.heads, self.feature_dim, frequency, frames
        )

        for repeat in self.repeats:
            # Every frequency bin gets a sequence over time.
            y = y.permute(0, 2, 3, 1).contiguous()  # [BH,F,T,Chead]
            y = repeat["time"](y)

            # Every time frame gets a sequence over frequency.
            y = y.permute(0, 2, 1, 3).contiguous()  # [BH,T,F,Chead]
            y = repeat["frequency"](y)

            # Restore channel-first layout for the next repeat
            y = y.permute(0, 3, 2, 1).contiguous()
            # [BH,T,F,Chead] -> [BH,Chead,F,T]

        return y.reshape(
            batch, self.heads, self.feature_dim, frequency, frames
        ).reshape(batch, channels, frequency, frames)


__all__ = ["DualAxisGRU", "IDPM"]
