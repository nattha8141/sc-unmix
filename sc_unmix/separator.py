"""Local TFC refinement, compressed attention and split-head dual-path GRUs.

The 128-channel encoder map is projected to 96 channels, refined, and
projected back before a learned scaled residual addition.
"""

from __future__ import annotations

import torch
from torch import nn

from .csa_fusion import CMHSA, LinearGate
from .idpm import IDPM
from .tfc import TFCBlock


def _group_count(channels: int, preferred: int = 8) -> int:
    """Return a useful GroupNorm group count that divides ``channels``."""

    channels = int(channels)
    preferred = int(preferred)
    for groups in range(min(preferred, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class TFC_IDPMSeparator(nn.Module):
    """No-TDF separator with one CSA-style attention block after TFC.

    Args:
        input_channels: Channel width of the deepest encoder feature map.
        separator_channels: Width used by the TFC and IDPM modules.
        frequency_bins: Number of frequency bins at the deepest level.
        tfc_layers: Number of residual local time-frequency convolutions.
        tfc_kernel_size: Odd 2-D kernel size used by each TFC layer.
        idpm_heads: Number of independent channel groups in IDPM.
        idpm_repeats: Number of time/frequency GRU pairs.
        idpm_hidden_multiplier: GRU hidden width divided by per-head width.
        attention_heads: Number of heads in the post-TFC compressed attention.
        attention_embedding_dim: Compressed Q/K embedding target used by CSA.

    Input and output shape:
        ``[B, input_channels, frequency_bins, frames]``.

    The TDF block is deliberately absent.  A single compressed temporal
    attention block is evaluated after TFC and before IDPM.  It reuses the
    same CMHSA and direct frequency-gate primitives as decoder CSA, but has a
    residual carry because there is no decoder/skip pair to fuse.  Frequency
    resampling is disabled, so the deepest 57-bin grid is preserved.
    """

    def __init__(
        self,
        input_channels: int,
        separator_channels: int,
        frequency_bins: int,
        tfc_layers: int = 3,
        tfc_kernel_size: int = 3,
        idpm_heads: int = 2,
        idpm_repeats: int = 3,
        idpm_hidden_multiplier: float = 2.0,
        attention_heads: int = 4,
        attention_embedding_dim: int = 256,
    ) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.separator_channels = int(separator_channels)
        self.frequency_bins = int(frequency_bins)
        self.tfc_layers = int(tfc_layers)
        self.tfc_kernel_size = int(tfc_kernel_size)
        self.idpm_heads = int(idpm_heads)
        self.idpm_repeats = int(idpm_repeats)
        self.idpm_hidden_multiplier = float(idpm_hidden_multiplier)
        self.attention_heads = int(attention_heads)
        self.attention_embedding_dim = int(attention_embedding_dim)

        if self.input_channels < 1 or self.separator_channels < 1:
            raise ValueError("channel widths must be positive")
        if self.frequency_bins < 1:
            raise ValueError("frequency_bins must be positive")
        if self.attention_heads < 1:
            raise ValueError("attention_heads must be positive")
        if self.separator_channels % self.attention_heads:
            raise ValueError("separator_channels must be divisible by attention_heads")
        if self.attention_embedding_dim < 1:
            raise ValueError("attention_embedding_dim must be positive")

        self.input_projection = nn.Sequential(
            nn.Conv2d(
                self.input_channels,
                self.separator_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(
                _group_count(self.separator_channels), self.separator_channels
            ),
            nn.SiLU(),
        )
        self.tfc = TFCBlock(
            channels=self.separator_channels,
            layers=self.tfc_layers,
            kernel_size=self.tfc_kernel_size,
        )
        self.post_tfc_attention = CMHSA(
            embedding_dim=self.attention_embedding_dim,
            in_channels=self.separator_channels,
            frequency_bins=self.frequency_bins,
            num_heads=self.attention_heads,
        )
        self.post_tfc_gate = LinearGate(
            channels=self.separator_channels,
            frequency_bins=self.frequency_bins,
            use_frequency_resampling=False,
        )
        self.idpm = IDPM(
            channels=self.separator_channels,
            heads=self.idpm_heads,
            repeats=self.idpm_repeats,
            hidden_multiplier=self.idpm_hidden_multiplier,
        )
        self.output_projection = nn.Conv2d(
            self.separator_channels,
            self.input_channels,
            kernel_size=1,
            bias=False,
        )
        # Start with a small contribution from the separator residual branch.
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Refine ``x`` and preserve the encoder feature-map shape."""

        if x.ndim != 4:
            raise RuntimeError(f"Expected [B,C,F,T], got {tuple(x.shape)}")
        if x.shape[1] != self.input_channels:
            raise RuntimeError("separator received the wrong channel count")
        if x.shape[2] != self.frequency_bins:
            raise RuntimeError("separator received the wrong frequency count")

        residual = x
        x = self.input_projection(x)
        x = self.tfc(x)
        # CSA uses [B,C,T,F]; the sparse path and TFC use [B,C,F,T].
        # The frequency gate is direct (no stride-2 down/up resampling), and
        # the gated attention is added as a residual local-to-global refinement.
        attention_input = x.permute(0, 1, 3, 2).contiguous()
        attention = self.post_tfc_attention(attention_input)
        gate = self.post_tfc_gate(attention_input)
        x = (attention_input + gate * attention).permute(0, 1, 3, 2).contiguous()
        x = self.idpm(x)
        x = self.output_projection(x)
        return residual + self.residual_scale * x


__all__ = ["TFC_IDPMSeparator"]
