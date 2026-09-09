"""Compressed self-attention skip fusion for SC-Unmix.

The attention implementation uses ``[B, C, T, F]`` while the original
SC-Unmix decoder uses ``[B, C, F, T]``. The :class:`CSAFusion` wrapper
preserves that layout and performs conversion at its boundary.
Only the deepest decoder fusion operation uses this module in the decoder; the
two higher-resolution decoder skips use SCNet's original ``FusionLayer``.
The post-TFC separator reuses the ``CMHSA`` and ``LinearGate``
primitives defined here, but does not change this decoder fusion contract.
Sparse upsampling and the encoder remain unchanged.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class ChannelLayerNorm(nn.Module):
    """LayerNorm over channels for a ``[B,C,T,F]`` feature map."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(int(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class LinearGate(nn.Module):
    """Learned frequency gate with optional frequency resampling.

    The released model uses the direct form: the frequency projection operates on
    the full 57-bin deepest map, without a stride-2 convolution or transposed
    convolution. The last axis is frequency in ``[B,C,T,F]`` layout.
    """

    def __init__(
        self,
        channels: int,
        frequency_bins: int,
        kernel_size: int = 2,
        stride: int = 2,
        use_frequency_resampling: bool = False,
    ) -> None:
        super().__init__()
        self.frequency_bins = int(frequency_bins)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.use_frequency_resampling = bool(use_frequency_resampling)
        self.padding = self.kernel_size // 2
        self.norm = ChannelLayerNorm(channels)
        if self.use_frequency_resampling:
            self.down = nn.Conv2d(
                channels,
                channels,
                kernel_size=(1, self.kernel_size),
                stride=(1, self.stride),
                padding=(0, self.padding),
                bias=False,
            )
            projection_bins = (
                self.frequency_bins + 2 * self.padding - self.kernel_size
            ) // self.stride + 1
            self.up = nn.ConvTranspose2d(
                channels,
                channels,
                kernel_size=(1, self.kernel_size),
                stride=(1, self.stride),
                padding=(0, self.padding),
                output_padding=(0, self.stride - 1),
                bias=False,
            )
        else:
            self.down = None
            self.up = None
            projection_bins = self.frequency_bins
        self.projection_frequency_bins = int(projection_bins)
        self.frequency_projection = nn.Linear(projection_bins, projection_bins)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.frequency_bins:
            raise RuntimeError(
                f"LinearGate expected F={self.frequency_bins}, " f"got F={x.shape[-1]}"
            )
        gate = self.down(x) if self.down is not None else x
        gate = self.norm(gate)
        gate = torch.sigmoid(self.frequency_projection(gate))
        if self.up is None:
            return gate
        gate = self.up(gate)
        difference = gate.shape[-1] - self.frequency_bins
        if difference > 0:
            left = difference // 2
            gate = gate[..., left : left + self.frequency_bins]
        elif difference < 0:
            missing = -difference
            gate = F.pad(gate, (missing // 2, missing - missing // 2, 0, 0))
        return gate


class CompressedTemporalAttention(nn.Module):
    """Compressed multi-head attention over the temporal axis.

    The query/key projection width is
    ``ceil(embedding_dim / frequency_bins)`` channels per head.  Attention is
    performed across frames using flattened channel-frequency features, while
    values retain the full channel width.
    """

    def __init__(
        self,
        embedding_dim: int,
        in_channels: int,
        frequency_bins: int,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        if in_channels % num_heads:
            raise ValueError("in_channels must be divisible by num_heads")
        self.in_channels = int(in_channels)
        self.frequency_bins = int(frequency_bins)
        self.num_heads = int(num_heads)
        self.channels_per_head = self.in_channels // self.num_heads
        self.query_channels_per_head = math.ceil(
            int(embedding_dim) / self.frequency_bins
        )
        self.query_channels = self.query_channels_per_head * self.num_heads

        self.query = nn.Sequential(
            nn.Conv2d(
                self.in_channels,
                self.query_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.PReLU(self.query_channels),
            ChannelLayerNorm(self.query_channels),
        )
        self.key = nn.Sequential(
            nn.Conv2d(
                self.in_channels,
                self.query_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.PReLU(self.query_channels),
            ChannelLayerNorm(self.query_channels),
        )
        self.value = nn.Sequential(
            nn.Conv2d(
                self.in_channels,
                self.in_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.PReLU(self.in_channels),
            ChannelLayerNorm(self.in_channels),
        )

    def _query_or_key(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, frames, frequency_bins = x.shape
        return (
            x.reshape(
                batch,
                self.num_heads,
                self.query_channels_per_head,
                frames,
                frequency_bins,
            )
            .permute(0, 1, 3, 2, 4)
            .reshape(batch, self.num_heads, frames, -1)
        )

    def _value(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, frames, frequency_bins = x.shape
        return (
            x.reshape(
                batch,
                self.num_heads,
                self.channels_per_head,
                frames,
                frequency_bins,
            )
            .permute(0, 1, 3, 2, 4)
            .reshape(batch, self.num_heads, frames, -1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.frequency_bins:
            raise RuntimeError(
                f"CSA expected F={self.frequency_bins}, got F={x.shape[-1]}"
            )
        query = self._query_or_key(self.query(x))
        key = self._query_or_key(self.key(x))
        value = self._value(self.value(x))
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=0.0,
            is_causal=False,
        )
        batch, _, frames, _ = attended.shape
        return (
            attended.reshape(
                batch,
                self.num_heads,
                frames,
                self.channels_per_head,
                self.frequency_bins,
            )
            .permute(0, 1, 3, 2, 4)
            .reshape(batch, self.in_channels, frames, self.frequency_bins)
        )


class CMHSA(nn.Module):
    """Compressed multi-head self-attention plus output projection."""

    def __init__(
        self,
        embedding_dim: int,
        in_channels: int,
        frequency_bins: int,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.attention = CompressedTemporalAttention(
            embedding_dim,
            in_channels,
            frequency_bins,
            num_heads,
        )
        self.output = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.PReLU(in_channels),
            ChannelLayerNorm(in_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.attention(x))


class GLU(nn.Module):
    """Pointwise gated linear unit used after CSA and the frequency gate."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.input = nn.Conv2d(channels, 2 * channels, kernel_size=1)
        self.output = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.input(x).chunk(2, dim=1)
        return self.output(value * torch.sigmoid(gate))


class _CSAFusionTimeFrequency(nn.Module):
    """Compressed self-attention and gating in ``[B,C,T,F]`` layout."""

    def __init__(
        self,
        embedding_dim: int,
        in_channels: int,
        frequency_bins: int,
        num_heads: int = 4,
        use_frequency_resampling: bool = False,
    ) -> None:
        super().__init__()
        self.cmhsa = CMHSA(
            embedding_dim,
            in_channels,
            frequency_bins,
            num_heads,
        )
        self.gate = LinearGate(
            in_channels,
            frequency_bins,
            use_frequency_resampling=use_frequency_resampling,
        )
        self.glu = GLU(in_channels)

    def forward(self, decoder: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if decoder.shape != skip.shape:
            raise RuntimeError(
                "CSA inputs must match, got "
                f"{tuple(decoder.shape)} and {tuple(skip.shape)}"
            )
        combined = decoder + skip
        return self.glu(self.cmhsa(combined) * self.gate(combined))


class CSAFusion(nn.Module):
    """CSA skip fusion preserving the ``[B,C,F,T]`` feature layout.

    Used at the deepest decoder skip. Layout permutations are required
    because the encoder/decoder store frequency before time, while attention stores
    time before frequency.
    """

    def __init__(
        self,
        embedding_dim: int,
        in_channels: int,
        frequency_bins: int,
        num_heads: int = 4,
        use_frequency_resampling: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.frequency_bins = int(frequency_bins)
        self.inner = _CSAFusionTimeFrequency(
            embedding_dim,
            in_channels,
            frequency_bins,
            num_heads,
            use_frequency_resampling,
        )

    def forward(self, decoder: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if decoder.ndim != 4 or decoder.shape != skip.shape:
            raise RuntimeError(
                "CSAFusion expects matching [B,C,F,T] tensors, got "
                f"{tuple(decoder.shape)} and {tuple(skip.shape)}"
            )
        if decoder.shape[1] != self.in_channels:
            raise RuntimeError(
                f"CSAFusion expected C={self.in_channels}, got C={decoder.shape[1]}"
            )
        if decoder.shape[2] != self.frequency_bins:
            raise RuntimeError(
                f"CSAFusion expected F={self.frequency_bins}, got F={decoder.shape[2]}"
            )
        decoder_tf = decoder.permute(0, 1, 3, 2).contiguous()
        skip_tf = skip.permute(0, 1, 3, 2).contiguous()
        fused_tf = self.inner(decoder_tf, skip_tf)
        return fused_tf.permute(0, 1, 3, 2).contiguous()


__all__ = [
    "CSAFusion",
    "CMHSA",
    "CompressedTemporalAttention",
    "LinearGate",
]
