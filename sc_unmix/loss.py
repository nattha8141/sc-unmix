"""SCNet complex-spectrogram RMSE objective for the vocals-only model."""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn


def source_activity_mask(
    target: torch.Tensor,
    activity_threshold: float = 1e-3,
) -> torch.Tensor:
    """Return ``[B]`` activity flags for stereo targets shaped ``[B,C,L]``."""
    if target.ndim != 3:
        raise ValueError(f"Expected target [B,C,L], got {tuple(target.shape)}")
    return target.float().square().mean(dim=(1, 2)).sqrt() >= activity_threshold


def frame_activity_fraction(
    target: torch.Tensor,
    frame_samples: int,
    activity_threshold: float = 1e-3,
) -> torch.Tensor:
    """Return the active short-frame fraction for every stereo example."""
    if target.ndim != 3:
        raise ValueError(f"Expected target [B,C,L], got {tuple(target.shape)}")
    frame_samples = max(int(frame_samples), 1)
    frame_count = max(math.ceil(target.shape[-1] / frame_samples), 1)
    padding = frame_count * frame_samples - target.shape[-1]
    if padding:
        target = F.pad(target, (0, padding))
    framed = target.float().reshape(
        target.shape[0],
        target.shape[1],
        frame_count,
        frame_samples,
    )
    frame_rms = framed.square().mean(dim=(1, 3)).sqrt()
    return (frame_rms >= activity_threshold).float().mean(dim=1)


def _group_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    *,
    empty_value: float = 0.0,
) -> torch.Tensor:
    if mask.any():
        return values[mask].mean()
    return values.new_tensor(empty_value)


class ComplexSTFTRMSELoss(nn.Module):
    """Reproduce SCNet's waveform-to-complex-STFT RMSE reduction.

    SCNet reconstructs waveform estimates, applies a second normalized STFT
    to both estimates and references, averages squared error over channel,
    frequency, time and real/imaginary coordinates, takes one square root per
    example, and finally averages examples uniformly.

    The public SCNet implementation does not pass a window to ``torch.stft``,
    which means a rectangular window. This class passes an explicit all-ones
    window to reproduce that operation without PyTorch's spectral-leakage
    warning. A tiny clamp is applied only at exactly-zero MSE to avoid the
    infinite derivative of ``sqrt(0)``.

    Target activity affects diagnostics only. Quiet examples remain in the
    objective with the same weight as active examples.
    """

    def __init__(
        self,
        n_fft: int = 4096,
        hop_length: int = 1024,
        win_length: int | None = None,
        activity_threshold: float = 1e-3,
        sqrt_epsilon: float = 1e-12,
    ) -> None:
        super().__init__()
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length or n_fft)
        self.activity_threshold = float(activity_threshold)
        self.sqrt_epsilon = float(sqrt_epsilon)
        if self.sqrt_epsilon <= 0:
            raise ValueError("sqrt_epsilon must be positive.")
        self.register_buffer(
            "_window",
            torch.ones(self.win_length),
            persistent=False,
        )

    def _complex_spectrum(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim != 3:
            raise ValueError(f"Expected waveform [B,C,L], got {tuple(waveform.shape)}")
        batch, channels, length = waveform.shape
        spectrum = torch.stft(
            waveform.reshape(batch * channels, length),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self._window.to(device=waveform.device, dtype=waveform.dtype),
            center=True,
            normalized=True,
            return_complex=True,
        )
        spectrum = torch.view_as_real(spectrum)
        return spectrum.reshape(
            batch,
            channels,
            spectrum.shape[-3],
            spectrum.shape[-2],
            2,
        )

    def forward(
        self,
        prediction_waveform: torch.Tensor,
        target_waveform: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if prediction_waveform.shape != target_waveform.shape:
            raise ValueError(
                "Prediction and target waveforms must have the same shape."
            )
        if prediction_waveform.ndim != 3:
            raise ValueError("Expected prediction and target [B,C,L].")

        # SCNet supervises a re-STFT of reconstructed audio, not the decoder's
        # internal spectrum. Float32 avoids unsupported half-precision FFTs.
        # The local MPS smoke run was verified with the re-STFT loss on CPU.
        # Preserve that compatibility path; autograd propagates the gradient
        # back to the MPS model through the device copy.
        loss_prediction = prediction_waveform
        loss_target = target_waveform
        if prediction_waveform.device.type == "mps":
            loss_prediction = prediction_waveform.to("cpu")
            loss_target = target_waveform.to("cpu")
        prediction_spectrum = self._complex_spectrum(loss_prediction.float())
        target_spectrum = self._complex_spectrum(loss_target.float())
        per_example_mse = F.mse_loss(
            prediction_spectrum,
            target_spectrum,
            reduction="none",
        ).mean(dim=(1, 2, 3, 4))
        per_example_rmse = per_example_mse.clamp_min(self.sqrt_epsilon).sqrt()
        total = per_example_rmse.mean()

        # Use the same device as the per-example STFT loss so boolean indexing
        # does not mix an MPS mask with CPU loss tensors.
        active = source_activity_mask(loss_target, self.activity_threshold)
        quiet = ~active
        return total, {
            "loss": total.detach(),
            "complex_rmse": total.detach(),
            "active_complex_rmse": _group_mean(per_example_rmse, active).detach(),
            "quiet_complex_rmse": _group_mean(per_example_rmse, quiet).detach(),
            "active_examples": active.sum().detach(),
            "quiet_examples": quiet.sum().detach(),
        }


def sdr_per_example(
    prediction: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Scale-dependent utterance SDR (the MDX uSDR/SNR definition)."""
    if prediction.shape != target.shape or target.ndim != 3:
        raise ValueError("Expected matching prediction/target [B,C,L].")
    # Apple MPS currently has no float64 implementation. Keep the original
    # float64 diagnostic precision on CPU/CUDA, but use float32 on MPS so the
    # validation/diagnostic SDR path can run during local Apple training.
    if prediction.device.type == "mps":
        prediction_work = prediction.float()
        target_work = target.float()
    else:
        prediction_work = prediction.double()
        target_work = target.double()
    target_energy = target_work.square().sum(dim=(1, 2))
    error_energy = (prediction_work - target_work).square().sum(dim=(1, 2))
    return (10.0 * torch.log10((target_energy + eps) / (error_energy + eps))).float()


def active_sdr(
    prediction: torch.Tensor,
    target: torch.Tensor,
    activity_threshold: float = 1e-3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return per-example SDR and a mask excluding silent targets."""
    return (
        sdr_per_example(prediction, target),
        source_activity_mask(target, activity_threshold),
    )


def quiet_leakage_rms(prediction: torch.Tensor) -> torch.Tensor:
    """Return output RMS for each example."""
    if prediction.ndim != 3:
        raise ValueError(f"Expected prediction [B,C,L], got {tuple(prediction.shape)}")
    return prediction.float().square().mean(dim=(1, 2)).sqrt()


__all__ = [
    "ComplexSTFTRMSELoss",
    "active_sdr",
    "frame_activity_fraction",
    "quiet_leakage_rms",
    "sdr_per_example",
    "source_activity_mask",
]
