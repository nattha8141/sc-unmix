"""Separate a file into predicted vocals and residual accompaniment."""

import argparse
from pathlib import Path

import julius
import numpy as np
import soundfile as sf
import torch

from .SCUnmix import SCUnmix
from .overlap_add import separate_in_chunks

SAMPLE_RATE = 44100


def load_model(checkpoint, device=None):
    """Load the released tensor checkpoint, checking every model weight."""
    if device is None:
        mps = getattr(torch.backends, "mps", None)
        device = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if mps is not None and mps.is_available() else "cpu")
        )
    if device == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError("MPS was requested, but Apple MPS is unavailable")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but CUDA is unavailable")
    device = torch.device(device)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = dict(payload["model_config"])
    separator = config.pop("separator")
    if separator != {
        "type": "TFC_IDPMSeparator",
        "channels": 96,
        "tfc_layers": 3,
        "post_tfc_attention": "CMHSA + direct frequency gate + residual",
        "post_tfc_attention_heads": 4,
        "post_tfc_attention_embedding_dim": 256,
        "idpm_heads": 2,
        "idpm_repeats": 3,
        "idpm_hidden_multiplier": 2.0,
    }:
        raise ValueError("This checkpoint does not describe the released separator")
    model = SCUnmix(**config)
    model.load_state_dict(payload["model_state"], strict=True)
    return model.to(device).eval()


def prepare_audio(audio, sample_rate):
    """Convert sample-major mono/stereo audio to stereo [2,L] at 44.1 kHz."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio[:, None]
    if audio.ndim != 2 or audio.shape[1] not in (1, 2):
        raise ValueError("Expected mono or stereo audio with shape [samples, channels]")
    if not len(audio) or not np.isfinite(audio).all() or sample_rate <= 0:
        raise ValueError(
            "Audio must be nonempty and finite with a positive sample rate"
        )
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    waveform = torch.from_numpy(np.ascontiguousarray(audio.T))
    if sample_rate != SAMPLE_RATE:
        waveform = julius.resample_frac(waveform, int(sample_rate), SAMPLE_RATE)
    if waveform.shape[-1] == 0:
        raise ValueError("Audio is too short after resampling")
    return waveform


def separate_audio(
    model,
    audio,
    sample_rate,
    *,
    segment_seconds=11.0,
    overlap=0.5,
    batch_size=1,
    amp="auto",
):
    """Return CPU [2,L] tensors. All outputs use the resampled mixture grid.

    The model predicts vocals only. Subtraction enforces additive consistency;
    it does not independently estimate or improve the accompaniment.
    """
    if segment_seconds <= 0 or batch_size < 1:
        raise ValueError("Segment duration and batch size must be positive")
    if amp not in {"auto", "none", "bfloat16", "float16"}:
        raise ValueError("Unsupported AMP mode")
    mixture = prepare_audio(audio, sample_rate)
    vocals = separate_in_chunks(
        model,
        mixture,
        sample_rate=SAMPLE_RATE,
        segment_seconds=segment_seconds,
        overlap=overlap,
        batch_size=batch_size,
        amp=amp,
    )
    return {"mixture": mixture, "vocals": vocals, "accompaniment": mixture - vocals}


def save_estimates(estimates, output_dir):
    """Save float WAVs without clipping or independent stem normalization."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: output_dir / f"{name}.wav" for name in ("vocals", "accompaniment")}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError(
            "Output stems already exist; choose a new output directory"
        )
    for name, path in paths.items():
        sf.write(
            path, estimates[name].detach().cpu().numpy().T, SAMPLE_RATE, subtype="FLOAT"
        )
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default=None)
    parser.add_argument("--segment-seconds", type=float, default=11.0)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--amp", choices=("auto", "none", "bfloat16", "float16"), default="auto"
    )
    args = parser.parse_args()
    model = load_model(args.checkpoint, args.device)
    audio, rate = sf.read(args.input, dtype="float32", always_2d=True)
    estimates = separate_audio(
        model,
        audio,
        rate,
        segment_seconds=args.segment_seconds,
        overlap=args.overlap,
        batch_size=args.batch_size,
        amp=args.amp,
    )
    for name, path in save_estimates(estimates, args.output_dir).items():
        print(f"{name}: {path.resolve()}")


if __name__ == "__main__":
    main()
