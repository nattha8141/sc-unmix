#!/usr/bin/env python3
"""SC-Unmix training with bounded sampling and full-track tiled validation.

Run ``python -m sc_unmix.train --config-path configs/train.yaml``.
Command-line options override YAML values. Relative YAML paths are resolved
against the configuration file, not the current working directory.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from sc_unmix.SCUnmix import SCUnmix
from sc_unmix.data import (
    EpochShuffleSampler,
    MUSDBVocalDataset,
)
from sc_unmix.ema import (
    ExponentialMovingAverage,
)
from sc_unmix.loss import (
    ComplexSTFTRMSELoss,
    active_sdr,
    frame_activity_fraction,
    quiet_leakage_rms,
    sdr_per_example,
)


EXPERIMENT = "sc-unmix"
CHECKPOINT_NAME = "training_state.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-path", "--config_path")
    parser.add_argument("--root")
    parser.add_argument("--output", "--save-path", "--save_path")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--seq-dur", type=float, default=11.0)
    parser.add_argument("--samples-per-track", type=int, default=9)
    parser.add_argument("--valid-windows-per-track", type=int, default=4)
    parser.add_argument("--valid-probe-windows", type=int, default=24)
    parser.add_argument("--valid-tracks", type=int, default=14)
    parser.add_argument(
        "--validation-manifest",
        default=None,
        help=(
            "JSON manifest listing held-out validation track directory names. "
            "When supplied, these tracks are excluded from every training stem pool."
        ),
    )
    parser.add_argument(
        "--diagnostic-every",
        type=int,
        default=2,
        help="Evaluate fixed clean-train and uniform-validation panels every N epochs.",
    )
    parser.add_argument("--diagnostic-train-tracks", type=int, default=14)
    parser.add_argument("--diagnostic-train-windows", type=int, default=2)
    parser.add_argument("--diagnostic-uniform-windows", type=int, default=4)
    parser.add_argument("--diagnostic-workers", type=int, default=2)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Physical NVIDIA L4 batch validated for the 11-second model.",
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help="Accumulate only if the desired effective example count requires it.",
    )
    parser.add_argument("--nb-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--no-pin-memory", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--nfft", type=int, default=4096)
    parser.add_argument("--nhop", type=int, default=1024)
    parser.add_argument("--win-length", type=int, default=None)

    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=0.0,
        help="Required absolute active-uSDR gain for checkpointing/early stopping.",
    )
    parser.add_argument("--lr-decay-patience", type=int, default=8)
    parser.add_argument("--lr-decay-gamma", type=float, default=0.5)
    parser.add_argument("--lr-decay-cooldown", type=int, default=2)
    parser.add_argument("--lr-min", type=float, default=1.25e-5)
    parser.add_argument("--lr-threshold", type=float, default=0.01)
    parser.add_argument("--lr-threshold-mode", choices=("rel", "abs"), default="abs")
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument(
        "--no-ema",
        action="store_true",
        help="Disable the default warm-started exponential moving average.",
    )
    parser.add_argument("--activity-threshold", type=float, default=1e-3)
    parser.add_argument("--active-probability", type=float, default=0.85)
    parser.add_argument("--activity-attempts", type=int, default=12)
    parser.add_argument("--activity-frame-duration", type=float, default=0.2)
    parser.add_argument("--min-active-frame-fraction", type=float, default=0.3)
    parser.add_argument("--max-quiet-frame-fraction", type=float, default=0.05)
    parser.add_argument("--coherent-mix-probability", type=float, default=0.5)
    parser.add_argument("--remix-gain-db", type=float, default=3.0)
    parser.add_argument(
        "--no-random-track-mix",
        action="store_true",
        help="Disable the default cross-song stem remixing.",
    )
    parser.add_argument(
        "--amp",
        choices=("auto", "bfloat16", "float16", "none"),
        default="auto",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
        help="Execution device. auto selects CUDA, then Apple MPS, then CPU.",
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help="Optional compatible checkpoint; omit for the recommended scratch run.",
    )
    parser.add_argument(
        "--init-scope",
        choices=("encoder", "all-compatible"),
        default="all-compatible",
        help="Select compatible encoder tensors or require the whole SC-Unmix model.",
    )
    parser.add_argument("--no-cuda", action="store_true")
    preliminary, _ = parser.parse_known_args()
    if preliminary.config_path:
        config_path = Path(preliminary.config_path).expanduser().resolve()
        with config_path.open() as stream:
            config = yaml.safe_load(stream)
        if not isinstance(config, dict):
            parser.error("YAML must contain a mapping of training option names")
        actions = {action.dest: action for action in parser._actions}
        for key, value in config.items():
            if key not in actions or key in {"help", "config_path"}:
                parser.error(f"Unknown YAML option: {key}")
            action = actions[key]
            if isinstance(action, argparse._StoreTrueAction):
                if not isinstance(value, bool):
                    parser.error(f"{key} must be a YAML boolean")
            elif value is not None and action.type:
                try:
                    config[key] = action.type(value)
                except (ValueError, TypeError):
                    parser.error(f"Invalid value for {key}: {value}")
            if action.choices and config[key] not in action.choices:
                parser.error(f"Invalid choice for {key}: {value}")
        for key in (
            "root",
            "output",
            "validation_manifest",
            "checkpoint",
            "init_checkpoint",
        ):
            if config.get(key):
                path = Path(config[key]).expanduser()
                config[key] = str((config_path.parent / path).resolve())
        parser.set_defaults(**config)
    args = parser.parse_args()
    if not args.root or not args.output:
        parser.error("root and output must be supplied through YAML or CLI")
    return args


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _mps_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())


def resolve_device(args: argparse.Namespace) -> torch.device:
    """Select CUDA, Apple MPS, or CPU while preserving ``--no-cuda``."""
    if args.no_cuda:
        if args.device not in {"auto", "cpu"}:
            raise ValueError("--no-cuda cannot be combined with --device mps/cuda")
        return torch.device("cpu")
    if args.device == "cpu":
        return torch.device("cpu")
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested, but CUDA is unavailable")
        return torch.device("cuda")
    if args.device == "mps":
        if not _mps_available():
            raise RuntimeError("--device mps requested, but Apple MPS is unavailable")
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if _mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def parameter_count(model: torch.nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def model_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Return the fixed architectural contract used for resume validation."""
    return {
        "sources": ["vocals"],
        "audio_channels": 2,
        "dims": [4, 32, 64, 128],
        "nfft": int(args.nfft),
        "hop_size": int(args.nhop),
        "win_size": int(args.win_length or args.nfft),
        "normalized": True,
        "band_SR": [0.175, 0.392, 0.433],
        "band_stride": [1, 4, 16],
        "band_kernel": [3, 4, 16],
        "conv_depths": [3, 2, 1],
        "compress": 4,
        "conv_kernel": 3,
        "fusion_dim": 1024,
        "fusion_attention_heads": 4,
        "deepest_csa_only": True,
        "separator": {
            "type": "TFC_IDPMSeparator",
            "channels": 96,
            "tfc_layers": 3,
            "post_tfc_attention": "CMHSA + direct frequency gate + residual",
            "post_tfc_attention_heads": 4,
            "post_tfc_attention_embedding_dim": 256,
            "idpm_heads": 2,
            "idpm_repeats": 3,
            "idpm_hidden_multiplier": 2.0,
        },
    }


def architecture_description() -> Dict[str, Any]:
    return {
        "encoder": "SCNet-derived sparse encoder",
        "separator": "96-channel TFC -> one 4-head CSA-style attention block -> IDPM H2/R3 -> scaled residual (no TDF)",
        "decoder": "SCNet sparse decoder with CSA only at the deepest skip and original FusionLayer above it",
        "output": "one direct complex vocal estimate",
    }


def build_model(args: argparse.Namespace, device: torch.device) -> SCUnmix:
    config = model_config(args)
    constructor = {key: value for key, value in config.items() if key != "separator"}
    return SCUnmix(**constructor).to(device)


def _state_dict(payload: Any) -> Dict[str, torch.Tensor]:
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint must be a dictionary.")
    for key in ("model_state", "state_dict", "best_state", "state"):
        if key in payload and isinstance(payload[key], dict):
            return payload[key]
    return payload


def load_compatible_weights(
    model: torch.nn.Module,
    path: str,
    device: torch.device,
    *,
    prefixes: tuple[str, ...] | None = ("encoder.",),
) -> Dict[str, Any]:
    payload = torch.load(path, map_location=device, weights_only=False)
    source = _state_dict(payload)
    destination = model.state_dict()
    compatible = {}
    for name, tensor in source.items():
        clean_name = name.removeprefix("module.")
        in_scope = prefixes is None or clean_name.startswith(prefixes)
        if (
            in_scope
            and clean_name in destination
            and destination[clean_name].shape == tensor.shape
        ):
            compatible[clean_name] = tensor
    destination.update(compatible)
    model.load_state_dict(destination, strict=True)
    print(
        f"Initialized {len(compatible)}/{len(destination)} tensors from {path}",
        flush=True,
    )
    return {
        "compatible_tensors": len(compatible),
        "model_tensors": len(destination),
        "checkpoint": str(Path(path).expanduser().resolve()),
        "prefixes": list(prefixes) if prefixes is not None else None,
    }


def resolve_resume(path: str | None, output: Path) -> Path | None:
    if path is None:
        candidate = output / CHECKPOINT_NAME
        return candidate if candidate.is_file() else None
    candidate = Path(path).expanduser()
    return candidate / CHECKPOINT_NAME if candidate.is_dir() else candidate


def atomic_torch_save(payload: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def resolve_amp(args: argparse.Namespace, device: torch.device) -> torch.dtype | None:
    if device.type != "cuda" or args.amp == "none":
        return None
    if args.amp == "bfloat16":
        return torch.bfloat16
    if args.amp == "float16":
        return torch.float16
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def make_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def make_optimizer(
    model: torch.nn.Module,
    args: argparse.Namespace,
    device: torch.device,
) -> torch.optim.Optimizer:
    options = {"lr": args.lr, "weight_decay": args.weight_decay}
    if device.type == "cuda":
        try:
            return torch.optim.AdamW(model.parameters(), fused=True, **options)
        except (TypeError, RuntimeError):
            pass
    return torch.optim.AdamW(model.parameters(), **options)


def make_loaders(args: argparse.Namespace):
    common = {
        "root": args.root,
        "seq_duration": args.seq_dur,
        "samples_per_track": args.samples_per_track,
        "valid_windows_per_track": args.valid_windows_per_track,
        "valid_probe_windows": args.valid_probe_windows,
        "sample_rate": args.sample_rate,
        "valid_tracks": args.valid_tracks,
        "validation_manifest": args.validation_manifest,
        "activity_threshold": args.activity_threshold,
        "activity_frame_duration": args.activity_frame_duration,
        "min_active_frame_fraction": args.min_active_frame_fraction,
        "max_quiet_frame_fraction": args.max_quiet_frame_fraction,
        "seed": args.seed,
    }
    train_data = MUSDBVocalDataset(
        split="train",
        random_track_mix=not args.no_random_track_mix,
        coherent_mix_probability=args.coherent_mix_probability,
        remix_gain_db=args.remix_gain_db,
        active_probability=args.active_probability,
        activity_attempts=args.activity_attempts,
        **common,
    )
    valid_data = MUSDBVocalDataset(
        split="valid",
        random_track_mix=False,
        coherent_mix_probability=1.0,
        remix_gain_db=0.0,
        active_probability=1.0,
        window_selection="active",
        full_track_validation=True,
        **common,
    )
    remix_train_data = MUSDBVocalDataset(
        split="train",
        random_track_mix=not args.no_random_track_mix,
        coherent_mix_probability=0.0,
        remix_gain_db=args.remix_gain_db,
        active_probability=1.0,
        activity_attempts=args.activity_attempts,
        max_tracks=args.diagnostic_train_tracks,
        samples_per_track=args.diagnostic_train_windows,
        **{key: value for key, value in common.items() if key != "samples_per_track"},
    )
    clean_train_data = MUSDBVocalDataset(
        split="train",
        random_track_mix=False,
        coherent_mix_probability=1.0,
        remix_gain_db=0.0,
        active_probability=1.0,
        clean_probe=True,
        window_selection="active",
        max_tracks=args.diagnostic_train_tracks,
        valid_windows_per_track=args.diagnostic_train_windows,
        **{
            key: value
            for key, value in common.items()
            if key != "valid_windows_per_track"
        },
    )
    valid_uniform_data = MUSDBVocalDataset(
        split="valid",
        random_track_mix=False,
        coherent_mix_probability=1.0,
        remix_gain_db=0.0,
        active_probability=1.0,
        window_selection="uniform",
        valid_windows_per_track=args.diagnostic_uniform_windows,
        **{
            key: value
            for key, value in common.items()
            if key != "valid_windows_per_track"
        },
    )
    sampler = EpochShuffleSampler(train_data, seed=args.seed)
    workers = max(args.nb_workers, 0)
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": workers,
        "pin_memory": (
            not args.no_pin_memory and torch.cuda.is_available() and not args.no_cuda
        ),
        "drop_last": False,
    }
    if workers > 0:
        loader_options["persistent_workers"] = True
        loader_options["prefetch_factor"] = max(args.prefetch_factor, 1)
    train_loader = DataLoader(train_data, sampler=sampler, **loader_options)
    valid_loader = DataLoader(valid_data, shuffle=False, **loader_options)
    diagnostic_options = dict(loader_options)
    diagnostic_workers = max(args.diagnostic_workers, 0)
    diagnostic_options["num_workers"] = diagnostic_workers
    if diagnostic_workers > 0:
        diagnostic_options["prefetch_factor"] = min(max(args.prefetch_factor, 1), 2)
        diagnostic_options["persistent_workers"] = True
    else:
        diagnostic_options.pop("prefetch_factor", None)
        diagnostic_options.pop("persistent_workers", None)
    clean_train_loader = DataLoader(
        clean_train_data,
        shuffle=False,
        **diagnostic_options,
    )
    remix_train_loader = DataLoader(
        remix_train_data,
        shuffle=False,
        **diagnostic_options,
    )
    valid_uniform_loader = DataLoader(
        valid_uniform_data,
        shuffle=False,
        **diagnostic_options,
    )
    return (
        train_data,
        remix_train_data,
        clean_train_data,
        valid_data,
        valid_uniform_data,
        train_loader,
        remix_train_loader,
        clean_train_loader,
        valid_loader,
        valid_uniform_loader,
    )


def run_epoch(
    model: SCUnmix,
    loader: DataLoader,
    criterion: ComplexSTFTRMSELoss,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    activity_threshold: float,
    activity_frame_samples: int | None = None,
    *,
    epoch: int,
    optimizer: torch.optim.Optimizer | None = None,
    scaler=None,
    ema: ExponentialMovingAverage | None = None,
    grad_clip: float = 0.0,
    grad_accum_steps: int = 1,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    if isinstance(loader.sampler, EpochShuffleSampler):
        loader.sampler.set_epoch(epoch)
    component_names = (
        "loss",
        "complex_rmse",
        "active_complex_rmse",
        "quiet_complex_rmse",
    )
    totals = {name: 0.0 for name in component_names}
    sample_count = 0
    sdr_sum = 0.0
    mixture_sdr_sum = 0.0
    sdr_improvement_sum = 0.0
    sdr_count = 0
    silent_leakage_sum = 0.0
    silent_input_sum = 0.0
    silent_attenuation_sum = 0.0
    silent_count = 0
    target_rms_sum = 0.0
    mixture_rms_sum = 0.0
    prediction_rms_sum = 0.0
    frame_activity_sum = 0.0
    active_frame_activity_sum = 0.0
    quiet_frame_activity_sum = 0.0
    grad_accum_steps = max(int(grad_accum_steps), 1)
    if training:
        if scaler is None:
            raise ValueError(
                "Training requires a gradient scaler (it may be disabled)."
            )
        optimizer.zero_grad(set_to_none=True)

    grad_context = torch.enable_grad if training else torch.no_grad
    with grad_context():
        for batch_index, (mixture_wave, target_wave) in enumerate(loader):
            mixture_wave = mixture_wave.to(device, non_blocking=True)
            target_wave = target_wave.to(device, non_blocking=True)
            mps_debug = device.type == "mps"
            if mps_debug and not torch.isfinite(mixture_wave).all().item():
                raise FloatingPointError(
                    f"Non-finite mixture batch at index {batch_index} on {device}"
                )
            if mps_debug and not torch.isfinite(target_wave).all().item():
                raise FloatingPointError(
                    f"Non-finite vocal target batch at index {batch_index} on {device}"
                )
            use_amp = device.type == "cuda" and amp_dtype is not None
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype if amp_dtype is not None else torch.float32,
                enabled=use_amp,
            ):
                # SC-Unmix owns its STFT, feature normalization, sparse path and
                # iSTFT. Keep that model path exactly as authored; only remove
                # the singleton source dimension for the vocal objective.
                prediction_wave = model(mixture_wave)[:, 0]
            if mps_debug and not torch.isfinite(prediction_wave).all().item():
                raise FloatingPointError(
                    f"Non-finite model output at batch {batch_index} on {device}"
                )
            loss, components = criterion(prediction_wave, target_wave)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss at batch {batch_index} on {device}: "
                    f"{float(loss.detach())}"
                )

            if training:
                scaler.scale(loss / grad_accum_steps).backward()
                last_batch = batch_index + 1 == len(loader)
                if (batch_index + 1) % grad_accum_steps == 0 or last_batch:
                    scaler.unscale_(optimizer)
                    if mps_debug:
                        bad_gradient = next(
                            (
                                name
                                for name, parameter in model.named_parameters()
                                if parameter.grad is not None
                                and not torch.isfinite(parameter.grad).all().item()
                            ),
                            None,
                        )
                        if bad_gradient is not None:
                            raise FloatingPointError(
                                f"Non-finite gradient after batch {batch_index} "
                                f"on {device} (first parameter: {bad_gradient})"
                            )
                    if grad_clip > 0:
                        clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    if mps_debug:
                        bad_parameter = next(
                            (
                                name
                                for name, parameter in model.named_parameters()
                                if not torch.isfinite(parameter).all().item()
                            ),
                            None,
                        )
                        if bad_parameter is not None:
                            raise FloatingPointError(
                                f"Non-finite parameter after optimizer step at batch "
                                f"{batch_index} on {device} (first parameter: {bad_parameter})"
                            )
                    optimizer.zero_grad(set_to_none=True)
                    if ema is not None:
                        ema.update(model)

            batch_size = mixture_wave.shape[0]
            sample_count += batch_size
            for key in component_names:
                totals[key] += float(components[key]) * batch_size
            with torch.no_grad():
                values, active = active_sdr(
                    prediction_wave,
                    target_wave,
                    activity_threshold,
                )
                mixture_values = sdr_per_example(mixture_wave, target_wave)
                target_rms = target_wave.float().square().mean(dim=(1, 2)).sqrt()
                mixture_rms = mixture_wave.float().square().mean(dim=(1, 2)).sqrt()
                prediction_rms = (
                    prediction_wave.float().square().mean(dim=(1, 2)).sqrt()
                )
                coverage = frame_activity_fraction(
                    target_wave,
                    activity_frame_samples or target_wave.shape[-1],
                    activity_threshold,
                )
                target_rms_sum += float(target_rms.sum())
                mixture_rms_sum += float(mixture_rms.sum())
                prediction_rms_sum += float(prediction_rms.sum())
                frame_activity_sum += float(coverage.sum())
                if active.any():
                    sdr_sum += float(values[active].sum())
                    mixture_sdr_sum += float(mixture_values[active].sum())
                    sdr_improvement_sum += float(
                        (values[active] - mixture_values[active]).sum()
                    )
                    active_frame_activity_sum += float(coverage[active].sum())
                    sdr_count += int(active.sum())
                quiet = ~active
                if quiet.any():
                    leakage = quiet_leakage_rms(prediction_wave)
                    silent_leakage_sum += float(leakage[quiet].sum())
                    silent_input_sum += float(mixture_rms[quiet].sum())
                    attenuation = 20.0 * torch.log10(
                        (mixture_rms[quiet] + 1e-8) / (prediction_rms[quiet] + 1e-8)
                    )
                    silent_attenuation_sum += float(attenuation.sum())
                    quiet_frame_activity_sum += float(coverage[quiet].sum())
                    silent_count += int(quiet.sum())

    metrics = {key: value / max(sample_count, 1) for key, value in totals.items()}
    metrics["active_sdr"] = sdr_sum / max(sdr_count, 1)
    metrics["active_mixture_sdr"] = mixture_sdr_sum / max(sdr_count, 1)
    metrics["active_sdr_improvement"] = sdr_improvement_sum / max(sdr_count, 1)
    metrics["active_examples"] = float(sdr_count)
    metrics["silent_leakage_rms"] = silent_leakage_sum / max(silent_count, 1)
    metrics["silent_input_rms"] = silent_input_sum / max(silent_count, 1)
    metrics["silent_attenuation_db"] = silent_attenuation_sum / max(silent_count, 1)
    metrics["silent_examples"] = float(silent_count)
    metrics["target_rms"] = target_rms_sum / max(sample_count, 1)
    metrics["mixture_rms"] = mixture_rms_sum / max(sample_count, 1)
    metrics["prediction_rms"] = prediction_rms_sum / max(sample_count, 1)
    metrics["active_frame_fraction"] = frame_activity_sum / max(sample_count, 1)
    metrics["active_examples_frame_fraction"] = active_frame_activity_sum / max(
        sdr_count, 1
    )
    metrics["quiet_examples_frame_fraction"] = quiet_frame_activity_sum / max(
        silent_count, 1
    )
    return metrics


def run_fixed_diagnostics(
    model: SCUnmix,
    remix_train_loader: DataLoader,
    clean_train_loader: DataLoader,
    valid_uniform_loader: DataLoader,
    criterion: ComplexSTFTRMSELoss,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    activity_threshold: float,
    activity_frame_samples: int,
    *,
    epoch: int,
) -> Dict[str, Dict[str, float]]:
    """Evaluate deterministic coherent probes without changing model state."""
    return {
        "train_remix": run_epoch(
            model,
            remix_train_loader,
            criterion,
            device,
            amp_dtype,
            activity_threshold,
            activity_frame_samples,
            epoch=epoch,
        ),
        "train_clean": run_epoch(
            model,
            clean_train_loader,
            criterion,
            device,
            amp_dtype,
            activity_threshold,
            activity_frame_samples,
            epoch=epoch,
        ),
        "valid_uniform": run_epoch(
            model,
            valid_uniform_loader,
            criterion,
            device,
            amp_dtype,
            activity_threshold,
            activity_frame_samples,
            epoch=epoch,
        ),
    }


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.grad_accum_steps < 1:
        raise ValueError("epochs, batch size, and accumulation must be positive")
    if (
        args.diagnostic_every < 1
        or args.diagnostic_train_tracks < 1
        or args.diagnostic_train_windows < 1
        or args.diagnostic_uniform_windows < 1
    ):
        raise ValueError("All diagnostic counts must be positive")
    activity_frame_samples = max(
        int(round(args.activity_frame_duration * args.sample_rate)), 1
    )
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = resolve_device(args)
    print(
        f"Using {torch.cuda.get_device_name(device) if device.type == 'cuda' else str(device).upper()}",
        flush=True,
    )
    try:
        torch.set_float32_matmul_precision("high")
    except (AttributeError, RuntimeError):
        pass
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model = build_model(args, device)
    active_model_config = model_config(args)
    active_architecture = architecture_description()
    resume = resolve_resume(args.checkpoint, output)
    initialization = None
    if args.init_checkpoint and resume is None:
        initialization = load_compatible_weights(
            model,
            args.init_checkpoint,
            device,
            prefixes=("encoder.",) if args.init_scope == "encoder" else None,
        )
        if (
            args.init_scope == "all-compatible"
            and initialization["compatible_tensors"] != initialization["model_tensors"]
        ):
            raise ValueError(
                "--init-scope all-compatible requires every model tensor to "
                "match. Omit --init-checkpoint for a scratch run."
            )
    elif args.init_checkpoint and resume is not None:
        print("Resume checkpoint found; --init-checkpoint is ignored.", flush=True)
    (
        train_data,
        remix_train_data,
        clean_train_data,
        valid_data,
        valid_uniform_data,
        train_loader,
        remix_train_loader,
        clean_train_loader,
        valid_loader,
        valid_uniform_loader,
    ) = make_loaders(args)
    criterion = ComplexSTFTRMSELoss(
        n_fft=args.nfft,
        hop_length=args.nhop,
        win_length=args.win_length,
        activity_threshold=args.activity_threshold,
    ).to(device)
    optimizer = make_optimizer(model, args, device)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=args.lr_decay_gamma,
        patience=args.lr_decay_patience,
        cooldown=args.lr_decay_cooldown,
        min_lr=args.lr_min,
        threshold=args.lr_threshold,
        threshold_mode=args.lr_threshold_mode,
    )
    amp_dtype = resolve_amp(args, device)
    scaler = make_scaler(device.type == "cuda" and amp_dtype == torch.float16)
    ema = None if args.no_ema else ExponentialMovingAverage(model, decay=args.ema_decay)

    start_epoch = 0
    best_sdr = -float("inf")
    best_epoch = 0
    bad_epochs = 0
    history = []
    initial_validation = None
    initial_diagnostics = None
    if resume is not None:
        if not resume.is_file():
            raise FileNotFoundError(resume)
        payload = torch.load(resume, map_location=device, weights_only=False)
        if payload.get("model_config") != active_model_config:
            raise ValueError("Resume checkpoint model_config does not match this run.")
        model.load_state_dict(payload["model_state"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state"])
        scheduler.load_state_dict(payload["scheduler_state"])
        if payload.get("scaler_state"):
            scaler.load_state_dict(payload["scaler_state"])
        if ema is not None:
            if payload.get("ema_state"):
                ema.load_state_dict(payload["ema_state"])
            else:
                ema = ExponentialMovingAverage(model, decay=args.ema_decay)
        start_epoch = int(payload["epoch"])
        best_sdr = float(payload.get("best_sdr", best_sdr))
        best_epoch = int(payload.get("best_epoch", 0))
        bad_epochs = int(payload.get("bad_epochs", 0))
        history = list(payload.get("history", []))
        initial_validation = payload.get("initial_validation")
        initial_diagnostics = payload.get("initial_diagnostics")
        print(f"Resumed {resume} at epoch {start_epoch}", flush=True)

    if resume is None and initialization is not None:
        if ema is None:
            initial_validation = run_epoch(
                model,
                valid_loader,
                criterion,
                device,
                amp_dtype,
                args.activity_threshold,
                activity_frame_samples,
                epoch=-1,
            )
            fixed_initial = run_fixed_diagnostics(
                model,
                remix_train_loader,
                clean_train_loader,
                valid_uniform_loader,
                criterion,
                device,
                amp_dtype,
                args.activity_threshold,
                activity_frame_samples,
                epoch=-1,
            )
        else:
            with ema.average_parameters(model):
                initial_validation = run_epoch(
                    model,
                    valid_loader,
                    criterion,
                    device,
                    amp_dtype,
                    args.activity_threshold,
                    activity_frame_samples,
                    epoch=-1,
                )
                fixed_initial = run_fixed_diagnostics(
                    model,
                    remix_train_loader,
                    clean_train_loader,
                    valid_uniform_loader,
                    criterion,
                    device,
                    amp_dtype,
                    args.activity_threshold,
                    activity_frame_samples,
                    epoch=-1,
                )
        initial_diagnostics = {
            "train_remix": fixed_initial["train_remix"],
            "train_clean": fixed_initial["train_clean"],
            "valid_clean_raw": initial_validation,
            "valid_clean": initial_validation,
            "valid_uniform": fixed_initial["valid_uniform"],
        }
        if initial_validation["active_examples"] < 1:
            raise RuntimeError("Validation contains no active vocal examples.")
        best_sdr = float(initial_validation["active_sdr"])
        scheduler.step(best_sdr)
        inference_state = (
            ema.averaged_model_state()
            if ema is not None
            else {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
        )
        initial_best_payload = {
            "experiment": EXPERIMENT,
            "epoch": 0,
            "best_epoch": 0,
            "best_sdr": best_sdr,
            "model_config": active_model_config,
            "architecture": active_architecture,
            "model_state": inference_state,
            "evaluation_weights": "ema" if ema is not None else "raw",
            "initial_diagnostics": initial_diagnostics,
        }
        atomic_torch_save(initial_best_payload, output / "vocals_best.pt")
        atomic_torch_save(inference_state, output / "vocals.pth")
        atomic_torch_save(
            {
                "best_state": inference_state,
                "best_nsdr": best_sdr,
                "epoch": 0,
                "model_config": active_model_config,
            },
            output / "best_model.th",
        )
        print(
            f"Initialization validation: vocal={best_sdr:.3f} dB; "
            "preserved as epoch-0 best.",
            flush=True,
        )
        print(
            "Initialization diagnostics: "
            f"train-remix={fixed_initial['train_remix']['active_sdr']:.2f} dB "
            f"(mix={fixed_initial['train_remix']['active_mixture_sdr']:.2f}, "
            f"gain={fixed_initial['train_remix']['active_sdr_improvement']:.2f}) | "
            f"train-clean={fixed_initial['train_clean']['active_sdr']:.2f} dB "
            f"(mix={fixed_initial['train_clean']['active_mixture_sdr']:.2f}, "
            f"gain={fixed_initial['train_clean']['active_sdr_improvement']:.2f}) | "
            f"valid-clean={initial_validation['active_sdr']:.2f} dB "
            f"(mix={initial_validation['active_mixture_sdr']:.2f}, "
            f"gain={initial_validation['active_sdr_improvement']:.2f}) | "
            f"valid-uniform={fixed_initial['valid_uniform']['active_sdr']:.2f} dB "
            f"quiet={int(fixed_initial['valid_uniform']['silent_examples'])}; "
            f"coverage remix/clean/valid="
            f"{fixed_initial['train_remix']['active_frame_fraction']:.2f}/"
            f"{fixed_initial['train_clean']['active_frame_fraction']:.2f}/"
            f"{initial_validation['active_frame_fraction']:.2f}",
            flush=True,
        )

    run_metadata = {
        "experiment": EXPERIMENT,
        "args": jsonable(vars(args)),
        "model_config": active_model_config,
        "architecture": active_architecture,
        "parameter_count": parameter_count(model),
        "loss_objective": (
            "SCNet per-example waveform-to-complex-STFT RMSE; uniform batch "
            "mean; 1e-12 zero-MSE sqrt guard"
        ),
        "loss_stft": {
            "n_fft": args.nfft,
            "hop_length": args.nhop,
            "win_length": args.win_length or args.nfft,
            "center": True,
            "normalized": True,
            "window": "explicit_rectangular_ones",
            "sqrt_guard": 1e-12,
        },
        "data_construction": {
            "coherent_original_mixture_probability": args.coherent_mix_probability,
            "cross_song_remix_probability": 1.0 - args.coherent_mix_probability,
            "remix_independent_gain_db": args.remix_gain_db,
            "activity_frame_duration": args.activity_frame_duration,
            "minimum_active_frame_fraction": args.min_active_frame_fraction,
            "maximum_quiet_frame_fraction": args.max_quiet_frame_fraction,
            "active_sample_probability": args.active_probability,
        },
        "checkpoint_selection": "validation active-vocal uSDR",
        "initialization": initialization,
        "initial_validation": initial_validation,
        "initial_diagnostics": initial_diagnostics,
        "device": str(device),
        "amp_dtype": str(amp_dtype),
        "effective_batch_size": args.batch_size * args.grad_accum_steps,
        "train_tracks": [record.path.name for record in train_data.tracks],
        "diagnostic_remix_tracks": [
            record.path.name for record in remix_train_data.tracks
        ],
        "diagnostic_train_tracks": [
            record.path.name for record in clean_train_data.tracks
        ],
        "valid_tracks": [record.path.name for record in valid_data.tracks],
        "validation_mode": (
            "full-track coverage via deterministic tail-aligned "
            f"{args.seq_dur:g}-second tiles"
        ),
        "validation_track_audio_seconds": sum(
            record.duration for record in valid_data.tracks
        ),
        "validation_tiled_audio_seconds": len(valid_data) * args.seq_dur,
        "validation_windows": len(valid_data),
        "train_examples_per_epoch": len(train_data),
        "legacy_one_second_stride_examples": sum(
            max(
                1,
                math.ceil(max(record.duration - args.seq_dur, 0.0) / 1.0) + 1,
            )
            for record in train_data.tracks
        ),
        "optimizer_steps_per_epoch": (len(train_loader) + args.grad_accum_steps - 1)
        // args.grad_accum_steps,
        "diagnostic_train_windows": len(clean_train_data),
        "diagnostic_uniform_validation_windows": len(valid_uniform_data),
    }
    (output / "separator.json").write_text(
        json.dumps(run_metadata, indent=2, sort_keys=True) + "\n"
    )
    (output / "commands.log").write_text(" ".join([sys.executable, *sys.argv]) + "\n")
    print(
        f"Parameters: {parameter_count(model):,}; deepest frequency bins: 57; "
        f"AMP: {amp_dtype}; physical/effective batch: "
        f"{args.batch_size}/{args.batch_size * args.grad_accum_steps}; workers: "
        f"{args.nb_workers} x prefetch {args.prefetch_factor}; diagnostics every "
        f"{args.diagnostic_every} epoch(s); coherent/remix="
        f"{args.coherent_mix_probability:.2f}/"
        f"{1.0 - args.coherent_mix_probability:.2f}; remix gain=±"
        f"{args.remix_gain_db:g} dB; active coverage≥"
        f"{args.min_active_frame_fraction:.2f}",
        flush=True,
    )
    print(
        f"Bounded epoch: {len(train_data)} examples, {len(train_loader)} batches, "
        f"{run_metadata['optimizer_steps_per_epoch']} optimizer steps; validation "
        f"covers all {len(valid_data.tracks)} held-out tracks using "
        f"{len(valid_data)} deterministic {args.seq_dur:g}-second tiles "
        f"({run_metadata['validation_track_audio_seconds'] / 60.0:.1f} minutes "
        "of unique audio).",
        flush=True,
    )
    print(
        "Loader diagnosis: the legacy 1-second stride would expose "
        f"{run_metadata['legacy_one_second_stride_examples']} examples/epoch "
        f"({run_metadata['legacy_one_second_stride_examples'] / len(train_data):.1f}x "
        "this bounded epoch).",
        flush=True,
    )
    print(
        "Architecture: SC-Unmix sparse encoder -> TFC x3 "
        "-> one 4-head post-TFC CSA-style attention block -> IDPM 2 heads x 3 "
        "repeats -> deepest-only direct-gate CSA + original FusionLayer decoder",
        flush=True,
    )

    for epoch in range(start_epoch, args.epochs):
        started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        train_started = time.perf_counter()
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            amp_dtype,
            args.activity_threshold,
            activity_frame_samples,
            epoch=epoch,
            optimizer=optimizer,
            scaler=scaler,
            ema=ema,
            grad_clip=args.grad_clip,
            grad_accum_steps=args.grad_accum_steps,
        )
        train_seconds = time.perf_counter() - train_started
        diagnostics_due = (
            epoch == start_epoch or (epoch + 1) % args.diagnostic_every == 0
        )
        fixed_diagnostics = None
        raw_valid_metrics = None
        diagnostic_seconds = 0.0
        validation_started = time.perf_counter()
        if ema is None:
            valid_metrics = run_epoch(
                model,
                valid_loader,
                criterion,
                device,
                amp_dtype,
                args.activity_threshold,
                activity_frame_samples,
                epoch=epoch,
            )
            raw_valid_metrics = valid_metrics
            if diagnostics_due:
                diagnostic_started = time.perf_counter()
                fixed_diagnostics = run_fixed_diagnostics(
                    model,
                    remix_train_loader,
                    clean_train_loader,
                    valid_uniform_loader,
                    criterion,
                    device,
                    amp_dtype,
                    args.activity_threshold,
                    activity_frame_samples,
                    epoch=epoch,
                )
                diagnostic_seconds += time.perf_counter() - diagnostic_started
        else:
            if diagnostics_due:
                raw_valid_metrics = run_epoch(
                    model,
                    valid_loader,
                    criterion,
                    device,
                    amp_dtype,
                    args.activity_threshold,
                    activity_frame_samples,
                    epoch=epoch,
                )
            with ema.average_parameters(model):
                valid_metrics = run_epoch(
                    model,
                    valid_loader,
                    criterion,
                    device,
                    amp_dtype,
                    args.activity_threshold,
                    activity_frame_samples,
                    epoch=epoch,
                )
                if diagnostics_due:
                    diagnostic_started = time.perf_counter()
                    fixed_diagnostics = run_fixed_diagnostics(
                        model,
                        remix_train_loader,
                        clean_train_loader,
                        valid_uniform_loader,
                        criterion,
                        device,
                        amp_dtype,
                        args.activity_threshold,
                        activity_frame_samples,
                        epoch=epoch,
                    )
                    diagnostic_seconds += time.perf_counter() - diagnostic_started
        validation_phase_seconds = time.perf_counter() - validation_started
        validation_seconds = max(
            validation_phase_seconds - diagnostic_seconds,
            0.0,
        )
        validation_passes = 2 if ema is not None and diagnostics_due else 1
        if valid_metrics["active_examples"] < 1:
            raise RuntimeError("Validation contains no active vocal examples.")
        scheduler.step(valid_metrics["active_sdr"])
        elapsed = time.perf_counter() - started
        peak_allocated_gib = (
            torch.cuda.max_memory_allocated(device) / 2**30
            if device.type == "cuda"
            else 0.0
        )
        peak_reserved_gib = (
            torch.cuda.max_memory_reserved(device) / 2**30
            if device.type == "cuda"
            else 0.0
        )
        improved = valid_metrics["active_sdr"] > best_sdr + args.early_stop_min_delta
        if improved:
            best_sdr = valid_metrics["active_sdr"]
            best_epoch = epoch + 1
            bad_epochs = 0
        else:
            bad_epochs += 1
        row = {
            "epoch": epoch + 1,
            "train": train_metrics,
            "valid": valid_metrics,
            "diagnostics": {
                "train_batches": train_metrics,
                "train_remix": (
                    fixed_diagnostics["train_remix"]
                    if fixed_diagnostics is not None
                    else None
                ),
                "train_clean": (
                    fixed_diagnostics["train_clean"]
                    if fixed_diagnostics is not None
                    else None
                ),
                "valid_clean_raw": raw_valid_metrics,
                "valid_clean": valid_metrics,
                "valid_uniform": (
                    fixed_diagnostics["valid_uniform"]
                    if fixed_diagnostics is not None
                    else None
                ),
            },
            "lr": optimizer.param_groups[0]["lr"],
            "train_seconds": train_seconds,
            "full_track_validation_seconds": validation_seconds,
            "full_track_validation_passes": validation_passes,
            "fixed_diagnostic_seconds": diagnostic_seconds,
            "seconds": elapsed,
            "peak_cuda_allocated_gib": peak_allocated_gib,
            "peak_cuda_reserved_gib": peak_reserved_gib,
        }
        history.append(row)
        checkpoint = {
            "experiment": EXPERIMENT,
            "epoch": epoch + 1,
            "best_epoch": best_epoch,
            "best_sdr": best_sdr,
            "bad_epochs": bad_epochs,
            "model_config": active_model_config,
            "architecture": active_architecture,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "ema_state": ema.state_dict() if ema is not None else None,
            "initial_validation": initial_validation,
            "initial_diagnostics": initial_diagnostics,
            "history": history,
        }
        atomic_torch_save(checkpoint, output / CHECKPOINT_NAME)
        if improved:
            inference_state = (
                ema.averaged_model_state()
                if ema is not None
                else {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()
                }
            )
            best_payload = {
                key: checkpoint[key]
                for key in (
                    "experiment",
                    "epoch",
                    "best_epoch",
                    "best_sdr",
                    "model_config",
                    "architecture",
                )
            }
            best_payload["model_state"] = inference_state
            best_payload["evaluation_weights"] = "ema" if ema is not None else "raw"
            atomic_torch_save(best_payload, output / "vocals_best.pt")
        record = dict(run_metadata)
        record.update(
            {
                "best_epoch": best_epoch,
                "best_validation_active_sdr": best_sdr,
                "epochs_trained": epoch + 1,
                "bad_epochs": bad_epochs,
                "history": history,
            }
        )
        (output / "vocals.json").write_text(
            json.dumps(jsonable(record), indent=2, sort_keys=True) + "\n"
        )
        print(
            f"Epoch {epoch + 1:03d}/{args.epochs}: "
            f"train loss={train_metrics['loss']:.5f} sdr={train_metrics['active_sdr']:.2f} | "
            f"valid loss={valid_metrics['loss']:.5f} sdr={valid_metrics['active_sdr']:.2f} dB | "
            f"lr={optimizer.param_groups[0]['lr']:.2g} "
            f"peak={peak_allocated_gib:.2f}/{peak_reserved_gib:.2f} GiB "
            f"time train/full-valid/diag/total="
            f"{train_seconds:.1f}/{validation_seconds:.1f}/"
            f"{diagnostic_seconds:.1f}/{elapsed:.1f}s "
            f"valid-passes={validation_passes} "
            f"{'*' if improved else ''}",
            flush=True,
        )
        if fixed_diagnostics is not None:
            train_remix = fixed_diagnostics["train_remix"]
            train_clean = fixed_diagnostics["train_clean"]
            valid_uniform = fixed_diagnostics["valid_uniform"]
            print(
                "  Panels: "
                f"train-remix mix/out/gain="
                f"{train_remix['active_mixture_sdr']:.2f}/"
                f"{train_remix['active_sdr']:.2f}/"
                f"{train_remix['active_sdr_improvement']:.2f} | "
                f"train-clean={train_clean['active_mixture_sdr']:.2f}/"
                f"{train_clean['active_sdr']:.2f}/"
                f"{train_clean['active_sdr_improvement']:.2f} | "
                f"valid-clean={valid_metrics['active_mixture_sdr']:.2f}/"
                f"{valid_metrics['active_sdr']:.2f}/"
                f"{valid_metrics['active_sdr_improvement']:.2f} | "
                f"valid-uniform={valid_uniform['active_mixture_sdr']:.2f}/"
                f"{valid_uniform['active_sdr']:.2f}/"
                f"{valid_uniform['active_sdr_improvement']:.2f}; "
                f"quiet n={int(valid_uniform['silent_examples'])} "
                f"leak={valid_uniform['silent_leakage_rms']:.3g} "
                f"atten={valid_uniform['silent_attenuation_db']:.2f} dB; "
                f"coverage remix/clean/valid="
                f"{train_remix['active_frame_fraction']:.2f}/"
                f"{train_clean['active_frame_fraction']:.2f}/"
                f"{valid_metrics['active_frame_fraction']:.2f}; "
                f"valid raw/EMA="
                f"{raw_valid_metrics['active_sdr']:.2f}/"
                f"{valid_metrics['active_sdr']:.2f}",
                flush=True,
            )
        if bad_epochs >= args.patience:
            print(
                f"Early stopping after {bad_epochs} epochs without SDR gain.",
                flush=True,
            )
            break
    print(f"Best validation SDR: {best_sdr:.3f} dB at epoch {best_epoch}", flush=True)


if __name__ == "__main__":
    main()
