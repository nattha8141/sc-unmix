"""Memory-bounded inference using the evaluation recipe's crossfades."""

import torch


def _crossfade_weight(
    length: int,
    overlap_samples: int,
    *,
    fade_in: bool,
    fade_out: bool,
) -> torch.Tensor:
    weight = torch.ones(length, dtype=torch.float32)
    fade_length = min(int(overlap_samples), length // 2)
    if fade_length:
        phase = torch.linspace(0.0, torch.pi / 2.0, fade_length + 2)[1:-1]
        fade = torch.sin(phase).square()
        if fade_in:
            weight[:fade_length] = fade
        if fade_out:
            weight[-fade_length:] = fade.flip(0)
    return weight


def _model_output(model: torch.nn.Module, batch: torch.Tensor) -> torch.Tensor:
    """Normalize SC-Unmix output to vocal waveform shape ``[B,2,L]``."""

    output = model(batch)
    if output.ndim == 4:
        if output.shape[1] < 1:
            raise RuntimeError("SC-Unmix returned zero source outputs.")
        output = output[:, 0]
    if output.ndim != 3 or output.shape[1] != 2:
        raise RuntimeError(
            f"Expected vocal output [B,2,L], received {tuple(output.shape)}"
        )
    return output


def separate_in_chunks(
    model: torch.nn.Module,
    waveform: torch.Tensor,
    *,
    sample_rate: int,
    segment_seconds: float,
    overlap: float,
    batch_size: int,
    amp: str,
) -> torch.Tensor:
    """Run memory-bounded overlap-add inference for ``[2,L]`` audio."""

    if waveform.ndim != 2 or waveform.shape[0] != 2:
        raise ValueError(f"Expected stereo [2,L] waveform, got {tuple(waveform.shape)}")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0, 1)")
    device = next(model.parameters()).device
    stft_config = getattr(model, "stft_config", {})
    n_fft = int(stft_config.get("n_fft", 4096))
    segment = max(int(round(segment_seconds * sample_rate)), n_fft)
    hop = max(int(round(segment * (1.0 - overlap))), 1)
    overlap_samples = segment - hop
    total_length = int(waveform.shape[-1])
    if total_length <= segment:
        starts = [0]
    else:
        starts = list(range(0, total_length - segment + 1, hop))
        final_start = total_length - segment
        if starts[-1] != final_start:
            starts.append(final_start)

    output = torch.zeros_like(waveform, dtype=torch.float32)
    weight_sum = torch.zeros(total_length, dtype=torch.float32)
    batch_size = max(int(batch_size), 1)
    amp_dtype: torch.dtype | None
    if device.type != "cuda" or amp == "none":
        amp_dtype = None
    elif amp == "float16":
        amp_dtype = torch.float16
    elif amp == "bfloat16":
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    with torch.inference_mode():
        for batch_start in range(0, len(starts), batch_size):
            selected = starts[batch_start : batch_start + batch_size]
            chunks = []
            valid_lengths = []
            for start in selected:
                chunk = waveform[:, start : start + segment]
                valid_lengths.append(int(chunk.shape[-1]))
                if chunk.shape[-1] < segment:
                    chunk = torch.nn.functional.pad(
                        chunk, (0, segment - int(chunk.shape[-1]))
                    )
                chunks.append(chunk)
            batch = torch.stack(chunks).to(device)
            use_amp = device.type == "cuda" and amp_dtype is not None
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype if amp_dtype is not None else torch.float32,
                enabled=use_amp,
            ):
                estimates = _model_output(model, batch).float().cpu()

            for local_index, (start, valid_length) in enumerate(
                zip(selected, valid_lengths)
            ):
                end = start + valid_length
                weight = _crossfade_weight(
                    valid_length,
                    overlap_samples,
                    fade_in=start > 0,
                    fade_out=end < total_length,
                )
                output[:, start:end] += (
                    estimates[local_index, :, :valid_length] * weight
                )
                weight_sum[start:end] += weight
    return output / weight_sum.clamp_min(1e-8)
