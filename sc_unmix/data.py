"""Bounded MUSDB18-HQ mixture/vocal windows for SC-Unmix.

Training indices include the epoch, so crop positions and remix gains change
on every pass without defining an epoch as a dense one-second scan. The
dataset mixes unchanged same-song examples with cross-song stem remixes and
uses short-frame vocal coverage for active/quiet sampling. When a validation
manifest is supplied, validation windows are deterministic and bounded, and
the listed songs are excluded from every training stem pool. The official test
split remains untouched.
"""

from __future__ import annotations

import math
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Tuple

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset, Sampler


SOURCE_NAMES = ("vocals", "drums", "bass", "other")


@dataclass(frozen=True)
class TrackRecord:
    path: Path
    sample_rate: int
    frames: int

    @property
    def duration(self) -> float:
        return self.frames / float(self.sample_rate)


def _audio_file(path: Path, stem: str) -> Path:
    for suffix in ("wav", "flac"):
        candidate = path / f"{stem}.{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(path / f"{stem}.wav")


def _record_track(path: Path, sample_rate: int) -> TrackRecord | None:
    try:
        files = [_audio_file(path, "mixture")]
        files.extend(_audio_file(path, stem) for stem in SOURCE_NAMES)
        infos = [sf.info(str(file)) for file in files]
    except (FileNotFoundError, RuntimeError, sf.LibsndfileError):
        return None
    if any(info.samplerate != sample_rate for info in infos):
        raise ValueError(f"All stems must be {sample_rate} Hz: {path}")
    return TrackRecord(path, sample_rate, min(int(info.frames) for info in infos))


def load_validation_manifest(path: str | Path) -> tuple[str, ...]:
    """Read and validate the deterministic validation-track manifest."""
    manifest_path = Path(path).expanduser().resolve()
    payload = json.loads(manifest_path.read_text())
    names = payload.get("validation_tracks") if isinstance(payload, dict) else None
    if not isinstance(names, list) or not names:
        raise ValueError(
            f"Manifest must contain a non-empty validation_tracks list: {manifest_path}"
        )
    normalized = tuple(str(name) for name in names)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"Manifest contains duplicate track names: {manifest_path}")
    return normalized


def discover_tracks(
    root: str | Path,
    *,
    split: str,
    sample_rate: int = 44100,
    valid_tracks: int = 14,
    validation_manifest: str | Path | None = None,
) -> List[TrackRecord]:
    root = Path(root).expanduser().resolve()
    base = root / ("train" if split == "valid" else split)
    if not base.is_dir():
        raise FileNotFoundError(f"Missing dataset split: {base}")
    records = []
    for path in sorted(item for item in base.iterdir() if item.is_dir()):
        record = _record_track(path, sample_rate)
        if record is not None:
            records.append(record)
    if split in {"train", "valid"}:
        if validation_manifest is not None:
            manifest_names = set(load_validation_manifest(validation_manifest))
            available_names = {record.path.name for record in records}
            missing = sorted(manifest_names - available_names)
            if missing:
                raise ValueError(
                    "Validation manifest names are missing or incomplete in "
                    f"{base}: {missing}"
                )
            if split == "valid":
                records = [
                    record for record in records if record.path.name in manifest_names
                ]
            else:
                # Exclude the complete songs from every training stem pool,
                # including cross-song accompaniment remixing.
                records = [
                    record
                    for record in records
                    if record.path.name not in manifest_names
                ]
        else:
            count = min(max(int(valid_tracks), 1), max(len(records) - 1, 1))
            records = records[-count:] if split == "valid" else records[:-count]
    if not records:
        raise RuntimeError(f"No complete MUSDB tracks found under {base}")
    return records


def _read_chunk(
    path: Path,
    start: int,
    frames: int,
    channels: int = 2,
) -> torch.Tensor:
    audio, _ = sf.read(
        str(path),
        start=max(int(start), 0),
        frames=max(int(frames), 1),
        dtype="float32",
        always_2d=True,
    )
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    if audio.shape[1] < channels:
        audio = np.pad(audio, ((0, 0), (0, channels - audio.shape[1])))
    audio = audio[:, :channels]
    if audio.shape[0] < frames:
        audio = np.pad(audio, ((0, frames - audio.shape[0]), (0, 0)))
    return torch.from_numpy(np.ascontiguousarray(audio[:frames].T))


def _rms(audio: torch.Tensor) -> float:
    return float(audio.float().square().mean().sqrt().item())


def _frame_activity_fraction(
    audio: torch.Tensor,
    *,
    frame_samples: int,
    threshold: float,
) -> float:
    """Fraction of short frames whose stereo RMS reaches ``threshold``."""
    if audio.ndim != 2:
        raise ValueError(f"Expected audio [C,L], got {tuple(audio.shape)}")
    frame_samples = max(int(frame_samples), 1)
    frame_count = max(math.ceil(audio.shape[-1] / frame_samples), 1)
    padding = frame_count * frame_samples - audio.shape[-1]
    if padding:
        audio = torch.nn.functional.pad(audio, (0, padding))
    framed = audio.float().reshape(audio.shape[0], frame_count, frame_samples)
    rms = framed.square().mean(dim=(0, 2)).sqrt()
    return float((rms >= threshold).float().mean().item())


class EpochShuffleSampler(Sampler[int]):
    """Shuffle reproducibly and encode the epoch in every dataset index."""

    def __init__(self, dataset: Dataset, seed: int = 42) -> None:
        self.dataset = dataset
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(len(self.dataset), generator=generator).tolist()
        offset = self.epoch * len(self.dataset)
        return iter(offset + index for index in order)

    def __len__(self) -> int:
        return len(self.dataset)


class MUSDBVocalDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        *,
        split: str,
        seq_duration: float = 11.0,
        samples_per_track: int = 16,
        valid_windows_per_track: int = 4,
        valid_probe_windows: int = 24,
        valid_probe_duration: float = 1.0,
        sample_rate: int = 44100,
        valid_tracks: int = 14,
        validation_manifest: str | Path | None = None,
        random_track_mix: bool = True,
        coherent_mix_probability: float = 0.5,
        remix_gain_db: float = 3.0,
        active_probability: float = 0.85,
        activity_threshold: float = 1e-3,
        activity_attempts: int = 12,
        activity_frame_duration: float = 0.2,
        min_active_frame_fraction: float = 0.3,
        max_quiet_frame_fraction: float = 0.05,
        channel_swap_probability: float = 0.5,
        clean_probe: bool = False,
        window_selection: str = "active",
        full_track_validation: bool = False,
        max_tracks: int | None = None,
        seed: int = 42,
    ) -> None:
        if split not in {"train", "valid", "test"}:
            raise ValueError("split must be train, valid, or test")
        if seq_duration <= 0:
            raise ValueError("seq_duration must be positive")
        if not 0.0 <= active_probability <= 1.0:
            raise ValueError("active_probability must be in [0,1]")
        if not 0.0 <= coherent_mix_probability <= 1.0:
            raise ValueError("coherent_mix_probability must be in [0,1]")
        if activity_frame_duration <= 0:
            raise ValueError("activity_frame_duration must be positive")
        if not 0.0 <= max_quiet_frame_fraction <= min_active_frame_fraction <= 1.0:
            raise ValueError(
                "Expected 0 <= max quiet fraction <= min active fraction <= 1"
            )
        if clean_probe and split != "train":
            raise ValueError("clean_probe is only meaningful for split='train'")
        if full_track_validation and split != "valid":
            raise ValueError("full_track_validation requires split='valid'")
        if window_selection not in {"active", "uniform"}:
            raise ValueError("window_selection must be 'active' or 'uniform'")
        self.root = Path(root).expanduser().resolve()
        self.split = split
        self.sample_rate = int(sample_rate)
        self.seq_frames = int(round(seq_duration * self.sample_rate))
        self.samples_per_track = max(int(samples_per_track), 1)
        self.valid_windows_per_track = max(int(valid_windows_per_track), 1)
        self.valid_probe_windows = max(
            int(valid_probe_windows), self.valid_windows_per_track
        )
        self.valid_probe_frames = min(
            max(int(round(valid_probe_duration * self.sample_rate)), 1),
            self.seq_frames,
        )
        self.clean_probe = bool(clean_probe)
        self.window_selection = str(window_selection)
        self.full_track_validation = bool(full_track_validation)
        self.random_track_mix = bool(
            random_track_mix and split == "train" and not self.clean_probe
        )
        self.coherent_mix_probability = float(coherent_mix_probability)
        self.remix_gain_db = max(float(remix_gain_db), 0.0)
        self.active_probability = float(active_probability)
        self.activity_threshold = max(float(activity_threshold), 0.0)
        self.activity_attempts = max(int(activity_attempts), 1)
        self.activity_frame_samples = max(
            int(round(activity_frame_duration * self.sample_rate)), 1
        )
        self.min_active_frame_fraction = float(min_active_frame_fraction)
        self.max_quiet_frame_fraction = float(max_quiet_frame_fraction)
        self.channel_swap_probability = float(channel_swap_probability)
        self.seed = int(seed)
        self.validation_manifest = (
            Path(validation_manifest).expanduser().resolve()
            if validation_manifest is not None
            else None
        )
        self.tracks = discover_tracks(
            self.root,
            split=split,
            sample_rate=self.sample_rate,
            valid_tracks=valid_tracks,
            validation_manifest=self.validation_manifest,
        )
        if max_tracks is not None:
            track_count = max(int(max_tracks), 1)
            if track_count < len(self.tracks):
                selector = random.Random(self.seed + 71_003)
                indices = sorted(selector.sample(range(len(self.tracks)), track_count))
                self.tracks = [self.tracks[index] for index in indices]
        self.full_track_windows = (
            self._build_full_track_windows() if self.full_track_validation else None
        )
        self.validation_starts = (
            self._build_validation_starts()
            if (split != "train" or self.clean_probe) and not self.full_track_validation
            else None
        )

    def __len__(self) -> int:
        if self.full_track_windows is not None:
            return len(self.full_track_windows)
        multiplier = (
            self.samples_per_track
            if self.split == "train" and not self.clean_probe
            else self.valid_windows_per_track
        )
        return len(self.tracks) * multiplier

    def _decode_index(self, encoded_index: int) -> Tuple[int, int]:
        if self.split != "train" or self.clean_probe:
            return 0, int(encoded_index)
        return divmod(int(encoded_index), len(self))

    def _rng(self, epoch: int, index: int, salt: int) -> random.Random:
        seed = self.seed + 1_000_003 * epoch + 10_007 * index + 97 * salt
        return random.Random(seed)

    def _candidate_starts(self, record: TrackRecord, count: int) -> List[int]:
        max_start = max(record.frames - self.seq_frames, 0)
        if max_start == 0 or count == 1:
            return [0]
        return [
            int(round(value))
            for value in np.linspace(0, max_start, num=count, dtype=np.float64)
        ]

    def _build_full_track_windows(self) -> List[Tuple[int, int]]:
        """Tile every validation song with fixed-size, L4-safe excerpts.

        The regular grid uses non-overlapping ``seq_frames`` steps. The last
        excerpt is aligned to the exact track end so no real audio is omitted
        and no artificial zero tail is scored. This can overlap the preceding
        excerpt by less than one excerpt, but keeps every model input the same
        duration used during training.
        """
        windows: List[Tuple[int, int]] = []
        for track_index, record in enumerate(self.tracks):
            max_start = max(record.frames - self.seq_frames, 0)
            starts = list(range(0, max_start + 1, self.seq_frames)) or [0]
            if starts[-1] != max_start:
                starts.append(max_start)
            windows.extend((track_index, start) for start in starts)
        return windows

    def _build_validation_starts(self) -> List[List[int]]:
        starts_by_track = []
        for record in self.tracks:
            if self.window_selection == "uniform":
                starts_by_track.append(
                    self._candidate_starts(record, self.valid_windows_per_track)
                )
                continue
            candidates = self._candidate_starts(record, self.valid_probe_windows)
            scored = []
            for start in candidates:
                probe_start = start + max(
                    (self.seq_frames - self.valid_probe_frames) // 2, 0
                )
                probe = _read_chunk(
                    _audio_file(record.path, "vocals"),
                    probe_start,
                    self.valid_probe_frames,
                )
                scored.append((_rms(probe), start))
            score_by_start = {}
            for score, start in scored:
                score_by_start[start] = max(score, score_by_start.get(start, 0.0))
            ranked = sorted(
                ((score, start) for start, score in score_by_start.items()),
                reverse=True,
            )
            active_starts = sorted(
                start
                for start, score in score_by_start.items()
                if score >= self.activity_threshold
            )
            if len(active_starts) >= self.valid_windows_per_track:
                positions = np.linspace(
                    0,
                    len(active_starts) - 1,
                    num=self.valid_windows_per_track,
                )
                selected = [
                    active_starts[int(round(position))] for position in positions
                ]
            else:
                selected = list(active_starts)
                for _, start in ranked:
                    if start not in selected:
                        selected.append(start)
                    if len(selected) == self.valid_windows_per_track:
                        break
                selected.sort()
            # Very short tracks may produce only one distinct candidate.  Keep
            # Dataset.__len__ and indexing valid by repeating the last window.
            if not selected:
                selected = [0]
            selected.extend(
                selected[-1:] * max(self.valid_windows_per_track - len(selected), 0)
            )
            starts_by_track.append(selected)
        return starts_by_track

    def _training_vocal(
        self,
        record: TrackRecord,
        epoch: int,
        index: int,
    ) -> Tuple[int, torch.Tensor]:
        rng = self._rng(epoch, index, 1)
        want_active = rng.random() < self.active_probability
        max_start = max(record.frames - self.seq_frames, 0)
        best: Tuple[float, int, torch.Tensor] | None = None
        for _ in range(self.activity_attempts):
            start = rng.randint(0, max_start) if max_start else 0
            vocals = _read_chunk(
                _audio_file(record.path, "vocals"), start, self.seq_frames
            )
            score = _frame_activity_fraction(
                vocals,
                frame_samples=self.activity_frame_samples,
                threshold=self.activity_threshold,
            )
            matches = (
                score >= self.min_active_frame_fraction
                if want_active
                else score <= self.max_quiet_frame_fraction
            )
            if matches:
                return start, vocals
            if (
                best is None
                or (want_active and score > best[0])
                or (not want_active and score < best[0])
            ):
                best = (score, start, vocals)
        assert best is not None
        return best[1], best[2]

    def _source_record(
        self,
        base: TrackRecord,
        epoch: int,
        index: int,
        source_index: int,
    ) -> TrackRecord:
        if source_index == 0 or not self.random_track_mix:
            return base
        rng = self._rng(epoch, index, 10 + source_index)
        return self.tracks[rng.randrange(len(self.tracks))]

    def __getitem__(self, encoded_index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.full_track_windows is not None:
            track_index, start = self.full_track_windows[int(encoded_index)]
            base = self.tracks[track_index]
            mixture = _read_chunk(
                _audio_file(base.path, "mixture"), start, self.seq_frames
            )
            vocals = _read_chunk(
                _audio_file(base.path, "vocals"), start, self.seq_frames
            )
            return mixture, vocals

        epoch, index = self._decode_index(encoded_index)
        multiplier = (
            self.samples_per_track
            if self.split == "train" and not self.clean_probe
            else self.valid_windows_per_track
        )
        track_index = (index // multiplier) % len(self.tracks)
        base = self.tracks[track_index]

        if self.split != "train" or self.clean_probe:
            assert self.validation_starts is not None
            window_index = index % self.valid_windows_per_track
            start = self.validation_starts[track_index][window_index]
            mixture = _read_chunk(
                _audio_file(base.path, "mixture"), start, self.seq_frames
            )
            vocals = _read_chunk(
                _audio_file(base.path, "vocals"), start, self.seq_frames
            )
            return mixture, vocals

        base_start, base_vocals = self._training_vocal(base, epoch, index)
        mix_rng = self._rng(epoch, index, 7)
        use_coherent = mix_rng.random() < self.coherent_mix_probability
        if use_coherent:
            mixture = _read_chunk(
                _audio_file(base.path, "mixture"),
                base_start,
                self.seq_frames,
            )
            vocals = base_vocals
            swap_rng = self._rng(epoch, index, 50)
            if swap_rng.random() < self.channel_swap_probability:
                mixture = mixture.flip(0)
                vocals = vocals.flip(0)
            peak = max(float(mixture.abs().max()), float(vocals.abs().max()), 1.0)
            return mixture / peak, vocals / peak

        sources = []
        for source_index, stem in enumerate(SOURCE_NAMES):
            record = self._source_record(base, epoch, index, source_index)
            if source_index == 0:
                audio = base_vocals
            else:
                max_start = max(record.frames - self.seq_frames, 0)
                if record.path == base.path and not self.random_track_mix:
                    start = base_start
                else:
                    rng = self._rng(epoch, index, 20 + source_index)
                    start = rng.randint(0, max_start) if max_start else 0
                audio = _read_chunk(
                    _audio_file(record.path, stem), start, self.seq_frames
                )
            if self.remix_gain_db:
                gain_rng = self._rng(epoch, index, 30 + source_index)
                gain_db = gain_rng.uniform(-self.remix_gain_db, self.remix_gain_db)
                audio = audio * math.pow(10.0, gain_db / 20.0)
            sources.append(audio)

        mixture = torch.stack(sources).sum(dim=0)
        vocals = sources[0]
        swap_rng = self._rng(epoch, index, 50)
        if swap_rng.random() < self.channel_swap_probability:
            mixture = mixture.flip(0)
            vocals = vocals.flip(0)
        peak = max(float(mixture.abs().max()), float(vocals.abs().max()), 1.0)
        return mixture / peak, vocals / peak


__all__ = [
    "EpochShuffleSampler",
    "MUSDBVocalDataset",
    "SOURCE_NAMES",
    "TrackRecord",
    "discover_tracks",
    "load_validation_manifest",
]
