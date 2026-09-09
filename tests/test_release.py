"""CPU checks: python -m unittest discover -s tests -v."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch

from sc_unmix.separate import load_model, prepare_audio, separate_audio, save_estimates
from sc_unmix.train import parse_args, resolve_device

ROOT = Path(__file__).resolve().parents[1]


class IdentityModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return x[:, None]


class ReleaseTests(unittest.TestCase):
    def test_yaml_and_cli_override(self):
        with patch.object(
            sys,
            "argv",
            [
                "train",
                "--config-path",
                str(ROOT / "configs/train.yaml"),
                "--epochs",
                "2",
            ],
        ):
            args = parse_args()
        self.assertEqual(args.epochs, 2)
        self.assertEqual(args.samples_per_track, 18)
        self.assertEqual(args.batch_size, 7)
        self.assertEqual(
            Path(args.validation_manifest), ROOT / "configs/validation_tracks.json"
        )

    def test_cpu_device_resolution(self):
        with patch.object(
            sys,
            "argv",
            [
                "train",
                "--config-path",
                str(ROOT / "configs/train.yaml"),
                "--device",
                "cpu",
            ],
        ):
            args = parse_args()
        self.assertEqual(resolve_device(args).type, "cpu")

    def test_unknown_yaml_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "bad.yaml"
            config.write_text("typo_option: 1\n")
            with patch.object(sys, "argv", ["train", "--config-path", str(config)]):
                with self.assertRaises(SystemExit):
                    parse_args()

    def test_overlap_add_and_residual(self):
        rng = np.random.default_rng(42)
        for length in (1, 8192, 19233):
            audio = rng.normal(0, 0.1, (length, 2)).astype("float32")
            estimates = separate_audio(
                IdentityModel(), audio, 44100, segment_seconds=0.2, amp="none"
            )
            torch.testing.assert_close(
                estimates["vocals"], estimates["mixture"], atol=1e-7, rtol=1e-6
            )
            torch.testing.assert_close(
                estimates["vocals"] + estimates["accompaniment"], estimates["mixture"]
            )

    def test_mono_resampling_and_validation(self):
        self.assertEqual(prepare_audio(np.zeros(4800), 48000).shape, (2, 4410))
        for audio in (np.zeros((3, 4)), np.zeros(0), np.array([np.nan])):
            with self.assertRaises(ValueError):
                prepare_audio(audio, 44100)

    def test_float_wav_no_clipping(self):
        stems = {
            "vocals": torch.full((2, 100), 1.8),
            "accompaniment": torch.full((2, 100), -0.8),
        }
        with tempfile.TemporaryDirectory() as directory:
            paths = save_estimates(stems, directory)
            audio, rate = sf.read(paths["vocals"])
            self.assertEqual(rate, 44100)
            self.assertAlmostEqual(float(audio.max()), 1.8, places=6)
            with self.assertRaises(FileExistsError):
                save_estimates(stems, directory)

    def test_checkpoint_forward_backward(self):
        torch.set_num_threads(2)
        model = load_model(ROOT / "checkpoints/vocals_best.pt", "cpu")
        self.assertEqual(sum(p.numel() for p in model.parameters()), 2055561)
        model.train()
        output = model(torch.randn(1, 2, 8192) * 0.01)
        self.assertEqual(output.shape, (1, 1, 2, 8192))
        output.square().mean().backward()
        self.assertTrue(
            all(
                torch.isfinite(p.grad).all()
                for p in model.parameters()
                if p.grad is not None
            )
        )


if __name__ == "__main__":
    unittest.main()
