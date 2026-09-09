"""Exercise the release CLI on tiny synthetic data, without touching real runs."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf

RELEASE = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    args = parser.parse_args()
    rng = np.random.default_rng(42)
    environment = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    with tempfile.TemporaryDirectory(prefix="sc-unmix-smoke-") as directory:
        root = Path(directory)
        for name in ("train_a", "train_b", "held_out"):
            track = root / "data/train" / name
            track.mkdir(parents=True)
            stems = [
                rng.normal(0, 0.02, (22050, 2)).astype("float32") for _ in range(4)
            ]
            for stem, audio in zip(
                ("vocals", "drums", "bass", "other", "mixture"), [*stems, sum(stems)]
            ):
                sf.write(track / f"{stem}.wav", audio, 44100, subtype="FLOAT")
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps({"validation_tracks": ["held_out"]}))
        command = [
            sys.executable,
            "-m",
            "sc_unmix.train",
            "--config-path",
            "configs/train.yaml",
            "--root",
            str(root / "data"),
            "--output",
            str(root / "run"),
            "--validation-manifest",
            str(manifest),
            "--epochs",
            "1",
            "--seq-dur",
            "0.2",
            "--samples-per-track",
            "1",
            "--batch-size",
            "1",
            "--nb-workers",
            "0",
            "--diagnostic-workers",
            "0",
            "--diagnostic-every",
            "1",
            "--diagnostic-train-tracks",
            "1",
            "--diagnostic-train-windows",
            "1",
            "--diagnostic-uniform-windows",
            "1",
            "--valid-tracks",
            "1",
            "--valid-windows-per-track",
            "1",
            "--valid-probe-windows",
            "2",
            "--amp",
            "none",
            "--device",
            args.device,
        ]
        subprocess.run(command, cwd=RELEASE, env=environment, check=True)
        assert (root / "run/training_state.pt").is_file()
        assert (root / "run/vocals_best.pt").is_file()
        assert not (root / "run/vocals.pth").exists()
        assert not (root / "run/best_model.th").exists()
        # Exercise automatic resume and the new inference checkpoint export.
        command[command.index("--epochs") + 1] = "2"
        subprocess.run(command, cwd=RELEASE, env=environment, check=True)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "sc_unmix.separate",
                "--checkpoint",
                str(root / "run/vocals_best.pt"),
                "--input",
                str(root / "data/train/held_out/mixture.wav"),
                "--output-dir",
                str(root / "stems"),
                "--segment-seconds",
                "0.2",
                "--device",
                args.device,
                "--amp",
                "none",
            ],
            cwd=RELEASE,
            env=environment,
            check=True,
        )
        vocals, _ = sf.read(root / "stems/vocals.wav")
        accompaniment, _ = sf.read(root / "stems/accompaniment.wav")
        mixture, _ = sf.read(root / "data/train/held_out/mixture.wav")
        error = float(np.max(np.abs(vocals + accompaniment - mixture)))
        assert error < 1e-6
        print("CLI output reconstruction max error:", error)
        print(
            "Training + full-track tiled validation + checkpoint save + inference CLI: PASS"
        )


if __name__ == "__main__":
    main()
