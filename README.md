# SC-Unmix

SC-Unmix is a lightweight vocal separation model based on SCNet, with
**2,055,561 parameters**. It combines a sparse encoder/decoder, TFC ×3,
compressed post-TFC attention (4 heads, embedding setting 256), and a split-head
BiGRU dual-path module with three repeats. The deepest skip uses direct-gate
CSA; the upper two use SCNet's original fusion layer.

The model predicts vocals. Accompaniment is obtained by subtracting predicted
vocals from the mixture, so its quality depends on the vocal estimate.

## Colab demo

1. Open `SC_Unmix.ipynb` in Google Colab and select a GPU runtime.
2. Run the cells and upload `sc-unmix.zip` when prompted (code and weights included).
3. Upload WAV/FLAC audio or select the MUSDB sample option.
4. Listen to previews and download both stems.

The notebook installs dependencies in Colab. No local installation, builder
script or training dataset is required. The MUSDB option downloads the sample
dataset through `musdb`, following the Open-Unmix demo workflow.

## Local installation and inference

Use Python 3.10 or newer. From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m sc_unmix.separate --checkpoint checkpoints/vocals_best.pt --input /path/to/song.flac --output-dir outputs/song
```

The repository/archive is **`sc-unmix`**; the Python package is **`sc_unmix`**
because hyphens cannot appear in Python import identifiers. Import the model
with `from sc_unmix import SCUnmix`; its source is `sc_unmix/SCUnmix.py`.

Defaults are 11-second segments, 50% overlap and batch size 1. Device selection
prefers CUDA, then Apple MPS, then CPU. Choose explicitly with `--device cpu`,
`--device mps` or `--device cuda`. CUDA supports automatic mixed precision;
`--amp none` selects FP32. MPS uses FP32.

Input is resampled to 44.1 kHz. Mono is duplicated to stereo; more than two
channels are rejected. `vocals.wav` and `accompaniment.wav` are float32 WAVs,
without clipping or independent normalization. Their sum reconstructs the
resampled mixture within floating-point rounding. Peaks can exceed 1; use a
common playback gain when needed. Existing stems are not overwritten. GPU work
is chunked; CPU RAM use still grows with track length.

```python
import soundfile as sf
from sc_unmix.separate import load_model, separate_audio, save_estimates

model = load_model("checkpoints/vocals_best.pt")
audio, rate = sf.read("song.flac", dtype="float32", always_2d=True)
estimates = separate_audio(model, audio, rate)
save_estimates(estimates, "outputs/song")
# mixture, vocals and accompaniment are CPU tensors shaped [2, samples].
```

Optional: `python -m pip install -e .` enables importing `sc_unmix` from other
directories. Checkpoint/config paths must still refer to their actual locations.
This installs the project locally; it does not publish it to PyPI.

## Training

Edit `configs/train.yaml`, especially `root` and `output`, then run:

```bash
python -m sc_unmix.train --config-path configs/train.yaml
```

CLI options override YAML:

```bash
python -m sc_unmix.train --config-path configs/train.yaml --root /datasets/musdb18hq --save-path runs/my_run
```

Relative YAML paths resolve against the YAML file; relative CLI paths use the
working directory. The architecture is fixed in code to match the checkpoint.
Training uses one device. Expected 44.1 kHz WAV or FLAC dataset layout:

```text
musdb18hq/train/Track name/
    mixture.wav
    vocals.wav
    drums.wav
    bass.wav
    other.wav
```

The manifest holds out the original 14 validation songs from every training
stem pool. Training retains epoch-varying sampling and vocal-only complex-STFT
RMSE. Full-track validation uses deterministic, tail-aligned 11-second tiles;
separate diagnostic panels use fixed windows.

The YAML starts fresh with 18 examples per track, physical/effective batch 7/7,
coherent/remix probabilities 0.10/0.90, initial LR 2e-4, and a maximum of 320
epochs. EMA, LR scheduling and early stopping are enabled. This configuration
does not reproduce the original multi-notebook continuation history exactly.

Outputs include `training_state.pt` for resuming, `vocals_best.pt` for inference,
JSON metrics/configuration, and `commands.log`. Redundant historical exports
`vocals.pth` and `best_model.th` have been removed. An existing output directory
with `training_state.pt` resumes automatically; use a new directory for scratch
training. Use `--checkpoint /path/to/training_state.pt` to resume explicitly,
with `--epochs` specifying the total target epoch count. The bundled inference
checkpoint does not contain optimizer state.

On Apple MPS, the re-STFT loss runs on CPU with gradients propagated back to the
MPS model, retaining the path that passed the local Apple smoke test. Extra MPS
checks identify non-finite inputs, outputs, gradients or parameters. Start with
batch size 1 for local checks. CPU is available via `--device cpu`.

## Checks and packaging

```bash
python -m unittest discover -s tests -v
python tools/smoke_test.py --device cpu
python tools/smoke_test.py --device mps
python tools/package_release.py
```

The smoke test creates temporary synthetic tracks and exercises training,
validation, checkpoint saving and inference, then removes the temporary files.
It checks implementation correctness rather than meaningful separation SDR.
Packaging writes `../sc-unmix.zip` from distribution files only, excluding
environments, datasets, outputs and runs. The notebook is the canonical demo;
packaging does not regenerate it or overwrite configuration.

| Location | Purpose |
| --- | --- |
| `sc_unmix/SCUnmix.py` | STFT, sparse encoder/decoder, fusion |
| `sc_unmix/separator.py` | Projection, TFC, attention and IDPM |
| `sc_unmix/csa_fusion.py` | Attention and gate primitives |
| `sc_unmix/tfc.py`, `sc_unmix/idpm.py` | Convolutions and dual-path GRUs |
| `sc_unmix/data.py`, `sc_unmix/loss.py` | Sampling, validation and objective |
| `sc_unmix/train.py`, `sc_unmix/ema.py` | Training and EMA |
| `sc_unmix/separate.py`, `sc_unmix/overlap_add.py` | Inference API and CLI |

## Attribution and provenance

SC-Unmix adapts the sparse encoder/decoder from
[SCNet](https://github.com/starrytong/SCNet), described in
[Sparse Compression Network for Music Source Separation](https://arxiv.org/abs/2401.13276).
The upstream MIT license and copyright remain in `LICENSE`.
The demo is inspired by [Open-Unmix](https://github.com/sigsep/open-unmix-pytorch).

Original experiment code:
`self-modified-SCNet-l4-deep-csa-direct-gate-no-tdf-full-track-validation-post-tfc-attention-vocals`.
Original checkpoint run:
`self_modified_scnet_l4_deep_csa_direct_gate_no_tdf_full_track_validation_post_tfc_attention_continued_e320_vocals`.

Checkpoint tensors and historical metadata are preserved. A model class rename
does not change state-dictionary keys. The historical separator type in
`model_config` is retained for checkpoint compatibility. New training runs use
`sc-unmix` as the experiment identifier. See `VERIFICATION.md` for checks and limits.
