# SC-Unmix verification

The rename preserves the trained network. Local CPU checks compare SC-Unmix
against the preceding package using the same saved weights:

- **2,055,561 parameters**, identical state-dictionary keys and tensor values.
- The bundled `vocals_best.pt` is byte-identical to the preceding checkpoint.
- A stereo 11-second FP32 input produces bit-identical predictions: maximum
  absolute difference **0.0**.
- Seven unit tests pass, covering checkpoint forward/backward, audio preparation,
  overlap-add, residual accompaniment, file output and configuration parsing.
- The standalone CLI smoke test exercises training, full-track tiled validation,
  checkpoint saving, automatic resume and inference from the new training export.
- Python sources pass static unused-name/import checks and consistent formatting.
- The notebook retains upload, MUSDB sample, playback and stem-download flows,
  updated to `sc-unmix.zip` and `sc_unmix` imports.
- Notebook schema validation and compilation of all code cells pass. A fresh
  extraction of the ZIP imports SCUnmix and loads its checkpoint successfully.
- A local wheel builds successfully from the extracted distribution and contains
  `sc_unmix/SCUnmix.py`. The ZIP contains 25 files, approximately 7.34 MiB.

The user confirmed a successful MPS smoke run before this rename. Its CPU
re-STFT loss path and MPS checks are retained; the renamed package was checked
locally on CPU. No new CUDA benchmark, separation score or live Colab browser
run is claimed here.

Cleanup includes clearer model/data/loss/EMA filenames, removal of unused
imports and redundant `vocals.pth`/`best_model.th` exports, and self-contained
packaging/smoke tools. Source comments retain genuine SCNet attribution. The
historical checkpoint metadata remains intact for compatibility and provenance.

The ZIP includes only distribution files and the inference checkpoint. Existing
virtual environments, training results, datasets and original experiment code
are excluded and preserved outside this GitHub folder. In the parent thesis
workspace, the former package directory remains available for the existing
Apple virtual environment and runs; use `sc-unmix` for future distribution.
