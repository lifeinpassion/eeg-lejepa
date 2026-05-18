# Setup on M1 — first run

Estimated time: 15–30 minutes (most of it the data download).

## 1. Prerequisites

You need conda (Miniconda or Anaconda) or a recent `python3.11`. Conda is recommended because MNE and SciPy compile cleanly through conda-forge.

If you don't have conda yet, get Miniconda for Apple Silicon: <https://docs.conda.io/en/latest/miniconda.html> (the macOS arm64 installer).

## 2. Create the environment

```bash
cd "/Users/billion/Downloads/SLM/eeg-slm"
conda create -n eeg-slm python=3.11 -y
conda activate eeg-slm
```

## 3. Install the project (editable mode)

```bash
pip install -e ".[dev,training]"
```

This pulls PyTorch (with MPS support on M1), MNE, the rest of the scientific stack, plus dev tools (pytest, ruff, ipykernel, jupytext) and the training extras (accelerate, wandb, safetensors).

Expected install size: ~3 GB. First install may take 5–10 minutes.

## 4. Verify the environment

```bash
make info
```

Expected output (your versions may differ slightly):

```
Python: 3.11.x
Platform: macOS-x.x-arm64-arm-64bit
PyTorch: 2.x.x
MNE: 1.x.x
MPS available: True
CUDA available: False
```

The important line is `MPS available: True`. If it says False, your PyTorch install didn't pick up the Apple Silicon wheel — try reinstalling with `pip install --force-reinstall torch`.

## 5. Run the unit tests

```bash
make test
```

These are smoke tests (imports, config parsing). They should all pass in under a second. The network-dependent download test is gated behind an env var — to include it:

```bash
EEG_SLM_RUN_NETWORK_TESTS=1 make test
```

## 6. Download the first dataset

```bash
make download
```

This pulls a few subject-runs from PhysioNet EEGMMIDB into `data/raw/`. The default config downloads subjects 1–3, runs 3, 4, 7, 8, 11, 12 (~50 MB total — small enough to iterate on; we'll scale up once Phase 1 architecture is real).

To download more subjects, edit `configs/default.yaml` or pass overrides:

```bash
python scripts/01_download_data.py --subjects 1 2 3 4 5 --runs 3 4 7 8
```

## 7. Run the first-look exploration

```bash
make explore
```

This runs `scripts/02_explore_data.py` end-to-end:

1. Loads subject 1 with motor-imagery runs
2. Preprocesses (bandpass 1–80 Hz, notch 60 Hz, average reference, resample to 200 Hz)
3. Plots the first 10 seconds across five 10-10 electrodes
4. Shows the power-spectral density
5. Cuts into 4-second pretraining-style chunks
6. Pushes a small batch to MPS and runs a trivial LayerNorm to confirm the device path

If everything works, you'll see two matplotlib windows and a final line like:

```
PyTorch tensor on mps: shape=(8, 64, 800), dtype=torch.float32
Output shape after LayerNorm: (8, 64, 800)
Output mean: ..., std: ...
```

## 8. Open the exploration as a notebook (optional)

The script is in jupytext "percent" format. If you prefer a Jupyter notebook:

```bash
jupyter lab
```

Then open `scripts/02_explore_data.py` in JupyterLab — with the jupytext extension (already installed) it renders as a notebook with executable cells.

## 9. Register the kernel (optional)

```bash
python -m ipykernel install --user --name eeg-slm --display-name "Python (eeg-slm)"
```

Now JupyterLab will offer "Python (eeg-slm)" as a kernel option.

---

## Troubleshooting

**`make info` says `MPS available: False`.** Re-install PyTorch:
```bash
pip install --force-reinstall --no-cache-dir torch
```

**MNE complains about a missing FreeSurfer environment.** Ignore — we don't use source-space methods. The warning is harmless.

**The download fails partway through.** PhysioNet rate-limits aggressive clients. Re-run `make download` — it's incremental and skips files already present.

**`pip install -e .` fails on `pyedflib`.** On older M1 macOS you may need:
```bash
conda install -c conda-forge pyedflib
```
Then re-run `pip install -e ".[dev,training]"`.

**Notch frequency wrong for your environment.** EEGMMIDB was recorded in the US (60 Hz). For your own data in CN/EU, change `notch_hz: 50.0` in `configs/default.yaml`.

---

## What's next (Session 2)

In our next session we'll work on:

1. Reading the LeWorldModel paper together and deciding Phase 1 framing (i) vs (ii)
2. Writing the patch tokenizer — turning continuous EEG into transformer-consumable tokens
3. Drafting the encoder skeleton (small, ~5–15M parameters)
4. Implementing SIGReg in PyTorch as a standalone module we can unit-test

Before then, please:

- Run through this setup once on your M1 and confirm `make explore` produces real plots
- Note any friction in `DECISIONS.md`
- If you have bandwidth, skim the LeWorldModel paper (arXiv:2603.19312) — it's short
- Answer at least the M1 RAM question in `ROADMAP.md §9 Open questions` so I know what scale we can train locally
