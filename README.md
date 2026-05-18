# eeg-slm

Small, edge-deployable, self-adapting EEG foundation models.

Part of the multi-phase EEG-SLM research and startup agenda described in `../ROADMAP.md`. This repo is the working codebase for **Phase 1: Small EEG Foundation Model** and will extend across subsequent phases (MoE, TTT personalization, quantum-classical hybrids).

## Quick start (on M1)

```bash
# 1. Clone or open this directory
cd "/Users/billion/Downloads/SLM/eeg-slm"

# 2. Create a fresh conda env (recommended) or venv
conda create -n eeg-slm python=3.11 -y
conda activate eeg-slm

# 3. Install in editable mode with dev + training extras
pip install -e ".[dev,training]"

# 4. Sanity-check the environment
make info
# Expected output should show MPS available: True on M1

# 5. Download the first dataset (~1 GB, takes a few minutes)
make download

# 6. Run the first-look exploration script
make explore
```

If you prefer a Jupyter notebook, the exploration script under `scripts/02_explore_data.py` is a [jupytext](https://jupytext.readthedocs.io/) percent-format `.py` file — open it in JupyterLab with the jupytext extension and it renders as a notebook.

## Project structure

```
eeg-slm/
├── README.md                    you are here
├── DECISIONS.md                 running log of architecture / scope decisions
├── pyproject.toml               Python project + dependencies
├── Makefile                     common commands (make download, make explore, ...)
├── .gitignore
├── configs/
│   └── default.yaml             default config values (data paths, preprocessing, etc.)
├── data/                        gitignored — populated by download scripts
├── notebooks/                   exploratory Jupyter notebooks
├── scripts/
│   ├── 01_download_data.py      pulls PhysioNet EEGMMIDB into data/raw
│   └── 02_explore_data.py       first-look plotting and statistics (jupytext)
├── src/
│   └── eeg_slm/
│       ├── data/
│       │   ├── loaders.py       dataset access (EEGMMIDB to start)
│       │   └── preprocessing.py standard EEG preprocessing (filter, re-ref, epoch)
│       ├── models/              architecture code (placeholder for now)
│       └── utils/
│           └── seeding.py       reproducibility helpers
└── tests/
    └── test_loaders.py          smoke tests for data loading
```

## Hardware targets

- **Local development:** Apple M1 with MPS backend (PyTorch)
- **Training:** AutoDL rented GPUs (RTX 4090 default, A100 for larger experiments)
- **Deployment (later phases):** iOS via Core ML, Android via TFLite, ESP32 via TFLite Micro

## Phase status

- [x] Phase 0: planning, repo bootstrap (this commit)
- [ ] Phase 1: small EEG foundation model — *in progress*
- [ ] Phase 2: SLA-MoE extension
- [ ] Phase 3: TTT personalization
- [ ] Phase 4: quantum-classical hybrid

See `../ROADMAP.md` for the full agenda, target venues, and timeline.

## License

To be decided. Treat as proprietary until otherwise specified.
