# Decisions log

A running log of consequential decisions made during the project. Each entry: date, decision, rationale, and what would cause us to revisit it.

---

## 2026-05-17 — Project bootstrap

**Decision:** Start Phase 1 work now, in parallel with dissertation defense prep.
**Rationale:** Dissertation is written; only defense rehearsal remains. Bandwidth is available.
**Revisit if:** Defense prep becomes more demanding than expected.

**Decision:** Repo lives at `/Users/billion/Downloads/SLM/eeg-slm/`, local-only initially.
**Rationale:** Avoid premature commitment to a hosting platform; push to private GitHub once we've picked a name and confirmed IP strategy.
**Revisit when:** Phase 1 produces something worth a private GitHub mirror (probably ~week 4 of Phase 1).

**Decision:** Python 3.11.
**Rationale:** Best wheel coverage on Apple Silicon as of mid-2026; supported by every dependency we care about.
**Revisit if:** A core dependency requires 3.12+.

**Decision:** PyTorch with MPS (M1 local) and CUDA (AutoDL) backends.
**Rationale:** Most EEG research uses PyTorch; HuggingFace transformers ecosystem is PyTorch-first; MPS is mature enough on M1 for dev-scale iteration.
**Revisit if:** JAX shows clear advantages for our JEPA training (LeWorldModel reference code is PyTorch as of writing).

**Decision:** Start with PhysioNet EEGMMIDB as the first dataset.
**Rationale:** Fully open (no credentialing), supported natively by MNE-Python (`mne.datasets.eegbci`), ~1 GB, 109 subjects, well-known in BCI literature so easy to cross-reference. Lets us validate the entire pipeline end-to-end before tackling the much larger TUH-EEG.
**Revisit when:** Phase 1 architecture is stable and we need scale (then TUH-EEG + SEED + DEAP).

**Decision:** Keep Phase 1 framings (i) "small EEG foundation model" and (ii) "small EEG predictive world model" both open at the infrastructure level.
**Rationale:** Data loading, preprocessing, evaluation, and deployment code is identical for both. The decision belongs after the LeWorldModel paper has been read and we've prototyped a JEPA training loop.
**Revisit when:** ~end of week 3 of Phase 1, after first JEPA training experiment.

**Decision:** Use a `src/` layout with editable install (`pip install -e .`).
**Rationale:** Standard modern Python packaging; avoids accidentally importing from cwd; matches what we'll need when we package for distribution.
**Revisit if:** A specific tool (e.g., a notebook server config) requires `pip install -e .` to be re-run after every code change in a way that's annoying.

**Decision:** Use `Makefile` for top-level commands rather than `task`, `just`, or similar.
**Rationale:** Universally available, no extra install; the targets we need are simple.
**Revisit if:** Cross-platform support (Windows collaborators) becomes important.

**Decision:** Scripts are `.py` files in jupytext "percent" format, not `.ipynb` notebooks.
**Rationale:** `.ipynb` files are JSON with embedded outputs — they pollute diffs and merge poorly. Jupytext `.py` files are diff-friendly and still render as notebooks in JupyterLab. Outputs go to PNGs / wandb, not into the notebook file.
**Revisit if:** A collaborator strongly prefers ipynb.

## 2026-05-17 — M1 hardware budget confirmed: 8 GB unified memory

**Decision:** Treat AutoDL as essential rather than optional. M1 is for development, debugging, small-scale prototyping (≤ ~15M parameters at small batch sizes), and final inference/demo. Real training goes to AutoDL.
**Rationale:** 8 GB unified memory is shared between OS, apps, Python process, and the MPS device. Effective ML working budget is ~2-3 GB after macOS Tahoe baseline. A 10-15M-parameter FP32 model fits comfortably; anything beyond that requires either BF16/INT8 or AutoDL.
**Revisit if:** Bill upgrades hardware or rents an Apple Silicon dev machine (Mac Mini M4 Pro 64 GB is the obvious upgrade path if needed).

**Decision:** Use precomputed teacher embeddings for distillation rather than running the teacher live during student training.
**Rationale:** Running LaBraM (~50-100M params) as a live teacher while also training a student is impractical at 8 GB. Precompute teacher embeddings once on AutoDL, save to disk, then train the student against the cached embeddings. This is a cleaner pipeline anyway — reproducible, debug-friendly, and decouples teacher and student training.
**Revisit if:** We move to a much larger M-series machine or have specific need for live teacher gradients (rare).

**Decision:** Storage strategy — local for small subsets, stream from AutoDL for full corpora.
**Rationale:** TUH-EEG full corpus is hundreds of GB; Bill has 72 GB free locally. The current ~50 MB EEGMMIDB subset is fine. When we scale to TUH-EEG, we'll work with a local 5-10 GB subset and run full-scale pretraining on AutoDL where the data lives.
**Revisit when:** Phase 1 scales beyond the EEGMMIDB-only proof of concept.

**Decision:** Default to small batch sizes for local prototyping (4-8); rely on gradient accumulation if effective batch size needs to be larger.
**Rationale:** Memory headroom is the binding constraint on M1 8GB. Gradient accumulation gets us effective large batches without OOM.
**Revisit if:** Profiling shows we're routinely OOM-ing or batch=4 is bottlenecking iteration speed.

## 2026-05-17 — MNE API fix (loaders.py)

**Change:** `eegbci.load_data(subject=..., update_path=...)` → `eegbci.load_data(subjects=[...])`.
**Why:** MNE ≥1.6 renamed the parameter to `subjects` (plural, list) and removed `update_path`. Bill is running MNE 1.12.1; my original code targeted the pre-1.6 API.
**Lesson:** Pin or constrain MNE in `pyproject.toml` more tightly once we know what we're depending on.

## 2026-05-17 — Bandpass default lowered to 70 Hz; Nyquist clamp added

**Change:** Default `bandpass_high_hz` from 80 → 70 Hz; added auto-clamp in `preprocess_raw` that warns and lowers `h_freq` if it's ≥ Nyquist of the source.
**Why:** EEGMMIDB samples at 160 Hz → Nyquist = 80 Hz. MNE requires `h_freq < Nyquist` strictly. 70 Hz is the safer default and matches LaBraM/EEGPT practice (surface-EEG gamma above ~70 Hz is mostly EMG contamination anyway).
**Note for later:** If we ever pretrain on a dataset with non-standard sampling rates (e.g., some clinical recordings at 100 Hz), the clamp will keep us safe but we should still set per-dataset defaults explicitly.

## 2026-05-17 — EEG amplitude convention: scale to µV at the data-layer boundary

**Decision:** `to_numpy(epochs, to_microvolts=True)` is the default — every batch leaves the data layer in µV, not V.
**Why:** Raw MNE values are in volts (~1e-5 V scale). `nn.LayerNorm`'s default `eps=1e-5` dominates the computed variance, so the layer effectively produces input/√eps rather than input/σ. Confirmed empirically on the first M1 run: LayerNorm output `std=0.0083` instead of expected ~1.0. This is the canonical EEG-ML gotcha and every published EEG-FM (LaBraM, EEGPT, NeuroLM) handles it implicitly.
**Also added:** `zscore_per_channel(x)` helper. This is the model-input convention we'll use in Phase 1 (matches EEGPT's input preprocessing).
**Revisit if:** We want to preserve absolute amplitude information for downstream tasks where it matters (e.g., seizure detection where signal magnitude correlates with severity). In that case, a per-channel scale factor learned during pretraining is the right approach.

---

*Future entries below.*
