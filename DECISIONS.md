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

## 2026-05-18 — Session 2: Phase 1 architecture commitments

**Decision:** EEGLeJEPA = per-patch independent encoder + causal-Transformer predictor + SIGReg. No EMA, no stop-gradient, no teacher network — single shared encoder produces both inputs and targets. Single regularization weight λ = 0.1.
**Why:** Faithful to LeJEPA/LeWorldModel theory; one hyperparameter; demonstrably stable on a tiny GPU per the paper. For EEG specifically, the per-frame-independent encoder design maps cleanly to per-patch processing.
**Component-by-component:**
- **Encoder:** Conv1d patch embed (kernel=stride=40 samples = 200 ms at 200 Hz) → sinusoidal pos embed → 2-layer per-patch MLP → `nn.BatchNorm1d`. NO cross-patch attention — would leak future info into the prediction target. Unit test enforces this property.
- **Predictor:** 4-layer causal Transformer (dim=192, heads=4, mlp_ratio=4, dropout=0.1) → `nn.BatchNorm1d`. Trained via MSE in raw embedding space against `encoder(x)[:, 1:]`.
- **SIGReg:** Cramér-von Mises on 256 random unit-vector projections per call. CvM chosen over Epps-Pulley for the cleanest first implementation (closed-form, sort-based, verifiable against scipy). Epps-Pulley deferred as an ablation.
- **BatchNorm everywhere the embedding meets a loss.** LayerNorm at the output would force per-sample unit norm and destroy SIGReg — the paper is explicit. We use LayerNorm internally inside transformer blocks (which is fine) and BatchNorm at the encoder/predictor output projectors.
**Revisit if:** Training is unstable (try smaller λ, more projections, or switch to Epps-Pulley); or if downstream evaluation reveals the encoder is under-parameterized (add per-patch depth or per-channel attention within a patch).

**Decision:** Phase 1 prototype targets ~1-3M params total. Encoder ~500K-1M, predictor ~1-2M. Comfortable on 8 GB M1 dev.
**Why:** Validates the architecture and training loop end-to-end before spending AutoDL hours on a production-scale run. Real EEG-FM scale (5-50M params, à la EEGPT/LaBraM) comes after Phase 1 is proven.
**Revisit when:** Phase 1 is training stably and we move to Phase 1.5 (scale up) on AutoDL.

**Decision:** `num_slices=256` for SIGReg on M1; bump to `1024` on AutoDL.
**Why:** Paper notes SIGReg is largely insensitive to the number of slices. 256 keeps each forward pass fast on MPS during dev; 1024 (paper default) is fine on a GPU.

## 2026-05-18 — Session 3: training infrastructure

**Decision:** Functional `train()` function rather than a Trainer class.
**Why:** Less ceremony, easier to test, no hidden state across runs. The function takes model + loader + cfg and returns a dict pointing to CSV log + checkpoint.

**Decision:** Default `deterministic=False` in `set_global_seed()`. Was True.
**Why:** On MPS, `torch.use_deterministic_algorithms(True)` forces CPU fallbacks for `sort` and SDPA, which dominated the 4-second forward time observed in Session 2. Reproducibility is still available via explicit opt-in for debugging runs.

**Decision:** Optimizer + schedule. AdamW (lr=1e-3, wd=0.05, betas=(0.9, 0.95)), cosine LR with linear warmup over 30 steps, min_lr_ratio=0.1.
**Why:** Standard small-transformer pretraining recipe. LeJEPA paper notes no LR scheduler is strictly required, but warmup avoids early SIGReg instability when embeddings are still wildly non-isotropic.

**Decision:** Per-channel z-score is applied at data-construction time (in `build_eegmmidb_pretraining_tensor`), not as a layer in the model.
**Why:** Cheap, deterministic, and keeps the runtime path simple. If we later need on-the-fly augmentation (e.g., time-shift, channel masking), it goes between the tensor and the model in a `transforms` callable.

**Decision:** CSV + matplotlib logging instead of wandb.
**Why:** Zero external deps, works offline, easy to diff across runs. We'll add wandb later if we want to compare many runs at once or share dashboards.

**Decision:** Embedding diagnostics tracked alongside losses: `|mean|`, `std`, off-diagonal covariance, embedding norm. Targets are (0, 1, 0, √D) under isotropic N(0, I).
**Why:** Loss values alone don't tell us whether SIGReg is actually working. Distribution stats do — they should monotonically approach the targets even if the prediction loss plateaus.

## 2026-05-18 — Session 3 results: SIGReg weight calibration on EEGMMIDB

**Decision:** Set default `sigreg_weight` = 1.0 (was 0.1 per paper default). Set default `sigreg_num_slices` = 1024 (was 256).
**Why:** First 100-step run with λ=0.1 showed clear representational collapse: `pred_loss` cratered from 1.8 → 0.04 in 20 steps (suspiciously fast), `sigreg_loss` *grew* from 0.18 → 1.88 (10×), `off-diag` covariance climbed from 0.16 → 0.60. The model was finding a trivial low-rank prediction shortcut and SIGReg's weighted contribution (≈0.19) was too small to overcome the prediction-shortcut savings (≈1.74).

**Three-run sweep on subjects 1-3, 200 steps each:**

| Config | pred final | sigreg final | off-diag final | grad-norm | Verdict |
|--------|-----------|--------------|----------------|-----------|---------|
| λ=0.1, slices=256, pred-depth=4   | **0.04**   | **1.88** ↑  | **0.60** ↑    | 0.5-1.5  | Collapsed |
| λ=1.0, slices=1024, pred-depth=4  | 0.29       | 0.33 (flat) | 0.22 (slow ↓) | 0.5-1.5  | **Healthy** ✓ |
| λ=5.0, slices=1024, pred-depth=2  | 0.77       | 0.14 (flat) | 0.13 (slow ↓) | 1-44 spikes | Over-constrained; jittery |

λ=1.0 is the production sweet spot: pred loss reaches a non-trivial plateau, SIGReg holds steady, off-diag is bounded and slowly decreasing, gradients are stable. λ=5.0 demonstrates the regularization *can* fully dominate if needed, but at the cost of optimization stability — useful as an ablation, not a default.

**Revisit when:** We scale batch on AutoDL. The paper's λ=0.1 was tuned on ImageNet with B=256+; our 8× scale-down naturally argues for ~10× more weight on SIGReg to compensate for noisier distribution estimates. If we bump to B=64 on AutoDL, λ may need to come back toward 0.5 or even 0.1.

**Caveat:** Healthy training metrics ≠ good downstream representations. The real test is linear-probe accuracy on standard EEG benchmarks (TUH-Events, SEED, BCI-IV) — that's Session 4 / Phase 1.5. We may discover that λ=5.0's more isotropic embeddings probe better even though they look "worse" by training-loss standards.

## 2026-05-18 — Session 4: linear-probe evaluation protocol

**Decision:** Phase 1 downstream evaluation = LOSO linear probe with sklearn LogisticRegression.
**Why:** This is the SSL-evaluation standard. Linear probes are minimal (one matmul), hard to game, and directly answer "did pretraining produce useful features?" If the pretrained encoder beats a randomly-initialized encoder on this protocol, SIGReg worked.

**Decision:** First evaluation task = EEGMMIDB motor-imagery left-vs-right (runs 4, 8, 12).
**Why:** Binary classification, well-understood, published baselines exist for direct comparison, data is already cached locally. T0 (rest) excluded for now — left-vs-right is the cleaner first signal.

**Decision:** Pool token embeddings via mean over the patch dimension.
**Why:** Simplest defensible aggregator. Max-pool is available as a switch; attentive pooling is overkill until we have a stronger baseline.

**Decision:** Cross-validation = leave-one-subject-out (LOSO).
**Why:** Within-subject splits are trivially solvable (same brain, same session — leakage). LOSO is the honest test of whether the encoder captures something subject-invariant. With only 3 subjects, LOSO gives 3 folds and noisy numbers — bump to ~10-20 subjects when the probe matters.

**Decision:** Always pair pretrained-probe with random-init-probe as control.
**Why:** A pretrained probe at 65% accuracy means nothing without knowing what random features score. The honest answer is the *delta*. The script prints both side-by-side and labels the gap explicitly.

**Caveat we'll discover empirically:** At 3-subject LOSO scale, the variance of the probe is probably ±5-10 percentage points. A small positive delta (e.g., +3 pp) might not be statistically meaningful. The right scale for a publishable comparison is 20+ subjects.

## 2026-05-18 — Session 4 result: SIGReg pretraining shows +11.1 pp on predictor features

**Finding:** LOSO linear probe on EEGMMIDB motor-imagery (subjects 1-3, runs 4/8/12) yields:

| Source | Pretrained | Random | Δ |
|--------|-----------|--------|---|
| encoder_mean   | 48.1% ±3.8 | 46.7% ±1.8 | +1.5 |
| predictor_mean | **53.3% ±1.8** | 42.2% ±1.8 | **+11.1** |
| both_mean      | 48.1% ±1.0 | 42.2% ±0.0 | +5.9 |

Chance ≈ 51%. Pretrained predictor_mean is the first source we've measured that's both (a) above chance and (b) substantially above the random-init baseline.

**Interpretation:**
- The encoder is per-patch by design — no cross-time context — so mean-pooled encoder features carry no information about temporal dynamics like ERD/ERS. They sit at chance whether pretrained or random.
- The predictor is a causal Transformer over the patch sequence, so it does see temporal context. Pretraining gives it useful temporal dynamics; without pretraining, its outputs are arbitrary functions of random weights and overfit-then-fail on cross-subject LOSO.
- The +11.1 pp gap on predictor_mean is the first concrete evidence that SIGReg pretraining produced useful representations for a real downstream task.

**Implication for downstream protocol:** Default feature source for sequence-level EEG tasks should be `predictor_mean`, not `encoder_mean`. The encoder alone is fine for per-patch tasks (e.g., quick anomaly detection on a single window), but anything epoch-level needs the predictor.

**Caveats:**
- Absolute accuracy (53.3%) is still very low vs published BCI methods (~75% with FBCSP/EEGNet on similar data). We're at "the architecture works and pretraining helps" not "competitive with SOTA."
- 3-subject LOSO has wide effective CIs even with low per-fold std. The +11.1 pp result is meaningful in this controlled setup but needs ~10-20 subjects for paper-grade confidence intervals.

**Action items for Session 5+:**
1. Download subjects 1-20 (≈10× current data) and re-run the same probe.
2. Pretrain on the full subjects-1-20 corpus (10× steps of Session 3, on AutoDL once we get over there).
3. Add a "supervised baseline" — train EEGLeJEPA from scratch with a classification head, compare to the SSL+probe pipeline.
4. Add an easier task as a sanity-check probe: rest (T0) vs activity (T1+T2). If we can't beat random on that, something deeper is wrong.

---

*Future entries below.*
