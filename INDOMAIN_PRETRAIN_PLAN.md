# In-Domain LeJEPA Pretraining for Seizure Transfer — Spec

Status: draft (2026-05-22). Owner: Ping. Sibling project: `../eeg-seizure`.

## 0. Why this exists

The EEGMMIDB-pretrained encoder (`runs/s11-eegmmidb-22ch-10k`) gave **zero cross-patient
transfer** to CHB-MIT seizure detection. Measured four ways on the 2-subject LOSO pilot
(`../eeg-seizure`, Session 18), all at the matched 160/40/192 config, LOSO-mean window AUROC:

| init / recipe | AUROC |
|---|---|
| scratch | 0.736 |
| pretrained, full fine-tune | 0.732 |
| pretrained, discriminative FT (warmup + low enc-LR) | 0.702 |
| **pretrained, frozen probe** | **0.648** |
| **random encoder, frozen probe** | **0.633** |

The decisive line is the bottom two: a *frozen pretrained* encoder barely beats a *frozen
random* one, so the LeJEPA features carry almost no cross-patient seizure-discriminative
structure. Cause: a double domain gap — EEGMMIDB is **healthy-subject motor imagery** on a
**monopolar 64→22-channel** subset, while CHB-MIT is **pediatric epilepsy** on a **bipolar
double-banana** montage. The 22 channels are a different *semantic* set, so the channel-mixing
patch-embed conv is a mismatched initialiser.

This spec re-pretrains LeJEPA **in-domain**: on clinical scalp EEG, in the **exact montage and
tokenisation the seizure detector uses**, so the encoder drops into the detector's *better*
256/64/128 config and has a real chance of transferring.

## 1. Hard constraints (non-negotiable)

1. **Tokenisation match with the detector.** Encoder must be `n_channels=22`, `patch_size=64`
   samples, `embed_dim=128`, at **256 Hz**. With a 4 s window that is 1024 samples → 16 patches.
   This is exactly `eeg-seizure`'s `DetectorConfig` defaults, so the checkpoint drops in with **no
   `--target-fs/--patch-size/--embed-dim` overrides** (and no 160 Hz downsample, which itself cost
   ~0.04 AUROC).
2. **Channel identity match.** The 22 bipolar channels must be in the **same order** as
   `eeg_seizure.data.chbmit.MONTAGES["full"]`, so the conv weights align channel-for-channel.
3. **No leakage.** The pretraining corpus must contain **no recording from any subject used as a
   CHB-MIT LOSO test patient** (see §3).
4. **Same input normalisation.** Per-window, per-channel z-score (matches the detector's
   `WindowTensorDataset` and `zscore_per_channel`).

## 2. Corpus choice

| corpus | open? | scale | montage | leakage | verdict |
|---|---|---|---|---|---|
| **CHB-MIT interictal** (disjoint subjects) | yes | ~900 h / 24 subj | **bipolar, exact match** | none if split fixed | **v1 — recommended** |
| Siena Scalp EEG (PhysioNet) | yes | ~14 adult epilepsy subj | 10-20 → convert to same bipolar pairs | none (disjoint from CHB-MIT) | scale-up / external check |
| TUH EEG (TUEG/TUSZ) | credentials | huge (>1 TB) | 10-20 → bipolar | none | long-term scale; impractical now |

**Recommendation:** v1 uses **CHB-MIT interictal from a disjoint subject pool** — it is open,
already partly downloaded, and uses the *exact* bipolar montage (no conversion, no population
shift), so it is the cleanest first test of whether in-domain helps at all. Add **Siena** later
for breadth and an external (adult) check; defer **TUH** until the approach is proven.

## 3. Leakage-safe subject split (critical, do this first)

Partition the 24 CHB-MIT subjects **once** into a pretraining pool **P** and an evaluation pool
**E**, fixed and logged in both projects' DECISIONS files.

- **E (evaluation, LOSO only):** the subjects already downloaded — `chb01, chb03, chb05, chb08,
  chb10, chb20, chb23` — plus any others reserved for the headline baseline. Never appears in P.
- **P (pretraining only):** a disjoint set, e.g. `chb02, chb04, chb06, chb07, chb09, chb11,
  chb13, chb14, chb15, chb17, chb18, chb19, chb22, chb24`. Their **interictal** windows feed SSL;
  they are **never** evaluated.

One pretraining run, zero leakage, no per-fold re-pretraining. (Alternative "Design A": pretrain
on Siena/TUH only and keep *all* CHB-MIT for eval — better eval breadth, at the cost of an
external download + a montage conversion step.)

## 4. Build the interictal corpus (new code, in `eeg-seizure`)

Keep the project dependency direction intact (`eeg-seizure` → `eeg-slm`, not the reverse) by
**building the tensor in `eeg-seizure`** — which already has the tested CHB-MIT loader, montage,
and windowing — and handing `eeg-slm` a precomputed array.

New `eeg-seizure/scripts/04_build_pretrain_corpus.py`:

- Stream P-pool subjects with the existing `iter_subject_records(..., montage="full",
  target_fs=256, win_sec=4.0, stride_sec=…)`.
- Keep **interictal windows only**: drop any window overlapping a seizure **± a guard band**
  (recommend ±300 s around each ictal event) to avoid pre/post-ictal contamination. Most CHB-MIT
  records are seizure-free, so interictal data is abundant.
- Per-window per-channel z-score (`stats.zscore_windows`), µV scaling already handled by the loader.
- **Subsample to a budget** (recommend 150k–300k windows) so the array fits in RAM on the AutoDL
  GPU box; draw evenly across P subjects so no one patient dominates.
- Save `data/pretrain/chbmit_interictal_P_256hz_4s.npz` with `X (N, 22, 1024) float32`, the
  ordered `channels` list, and `meta` (subjects, fs, win, guard, notch). Notch = **60 Hz**
  (CHB-MIT is US mains).

This reuses tested code; the only new logic is "interictal selection + guard band + budgeted
subsample," which is small and unit-testable in the sandbox (pure numpy on synthetic labels).

## 5. eeg-slm changes (small, concrete)

1. **Compact preset.** `models/jepa.py`: add `EEGLeJEPAConfig.compact()` → `encoder.embed_dim=128`,
   `predictor.embed_dim=128` (keep `predictor.depth=4`, `encoder.mlp_depth=2`). The existing
   `base()`=192 / `large()`=256 don't give 128, and `EEGLeJEPA.__init__` enforces
   `encoder.embed_dim == predictor.embed_dim`, so both must be set together.
2. **Configurable patch size.** `scripts/04_train.py` line ~126 hardcodes
   `model_cfg.encoder.patch_size = 40`. Replace with a `--patch-size` flag (default 40 for
   back-compat) and pass **64** for this run. Optionally derive it as `int(0.25 * resample_hz)`.
3. **`--model-size compact`** in `04_train.py`'s existing `--model-size {base,large}` choices.
4. **Precomputed-tensor loader.** Add `--npy PATH` to `04_train.py` that loads the `.npz` from §4
   instead of calling `build_eegmmidb_pretraining_tensor`, wraps it in `EEGTensorDataset`, and sets
   `n_channels` from the array. (Bypasses EEGMMIDB entirely; no cross-project import needed.)
5. **Config.** New `configs/seizure_indomain.yaml`: `preprocessing.resample_hz: 256`,
   `epoch_length_s: 4.0`; `paths.data_root` unused when `--npy` is given; `training` block as base.

Minimal diff footprint: one classmethod, ~3 CLI flags, one yaml. No change to the encoder,
predictor, SIGReg, or trainer.

## 6. Training recipe (AutoDL GPU)

- Model: `compact` (embed_dim=128, patch_size=64, n_channels=22) — encoder ≈ the detector's
  reused encoder, so the checkpoint is a verbatim drop-in.
- Objective unchanged: LeJEPA next-patch MSE + λ·SIGReg, no stop-grad / EMA / teacher.
- Batch: GPU allows **256–512** (SIGReg likes large batches; revisit the `sigreg_weight=1.0`
  small-batch hack — sweep λ ∈ {0.1, 0.3, 1.0} briefly, paper default is 0.1).
- Steps: start **20k–30k**. Heed the Session-8 finding (sample-exposure saturation ~tens of M);
  don't overspend compute. LR 1e-3, warmup ~30–100, weight_decay 0.05, `--bf16`.
- Output: `runs/seizure-indomain-256-64-128/model_final.pt`.
- Follow `AUTODL.md`; **关机 (shut down the instance) when done.**

Example:

```
python scripts/04_train.py \
    --config configs/seizure_indomain.yaml \
    --model-size compact --patch-size 64 \
    --npy data/pretrain/chbmit_interictal_P_256hz_4s.npz \
    --steps 20000 --batch-size 256 --bf16 \
    --out runs/seizure-indomain-256-64-128
```

## 7. Evaluation — does it transfer? (run on E only)

Re-run the exact battery used for EEGMMIDB, on the **E** pool, at the detector's native
256/64/128 (no overrides), so results compare directly to the scratch baseline:

```
# frozen probe (pretrained) vs frozen probe (random control) — the decisive test
python scripts/02_train_detector.py --data-root data/raw --subjects <E...> --loso --montage full \
    --epochs 20 --device mps --freeze-encoder \
    --pretrained ../eeg-slm/runs/seizure-indomain-256-64-128/model_final.pt \
    --out runs/indomain_probe_pre.json
python scripts/02_train_detector.py --data-root data/raw --subjects <E...> --loso --montage full \
    --epochs 20 --device mps --freeze-encoder --out runs/indomain_probe_rand.json

# discriminative fine-tune vs scratch
python scripts/02_train_detector.py --data-root data/raw --subjects <E...> --loso --montage full \
    --epochs 20 --device mps \
    --pretrained ../eeg-slm/runs/seizure-indomain-256-64-128/model_final.pt \
    --encoder-lr 1e-5 --head-warmup-epochs 3 --out runs/indomain_discft.json

python scripts/03_summarize_runs.py runs/d0_full*.json runs/indomain_*.json
```

## 8. Decision gates

- **Gate 1 — features carry signal:** frozen-pretrained probe beats frozen-random by a clear
  margin (the EEGMMIDB run failed here: 0.648 vs 0.633). If it fails again, the corpus is too
  small / still mismatched → escalate to Siena+TUH before spending more.
- **Gate 2 — beats scratch:** fine-tuned LOSO window AUROC > the **256/64/128 scratch baseline
  (~0.78 on the pilot; re-measure on the full E pool)**. If yes, in-domain pretraining is a paper
  contribution and becomes the default init. If no, keep it as an honest negative and lean on the
  TTT/personalisation arm.
- **TTT check:** even if cross-patient is flat, test the patient-specific regime — the EEGMMIDB
  probe was *in-distribution* better than random (fold-1 calib AUPRC 0.262 vs 0.097), hinting
  pretraining may help personalisation (D4/D5) more than zero-shot LOSO.

## 9. Risks & notes

- **Interictal-only SSL** learns the "normal clinical EEG" manifold; the downstream head learns
  the ictal boundary. The encoder never needing to *see* seizures during SSL is fine and avoids
  ictal leakage entirely.
- **Population shift:** CHB-MIT is pediatric; Siena/TUH adult. Note when mixing corpora.
- **Mains frequency:** notch 60 Hz for CHB-MIT, 50 Hz for Siena.
- **Wearable montages (4 ch / 1 ch)** need their own pretrained encoders (different `n_channels`);
  use `04_train.py --channel-subset` on the corresponding bipolar pairs. Parallel extension, not v1.
- **Don't overspend AutoDL.** Watch the train curve for saturation; 20–30k steps is a starting
  point, not a target.

## 10. Build order (checklist)

1. [ ] Fix and log the P/E subject split in both DECISIONS files.
2. [x] `eeg-seizure/scripts/04_build_pretrain_corpus.py` + `data/pretrain.py` (interictal mask,
   reservoir, disjointness guard) + 7 sandbox tests. **Done 2026-05-22.**
3. [ ] Build `chbmit_interictal_P_256hz_4s.npz` on the Mac/AutoDL from P subjects.
4. [x] `eeg-slm`: `EEGLeJEPAConfig.compact()`, `--patch-size`, `--model-size compact`, `--npy`,
   `configs/seizure_indomain.yaml` + `test_compact_preset...`. **Done 2026-05-22.**
5. [ ] Pretrain on AutoDL → `runs/seizure-indomain-256-64-128/model_final.pt`. 关机.
6. [ ] Eval battery on E (frozen probe ± random control, disc-FT) → `03_summarize_runs.py`.
7. [ ] Pass/fail against the two gates; record verdict.
