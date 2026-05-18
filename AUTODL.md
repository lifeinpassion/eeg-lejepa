# AutoDL scale-up guide

The Session 6 workflow: provision a CUDA GPU on AutoDL, sync the repo + cached EEG data up to the instance, train at scale, pull the checkpoint back, run probes locally on M1.

Estimated time end-to-end: 30–60 minutes (most of it instance provisioning + rsync). Training itself is ~10–20 minutes on a 4090.

---

## 1. Local prep (before SSHing into AutoDL)

We download data on M1 (reliable, fast PhysioNet access) and rsync it up — far more reliable than fighting PhysioNet from a Chinese cloud instance.

### 1.1 Optionally download more subjects locally

You already have subjects 1-20 × runs (4, 8, 12). For Session 6 it's worth fetching more — say 1-50:

```bash
python scripts/01b_download_range.py --start 21 --end 50
# ~30 subjects × 3 runs × ~6 MB ≈ 540 MB, ~10-20 minutes
```

If you want to push further (full corpus is 109 subjects × 14 runs), this can run in background while you set up AutoDL. The training script will just use whatever is on disk; no need for all data to be present before starting AutoDL setup.

### 1.2 Verify the repo still trains locally

Quick sanity check that the recent `--device` / `--num-workers` additions didn't break the M1 path:

```bash
python scripts/04_train.py --steps 20 --subjects 1 2 3 --runs 4 8 12 --out /tmp/sanity --no-plot
# Should run 20 steps in <1 minute, no errors
rm -rf /tmp/sanity
```

---

## 2. AutoDL instance setup

### 2.1 Provision

On [autodl.com](https://www.autodl.com):

- **Region**: pick whichever is fastest from your network. Most regions stock 4090s.
- **GPU** (in rough order of preference for our 2.86M-param workload):
  - **RTX 5090** (32 GB, Blackwell, ~30% faster than 4090) — first choice if in stock
  - **RTX 4090 / 4090D** (24 GB) — perfect fit if available
  - **RTX PRO 6000** (96 GB Blackwell) — overkill but fine
  - **vGPU-32GB / vGPU-48GB** — virtualized, cheaper, occasional driver quirks but usually fine for PyTorch
  - **RTX 3090 / 3080 Ti** (24/12 GB) — older but solid fallback
  - Skip: H800 (overkill + expensive), V100/A800 (older Ampere/Volta), Moore Threads, Huawei Ascend (non-CUDA toolchains)
- **Image**: official PyTorch image — **PyTorch ≥ 2.5 / CUDA ≥ 12.4 / Python 3.11** if picking a 5090 or Blackwell card (needs sm_120 support). For 4090/3090 anything PyTorch 2.4+ works.
- **Disk**: default is fine — we need ~5 GB for code + data + checkpoints.
- **Hourly billing** ("按量计费"), not subscription. Stop the instance when done so you stop paying.

Cost: ~3-5 RMB/hr for a 4090. A full Session 6 (data upload + training + download) is typically <30 min of running time → <2 RMB.

### 2.2 Get the SSH command

After instance starts, AutoDL gives you something like:

```
ssh -p 12345 root@connect.westa.seetacloud.com
ssh -p 36601 root@connect.westd.seetacloud.com
```

…plus a password. Save both somewhere convenient.

---

## 3. Sync code + data up

From your M1 (in a separate terminal — keep the AutoDL one for SSH):

```bash
# Replace the host/port with what AutoDL gave you
AUTODL_HOST="root@connect.westd.seetacloud.com"
AUTODL_PORT=36601

# 3.1 Sync the repo (excludes top-level runs/, data/, .git for speed)
# NOTE: the leading slash on /runs/ and /data/ anchors them to the rsync
# source root — without it, rsync would also exclude src/eeg_slm/data/
# (the Python submodule), which would break the install.
rsync -avz -e "ssh -p $AUTODL_PORT" \
    --exclude='/runs/' --exclude='/data/' --exclude='.git/' \
    --exclude='__pycache__/' --exclude='*.egg-info' \
    "/Users/billion/Downloads/SLM/eeg-slm/" \
    "$AUTODL_HOST:/root/eeg-slm/"

rsync -avz -e "ssh -p 36601" \
    "/Users/billion/Downloads/SLM/eeg-slm/src/" \
    "root@connect.westd.seetacloud.com:/root/eeg-slm/src/"

# 3.2 Sync the EEG data (this is the chunky one — ~300 MB-1 GB depending on subjects)
rsync -avz -e "ssh -p $AUTODL_PORT" --progress \
    "/Users/billion/Downloads/SLM/eeg-slm/data/" \
    "$AUTODL_HOST:/root/eeg-slm/data/"
```

`rsync -avz` is incremental, so re-running is cheap. The data rsync should hit a few MB/s on a decent connection.

---

## 4. AutoDL environment

SSH into the instance:

```bash
ssh -p $AUTODL_PORT $AUTODL_HOST
```

Once in:

```bash
cd /root/eeg-slm

# Verify Python + PyTorch + CUDA
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
# Expect: CUDA: True NVIDIA GeForce RTX 4090

# Install our package in editable mode + extras
pip install -e ".[dev,training]" --quiet

# Sanity check the env (this is the same `make info` we ran on M1)
make info
# Expect: CUDA available: True, MPS available: False (or absent)

# Quick smoke test — should run a 20-step training on CUDA
python scripts/04_train.py --steps 20 --subjects 1 2 3 --runs 4 8 12 \
    --out /tmp/sanity --no-plot
```

If the 20-step smoke test runs at ~0.1 s/step (vs ~0.4 s/step on M1), CUDA is working and you're ready for the real run.

---

## 5. The Session 6 run

The full scale-up. Recommended config:

```bash
# Adjust --subjects to whatever you actually have on disk
python scripts/04_train.py \
    --steps 10000 \
    --subjects $(seq 1 50) \
    --runs 4 8 12 \
    --batch-size 64 \
    --num-workers 4 \
    --bf16 \
    --device cuda \
    --out runs/lambda-1.0-s50-10k
```

Hyperparameter rationale:
- `--steps 10000` — 2× our M1 5000-step run; 4090 should chew through this in 10-15 min.
- `--batch-size 64` — 8× our M1 batch=8. Gives SIGReg 8× more samples per call, so the estimate is much tighter.
- `--num-workers 4` — CUDA-friendly multi-process dataloading.
- `--bf16` — bf16 autocast. ~2× speedup on A/RTX-series with negligible accuracy impact.
- `--device cuda` — explicit, in case auto-detect fails.

Expected throughput: 4090 should do ~10–15 steps/sec at batch=64 with bf16. 10k steps ≈ 15 min wall-clock.

**SIGReg λ note:** With 8× larger batch, our M1-calibrated λ=1.0 might be too strong. If the AutoDL run shows pred_loss never dropping below 0.3 or SIGReg growing unboundedly, retry with `--sigreg-weight 0.3` or `--sigreg-weight 0.1`. The latter is the paper's default and may be the right setting at production batch sizes.

---

## 6. Pull results back

From M1 (separate terminal):

```bash
# Copy the trained checkpoint and CSV log back
rsync -avz -e "ssh -p $AUTODL_PORT" --progress \
    "$AUTODL_HOST:/root/eeg-slm/runs/lambda-1.0-s50-10k/" \
    "/Users/billion/Downloads/SLM/eeg-slm/runs/lambda-1.0-s50-10k/"

rsync -avz -e "ssh -p 36601" --progress \
    "root@connect.westd.seetacloud.com:/root/eeg-slm/runs/lambda-1.0-s50-10k/" \
    "/Users/billion/Downloads/SLM/eeg-slm/runs/lambda-1.0-s50-10k/"
```

Checkpoint size: ~12 MB for our 2.86M-param model. Fast to copy.

---

## 7. Re-probe on M1

The probe runs locally on M1 (uses cached EEG, no need to re-sync data):

```bash
python scripts/05_linear_probe.py \
    --ckpt runs/lambda-1.0-s50-10k/model_final.pt \
    --subjects $(seq 1 20)

python scripts/05_linear_probe.py \
    --ckpt runs/lambda-1.0-s50-10k/model_final.pt \
    --subjects $(seq 1 20) \
    --task rest_vs_activity
```

(Probe still uses subjects 1-20 for direct comparison to Session 5 LOSO results.)

---

## 8. Stop the AutoDL instance

**Don't forget**. AutoDL bills per minute when running. From the AutoDL web console: "关机" (shut down). Instance state is preserved if you want to resume later; storage is billed separately at trivial rates.

---

## Troubleshooting

**`torch.cuda.is_available() = False`** → wrong PyTorch image. Reprovision with a CUDA-enabled image.

**`OOM at batch=64`** → 4090 has 24 GB which is plenty for our 2.86M-param model, but if you somehow OOM, drop to `--batch-size 32`.

**`pip install -e` fails on a missing system lib** → some AutoDL images don't have `libsndfile`. `apt-get update && apt-get install -y libsndfile1`.

**`rsync` keeps prompting for password** → set up an SSH key once:
```bash
ssh-copy-id -p $AUTODL_PORT $AUTODL_HOST
```

**Slow training on AutoDL (worse than M1)** → check `nvidia-smi` while training. If GPU utilization is <50%, the bottleneck is dataloading — bump `--num-workers` to 8.

**bf16 makes losses go NaN** → drop `--bf16`, fall back to fp32. Some PyTorch + driver combinations have flaky bf16 on RTX cards.

---

## What success looks like

A clean Session 6 run produces:

1. `runs/lambda-1.0-s50-10k/model_final.pt` (~12 MB checkpoint) on M1
2. `runs/lambda-1.0-s50-10k/train_log.csv` + `train_curves.png` (loss curves)
3. Probe results we can compare to Session 5:

| Pretraining | left_right enc Δ | rest_vs_activity enc Δ | best AUC |
|-------------|--------------------|-------------------------|----------|
| 20 subj / 5000 steps (M1)  | +7.9 | +8.8 | 0.740 |
| **50 subj / 10000 steps (AutoDL)** | **?** | **?** | **?** |

If the trend continues (more data → wider gap), we have a clean scaling-law story. If it plateaus, the next bottleneck is architecture (larger encoder), not data.
