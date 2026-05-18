# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
# ---

# %% [markdown]
# # First look at EEGMMIDB
#
# Run via `make explore` or open in JupyterLab (with the jupytext extension installed).
# This is the first end-to-end pipeline check: load → preprocess → visualize → spectrum.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from eeg_slm.data.loaders import EEGMMIDBLoader, RUNS_MOTOR_IMAGERY_HANDS
from eeg_slm.data.preprocessing import (
    PreprocessingConfig, preprocess_raw, fixed_length_epochs, to_numpy,
    zscore_per_channel,
)
from eeg_slm.utils.seeding import set_global_seed, get_device

# %% [markdown]
# ## Config

# %%
cfg = yaml.safe_load(Path("configs/default.yaml").read_text())
set_global_seed(cfg["training"]["seed"])
device = get_device(cfg["training"]["device"])
print(f"Device: {device}")
print(f"Config: subjects={cfg['dataset']['subjects']}, runs={cfg['dataset']['runs']}")

# %% [markdown]
# ## Load one subject

# %%
loader = EEGMMIDBLoader(data_root=Path(cfg["paths"]["data_root"]))
subject = cfg["dataset"]["subjects"][0]
raw = loader.load_raw(subject=subject, runs=list(RUNS_MOTOR_IMAGERY_HANDS))
print(raw)
print(f"  Channels: {raw.info['nchan']}")
print(f"  Sampling: {raw.info['sfreq']} Hz")
print(f"  Duration: {raw.n_times / raw.info['sfreq']:.1f} s")

# %% [markdown]
# ## Preprocess

# %%
pp = PreprocessingConfig(
    bandpass_low_hz=cfg["preprocessing"]["bandpass_low_hz"],
    bandpass_high_hz=cfg["preprocessing"]["bandpass_high_hz"],
    notch_hz=cfg["preprocessing"]["notch_hz"],
    reference=cfg["preprocessing"]["reference"],
    resample_hz=cfg["preprocessing"]["resample_hz"],
    epoch_length_s=cfg["preprocessing"]["epoch_length_s"],
    epoch_overlap_s=cfg["preprocessing"]["epoch_overlap_s"],
)
raw_pp = preprocess_raw(raw, pp)
print(f"After preprocessing: {raw_pp.info['sfreq']} Hz, "
      f"{raw_pp.n_times / raw_pp.info['sfreq']:.1f} s")

# %% [markdown]
# ## Plot a few seconds of a few channels

# %%
seconds_to_plot = 10
samples_to_plot = int(seconds_to_plot * raw_pp.info["sfreq"])
channels_to_plot = ["FP1", "FZ", "CZ", "PZ", "O1"]
available = [ch for ch in channels_to_plot if ch in raw_pp.ch_names]
print(f"Plotting channels: {available}")

times = np.arange(samples_to_plot) / raw_pp.info["sfreq"]
data, _ = raw_pp[available, :samples_to_plot]

fig, axes = plt.subplots(len(available), 1, figsize=(10, 1.5 * len(available)),
                         sharex=True)
for ax, name, trace in zip(axes, available, data):
    ax.plot(times, trace * 1e6)  # convert V to µV
    ax.set_ylabel(f"{name}\n(µV)")
    ax.grid(True, alpha=0.3)
axes[-1].set_xlabel("Time (s)")
fig.suptitle(f"EEGMMIDB subject {subject:03d} — first {seconds_to_plot}s")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Power spectrum

# %%
psd = raw_pp.compute_psd(fmin=1.0, fmax=40.0, verbose="ERROR")
fig = psd.plot(picks="eeg", show=False)
fig.suptitle(f"PSD — subject {subject:03d}")
plt.show()

# %% [markdown]
# ## Cut into fixed-length epochs (pretraining-style chunks)

# %%
epochs = fixed_length_epochs(raw_pp, pp)

# Default: scale to microvolts. Raw MNE values are in volts (~1e-5), which is
# below the default eps of nn.LayerNorm — leaving them in volts makes
# downstream normalization layers numerically unstable. See DECISIONS.md.
X = to_numpy(epochs, to_microvolts=True)
print(f"Epochs tensor shape: {X.shape}  (n_epochs, n_channels, n_times)")
print(f"  Memory: {X.nbytes / 1024**2:.2f} MB")
print(f"  Per-epoch: {X.shape[1]} channels x {X.shape[2]} samples = "
      f"{X.shape[2] / raw_pp.info['sfreq']:.1f}s")
print(f"  Amplitude range (µV): "
      f"min={X.min():+.1f}, max={X.max():+.1f}, "
      f"mean={X.mean():+.2f}, std={X.std():.2f}")

# Per-channel z-score (what we'll feed the model in Phase 1)
X_z = zscore_per_channel(X)
print(f"  After per-channel z-score: "
      f"mean={X_z.mean():+.4f}, std={X_z.std():.4f}")

# %% [markdown]
# ## Quick PyTorch sanity check

# %%
import torch

x_torch = torch.from_numpy(X_z[:8]).to(device)
print(f"PyTorch tensor on {device}: shape={tuple(x_torch.shape)}, dtype={x_torch.dtype}")

# LayerNorm should now produce ~unit-variance output because input is already
# z-scored. If you see std ~1.0, the device path is healthy AND the amplitude
# convention is correct.
ln = torch.nn.LayerNorm(X_z.shape[-1]).to(device)
out = ln(x_torch)
print(f"Output shape after LayerNorm: {tuple(out.shape)}")
print(f"Output mean: {out.mean().item():+.4f}, std: {out.std().item():.4f}")
print(f"  (expect std close to 1.0 — anything <0.5 indicates an amplitude issue)")

# %% [markdown]
# ## Next steps
#
# - Read LeWorldModel paper (arXiv:2603.19312) and the EEG-FM critical review
# - Decide Phase 1 framing (i) vs (ii) — see ROADMAP.md §3 Phase 1
# - Implement the patch tokenizer and the JEPA encoder skeleton
# - Wire up SIGReg as a candidate regularizer
