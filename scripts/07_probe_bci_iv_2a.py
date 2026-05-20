"""Linear probe on BCI Competition IV Dataset 2a (4-class motor imagery).

Three modes:
  1. RANDOM (no --ckpt)         : random-init encoder, baseline only.
  2. PRETRAINED-MATCHED (--ckpt) : encoder pretrained on EEGMMIDB *with the same*
                                   22-channel intersection, drop-in compatible.
  3. PRETRAINED-FULL (--ckpt + --pad-channels)
                                : encoder pretrained on 64-channel EEGMMIDB;
                                  BCI-IV-2a's 22 channels mapped into the
                                  corresponding 64-channel positions with zeros
                                  elsewhere. Hacky baseline for "out-of-the-box"
                                  cross-dataset transfer without re-pretraining.

LOSO across 9 BCI-IV-2a subjects on the T (training) session.

Example:
    # Baseline: random init, no pretraining
    python scripts/07_probe_bci_iv_2a.py

    # Pretrained matched: encoder trained on 22-channel EEGMMIDB
    python scripts/07_probe_bci_iv_2a.py --ckpt runs/s10-eegmmidb-22ch/model_final.pt

    # Quick test with our existing 64-channel checkpoint via channel padding
    python scripts/07_probe_bci_iv_2a.py --ckpt runs/s7-lambda-1.0/model_final.pt --pad-channels
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from rich.console import Console
from rich.table import Table

from eeg_slm.data.bci_iv_2a import (
    BCI_IV_2A_EEG_CHANNELS,
    BCIIV2aLoader,
    SUBJECT_IDS,
    build_bci_iv_2a_dataset,
    channel_intersection_indices,
)
from eeg_slm.eval import (
    extract_features_jepa,
    linear_probe_loso_from_features,
)
from eeg_slm.models import EEGLeJEPA, EEGLeJEPAConfig
from eeg_slm.utils.seeding import get_device, set_global_seed

console = Console()

# Canonical EEGMMIDB 64-channel order (matches what our loader produces after normalization).
# This is the standard 10-10 layout MNE assigns when reading EEGMMIDB EDF files.
EEGMMIDB_CHANNELS_64 = (
    "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6",
    "C5",  "C3",  "C1",  "CZ",  "C2",  "C4",  "C6",
    "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6",
    "FP1", "FPZ", "FP2",
    "AF7", "AF3", "AFZ", "AF4", "AF8",
    "F7",  "F5",  "F3",  "F1",  "FZ",  "F2",  "F4",  "F6",  "F8",
    "FT7", "FT8", "T7",  "T8",  "T9",  "T10",
    "TP7", "TP8",
    "P7",  "P5",  "P3",  "P1",  "PZ",  "P2",  "P4",  "P6",  "P8",
    "PO7", "PO3", "POZ", "PO4", "PO8",
    "O1",  "OZ",  "O2",  "IZ",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    p.add_argument("--ckpt", type=Path, default=None,
                   help="Path to a pretrained EEGLeJEPA checkpoint. If omitted, only the "
                        "random-init baseline is reported.")
    p.add_argument("--pad-channels", action="store_true",
                   help="If --ckpt is a 64-channel model, pad BCI-IV-2a's 22 channels into "
                        "64-channel positions (zeros elsewhere) so it can be applied without "
                        "re-pretraining.")
    p.add_argument("--data-root", type=Path, default=Path("data/raw/bci_iv_2a"))
    p.add_argument("--subjects", type=int, nargs="*", default=list(SUBJECT_IDS))
    p.add_argument("--session", default="T", choices=["T"])  # E support later
    p.add_argument("--C", type=float, default=1.0)
    return p.parse_args()


def _build_eegmmidb_to_bci_pad_indices() -> list[int]:
    """Indices into EEGMMIDB-64 channel order that correspond to each BCI-IV-2a channel."""
    src_to_idx = {ch.upper(): i for i, ch in enumerate(EEGMMIDB_CHANNELS_64)}
    return [src_to_idx[ch.upper()] for ch in BCI_IV_2A_EEG_CHANNELS]


def _pad_bci_to_64(X22: np.ndarray) -> np.ndarray:
    """Map (N, 22, T) BCI-IV-2a data into (N, 64, T) EEGMMIDB-layout with zeros."""
    n, _, t = X22.shape
    X64 = np.zeros((n, 64, t), dtype=np.float32)
    target_idx = _build_eegmmidb_to_bci_pad_indices()
    for j, src_pos in enumerate(target_idx):
        X64[:, src_pos] = X22[:, j]
    return X64


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    set_global_seed(cfg["training"]["seed"], deterministic=False)
    device = get_device(cfg["training"]["device"])
    console.print(f"[bold]Device:[/bold] {device}")

    # 1. Download (if needed) + build the 22-channel dataset
    console.print(f"\n[bold]Loading BCI-IV-2a (T session)[/bold] subjects={args.subjects}")
    BCIIV2aLoader(data_root=args.data_root)  # ensures directory exists
    X, y, sids = build_bci_iv_2a_dataset(
        subjects=args.subjects, data_root=args.data_root, session=args.session,
    )
    console.print(
        f"  {X.shape} epochs (N, C, T), {len(np.unique(y))} classes, "
        f"{len(np.unique(sids))} subjects "
        f"(per-class counts: {np.bincount(y).tolist()})"
    )

    # 2. Determine target architecture from checkpoint (or use base for random)
    state = None
    if args.ckpt is not None:
        if not args.ckpt.exists():
            raise FileNotFoundError(args.ckpt)
        state = torch.load(args.ckpt, map_location="cpu")
        # Sniff encoder.patch_embed weight to get input channels + embed_dim
        for k, v in state.items():
            if "patch_embed.weight" in k:
                ckpt_in_channels = v.shape[1]    # Conv1d: (out, in, kernel)
                ckpt_embed_dim = v.shape[0]
                break
        else:
            raise ValueError("Could not detect architecture from checkpoint.")
        is_large = ckpt_embed_dim == 256
        console.print(
            f"[bold]Checkpoint:[/bold] {args.ckpt.name}  "
            f"({ckpt_in_channels} in-channels, embed_dim={ckpt_embed_dim}, "
            f"{'large' if is_large else 'base'})"
        )
    else:
        ckpt_in_channels = 22  # default to matched architecture
        is_large = False
        console.print("[dim]No --ckpt given: running RANDOM-only baseline at 22 channels.[/dim]")

    # 3. Match data to architecture
    if ckpt_in_channels == 22:
        X_for_model = X  # native BCI-IV-2a 22 channels
        in_channels = 22
    elif ckpt_in_channels == 64 and args.pad_channels:
        X_for_model = _pad_bci_to_64(X)
        in_channels = 64
        console.print("  [yellow]padding 22→64 channels with zeros for cross-architecture transfer[/yellow]")
    else:
        raise ValueError(
            f"Checkpoint expects {ckpt_in_channels} input channels, "
            f"but BCI-IV-2a has 22. Either pretrain on the 22-channel intersection or "
            f"pass --pad-channels to zero-pad."
        )

    # 4. Build model configs
    def _make_cfg(size: str, n_channels: int) -> EEGLeJEPAConfig:
        cfg = EEGLeJEPAConfig.large() if size == "large" else EEGLeJEPAConfig.base()
        cfg.encoder.n_channels = n_channels
        cfg.encoder.patch_size = 40
        return cfg

    target_size = "large" if is_large else "base"
    model_cfg = _make_cfg(target_size, in_channels)

    # 5. Probes
    results: dict[str, object] = {}

    sources_to_probe = ["encoder_mean", "predictor_mean", "both_mean"]

    # Random baseline (always run)
    console.print("\n[bold]Random-init baseline[/bold]")
    torch.manual_seed(cfg["training"]["seed"] + 1)
    rand_model = EEGLeJEPA(model_cfg)
    for src in sources_to_probe:
        feats = extract_features_jepa(rand_model, X_for_model, source=src,
                                       device=device, batch_size=8)
        res = linear_probe_loso_from_features(feats, y, sids, C=args.C)
        results[f"random/{src}"] = res
        console.print(
            f"  random / {src:14s}: acc={res.mean_accuracy:.3f} "
            f"(macro-F1={res.mean_macro_f1:.3f}; macro-AUC={res.mean_auc:.3f})"
        )

    # Pretrained (if ckpt given)
    if state is not None:
        console.print(f"\n[bold]Pretrained:[/bold] {args.ckpt.name}")
        pre_model = EEGLeJEPA(model_cfg)
        missing, unexpected = pre_model.load_state_dict(state, strict=False)
        if missing or unexpected:
            console.print(f"  [yellow]missing {len(missing)} / unexpected {len(unexpected)}[/yellow]")
        for src in sources_to_probe:
            feats = extract_features_jepa(pre_model, X_for_model, source=src,
                                           device=device, batch_size=8)
            res = linear_probe_loso_from_features(feats, y, sids, C=args.C)
            results[f"pretrained/{src}"] = res
            rand = results[f"random/{src}"]
            delta_pp = (res.mean_accuracy - rand.mean_accuracy) * 100
            console.print(
                f"  pretrained / {src:14s}: acc={res.mean_accuracy:.3f} "
                f"(Δ {delta_pp:+5.1f}pp; macro-AUC={res.mean_auc:.3f})"
            )

    # 6. Pretty summary table
    table = Table(
        title=f"BCI-IV-2a LOSO ({len(args.subjects)} subjects, 4-class)",
        show_header=True, header_style="bold",
    )
    table.add_column("Source")
    table.add_column("Init")
    table.add_column("Acc / Bal-Acc / Macro-F1 / Macro-AUC", justify="right")
    table.add_column("Δ acc pp", justify="right")
    chance = float(max(np.bincount(y)) / len(y))
    for src in sources_to_probe:
        for init in (["random", "pretrained"] if state is not None else ["random"]):
            r = results[f"{init}/{src}"]
            cell = (f"{r.mean_accuracy:.3f} / {r.mean_balanced_accuracy:.3f} / "
                    f"{r.mean_macro_f1:.3f} / {r.mean_auc:.3f}")
            if init == "pretrained":
                rand = results[f"random/{src}"]
                delta = (r.mean_accuracy - rand.mean_accuracy) * 100
                delta_str = (f"[green]+{delta:.1f}[/green]" if delta > 2 else
                             f"[red]{delta:.1f}[/red]" if delta < -2 else f"[yellow]{delta:+.1f}[/yellow]")
            else:
                delta_str = "—"
            table.add_row(src, init, cell, delta_str)
    table.add_row("[dim]chance[/dim]", "[dim]—[/dim]",
                  f"[dim]{chance:.3f} / 0.250 / 0.250 / 0.500[/dim]", "")
    console.print()
    console.print(table)


if __name__ == "__main__":
    main()
