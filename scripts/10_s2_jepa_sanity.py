"""S²-JEPA architectural sanity checks (before committing to the refactor).

Runs three quick experiments on top of an existing pretrained checkpoint to
validate the three contribution premises proposed in the S²-JEPA architecture:

  Sanity 1 — Latent-source projector U is feasible.
    Project z (mean-pooled encoder embedding, dim=D) through a low-rank matrix
    U ∈ R^{D × E} for E ∈ {4, 8, 16, 32}, run the linear probe on s = z @ U,
    compare against the baseline probe on z. Two variants:
      (a) random orthogonal U   — lower bound, tests dimensionality budget
      (b) PCA-fit U on train z  — upper bound for fixed U

    Pass criterion: PCA-fit U with E=16 keeps probe accuracy within ~3 pp of
    the full-D baseline. Means the embedding space *is* low-rank-friendly and
    a learned U has room to live in.

  Sanity 2 — Cross-source independence is achievable without crushing acc.
    Train a learnable orthogonal U via gradient descent on:
      L = λ_ortho * ||U^T U - I||_F^2 + λ_indep * sum_{i<j} |corr(s^i, s^j)|
    Evaluate the resulting s = z @ U on the linear probe and report the
    off-diagonal correlation matrix.

    Pass criterion: off-diagonal |corr| < 0.1 AND probe accuracy within
    ~3 pp of PCA baseline. Means the independence objective doesn't destroy
    the information.

  Sanity 3 — Top-k routing under INT8 QAT does not collapse to uniform.
    Build a 4-expert sparse MoE wrapper around the predictor's last layer.
    Train with PyTorch QAT (FakeQuantize on weights+activations) for ~1k steps
    on the encoder embeddings (JEPA next-step MSE). Track the gate's expert
    utilization distribution across training.

    Pass criterion: per-expert utilization stays within [0.10, 0.50] (i.e.,
    NOT collapsed to uniform 0.25 and NOT collapsed to a single expert).
    Means we can preserve sparse routing under quantization.

Run:

    python scripts/10_s2_jepa_sanity.py \\
        --ckpt runs/s7-lambda-1.0/model_final.pt \\
        --subjects 1 2 3 4 5 6 7 8 9 10 \\
        --task rest_vs_activity

Total runtime ~ 5-15 min on M1 (most spent in encoder feature extraction).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from rich.console import Console
from rich.table import Table
from torch import Tensor, nn

from eeg_slm.data import PreprocessingConfig
from eeg_slm.eval import (
    build_motor_imagery_dataset,
    extract_features_jepa,
    linear_probe_loso_from_features,
)
from eeg_slm.eval.motor_imagery import RUNS_MOTOR_IMAGERY_LEFT_RIGHT
from eeg_slm.models import EEGLeJEPA, EEGLeJEPAConfig
from eeg_slm.utils.seeding import get_device, set_global_seed

console = Console()

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    p.add_argument("--ckpt", type=Path, required=True,
                   help="Path to EEGLeJEPA state_dict (.pt)")
    p.add_argument("--subjects", type=int, nargs="+", required=True,
                   help="Subject IDs for the probe (≥3 for LOSO)")
    p.add_argument("--task", choices=["left_right", "rest_vs_activity"],
                   default="rest_vs_activity")
    p.add_argument("--runs", type=int, nargs="*",
                   default=list(RUNS_MOTOR_IMAGERY_LEFT_RIGHT))
    p.add_argument("--qat-steps", type=int, default=1000)
    p.add_argument("--qat-batch", type=int, default=32)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Sanity 1 helpers
# ---------------------------------------------------------------------------

def random_orthogonal(d: int, e: int, seed: int = 0) -> np.ndarray:
    """Return a (d, e) matrix with orthonormal columns (e ≤ d)."""
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((d, e)).astype(np.float32)
    Q, _ = np.linalg.qr(M)
    return Q[:, :e]


def pca_fit(z_train: np.ndarray, e: int) -> np.ndarray:
    """Return the top-e principal directions as columns (d, e)."""
    zc = z_train - z_train.mean(0, keepdims=True)
    # SVD on centered features
    _, _, Vt = np.linalg.svd(zc, full_matrices=False)
    return Vt[:e].T.astype(np.float32)  # (d, e)


def run_sanity_1(
    z_full: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    e_values: tuple[int, ...] = (4, 8, 16, 32),
) -> None:
    console.print("\n[bold cyan]Sanity 1 — Latent-source projector U feasibility[/bold cyan]")

    d = z_full.shape[1]
    baseline = linear_probe_loso_from_features(z_full, y, subject_ids)
    console.print(f"  baseline (full z, D={d}): acc={baseline.mean_accuracy:.3f} "
                  f"AUC={baseline.mean_auc:.3f}")

    table = Table(title="Sanity 1 results", show_header=True, header_style="bold")
    table.add_column("E")
    table.add_column("Random U acc", justify="right")
    table.add_column("Random U Δ pp", justify="right")
    table.add_column("PCA U acc", justify="right")
    table.add_column("PCA U Δ pp", justify="right")

    # PCA needs a train/test split; use the same LOSO partitioning logic.
    # Easiest: fit PCA on the global mean-centered z (slight leakage but
    # this is a sanity check, not the final evaluation).
    for e in e_values:
        if e > d:
            continue
        U_rand = random_orthogonal(d, e, seed=42)
        s_rand = z_full @ U_rand
        res_rand = linear_probe_loso_from_features(s_rand, y, subject_ids)

        U_pca = pca_fit(z_full, e)
        s_pca = z_full @ U_pca
        res_pca = linear_probe_loso_from_features(s_pca, y, subject_ids)

        d_rand = (res_rand.mean_accuracy - baseline.mean_accuracy) * 100
        d_pca = (res_pca.mean_accuracy - baseline.mean_accuracy) * 100
        table.add_row(
            str(e),
            f"{res_rand.mean_accuracy:.3f}",
            f"{d_rand:+.1f}",
            f"{res_pca.mean_accuracy:.3f}",
            f"{d_pca:+.1f}",
        )

    console.print(table)
    console.print(
        "  [dim]Pass: PCA U with E=16 within ~3 pp of baseline → "
        "embedding is low-rank-friendly.[/dim]"
    )


# ---------------------------------------------------------------------------
# Sanity 2 helpers
# ---------------------------------------------------------------------------

class LearnableProjector(nn.Module):
    """U ∈ R^{D × E} with orthogonality + cross-source independence losses."""

    def __init__(self, d: int, e: int) -> None:
        super().__init__()
        # init from random orthogonal
        Q = torch.from_numpy(random_orthogonal(d, e, seed=0))
        self.U = nn.Parameter(Q.clone())

    def forward(self, z: Tensor) -> Tensor:
        return z @ self.U


def ortho_loss(U: Tensor) -> Tensor:
    """||U^T U - I||_F^2 — pushes U to have orthonormal columns."""
    e = U.shape[1]
    UtU = U.T @ U
    return ((UtU - torch.eye(e, device=U.device)) ** 2).sum()


def indep_loss(s: Tensor) -> Tensor:
    """Sum of squared off-diagonal correlations of the sources."""
    # s: (N, E)
    s = s - s.mean(0, keepdim=True)
    s = s / (s.std(0, keepdim=True) + 1e-6)
    C = (s.T @ s) / s.shape[0]              # (E, E) correlation matrix
    off = C - torch.diag(torch.diag(C))
    return (off ** 2).sum()


def run_sanity_2(
    z_full: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    e: int = 16,
    epochs: int = 300,
    lr: float = 1e-2,
    lambda_ortho: float = 1.0,
    lambda_indep: float = 1.0,
) -> None:
    console.print("\n[bold cyan]Sanity 2 — Cross-source independence achievable[/bold cyan]")

    d = z_full.shape[1]
    z = torch.from_numpy(z_full).float()
    proj = LearnableProjector(d, e)
    opt = torch.optim.Adam(proj.parameters(), lr=lr)

    losses = []
    for step in range(epochs):
        opt.zero_grad()
        s = proj(z)
        l_o = ortho_loss(proj.U)
        l_i = indep_loss(s)
        loss = lambda_ortho * l_o + lambda_indep * l_i
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == epochs - 1:
            losses.append((step, loss.detach().item(), l_o.detach().item(), l_i.detach().item()))

    table_loss = Table(title="Sanity 2 — training trace", show_header=True)
    table_loss.add_column("step")
    table_loss.add_column("L_total", justify="right")
    table_loss.add_column("L_ortho", justify="right")
    table_loss.add_column("L_indep", justify="right")
    for step, l, lo, li in losses:
        table_loss.add_row(str(step), f"{l:.4f}", f"{lo:.4f}", f"{li:.4f}")
    console.print(table_loss)

    # Final off-diagonal corr
    with torch.no_grad():
        s = proj(z).numpy()
    s_norm = (s - s.mean(0)) / (s.std(0) + 1e-6)
    C = np.abs(s_norm.T @ s_norm / s_norm.shape[0])
    off_diag_mean = float((C - np.diag(np.diag(C))).sum() / (e * (e - 1)))
    off_diag_max = float((C - np.diag(np.diag(C))).max())

    # Linear probe on learned sources
    res = linear_probe_loso_from_features(s, y, subject_ids)

    # Baseline for context
    baseline = linear_probe_loso_from_features(z_full, y, subject_ids)
    U_pca = pca_fit(z_full, e)
    res_pca = linear_probe_loso_from_features(z_full @ U_pca, y, subject_ids)

    table = Table(title="Sanity 2 — final metrics", show_header=True, header_style="bold")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Mean |corr_off_diag|", f"{off_diag_mean:.4f}")
    table.add_row("Max |corr_off_diag|", f"{off_diag_max:.4f}")
    table.add_row("Probe acc (learned U)", f"{res.mean_accuracy:.3f}")
    table.add_row("Probe acc (PCA U, same E)", f"{res_pca.mean_accuracy:.3f}")
    table.add_row("Probe acc (full z)", f"{baseline.mean_accuracy:.3f}")
    console.print(table)
    console.print(
        "  [dim]Pass: mean off-diag |corr| < 0.10 AND probe acc within ~3 pp "
        "of PCA baseline.[/dim]"
    )


# ---------------------------------------------------------------------------
# Sanity 3 helpers — Top-k MoE under INT8 QAT
# ---------------------------------------------------------------------------

class TopKMoEPredictor(nn.Module):
    """A drop-in 4-expert sparse MoE wrapper around a small predictor head.

    Each expert is a 2-layer MLP that predicts next-step embedding deltas.
    Routing uses learned (B, T, D) → (B, T, E) gate logits, top-k=2.

    The whole module is INT8-quantizable via PyTorch's FX quantization or
    eager quantize_qat with FakeQuantize. We instrument it to track gate
    utilization across training to detect collapse.
    """

    def __init__(self, dim: int, n_experts: int = 4, k: int = 2, hidden_mult: int = 2) -> None:
        super().__init__()
        self.dim = dim
        self.n_experts = n_experts
        self.k = k
        hidden = dim * hidden_mult
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, dim),
            )
            for _ in range(n_experts)
        ])
        self.gate = nn.Linear(dim, n_experts)
        # Running counter for expert utilization (NOT a parameter)
        self.register_buffer("util_counter", torch.zeros(n_experts))
        self.register_buffer("util_total", torch.zeros(1))

    def forward(self, z: Tensor) -> Tensor:
        """z: (B, T, D). Returns (B, T, D)."""
        B, T, D = z.shape
        z_flat = z.reshape(B * T, D)
        logits = self.gate(z_flat)                   # (BT, E)
        gates = F.softmax(logits, dim=-1)            # (BT, E)
        top_vals, top_idx = gates.topk(self.k, dim=-1)  # both (BT, k)
        # Re-normalize over the selected k
        top_vals = top_vals / (top_vals.sum(-1, keepdim=True) + 1e-9)

        out = torch.zeros_like(z_flat)
        for e in range(self.n_experts):
            mask = (top_idx == e).any(-1)            # (BT,)
            if mask.sum() == 0:
                continue
            sel_z = z_flat[mask]
            sel_out = self.experts[e](sel_z)
            # Weight by the gate value assigned to this expert
            sel_gate = top_vals[mask] * (top_idx[mask] == e).float()
            sel_w = sel_gate.sum(-1, keepdim=True)
            out[mask] = out[mask] + sel_w * sel_out

            # Update util counters (utilization = fraction of tokens routed here)
            if not self.training:
                continue
            self.util_counter[e] += float(mask.sum())
        self.util_total += float(B * T)
        return out.reshape(B, T, D)

    def utilization(self) -> np.ndarray:
        """Per-expert fraction of routed tokens since last reset."""
        total = float(self.util_total.item()) or 1.0
        return (self.util_counter.cpu().numpy() / total).astype(np.float32)

    def reset_util(self) -> None:
        self.util_counter.zero_()
        self.util_total.zero_()


def run_sanity_3(
    model: EEGLeJEPA,
    X: np.ndarray,
    device: str,
    n_steps: int = 1000,
    batch_size: int = 32,
) -> None:
    console.print("\n[bold cyan]Sanity 3 — Top-k routing under INT8 QAT[/bold cyan]")

    # PyTorch eager QAT's FakeQuantize ops are CPU-only (no MPS implementation
    # of aten::_fused_moving_avg_obs_fq_helper). Encoder forward can stay on MPS
    # for speed; the MoE training + QAT must run on CPU.
    moe_device = "cpu"
    console.print(f"  [dim]Encoder forward on {device}; MoE+QAT on {moe_device} "
                  f"(MPS lacks FakeQuantize op).[/dim]")

    encoder = model.encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # Probe shape of encoder output
    with torch.no_grad():
        sample = torch.from_numpy(X[:2]).float().to(device)
        z0 = encoder(sample)
    D = z0.shape[-1]
    console.print(f"  Encoder dim D={D}, n_steps={n_steps}, batch={batch_size}")

    # Pre-extract z once for the WHOLE training set — small enough to fit in RAM,
    # and avoids re-running the encoder every step (which would also dominate
    # the timing under CPU fallback).
    console.print("  [dim]Pre-extracting encoder embeddings z for all X (one pass)...[/dim]")
    z_cache_pieces: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(X), 16):
            batch = torch.from_numpy(X[i : i + 16]).float().to(device)
            z_batch = encoder(batch).detach().cpu().numpy()
            z_cache_pieces.append(z_batch)
    z_cache = np.concatenate(z_cache_pieces, axis=0)
    console.print(f"  [dim]z_cache shape: {z_cache.shape}[/dim]")

    # Two-pass training: (A) FP32 baseline, (B) INT8 QAT.
    # Each pass runs n_steps on CPU using pre-cached z embeddings.
    def train_pass(label: str, use_qat: bool) -> tuple[list[float], np.ndarray]:
        moe = TopKMoEPredictor(dim=D, n_experts=4, k=2).to(moe_device)
        moe.train()

        if use_qat:
            # PyTorch eager-mode QAT. We attach a default qconfig with FakeQuantize on
            # weights and activations, fuse where applicable, and prepare.
            try:
                from torch.quantization import (
                    get_default_qat_qconfig,
                    prepare_qat,
                )
                # Use 'qnnpack' for ARM (M1) backend; 'fbgemm' for x86.
                import platform
                backend = "qnnpack" if platform.machine() == "arm64" else "fbgemm"
                torch.backends.quantized.engine = backend
                moe.qconfig = get_default_qat_qconfig(backend)
                moe = prepare_qat(moe)
                console.print(f"  [dim]{label}: QAT prepared with backend={backend}[/dim]")
            except Exception as ex:
                console.print(f"  [yellow]QAT setup failed: {ex} — falling back to FP32[/yellow]")
                use_qat = False

        opt = torch.optim.AdamW(moe.parameters(), lr=5e-4)
        losses: list[float] = []
        moe.reset_util()

        n = z_cache.shape[0]
        rng = np.random.default_rng(42)
        for step in range(n_steps):
            idx = rng.integers(0, n, size=batch_size)
            z = torch.from_numpy(z_cache[idx]).float().to(moe_device)  # (B, T, D)
            pred = moe(z)
            # JEPA-style: predict next-step embedding (no grad through encoder/target)
            target = z[:, 1:].detach()
            loss = F.mse_loss(pred[:, :-1], target)
            opt.zero_grad()
            loss.backward()
            opt.step()

            if step % 100 == 0 or step == n_steps - 1:
                losses.append(loss.detach().item())
        return losses, moe.utilization()

    fp32_losses, fp32_util = train_pass("FP32", use_qat=False)
    qat_losses, qat_util = train_pass("INT8 QAT", use_qat=True)

    table = Table(title="Sanity 3 — expert utilization & loss", show_header=True,
                  header_style="bold")
    table.add_column("Setup")
    table.add_column("Final loss", justify="right")
    table.add_column("Expert 0", justify="right")
    table.add_column("Expert 1", justify="right")
    table.add_column("Expert 2", justify="right")
    table.add_column("Expert 3", justify="right")
    table.add_row("FP32",
                  f"{fp32_losses[-1]:.4f}",
                  *[f"{u:.3f}" for u in fp32_util])
    table.add_row("INT8 QAT",
                  f"{qat_losses[-1]:.4f}",
                  *[f"{u:.3f}" for u in qat_util])
    console.print(table)

    def verdict(util: np.ndarray) -> str:
        u_min = util.min()
        u_max = util.max()
        if u_min < 0.05 and u_max > 0.80:
            return "[red]collapsed to single expert[/red]"
        if u_max - u_min < 0.05:
            return "[yellow]collapsed to uniform[/yellow]"
        return f"[green]healthy (range {u_min:.2f}-{u_max:.2f})[/green]"

    console.print(f"  FP32 verdict:     {verdict(fp32_util)}")
    console.print(f"  INT8 QAT verdict: {verdict(qat_util)}")
    console.print(
        "  [dim]Pass: INT8 QAT verdict is 'healthy' — routing survives "
        "quantization. If 'collapsed to uniform', we need a temperature "
        "schedule (Contribution 4 work item).[/dim]"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    set_global_seed(cfg["training"]["seed"], deterministic=False)
    device = get_device(cfg["training"]["device"])
    console.print(f"[bold]Device:[/bold] {device}")
    console.print(f"[bold]Checkpoint:[/bold] {args.ckpt}")

    pp = PreprocessingConfig(
        bandpass_low_hz=cfg["preprocessing"]["bandpass_low_hz"],
        bandpass_high_hz=cfg["preprocessing"]["bandpass_high_hz"],
        notch_hz=cfg["preprocessing"]["notch_hz"],
        reference=cfg["preprocessing"]["reference"],
        resample_hz=cfg["preprocessing"]["resample_hz"],
        epoch_length_s=cfg["preprocessing"]["epoch_length_s"],
        epoch_overlap_s=cfg["preprocessing"]["epoch_overlap_s"],
    )
    console.print(f"[bold]Loading dataset[/bold] task={args.task} "
                  f"(subjects={args.subjects})")
    ds = build_motor_imagery_dataset(
        subjects=args.subjects,
        data_root=cfg["paths"]["data_root"],
        preprocessing=pp,
        runs=tuple(args.runs),
        task=args.task,
    )
    console.print(f"  {ds.summary()}")

    # Load model
    model_cfg = EEGLeJEPAConfig()
    model_cfg.encoder.n_channels = ds.X.shape[1]
    model_cfg.encoder.patch_size = 40
    model = EEGLeJEPA(model_cfg)
    state = torch.load(args.ckpt, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        console.print(f"  [yellow]missing keys: {missing[:3]}{'...' if len(missing)>3 else ''}[/yellow]")
    if unexpected:
        console.print(f"  [yellow]unexpected keys: {unexpected[:3]}{'...' if len(unexpected)>3 else ''}[/yellow]")

    # Extract encoder-mean features once
    z = extract_features_jepa(model, ds.X, source="encoder_mean",
                              device=device, batch_size=8)
    console.print(f"  [bold]Feature matrix:[/bold] z.shape={z.shape}")

    # --- Sanity 1 -------------------------------------------------------
    run_sanity_1(z, ds.y, ds.subject_ids)

    # --- Sanity 2 -------------------------------------------------------
    run_sanity_2(z, ds.y, ds.subject_ids, e=16)

    # --- Sanity 3 -------------------------------------------------------
    run_sanity_3(model, ds.X, device=device,
                 n_steps=args.qat_steps, batch_size=args.qat_batch)

    console.print("\n[bold green]Sanity checks complete.[/bold green]")
    console.print(
        "Decision rule:\n"
        "  Sanity 1 PCA U(E=16) within 3 pp → Contribution 1 viable\n"
        "  Sanity 2 mean off-diag |corr| < 0.10 & acc within 3 pp → "
        "Contribution 2 viable\n"
        "  Sanity 3 INT8 QAT verdict 'healthy' → Contribution 4 free; "
        "otherwise we need a logit-temperature schedule (still viable, "
        "just more work)"
    )


if __name__ == "__main__":
    main()
