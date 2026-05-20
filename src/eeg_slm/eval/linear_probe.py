"""Linear-probe evaluation.

Given an encoder (frozen) and a labeled dataset (X, y, subject_ids):

  1. Run X through the encoder to get token embeddings (N, T_patches, D).
  2. Mean-pool over tokens to get (N, D) features.
  3. Train a sklearn LogisticRegression on the features.
  4. Evaluate cross-subject (leave-one-subject-out).

This is the standard SSL evaluation protocol — minimal, principled, and
hard to game. If features are useful for the downstream task, a linear
classifier on top will be measurably better than chance / random features.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from torch import Tensor, nn

FeatureSource = Literal[
    "encoder_mean",      # mean-pooled encoder embeddings (per-patch, no temporal context)
    "predictor_mean",    # mean-pooled predictor hidden states (with causal temporal context)
    "both_mean",         # concat of encoder_mean and predictor_mean
    "predictor_last",    # last token from the predictor (has seen the full sequence)
    "predictor_concat",  # flatten ALL predictor outputs (high-dim, needs strong reg)
]


@dataclass
class LinearProbeResult:
    fold_accuracies: list[float] = field(default_factory=list)
    fold_aucs: list[float] = field(default_factory=list)
    fold_balanced_accuracies: list[float] = field(default_factory=list)
    fold_macro_f1s: list[float] = field(default_factory=list)
    fold_subjects: list[int] = field(default_factory=list)
    chance: float = 0.5
    n_classes: int = 2

    @property
    def mean_accuracy(self) -> float:
        return float(np.mean(self.fold_accuracies)) if self.fold_accuracies else float("nan")

    @property
    def std_accuracy(self) -> float:
        return float(np.std(self.fold_accuracies)) if self.fold_accuracies else float("nan")

    @property
    def mean_auc(self) -> float:
        return float(np.mean(self.fold_aucs)) if self.fold_aucs else float("nan")

    @property
    def mean_balanced_accuracy(self) -> float:
        if not self.fold_balanced_accuracies:
            return float("nan")
        return float(np.mean(self.fold_balanced_accuracies))

    @property
    def mean_macro_f1(self) -> float:
        if not self.fold_macro_f1s:
            return float("nan")
        return float(np.mean(self.fold_macro_f1s))

    def summary(self) -> str:
        if not self.fold_accuracies:
            return "(no folds)"
        per_fold = ", ".join(
            f"s{s:03d}={a:.3f}" for s, a in zip(self.fold_subjects, self.fold_accuracies)
        )
        if self.n_classes <= 2:
            head = (f"LOSO accuracy = {self.mean_accuracy:.3f} ± {self.std_accuracy:.3f}  "
                    f"(AUC = {self.mean_auc:.3f}; chance = {self.chance:.2f})")
        else:
            head = (f"LOSO accuracy = {self.mean_accuracy:.3f} ± {self.std_accuracy:.3f}  "
                    f"(balanced acc = {self.mean_balanced_accuracy:.3f}; "
                    f"macro-F1 = {self.mean_macro_f1:.3f}; "
                    f"macro-AUC = {self.mean_auc:.3f}; "
                    f"chance = {self.chance:.2f})")
        return head + f"\n  per-fold: {per_fold}"


@torch.no_grad()
def extract_features_jepa(
    model: nn.Module,
    X: np.ndarray,
    source: FeatureSource = "encoder_mean",
    device: str = "cpu",
    batch_size: int = 8,
) -> np.ndarray:
    """Extract features from a full EEGLeJEPA model using a specified source.

    Lets us probe NOT just the encoder (per-patch, no time context) but also
    the predictor (causal Transformer with temporal context) — which is the
    natural feature source for sequence-level downstream tasks per the
    LeWM design philosophy.
    """
    model.eval()
    model.to(device)
    encoder = model.encoder
    predictor = getattr(model, "predictor", None)
    if source != "encoder_mean" and predictor is None:
        raise ValueError(f"Source '{source}' needs a predictor; model has none.")

    X_t = torch.from_numpy(X).float()
    pieces: list[np.ndarray] = []
    for i in range(0, len(X_t), batch_size):
        batch = X_t[i : i + batch_size].to(device)
        z_enc = encoder(batch)                       # (B, T, D)
        if source == "encoder_mean":
            feats = z_enc.mean(dim=1)
        else:
            z_pred = predictor(z_enc)                 # (B, T, D)
            if source == "predictor_mean":
                feats = z_pred.mean(dim=1)
            elif source == "predictor_last":
                feats = z_pred[:, -1]
            elif source == "predictor_concat":
                feats = z_pred.reshape(z_pred.shape[0], -1)
            elif source == "both_mean":
                feats = torch.cat([z_enc.mean(dim=1), z_pred.mean(dim=1)], dim=-1)
            else:
                raise ValueError(f"Unknown source: {source}")
        pieces.append(feats.detach().cpu().numpy())
    return np.concatenate(pieces, axis=0).astype(np.float32, copy=False)


@torch.no_grad()
def extract_features(
    encoder: nn.Module,
    X: np.ndarray,
    device: str = "cpu",
    batch_size: int = 8,
    pool: Literal["mean", "max"] = "mean",
) -> np.ndarray:
    """Forward X through `encoder`, pool token embeddings, return (N, D) features.

    Parameters
    ----------
    encoder
        Any module whose forward(x: (B, C, T)) returns (B, T_patches, D).
        Typically an `EEGEncoder` or `EEGLeJEPA.encoder`.
    X
        (N, C, T) preprocessed EEG.
    device
        torch device string.
    batch_size
        Forward batch size.
    pool
        "mean" or "max" over the token dimension.
    """
    encoder.eval()
    encoder.to(device)
    X_t = torch.from_numpy(X).float()
    pieces: list[np.ndarray] = []
    for i in range(0, len(X_t), batch_size):
        batch = X_t[i : i + batch_size].to(device)
        z = encoder(batch)  # (B, T, D)
        if pool == "mean":
            pooled = z.mean(dim=1)
        elif pool == "max":
            pooled = z.amax(dim=1)
        else:
            raise ValueError(f"Unknown pool: {pool}")
        pieces.append(pooled.detach().cpu().numpy())
    return np.concatenate(pieces, axis=0).astype(np.float32, copy=False)


def linear_probe_loso_from_features(
    features: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    C: float = 1.0,
    max_iter: int = 2000,
) -> LinearProbeResult:
    """LOSO probe on pre-extracted features (skips the encoder forward pass).

    Handles both binary and multi-class labels. For multi-class:
        - AUC is one-vs-rest macro-averaged
        - Additionally tracks balanced_accuracy and macro-F1
    """
    unique_subjects = np.unique(subject_ids)
    if len(unique_subjects) < 2:
        raise ValueError(
            "LOSO requires at least 2 distinct subjects; "
            f"got {len(unique_subjects)} ({unique_subjects.tolist()})."
        )

    classes_overall = np.unique(y)
    n_classes = int(classes_overall.size)

    result = LinearProbeResult(
        chance=float(max(np.bincount(y)) / len(y)),
        n_classes=n_classes,
    )

    for test_subject in unique_subjects:
        test_mask = subject_ids == test_subject
        train_mask = ~test_mask
        # Need all classes represented in train (otherwise classifier is biased);
        # need at least 2 distinct classes in test (otherwise AUC undefined).
        if len(np.unique(y[train_mask])) < n_classes or len(np.unique(y[test_mask])) < 2:
            continue
        clf = LogisticRegression(C=C, max_iter=max_iter)
        clf.fit(features[train_mask], y[train_mask])
        preds = clf.predict(features[test_mask])
        probas = clf.predict_proba(features[test_mask])

        result.fold_subjects.append(int(test_subject))
        result.fold_accuracies.append(float(accuracy_score(y[test_mask], preds)))

        if n_classes == 2:
            # Use prob of positive class
            result.fold_aucs.append(float(roc_auc_score(y[test_mask], probas[:, 1])))
        else:
            # Multi-class: macro-averaged OvR AUC. Need probas for ALL classes present
            # in test fold's labels; we pass labels= so sklearn aligns columns correctly.
            try:
                auc = roc_auc_score(
                    y[test_mask], probas,
                    multi_class="ovr", average="macro",
                    labels=classes_overall,
                )
            except ValueError:
                # Falls through if some class missing in test fold; OvR macro can't be computed
                auc = float("nan")
            result.fold_aucs.append(float(auc))
            result.fold_balanced_accuracies.append(
                float(balanced_accuracy_score(y[test_mask], preds))
            )
            result.fold_macro_f1s.append(
                float(f1_score(y[test_mask], preds, average="macro", labels=classes_overall))
            )

    return result


def linear_probe_loso(
    encoder: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    device: str = "cpu",
    batch_size: int = 8,
    pool: Literal["mean", "max"] = "mean",
    C: float = 1.0,
    max_iter: int = 2000,
) -> LinearProbeResult:
    """Leave-one-subject-out linear probe on the encoder.

    For each unique subject in `subject_ids`, train a LogisticRegression on
    the remaining subjects' features and evaluate on the held-out subject.
    Returns per-fold accuracies + AUCs and their means.
    """
    features = extract_features(encoder, X, device=device, batch_size=batch_size, pool=pool)
    return linear_probe_loso_from_features(features, y, subject_ids, C=C, max_iter=max_iter)
