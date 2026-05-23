"""Supervised LOSO baseline training.

Mirror image of `linear_probe_loso_from_features` but for an end-to-end
supervised classifier: for each LOSO fold, build a fresh EEGClassifier,
train it on the training subjects with cross-entropy, evaluate on the
held-out subject. This isolates the SSL contribution.

Keeps the training loop minimal — no scheduler, no SIGReg, no mixed precision
(at our scale on M1 / single GPU, the overhead isn't worth it for short runs).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from torch import Tensor, nn

from eeg_slm.eval.linear_probe import LinearProbeResult


@dataclass
class SupervisedTrainConfig:
    n_steps: int = 500
    batch_size: int = 16
    learning_rate: float = 5e-4         # gentler than SSL's 1e-3 — random-init transformer
                                        # + cross-entropy at tiny data scale is fragile
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    warmup_steps: int = 50              # linear warmup; clamps to n_steps//5 for short runs
    eval_batch_size: int = 32


def _eval_classifier(
    model: nn.Module, X: np.ndarray, y: np.ndarray, device: str, batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference on (X, y), return (preds, probas).

    Defensively clamps NaN/Inf logits so a single diverged fold doesn't crash
    the whole LOSO run — sklearn's roc_auc_score refuses non-finite values.
    """
    model.eval()
    n = len(X)
    pieces_pred: list[np.ndarray] = []
    pieces_prob: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, n, batch_size):
            x = torch.from_numpy(X[i : i + batch_size]).float().to(device)
            logits = model(x)
            # Replace NaN/Inf with safe values BEFORE softmax to avoid NaN probas
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            preds = probs.argmax(axis=-1)
            pieces_pred.append(preds)
            pieces_prob.append(probs)
    return np.concatenate(pieces_pred), np.concatenate(pieces_prob, axis=0)


def _train_one_fold(
    model: nn.Module,
    X_train: np.ndarray, y_train: np.ndarray,
    cfg: SupervisedTrainConfig,
    device: str,
    seed: int = 0,
) -> bool:
    """In-place training of `model` on (X_train, y_train) for cfg.n_steps.

    Returns True if training completed cleanly, False if loss went non-finite
    and we bailed (in which case the partially-trained model is still usable
    for evaluation but caller should expect lower accuracy).
    """
    model.to(device).train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
    )
    # Linear warmup, clamped to at most n_steps // 5 for short runs.
    warmup = min(cfg.warmup_steps, max(1, cfg.n_steps // 5))
    def lr_lambda(step: int) -> float:
        return float(step + 1) / float(warmup) if step < warmup else 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    rng = np.random.default_rng(seed)
    n = len(X_train)
    X_t = torch.from_numpy(X_train).float()
    y_t = torch.from_numpy(y_train).long()

    for step in range(cfg.n_steps):
        idx = rng.choice(n, size=cfg.batch_size, replace=(n < cfg.batch_size))
        x = X_t[idx].to(device, non_blocking=True)
        target = y_t[idx].to(device, non_blocking=True)
        logits = model(x)
        loss = F.cross_entropy(logits, target)
        if not torch.isfinite(loss):
            print(f"    [yellow]WARNING:[/yellow] non-finite loss at step {step}; bailing fold")
            return False
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        scheduler.step()
    return True


def supervised_loso(
    model_factory,
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    cfg: SupervisedTrainConfig,
    device: str = "cpu",
    seed: int = 42,
    verbose_each_fold: bool = True,
) -> LinearProbeResult:
    """Train + evaluate `model_factory()` once per LOSO fold; return per-fold metrics.

    Parameters
    ----------
    model_factory : callable returning a fresh nn.Module per call.
        Each fold starts with a freshly initialized model.
    """
    unique_subjects = np.unique(subject_ids)
    if len(unique_subjects) < 2:
        raise ValueError("Need ≥2 subjects for LOSO.")
    classes_overall = np.unique(y)
    n_classes = int(classes_overall.size)

    result = LinearProbeResult(
        chance=float(max(np.bincount(y)) / len(y)),
        n_classes=n_classes,
    )

    for fold_i, test_subject in enumerate(unique_subjects):
        test_mask = subject_ids == test_subject
        train_mask = ~test_mask
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        if len(np.unique(y_train)) < n_classes or len(np.unique(y_test)) < 2:
            continue

        torch.manual_seed(seed + fold_i)
        model = model_factory()
        _train_one_fold(model, X_train, y_train, cfg, device=device, seed=seed + fold_i)
        preds, probas = _eval_classifier(model, X_test, y_test, device=device,
                                          batch_size=cfg.eval_batch_size)

        acc = float(accuracy_score(y_test, preds))
        result.fold_subjects.append(int(test_subject))
        result.fold_accuracies.append(acc)

        if n_classes == 2:
            try:
                auc = float(roc_auc_score(y_test, probas[:, 1]))
            except ValueError:
                auc = float("nan")
        else:
            try:
                auc = float(roc_auc_score(
                    y_test, probas, multi_class="ovr", average="macro",
                    labels=classes_overall,
                ))
            except ValueError:
                auc = float("nan")
            result.fold_balanced_accuracies.append(
                float(balanced_accuracy_score(y_test, preds))
            )
            result.fold_macro_f1s.append(
                float(f1_score(y_test, preds, average="macro", labels=classes_overall))
            )
        result.fold_aucs.append(auc)

        if verbose_each_fold:
            print(f"  fold s{int(test_subject):03d}: acc={acc:.3f}  auc={auc:.3f}")

        # Free GPU memory between folds
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    return result
