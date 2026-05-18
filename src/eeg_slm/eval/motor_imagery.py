"""Build a labeled motor-imagery dataset from PhysioNet EEGMMIDB.

EEGMMIDB annotations (Schalk et al. 2004 / PhysioNet conventions):
  T0  — rest
  T1  — open / close left fist (or imagine doing so)
  T2  — open / close right fist (or imagine doing so)

Motor-imagery runs (vs motor-execution runs):
  4, 8, 12  — imagine left/right fist movement
  6, 10, 14 — imagine left/right or both feet/fist (we skip the feet/two-fist runs)

For a clean left-vs-right binary task we use runs {4, 8, 12} by default.

The output is event-locked to a window of length `epoch_length_s` starting at
each T1/T2 event, matching the pretraining tokenizer's expected sample count.
T0 (rest) events are excluded — they're not part of the classification task.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from mne import Epochs, events_from_annotations

from eeg_slm.data.loaders import EEGMMIDBLoader
from eeg_slm.data.preprocessing import (
    PreprocessingConfig,
    preprocess_raw,
    to_numpy,
    zscore_per_channel,
)

# Canonical left-vs-right motor-imagery runs in EEGMMIDB
RUNS_MOTOR_IMAGERY_LEFT_RIGHT = (4, 8, 12)

# Label conventions per task
EEGMMIDB_MI_LABELS = {"left_fist": 0, "right_fist": 1}
EEGMMIDB_REST_VS_ACTIVITY_LABELS = {"rest": 0, "activity": 1}

# Supported probe tasks
Task = Literal["left_right", "rest_vs_activity"]


@dataclass
class MIDataset:
    """Output of build_motor_imagery_dataset."""

    X: np.ndarray         # (N, C, T) float32, already z-scored
    y: np.ndarray         # (N,) int — 0=left, 1=right
    subject_ids: np.ndarray  # (N,) int — subject number for each epoch

    def summary(self) -> str:
        n = len(self.X)
        per_subj = np.bincount(self.subject_ids.astype(int))
        nonzero = per_subj[per_subj > 0]
        n_left = int((self.y == 0).sum())
        n_right = int((self.y == 1).sum())
        return (
            f"MIDataset: {n} epochs across {len(nonzero)} subjects "
            f"({n_left} left, {n_right} right). "
            f"X={self.X.shape}, per-subject min/median/max = "
            f"{nonzero.min()}/{int(np.median(nonzero))}/{nonzero.max()}."
        )


def _extract_event_locked(
    raw, epoch_length_s: float, task: Task = "left_right",
) -> tuple[Epochs, np.ndarray]:
    """Event-lock to task-relevant events with a window matching pretraining length.

    Parameters
    ----------
    raw
        Preprocessed MNE Raw.
    epoch_length_s
        Window length per epoch.
    task
        - "left_right": include T1 (left=0) and T2 (right=1); exclude T0 (rest).
        - "rest_vs_activity": include T0 (rest=0), T1/T2 (activity=1).

    Returns the MNE Epochs object and a (n_epochs,) array of 0/1 labels.
    """
    sfreq = raw.info["sfreq"]
    tmax = epoch_length_s - 1.0 / sfreq

    events, event_id_map = events_from_annotations(raw, verbose="ERROR")

    if task == "left_right":
        if "T1" not in event_id_map or "T2" not in event_id_map:
            raise ValueError(f"Expected T1, T2 annotations; found {list(event_id_map)}.")
        use_event_id = {"T1": event_id_map["T1"], "T2": event_id_map["T2"]}
        label_map = {use_event_id["T1"]: 0, use_event_id["T2"]: 1}
    elif task == "rest_vs_activity":
        missing = [k for k in ("T0", "T1", "T2") if k not in event_id_map]
        if missing:
            raise ValueError(f"Missing annotations {missing}; found {list(event_id_map)}.")
        use_event_id = {k: event_id_map[k] for k in ("T0", "T1", "T2")}
        label_map = {
            use_event_id["T0"]: 0,                # rest
            use_event_id["T1"]: 1,                # activity
            use_event_id["T2"]: 1,
        }
    else:
        raise ValueError(f"Unknown task: {task}")

    epochs = Epochs(
        raw, events, event_id=use_event_id,
        tmin=0.0, tmax=tmax,
        baseline=None, preload=True, verbose="ERROR",
    )
    y = np.array([label_map[c] for c in epochs.events[:, 2]], dtype=np.int64)
    return epochs, y


def build_motor_imagery_dataset(
    subjects: list[int],
    data_root: str | Path,
    preprocessing: PreprocessingConfig,
    runs: tuple[int, ...] = RUNS_MOTOR_IMAGERY_LEFT_RIGHT,
    task: Task = "left_right",
    to_microvolts: bool = True,
    zscore: bool = True,
    balance_rest: bool = True,
) -> MIDataset:
    """Load + preprocess + event-lock motor-imagery EEG from one or more subjects.

    Parameters
    ----------
    task
        "left_right" (default) or "rest_vs_activity". See `_extract_event_locked`.
    balance_rest
        For "rest_vs_activity": each run has ~2× more T0 (rest) events than
        T1+T2 combined. If True (default), subsample rest events per subject
        so the two classes are balanced. Set False to keep all rest events.
    """
    loader = EEGMMIDBLoader(data_root=Path(data_root))
    X_pieces: list[np.ndarray] = []
    y_pieces: list[np.ndarray] = []
    s_pieces: list[np.ndarray] = []

    rng = np.random.default_rng(seed=42)  # deterministic subsampling

    for subject in subjects:
        raw = loader.load_raw(subject=subject, runs=list(runs))
        raw_pp = preprocess_raw(raw, preprocessing)
        epochs, y = _extract_event_locked(raw_pp, preprocessing.epoch_length_s, task=task)
        X = to_numpy(epochs, to_microvolts=to_microvolts)
        if zscore:
            X = zscore_per_channel(X)

        if task == "rest_vs_activity" and balance_rest:
            n_act = int((y == 1).sum())
            rest_idx = np.where(y == 0)[0]
            if len(rest_idx) > n_act:
                keep_rest = rng.choice(rest_idx, size=n_act, replace=False)
                keep_all = np.sort(np.concatenate([keep_rest, np.where(y == 1)[0]]))
                X = X[keep_all]
                y = y[keep_all]

        X_pieces.append(X)
        y_pieces.append(y)
        s_pieces.append(np.full(len(X), subject, dtype=np.int64))

    return MIDataset(
        X=np.concatenate(X_pieces, axis=0).astype(np.float32, copy=False),
        y=np.concatenate(y_pieces, axis=0),
        subject_ids=np.concatenate(s_pieces, axis=0),
    )
