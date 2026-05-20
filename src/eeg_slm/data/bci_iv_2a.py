"""BCI Competition IV Dataset 2a loader (.mat format from BNCI Horizon 2020).

Standard 4-class motor-imagery benchmark (left hand, right hand, feet, tongue).
9 subjects × 2 sessions (T = training, E = evaluation). 22 EEG channels + 3 EOG
at 250 Hz native sampling.

The dataset is hosted as Matlab `.mat` files at the BNCI Horizon 2020 archive.
The bbci.de mirror has `.gdf` files but those require a manual click-through
agreement and cannot be auto-downloaded. We use scipy.io.loadmat.

The 22 EEG channels are a subset of EEGMMIDB's 64-channel 10-10 layout, which
enables cross-dataset transfer from EEGMMIDB-pretrained encoders. The label
files in the T session contain ground-truth class labels for all trials.

References
----------
Brunner, C. et al. (2008). BCI Competition 2008 - Graz data set A.
http://bnci-horizon-2020.eu/database/data-sets/001-2014/
"""

from __future__ import annotations

import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import mne
import numpy as np
import scipy.io as sio

from eeg_slm.data.preprocessing import zscore_per_channel


def _make_progress_hook(filename: str):
    """Build a `reporthook` for `urllib.request.urlretrieve` that prints
    an in-place progress line every ~0.5s.

    BNCI Horizon's server is slow (often <500 KB/s), so per-file downloads
    can take minutes. Without progress feedback the script looks hung.
    """
    start_time = time.time()
    last_print = [0.0]

    def hook(block_num: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            # Some servers don't report size; just emit a tick every 5s.
            now = time.time()
            if now - last_print[0] >= 5.0:
                last_print[0] = now
                downloaded_mb = (block_num * block_size) / 1e6
                print(f"\r    {filename}: {downloaded_mb:.1f} MB", end="", flush=True)
            return

        now = time.time()
        downloaded = min(block_num * block_size, total_size)
        is_done = downloaded >= total_size
        if not is_done and now - last_print[0] < 0.5:
            return
        last_print[0] = now
        elapsed = max(now - start_time, 0.01)
        pct = (downloaded * 100) // total_size
        mb, total_mb = downloaded / 1e6, total_size / 1e6
        speed = (downloaded / 1e6) / elapsed
        print(f"\r    {filename}: {pct:3d}%  {mb:5.1f} / {total_mb:.1f} MB  "
              f"({speed:.2f} MB/s)", end="", flush=True)
        if is_done:
            print()  # newline when complete

    return hook

BNCI_BASE_URL = "http://bnci-horizon-2020.eu/database/data-sets/001-2014"
SUBJECT_IDS: tuple[int, ...] = tuple(range(1, 10))  # A01 .. A09
SESSIONS: tuple[str, ...] = ("T", "E")

# Channel ordering as it appears in the BCI-IV-2a .mat files (first 22 are EEG).
# These names are also all present in EEGMMIDB's 64-channel layout, enabling
# direct channel-intersection transfer.
BCI_IV_2A_EEG_CHANNELS: tuple[str, ...] = (
    "FZ",
    "FC3", "FC1", "FCZ", "FC2", "FC4",
    "C5", "C3", "C1", "CZ", "C2", "C4", "C6",
    "CP3", "CP1", "CPZ", "CP2", "CP4",
    "P1", "PZ", "P2",
    "POZ",
)

# 4-class motor-imagery labels (0-indexed for consistency across this codebase).
# The .mat files use 1-indexed labels {1, 2, 3, 4}; we subtract 1.
BCI_IV_2A_LABELS: dict[str, int] = {
    "left_hand": 0,
    "right_hand": 1,
    "feet": 2,
    "tongue": 3,
}


# ---------- downloader -------------------------------------------------------


@dataclass
class BCIIV2aLoader:
    """Downloader/cache for BCI Competition IV Dataset 2a (.mat from BNCI Horizon).

    Each file (one subject × one session) is ~40-50 MB. Full T-session corpus
    (9 files) is ~400 MB. Downloads cached + idempotent.
    """

    data_root: Path = field(default_factory=lambda: Path("data/raw/bci_iv_2a"))

    def __post_init__(self) -> None:
        self.data_root = Path(self.data_root).expanduser().resolve()
        self.data_root.mkdir(parents=True, exist_ok=True)

    def _file_path(self, subject: int, session: str) -> Path:
        return self.data_root / f"A{subject:02d}{session}.mat"

    def _url(self, subject: int, session: str) -> str:
        return f"{BNCI_BASE_URL}/A{subject:02d}{session}.mat"

    def download_subject(self, subject: int, sessions: Iterable[str] = SESSIONS) -> None:
        """Download a subject's `.mat` files. Skips files already on disk.

        Shows live progress via stderr-style in-place updates. BNCI Horizon is
        slow (~200-700 KB/s), so each ~40 MB file takes several minutes.
        """
        for session in sessions:
            path = self._file_path(subject, session)
            if path.exists():
                continue
            url = self._url(subject, session)
            print(f"  downloading {url}")
            hook = _make_progress_hook(path.name)
            urllib.request.urlretrieve(url, path, reporthook=hook)

    def download_all_t(self) -> None:
        """Convenience: download every subject's T session (~400 MB)."""
        for subject in SUBJECT_IDS:
            self.download_subject(subject, sessions=("T",))


# ---------- .mat structure parsing -------------------------------------------


def _extract_mi_runs(mat_obj: dict) -> list[dict]:
    """Find runs that contain motor-imagery trials.

    BCI-IV-2a's T session has 9 runs total: 1-3 are eye-movement / baseline,
    4-9 are the 6 MI runs (288 trials = 48 per class × 6 runs, with class
    distribution actually balanced per run: 12 each of the 4 classes).

    Each run dict has fields X (samples, n_channels), trial (start indices),
    y (1-indexed class labels), fs (sampling frequency).
    """
    raw_runs = mat_obj["data"]
    # `simplify_cells=True` gives us either a list-of-dicts or a 1-D numpy array
    # of dicts depending on file structure; normalize.
    try:
        iterator = list(raw_runs)
    except TypeError:
        iterator = [raw_runs]

    mi_runs: list[dict] = []
    for run in iterator:
        # Some runs might be MATLAB structs that came through as object arrays;
        # treat as dict-like.
        try:
            trials = np.atleast_1d(np.asarray(run["trial"]).ravel())
            labels = np.atleast_1d(np.asarray(run["y"]).ravel())
        except (KeyError, TypeError, IndexError):
            continue
        if len(trials) == 0 or len(labels) == 0:
            continue
        # MI labels are in {1, 2, 3, 4}; eye-movement runs have y all 0 or absent
        unique = set(int(v) for v in labels if int(v) > 0)
        if unique != {1, 2, 3, 4}:
            # Some runs might have a subset; only keep runs with all 4 MI classes
            # to avoid mixing baseline/non-MI runs.
            if not unique.issubset({1, 2, 3, 4}) or len(unique) < 2:
                continue
        mi_runs.append(run)
    return mi_runs


def _detect_unit_scale(X: np.ndarray) -> float:
    """Heuristic: detect whether X is in volts or microvolts.

    EEG amplitudes are typically ±50-200 µV (= ±5e-5 to ±2e-4 V). If the array's
    99th-percentile absolute value is in the µV range (>0.1, <10000), assume µV;
    if it's in the V range (<1e-3), assume V. Used to convert to V for MNE.
    """
    p99 = float(np.percentile(np.abs(X[np.isfinite(X)]), 99))
    if p99 < 1e-3:
        return 1.0          # already in V
    return 1e-6             # was in µV, multiply to get V


# ---------- main loader function ---------------------------------------------


def load_motor_imagery_subject(
    subject: int,
    data_root: str | Path = "data/raw/bci_iv_2a",
    session: str = "T",
    bandpass: tuple[float, float] = (1.0, 70.0),
    notch_hz: float | None = 50.0,             # BCI-IV-2a recorded in EU (50 Hz mains)
    resample_hz: float = 200.0,
    epoch_length_s: float = 4.0,
    zscore: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Load + preprocess + epoch one BCI-IV-2a subject's session.

    Returns
    -------
    X : (n_epochs, 22, n_times) float32, in µV (optionally per-channel z-scored)
        with n_times = int(epoch_length_s * resample_hz).
    y : (n_epochs,) int64, with class labels per `BCI_IV_2A_LABELS` (0-indexed).
    """
    loader = BCIIV2aLoader(data_root=Path(data_root))
    path = loader._file_path(subject, session)
    if not path.exists():
        loader.download_subject(subject, sessions=(session,))

    mat = sio.loadmat(str(path), simplify_cells=True)
    mi_runs = _extract_mi_runs(mat)
    if not mi_runs:
        raise ValueError(
            f"No MI runs found in A{subject:02d}{session}.mat. "
            f"Available keys in .mat: {list(mat.keys())}"
        )

    # Concatenate runs into a continuous signal, adjusting trial indices
    X_runs: list[np.ndarray] = []
    trial_starts_concat: list[int] = []
    labels_concat: list[int] = []
    native_sfreq: float | None = None
    sample_offset = 0

    for run in mi_runs:
        X_run = np.asarray(run["X"], dtype=np.float64)  # (samples, 25) typically
        if X_run.ndim != 2:
            raise ValueError(f"Unexpected X shape {X_run.shape} in run.")
        # Restrict to first 22 channels (EEG; drop 3 EOG)
        X_eeg = X_run[:, :22].T  # → (22, samples)
        trial_starts = np.asarray(run["trial"]).ravel().astype(int)
        y_run = np.asarray(run["y"]).ravel().astype(int)

        # Drop any trials/labels with invalid labels (some baseline runs may
        # have zero labels mixed in)
        valid = (y_run >= 1) & (y_run <= 4)
        trial_starts = trial_starts[valid]
        y_run = y_run[valid]

        if len(trial_starts) == 0:
            continue

        if native_sfreq is None:
            native_sfreq = float(run.get("fs", 250))

        X_runs.append(X_eeg)
        trial_starts_concat.extend((trial_starts + sample_offset).tolist())
        labels_concat.extend(y_run.tolist())
        sample_offset += X_eeg.shape[1]

    if not X_runs:
        raise ValueError(f"No valid MI trials extracted from A{subject:02d}{session}.")

    X_concat = np.concatenate(X_runs, axis=1)               # (22, total_samples)
    trial_starts_arr = np.asarray(trial_starts_concat, dtype=int)
    labels_arr = np.asarray(labels_concat, dtype=int)
    assert native_sfreq is not None

    # Detect unit (µV vs V) and convert to V for MNE
    scale = _detect_unit_scale(X_concat)
    X_in_volts = X_concat * scale

    info = mne.create_info(
        ch_names=list(BCI_IV_2A_EEG_CHANNELS),
        sfreq=native_sfreq,
        ch_types="eeg",
    )
    raw = mne.io.RawArray(X_in_volts, info, verbose="ERROR")

    # Bandpass (clamped to source Nyquist via the standard utility)
    nyquist = native_sfreq / 2.0
    h_freq = min(bandpass[1], nyquist - max(1.0, nyquist * 0.05))
    raw.filter(l_freq=bandpass[0], h_freq=h_freq, fir_design="firwin", verbose="ERROR")
    if notch_hz is not None and notch_hz < nyquist:
        raw.notch_filter(freqs=notch_hz, fir_design="firwin", verbose="ERROR")
    raw.set_eeg_reference("average", projection=False, verbose="ERROR")

    # Resample (and rescale trial indices proportionally)
    if abs(native_sfreq - resample_hz) > 1e-3:
        raw.resample(resample_hz, verbose="ERROR")
        trial_starts_arr = (trial_starts_arr * resample_hz / native_sfreq).round().astype(int)

    # Build events for MNE Epochs
    events = np.column_stack([
        trial_starts_arr,
        np.zeros(len(trial_starts_arr), dtype=int),
        labels_arr,
    ])
    event_id = {f"class_{i}": i for i in range(1, 5)}  # 1=L, 2=R, 3=feet, 4=tongue

    sfreq_out = raw.info["sfreq"]
    tmax = epoch_length_s - 1.0 / sfreq_out
    epochs = mne.Epochs(
        raw, events, event_id=event_id,
        tmin=0.0, tmax=tmax,
        baseline=None, preload=True, reject=None, verbose="ERROR",
    )

    X = epochs.get_data().astype(np.float32) * 1e6      # back to µV
    # Convert 1-indexed event codes to 0-indexed class labels
    y = (epochs.events[:, 2] - 1).astype(np.int64)
    if zscore:
        X = zscore_per_channel(X)
    return X, y


def build_bci_iv_2a_dataset(
    subjects: Iterable[int] = SUBJECT_IDS,
    data_root: str | Path = "data/raw/bci_iv_2a",
    session: str = "T",
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (X, y, subject_ids) across multiple subjects for LOSO probing."""
    Xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    sids: list[np.ndarray] = []
    for s in subjects:
        X, y = load_motor_imagery_subject(s, data_root=data_root, session=session, **kwargs)
        Xs.append(X)
        ys.append(y)
        sids.append(np.full(len(X), s, dtype=np.int64))
    return (
        np.concatenate(Xs, axis=0),
        np.concatenate(ys, axis=0),
        np.concatenate(sids, axis=0),
    )


# ---------- channel intersection utility -------------------------------------


def channel_intersection_indices(
    source_channels: Iterable[str],
    target_channels: Iterable[str] = BCI_IV_2A_EEG_CHANNELS,
) -> list[int]:
    """Indices into `source_channels` that match `target_channels`, in target order.

    Raises if any target channel is missing in source. Comparison is case-insensitive
    and dot-stripped (matching EEGMMIDB's "FP1." → "FP1" normalization).
    """
    def _norm(s: str) -> str:
        return s.replace(".", "").upper()

    src_index: dict[str, int] = {}
    for i, ch in enumerate(source_channels):
        src_index[_norm(ch)] = i
    target_norm = [_norm(ch) for ch in target_channels]
    missing = [ch for ch in target_norm if ch not in src_index]
    if missing:
        raise ValueError(
            f"Channels not in source ({len(missing)}): {missing}. "
            f"Source has {len(src_index)} channels."
        )
    return [src_index[ch] for ch in target_norm]


def restrict_to_bci_channels(
    X: np.ndarray, source_channels: Iterable[str]
) -> np.ndarray:
    """Restrict X (..., C_src, T) to the 22 BCI-IV-2a EEG channels, in BCI order."""
    idx = channel_intersection_indices(source_channels, BCI_IV_2A_EEG_CHANNELS)
    return X[..., idx, :]
