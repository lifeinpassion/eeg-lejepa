"""Standard EEG preprocessing.

These steps are intentionally conservative and match what LaBraM, EEGPT, and
most EEG-FM training pipelines do as a first pass:

1. Bandpass filter (e.g., 1-80 Hz)
2. Notch filter at power-line frequency (50 Hz in EU/CN, 60 Hz in US)
3. Re-reference (commonly average reference for foundation-model pretraining)
4. Resample to a target rate (e.g., 200 Hz to roughly match LaBraM's choice)
5. Epoch into fixed-length windows for batched training

The functions take MNE Raw objects and return MNE Raw / Epochs objects so the
pipeline stays interoperable with the broader EEG ecosystem.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from mne import Epochs, events_from_annotations, make_fixed_length_events
from mne.io import BaseRaw


@dataclass
class PreprocessingConfig:
    bandpass_low_hz: float = 1.0
    bandpass_high_hz: float = 80.0
    notch_hz: float | None = 60.0
    reference: str | list[str] | None = "average"  # "average", channel list, or None
    resample_hz: float | None = 200.0
    epoch_length_s: float = 4.0
    epoch_overlap_s: float = 0.0


def preprocess_raw(raw: BaseRaw, cfg: PreprocessingConfig) -> BaseRaw:
    """Apply filtering, notch, re-referencing, and resampling to a Raw object.

    Returns a new Raw object (the input is copied first).
    """
    raw = raw.copy().load_data(verbose="ERROR")

    # Bandpass — clamp h_freq strictly below Nyquist of the source signal
    # (filter is applied before resampling, so Nyquist is based on raw.info["sfreq"]).
    nyquist = raw.info["sfreq"] / 2.0
    h_freq = cfg.bandpass_high_hz
    if h_freq is not None and h_freq >= nyquist:
        clamped = max(1.0, nyquist - max(1.0, nyquist * 0.05))  # at least 1 Hz below Nyquist
        warnings.warn(
            f"Requested bandpass high freq {h_freq} Hz is at or above Nyquist "
            f"({nyquist} Hz). Clamping to {clamped} Hz.",
            stacklevel=2,
        )
        h_freq = clamped

    raw.filter(
        l_freq=cfg.bandpass_low_hz,
        h_freq=h_freq,
        fir_design="firwin",
        verbose="ERROR",
    )

    # Notch
    if cfg.notch_hz is not None:
        raw.notch_filter(freqs=cfg.notch_hz, fir_design="firwin", verbose="ERROR")

    # Reference
    if cfg.reference == "average":
        raw.set_eeg_reference("average", projection=False, verbose="ERROR")
    elif isinstance(cfg.reference, list):
        raw.set_eeg_reference(cfg.reference, projection=False, verbose="ERROR")
    # else: leave the reference unchanged

    # Resample
    if cfg.resample_hz is not None and abs(raw.info["sfreq"] - cfg.resample_hz) > 1e-3:
        raw.resample(cfg.resample_hz, verbose="ERROR")

    return raw


def fixed_length_epochs(raw: BaseRaw, cfg: PreprocessingConfig) -> Epochs:
    """Cut a preprocessed Raw into overlapping fixed-length epochs.

    Useful for self-supervised pretraining where we just want chunks of
    continuous EEG rather than event-aligned trials.
    """
    duration = cfg.epoch_length_s
    overlap = cfg.epoch_overlap_s
    events = make_fixed_length_events(raw, duration=duration, overlap=overlap)
    return Epochs(
        raw, events,
        tmin=0.0,
        tmax=duration - 1.0 / raw.info["sfreq"],
        baseline=None,
        preload=True,
        verbose="ERROR",
    )


def event_locked_epochs(
    raw: BaseRaw,
    tmin: float = -0.5,
    tmax: float = 4.0,
    event_id: dict[str, int] | None = None,
) -> Epochs:
    """Cut Raw into event-locked epochs using MNE annotations.

    Useful for downstream classification benchmarks (e.g., motor-imagery
    left-hand vs right-hand in EEGMMIDB).
    """
    events, found_event_id = events_from_annotations(raw, verbose="ERROR")
    use_event_id = event_id or found_event_id
    return Epochs(
        raw, events,
        event_id=use_event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
        verbose="ERROR",
    )


def to_numpy(epochs: Epochs, to_microvolts: bool = True) -> np.ndarray:
    """Return epochs as a (n_epochs, n_channels, n_times) float32 array.

    Parameters
    ----------
    epochs
        MNE Epochs object.
    to_microvolts
        If True (default), multiply by 1e6 so values are in µV instead of V.
        EEG amplitudes in volts are ~1e-5, which is below the default epsilon
        of normalization layers like LayerNorm — leaving values in V makes the
        model effectively learn from noise. Scaling to µV puts values in a
        more natural range (~±100) where downstream normalization layers
        behave correctly. Every EEG-FM paper (LaBraM, EEGPT, NeuroLM, ...) does
        this implicitly via per-channel z-scoring or explicit µV conversion.
    """
    data = epochs.get_data().astype(np.float32, copy=False)
    if to_microvolts:
        data = data * 1e6
    return data


def zscore_per_channel(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Per-channel z-score normalization for (n_epochs, n_channels, n_times) arrays.

    Computes mean and std across the time dimension for each (epoch, channel)
    independently. This is what EEGPT does by default and is a good model-input
    default for foundation models.
    """
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    return ((x - mean) / (std + eps)).astype(np.float32, copy=False)
