"""Dataset loaders.

Initial dataset: PhysioNet EEG Motor Movement/Imagery Database (EEGMMIDB).

Reference
---------
Schalk, G. et al. (2004). BCI2000: A General-Purpose Brain-Computer Interface
(BCI) System. IEEE Transactions on Biomedical Engineering.
Goldberger, A. L. et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet:
Components of a New Research Resource for Complex Physiologic Signals.

109 subjects performed 14 runs each (rest + motor execution/imagery tasks),
sampled at 160 Hz across 64 channels (10-10 system).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mne
from mne.datasets import eegbci
from mne.io import BaseRaw


# Standard EEGMMIDB run codes
#   1, 2         : baseline (eyes open, eyes closed)
#   3, 7, 11     : Task 1 — open/close left or right fist (motor execution)
#   4, 8, 12     : Task 2 — imagine opening/closing left or right fist
#   5, 9, 13     : Task 3 — open/close both fists or both feet (motor execution)
#   6, 10, 14    : Task 4 — imagine opening/closing both fists or both feet
RUNS_BASELINE_EYES_OPEN = (1,)
RUNS_BASELINE_EYES_CLOSED = (2,)
RUNS_MOTOR_IMAGERY_HANDS = (4, 8, 12)
RUNS_MOTOR_EXECUTION_HANDS = (3, 7, 11)
RUNS_MOTOR_IMAGERY_FEET = (6, 10, 14)
RUNS_MOTOR_EXECUTION_FEET = (5, 9, 13)


@dataclass
class EEGMMIDBLoader:
    """Lightweight wrapper around mne.datasets.eegbci.

    Parameters
    ----------
    data_root
        Directory in which to download/cache the dataset. MNE will create
        subdirectories as needed.
    """

    data_root: Path = field(default_factory=lambda: Path("data/raw"))

    def __post_init__(self) -> None:
        self.data_root = Path(self.data_root).expanduser().resolve()
        self.data_root.mkdir(parents=True, exist_ok=True)

    def download_subjects(self, subjects: list[int], runs: list[int]) -> None:
        """Download specified (subject, run) combinations to data_root.

        Idempotent: skips files already present.
        """
        # MNE >=1.6 takes `subjects=` (plural, list); older versions used `subject=`.
        eegbci.load_data(subjects=subjects, runs=runs, path=str(self.data_root))

    def load_raw(
        self,
        subject: int,
        runs: list[int],
        preload: bool = True,
    ) -> BaseRaw:
        """Load and concatenate the specified runs for one subject."""
        paths = eegbci.load_data(
            subjects=[subject], runs=runs, path=str(self.data_root)
        )
        raws = [mne.io.read_raw_edf(p, preload=preload, verbose="ERROR") for p in paths]
        raw = mne.concatenate_raws(raws)

        # Normalize channel names: MNE's EEGMMIDB has trailing dots ("Fc5." etc.)
        renamed = {ch: ch.replace(".", "").upper() for ch in raw.ch_names}
        raw.rename_channels(renamed)

        # Tag channels with the 10-05 montage where possible
        try:
            montage = mne.channels.make_standard_montage("standard_1005")
            raw.set_montage(montage, match_case=False, on_missing="warn", verbose="ERROR")
        except Exception:
            pass

        return raw


def load_eegmmidb_subject(
    subject: int,
    runs: list[int] | None = None,
    data_root: str | Path = "data/raw",
    preload: bool = True,
) -> BaseRaw:
    """Convenience function: load one subject's runs in one call."""
    if runs is None:
        runs = list(RUNS_MOTOR_IMAGERY_HANDS)
    loader = EEGMMIDBLoader(data_root=Path(data_root))
    loader.download_subjects([subject], runs)
    return loader.load_raw(subject, runs, preload=preload)
