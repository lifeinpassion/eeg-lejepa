"""Download PhysioNet EEGMMIDB data for an initial set of subjects.

Run from the project root:

    python scripts/01_download_data.py

The download is incremental — running again skips files already present.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from rich.console import Console

from eeg_slm.data.loaders import EEGMMIDBLoader

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--subjects", type=int, nargs="*", default=None,
                        help="Override config subject list")
    parser.add_argument("--runs", type=int, nargs="*", default=None,
                        help="Override config run list")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    subjects = args.subjects if args.subjects is not None else cfg["dataset"]["subjects"]
    runs = args.runs if args.runs is not None else cfg["dataset"]["runs"]
    data_root = Path(cfg["paths"]["data_root"])

    console.print(f"[bold]Downloading EEGMMIDB[/bold] to [cyan]{data_root}[/cyan]")
    console.print(f"  Subjects: {subjects}")
    console.print(f"  Runs:     {runs}")

    loader = EEGMMIDBLoader(data_root=data_root)
    loader.download_subjects(subjects=subjects, runs=runs)

    # Quick sanity check: confirm one subject is loadable
    sanity_subject = subjects[0]
    raw = loader.load_raw(subject=sanity_subject, runs=runs[:1])
    console.print(
        f"\n[green]OK[/green] — loaded subject {sanity_subject:03d}, run {runs[0]}: "
        f"{raw.info['nchan']} channels, "
        f"{raw.n_times / raw.info['sfreq']:.1f}s at {raw.info['sfreq']} Hz"
    )


if __name__ == "__main__":
    main()
