"""Download a range of EEGMMIDB subjects' motor-imagery runs (4, 8, 12) + rest.

Convenience wrapper around scripts/01_download_data.py for bulk downloads.
Idempotent — skips files already present.

Example:

    # Download subjects 4-20, MI runs only
    python scripts/01b_download_range.py --start 4 --end 20

    # Include the baseline rest runs as well (for rest-vs-activity probe)
    python scripts/01b_download_range.py --start 4 --end 20 --include-baseline

Sizes (approximate):
    1 subject  × 3 MI runs  ≈ 18 MB
    1 subject  × 4 MI+rest runs ≈ 30 MB
    17 subjects × 3 MI runs ≈ 300 MB
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from rich.console import Console

from eeg_slm.data.loaders import EEGMMIDBLoader

console = Console()

MI_RUNS = [4, 8, 12]              # left/right fist imagery
BASELINE_RUNS = [1, 2]            # eyes-open + eyes-closed rest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    p.add_argument("--start", type=int, required=True, help="First subject ID (inclusive)")
    p.add_argument("--end", type=int, required=True, help="Last subject ID (inclusive)")
    p.add_argument("--runs", type=int, nargs="*", default=None,
                   help="Override the default runs (4 8 12) with a custom list")
    p.add_argument("--include-baseline", action="store_true",
                   help="Also download runs 1, 2 (resting-state baseline)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    data_root = Path(cfg["paths"]["data_root"])

    runs = args.runs if args.runs is not None else MI_RUNS.copy()
    if args.include_baseline:
        runs = sorted(set(runs + BASELINE_RUNS))

    subjects = list(range(args.start, args.end + 1))
    console.print(
        f"[bold]Downloading[/bold] {len(subjects)} subjects (s{args.start:03d}-s{args.end:03d}), "
        f"runs {runs} → [cyan]{data_root}[/cyan]"
    )
    console.print(
        f"  estimated size: ~{len(subjects) * len(runs) * 6:.0f} MB "
        f"(~6 MB per file). Skips files already on disk."
    )

    loader = EEGMMIDBLoader(data_root=data_root)
    loader.download_subjects(subjects=subjects, runs=runs)

    console.print(f"[green]Done.[/green] {len(subjects) * len(runs)} files in {data_root}")


if __name__ == "__main__":
    main()
