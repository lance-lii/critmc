"""Shared helpers for the experiment scripts.

Every experiment writes

* raw time series to ``$CRITMC_DATA/<exp>/`` (default ``data/<exp>/``, git-ignored;
  existing files are reused, so interrupted campaigns resume where they stopped),
* numerical results to ``results/<exp>.json``,
* figures to ``docs/figures/<exp>_*.png``.

Run with ``--quick`` for a reduced-statistics smoke run (minutes instead of hours).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np

import critmc
from critmc.io import to_jsonable

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "docs" / "figures"


def data_dir(name: str) -> Path:
    base = Path(os.environ.get("CRITMC_DATA", ROOT / "data"))
    path = base / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_args(description: str) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--quick", action="store_true", help="reduced statistics for a fast smoke run")
    p.add_argument(
        "--workers", type=int, default=None, help="worker processes (default: all cores - 1)"
    )
    p.add_argument("--seed", type=int, default=20260924, help="root seed of all random streams")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--fresh", action="store_true", help="recompute cached exact results (E02)")
    p.add_argument("--replot", action="store_true", help="redraw figures from results/*.json only")
    args = p.parse_args()
    global _QUICK
    _QUICK = bool(args.quick)
    return args


def load_results(name: str, args: argparse.Namespace) -> dict:
    """Stored results of an experiment (for ``--replot``)."""
    path = RESULTS / f"{name}{'_quick' if args.quick else ''}.json"
    return json.loads(path.read_text())


_QUICK = False  # set by parse_args; quick-mode figures never overwrite the real ones


def figure_path(name: str, suffix: str) -> Path:
    return FIGURES / f"{name}_{suffix}{'_quick' if _QUICK else ''}.png"


def save_results(name: str, payload: dict, args: argparse.Namespace, started: float) -> Path:
    meta = {
        "experiment": name,
        "critmc_version": critmc.__version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "quick": bool(args.quick),
        "seed": args.seed,
        "wall_time_s": round(time.time() - started, 1),
    }
    path = RESULTS / f"{name}{'_quick' if args.quick else ''}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable({"meta": meta, **payload}), indent=2) + "\n")
    return path


def banner(text: str) -> None:
    print(f"\n== {text}", flush=True)
