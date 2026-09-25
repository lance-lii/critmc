"""Run every experiment in sequence (reproduces results/ and docs/figures/).

    python experiments/run_all.py            # full campaign (several hours on 8 cores)
    python experiments/run_all.py --quick    # smoke run (minutes)
    python experiments/run_all.py e03 e07    # a subset, by prefix

Extra options (``--workers``, ``--seed``) are forwarded to every script.
Raw data are cached, so an interrupted campaign resumes where it stopped.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = sorted(p for p in HERE.glob("e[0-9][0-9]_*.py"))


def main(argv: list[str]) -> int:
    selected = [a for a in argv if not a.startswith("-") and not a.lstrip("-").isdigit()]
    forwarded = [a for a in argv if a not in selected]
    scripts = [s for s in SCRIPTS if not selected or any(s.name.startswith(p) for p in selected)]
    failures = []
    for script in scripts:
        print(f"\n######## {script.name} {' '.join(forwarded)}", flush=True)
        t0 = time.time()
        code = subprocess.call([sys.executable, str(script), *forwarded], cwd=HERE)
        print(f"######## {script.name}: exit {code} after {time.time() - t0:.0f} s", flush=True)
        if code:
            failures.append(script.name)
    tables = [sys.executable, str(HERE / "make_tables.py")] + (
        ["--quick"] if "--quick" in forwarded else []
    )
    subprocess.call(tables, cwd=HERE)
    if failures:
        print("failed:", ", ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
