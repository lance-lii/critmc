"""Parallel, resumable execution of simulation batches.

A :class:`Task` is a small, picklable description of one Markov chain.  Its
random stream is derived from ``(seed, *keys)``, so every task is reproducible
on its own, independently of scheduling or of the number of workers.  When an
output directory is given each result is written to ``<dir>/<name>.npz`` as
soon as it finishes, and tasks whose file already exists are skipped, which
makes long campaigns restartable after interruption.
"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .models import build_model
from .rng import make_rng
from .sim import Simulation, TimeSeries

__all__ = ["Task", "run_task", "run_tasks", "parallel_map", "default_workers"]


@dataclass(frozen=True)
class Task:
    """One canonical Markov chain.

    ``model`` is a model spec such as ``{"model": "ising", "lattice": "square", "L": 32}``.
    """

    model: dict[str, Any]
    beta: float
    updates: str
    n_therm: int
    n_meas: int
    seed: int
    keys: tuple = ()
    start: str = "random"
    name: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def filename(self) -> str:
        if self.name:
            return f"{self.name}.npz"
        tag = "_".join(str(k) for k in self.keys) or "run"
        return f"{self.model['model']}_L{self.model['L']}_b{self.beta:.8f}_{tag}.npz"


def run_task(task: Task) -> TimeSeries:
    """Execute a task in the current process."""
    model = build_model(task.model)
    rng = make_rng(task.seed, *task.keys)
    sim = Simulation(model, task.beta, task.updates, seed=rng, start=task.start)
    sim.thermalize(task.n_therm)
    ts = sim.run(task.n_meas)
    ts.meta.update({"task": asdict(task), "tuning": sim.tuning()})
    return ts


def _run_and_save(task: Task, out_dir: str | None) -> tuple[Task, TimeSeries | str]:
    ts = run_task(task)
    if out_dir is None:
        return task, ts
    path = Path(out_dir) / task.filename
    tmp = path.with_suffix(".part.npz")
    ts.save(tmp)
    os.replace(tmp, path)
    return task, str(path)


_BYTES_PER_WORKER = 1 << 30  # numpy + scipy + numba/LLVM commit about 0.5-1 GB per process


def _available_memory() -> int | None:
    """Memory that new worker processes can commit, or ``None`` if unknown."""
    try:
        if sys.platform == "win32":
            import ctypes

            class _Status(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = _Status()
            status.dwLength = ctypes.sizeof(_Status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                # on Windows the commit limit, not physical memory, kills workers first
                return int(status.ullAvailPageFile)
            return None
        return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError):
        return None


def default_workers() -> int:
    """One worker per core minus one, capped by the memory available for new processes."""
    workers = max(1, (os.cpu_count() or 2) - 1)
    avail = _available_memory()
    if avail is not None:
        workers = min(workers, max(1, avail // _BYTES_PER_WORKER - 1))
    return int(workers)


def run_tasks(
    tasks: Sequence[Task],
    *,
    workers: int | None = None,
    out_dir: str | Path | None = None,
    progress: bool = True,
) -> list[TimeSeries]:
    """Run tasks in parallel and return their time series in task order.

    With ``out_dir`` results are cached on disk and existing files are reused.
    """
    workers = default_workers() if workers is None else max(1, int(workers))
    out = str(out_dir) if out_dir is not None else None
    if out is not None:
        Path(out).mkdir(parents=True, exist_ok=True)
        for stale in Path(out).glob("*.part.npz"):  # left behind by an interrupted worker
            stale.unlink()
    results: dict[int, TimeSeries] = {}
    pending = []
    for i, t in enumerate(tasks):
        path = Path(out) / t.filename if out is not None else None
        if path is not None and path.exists():
            results[i] = TimeSeries.load(path)
        else:
            pending.append(i)

    t0 = time.perf_counter()
    if pending:
        if workers == 1 or len(pending) == 1:
            for n_done, i in enumerate(pending, 1):
                _, res = _run_and_save(tasks[i], out)
                results[i] = TimeSeries.load(res) if isinstance(res, str) else res
                if progress:
                    _report(n_done, len(pending), t0)
        else:
            with ProcessPoolExecutor(max_workers=min(workers, len(pending))) as pool:
                futures = {pool.submit(_run_and_save, tasks[i], out): i for i in pending}
                for n_done, fut in enumerate(as_completed(futures), 1):
                    i = futures[fut]
                    _, res = fut.result()
                    results[i] = TimeSeries.load(res) if isinstance(res, str) else res
                    if progress:
                        _report(n_done, len(pending), t0)
    return [results[i] for i in range(len(tasks))]


def parallel_map(
    fn: Callable, items: Iterable, *, workers: int | None = None, progress: bool = True
) -> list:
    """Order-preserving parallel map for arbitrary picklable top-level functions."""
    items = list(items)
    workers = default_workers() if workers is None else max(1, int(workers))
    t0 = time.perf_counter()
    if workers == 1 or len(items) <= 1:
        out = []
        for k, it in enumerate(items, 1):
            out.append(fn(it))
            if progress:
                _report(k, len(items), t0)
        return out
    results: dict[int, Any] = {}
    with ProcessPoolExecutor(max_workers=min(workers, len(items))) as pool:
        futures = {pool.submit(fn, it): i for i, it in enumerate(items)}
        for k, fut in enumerate(as_completed(futures), 1):
            results[futures[fut]] = fut.result()
            if progress:
                _report(k, len(items), t0)
    return [results[i] for i in range(len(items))]


def _report(done: int, total: int, t0: float) -> None:
    elapsed = time.perf_counter() - t0
    eta = elapsed / done * (total - done)
    sys.stderr.write(f"\r  {done}/{total} done, {elapsed:7.1f}s elapsed, ~{eta:7.1f}s left")
    if done == total:
        sys.stderr.write("\n")
    sys.stderr.flush()
