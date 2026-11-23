import numpy as np
import pytest

from critmc.runner import StaleCacheError, Task, run_task, run_tasks

MODEL = {"model": "ising", "lattice": "square", "L": 8}


def make_task(beta=0.3, n_meas=200, seed=1, **kw):
    return Task(MODEL, beta, "metropolis", n_therm=50, n_meas=n_meas, seed=seed, name="run", **kw)


def test_task_is_reproducible_from_its_seed_and_keys():
    a = run_task(make_task(keys=("x", 3)))
    b = run_task(make_task(keys=("x", 3)))
    c = run_task(make_task(keys=("x", 4)))
    np.testing.assert_array_equal(a["E"], b["E"])
    assert not np.array_equal(a["E"], c["E"])


def test_cached_result_is_reused(tmp_path):
    first = run_tasks([make_task()], workers=1, out_dir=tmp_path, progress=False)[0]
    path = tmp_path / "run.npz"
    stamp = path.stat().st_mtime_ns
    again = run_tasks([make_task()], workers=1, out_dir=tmp_path, progress=False)[0]
    assert path.stat().st_mtime_ns == stamp  # not rewritten
    np.testing.assert_array_equal(first["E"], again["E"])


@pytest.mark.parametrize(
    "changed, field",
    [
        (dict(beta=0.6), "beta"),
        (dict(n_meas=400), "n_meas"),
        (dict(seed=2), "seed"),
        (dict(keys=("other",)), "keys"),
    ],
)
def test_cache_from_a_different_task_is_refused(tmp_path, changed, field):
    run_tasks([make_task()], workers=1, out_dir=tmp_path, progress=False)
    with pytest.raises(StaleCacheError, match=field):
        run_tasks([make_task(**changed)], workers=1, out_dir=tmp_path, progress=False)


def test_cache_without_a_task_record_is_refused(tmp_path):
    ts = run_task(make_task())
    del ts.meta["task"]
    ts.save(tmp_path / "run.npz")
    with pytest.raises(StaleCacheError, match="does not record"):
        run_tasks([make_task()], workers=1, out_dir=tmp_path, progress=False)


def test_results_come_back_in_task_order(tmp_path):
    tasks = [
        Task(MODEL, b, "metropolis", n_therm=20, n_meas=50, seed=1, keys=(k,), name=f"r{k}")
        for k, b in enumerate((0.2, 0.4, 0.6))
    ]
    runs = run_tasks(tasks, workers=2, out_dir=tmp_path, progress=False)
    assert [r.beta for r in runs] == [0.2, 0.4, 0.6]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["r0.npz", "r1.npz", "r2.npz"]
