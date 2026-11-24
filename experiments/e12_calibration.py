"""E12 - Many-seed calibration of the samplers and of the Gamma-method error bars.

E01 compares one run per algorithm and temperature with the exact 16 x 16
solution.  That gives 40 pulls per algorithm, enough to see a bias of a third
of an error bar and no more.  Here the same runs are repeated with 200
independent random streams, which separates the two things a pull mixes:

* the sampler: the mean of 200 estimates has an error measured from their
  scatter alone and resolves a bias 14 times smaller than a single run;
* the error bar: the scatter of the 200 estimates is the true error of one
  run, to be compared with the error the Gamma-method quoted for it.

The comparison is made for the energy and for the specific heat (a non-linear
function of two means), and repeated for several values of the windowing
parameter ``S`` of the Gamma-method.  The raw time series are not kept: every
chain is reduced to its estimates and errors as soon as it finishes.

One long chain per algorithm and temperature then explains what the windows
show.  It is long enough to sum the autocorrelation function far out into its
tail, which gives the true integrated time; cut into pieces of campaign length
it also gives the integrated time the automatic window finds on average.  The
quoted errors are short by the square root of their ratio, a prediction that is
compared with the measured ratio of scatter to quoted error.
"""

from __future__ import annotations

import time

import numpy as np
from common import banner, figure_path, load_results, parse_args, run_dir, save_results
from e01_validation import ALGORITHMS, LABELS, L, run_length, temperatures
from e01_validation import NAME as E01
from scipy import stats

from critmc import plotting as P
from critmc.exact import kaufman
from critmc.io import load_npz, save_npz, to_jsonable
from critmc.runner import StaleCacheError, Task, parallel_map, run_task
from critmc.stats import autocorrelation, calibration, gamma_method, pull_statistics

NAME = "e12_calibration"
N_SEEDS = 200
WINDOWS = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0)  # Gamma-method windowing parameter S
DEFAULT = WINDOWS.index(1.5)  # the default of gamma_method, used by every other study
OBSERVABLES = {"e": "energy", "c": "specific heat"}
COLUMNS = ("e", "de", "tau_e", "c", "dc", "tau_c")
MODEL = {"model": "ising", "lattice": "square", "L": L}
TAIL_RUNS = 40  # length of the long chains, in campaign runs
TAIL_WINDOW = 20.0  # the long chains are summed out to this many integrated times
TAIL_COLUMNS = ("tau_int", "tau_int_error", "tail_time")


def reduce(e: np.ndarray, beta: float) -> np.ndarray:
    """Estimates, errors and integrated times of one energy series, for every window."""
    n = L * L
    col = np.column_stack([e, e * e])
    out = np.empty((len(WINDOWS), len(COLUMNS)))
    for k, S in enumerate(WINDOWS):
        ge = gamma_method(e / n, S=S)
        gc = gamma_method(col, lambda a: beta**2 * (a[1] - a[0] ** 2) / n, S=S)
        out[k] = ge.value, ge.error, ge.tau_int, gc.value, gc.error, gc.tau_int
    return out


def run_seed(job) -> np.ndarray:
    """All temperatures of one algorithm for one random stream, reduced to estimates."""
    algo, replica, seed, quick = job
    temps = temperatures()
    n_meas = run_length(algo, quick)
    out = np.empty((len(temps), len(WINDOWS), len(COLUMNS)))
    for i, T in enumerate(temps):
        task = Task(
            MODEL,
            1.0 / T,
            algo,
            n_therm=n_meas // 10,
            n_meas=n_meas,
            seed=seed,
            keys=(NAME, algo, i, replica),
        )
        out[i] = reduce(np.asarray(run_task(task)["E"], dtype=float), 1.0 / T)
    return out


def cached(path, config: dict, compute, n: int) -> np.ndarray:
    """First ``n`` rows of a cached array, computing and storing the rows that are missing."""
    done = None
    if path.exists():
        arrays, meta = load_npz(path)
        if meta.get("config") != config:
            raise StaleCacheError(f"{path} was produced with other settings; delete it to rerun")
        done = arrays["runs"][:n]
    have = 0 if done is None else len(done)
    if have < n:
        new = np.array(compute(range(have, n)))
        done = new if done is None else np.concatenate([done, new])
        tmp = path.with_suffix(".part.npz")
        save_npz(tmp, {"runs": done}, {"config": config})
        tmp.replace(path)
    return done


def settings(args, **extra) -> dict:
    return to_jsonable(
        {
            "L": L,
            "temperatures": temperatures(),
            "seed": args.seed,
            "windows": WINDOWS,
            "columns": COLUMNS,
            **extra,
        }
    )


def collect(algo: str, n_seeds: int, args) -> np.ndarray:
    """Reduced runs of one algorithm, shape (seeds, temperatures, windows, columns).

    Finished seeds are cached, so an interrupted or enlarged campaign only runs
    what is missing.
    """
    config = settings(args, algorithm=algo, n_meas=run_length(algo, args.quick))
    return cached(
        run_dir(NAME, args) / f"{algo}.npz",
        config,
        lambda missing: parallel_map(
            run_seed, [(algo, r, args.seed, args.quick) for r in missing], workers=args.workers
        ),
        n_seeds,
    )


def pooled(ratio: np.ndarray, error: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Root mean square over temperatures (axis 0) of a ratio of errors, with its error."""
    r2 = ratio**2
    dr2 = 2.0 * ratio * error
    out = np.sqrt(r2.mean(axis=0))
    return out, np.sqrt(np.sum(dr2**2, axis=0)) / r2.shape[0] / (2.0 * out)


def analyse(runs: np.ndarray, exact: dict) -> tuple[dict, np.ndarray]:
    """Calibration of both observables; also the pulls, shape (seeds, 2, temperatures, windows)."""
    column = {name: runs[..., j] for j, name in enumerate(COLUMNS)}
    out, pulls = {}, []
    for obs, ref in (("e", exact["u"]), ("c", exact["c"])):
        value, error = column[obs], column["d" + obs]
        cal = calibration(value, error, ref[:, None])
        z = (value - ref[:, None]) / error
        pulls.append(z)
        pool = pull_statistics(z.reshape(-1, z.shape[-1]))
        # the mean pull to first order: an offset of the estimate, plus the tendency
        # of an estimate and its own error bar to fluctuate together
        typical = error.mean(axis=0)
        covariance = ((value - value.mean(axis=0)) * (error - typical)).mean(axis=0)
        from_bias = ((value.mean(axis=0) - ref[:, None]) / typical).mean(axis=0)
        from_correlation = (-covariance / typical**2).mean(axis=0)
        ratio, dratio = pooled(cal.error_ratio, cal.error_ratio_error)
        chi2 = float(np.sum(cal.bias_normal[:, DEFAULT] ** 2))
        out[obs] = {
            # the estimator (independent of the window: S only changes the error)
            "mean": cal.mean[:, DEFAULT],
            "mean_error": cal.mean_error[:, DEFAULT],
            "bias_z": cal.bias_z[:, DEFAULT],
            "bias_chi2": chi2,
            "bias_dof": int(cal.bias_z.shape[0]),
            "bias_p": float(stats.chi2.sf(chi2, cal.bias_z.shape[0])),
            # the error bar, by temperature and window
            "tau_int": column["tau_" + obs].mean(axis=0),  # by temperature and window
            "error_ratio": cal.error_ratio,
            "error_ratio_error": cal.error_ratio_error,
            "correlation": cal.correlation,
            "pull_mean": cal.pulls.mean,
            "pull_mean_error": cal.pulls.mean_error,
            "pull_sd": cal.pulls.sd,
            "pull_sd_error": cal.pulls.sd_error,
            # ... and combined over temperatures, by window
            "pooled": {
                "n": pool.n,
                "error_ratio": ratio,
                "error_ratio_error": dratio,
                "pull_mean": pool.mean,
                "pull_mean_error": pool.mean_error,
                "pull_mean_from_bias": from_bias,
                "pull_mean_from_correlation": from_correlation,
                "pull_sd": pool.sd,
                "pull_sd_error": pool.sd_error,
                "within_1": pool.within_1,
                "within_2": pool.within_2,
                "beyond_3": pool.beyond_3,
            },
        }
    return out, np.stack(pulls, axis=1)


def tail_job(job) -> np.ndarray:
    """One long chain: true integrated times, and the ones campaign-length pieces find.

    Returns, for the energy and the specific heat, the ``TAIL_COLUMNS`` followed by
    the mean over the pieces of the Gamma-method integrated time for every window,
    and the errors of those means.
    """
    algo, i, seed, quick = job
    n_run = run_length(algo, quick)
    n_meas = TAIL_RUNS * n_run
    beta = 1.0 / temperatures()[i]
    task = Task(
        MODEL,
        beta,
        algo,
        n_therm=n_meas // 100,
        n_meas=n_meas,
        seed=seed,
        keys=(NAME, "tail", algo, i),
    )
    e = np.asarray(run_task(task)["E"], dtype=float)
    out = np.empty((len(OBSERVABLES), len(TAIL_COLUMNS) + 2 * len(WINDOWS)))
    # to first order the specific heat is the mean of (E - <E>)^2
    for j, series in enumerate((e, (e - e.mean()) ** 2)):
        rho = autocorrelation(series, max_lag=min(n_meas // 50, 5000))
        lags = np.arange(1, rho.size)
        running = 0.5 + np.cumsum(rho[1:])
        far = lags >= TAIL_WINDOW * running
        window = int(np.argmax(far)) if far.any() else lags.size - 1
        tau = running[window]
        # decay time of the tail: slope of ln(rho) while rho falls from 0.1 to 0.01
        start, stop = int(np.argmax(rho < 0.1)), int(np.argmax(rho < 0.01))
        decay = np.nan
        if stop - start >= 3:
            decay = -1.0 / np.polyfit(np.arange(start, stop), np.log(rho[start:stop]), 1)[0]
        out[j, :3] = tau, tau * np.sqrt(2.0 * (2 * lags[window] + 1) / n_meas), decay
    pieces = np.array([reduce(x, beta) for x in e.reshape(TAIL_RUNS, n_run)])
    found = pieces[:, :, [COLUMNS.index("tau_e"), COLUMNS.index("tau_c")]]  # (pieces, windows, 2)
    out[:, 3 : 3 + len(WINDOWS)] = found.mean(axis=0).T
    out[:, 3 + len(WINDOWS) :] = found.std(axis=0, ddof=1).T / np.sqrt(TAIL_RUNS)
    return out


def tail_study(table: dict, args) -> dict:
    """Does the tail of the autocorrelation function account for the error-bar size?

    The square of a quoted error is proportional to the integrated time the
    Gamma-method found.  If the long chain gives the true time ``tau`` and
    campaign-length runs find ``tau_hat`` on average, their errors are short by
    ``sqrt(tau / tau_hat)``.
    """
    n_temps, n_w = len(temperatures()), len(WINDOWS)
    out = {"runs": TAIL_RUNS, "window": TAIL_WINDOW, "algorithms": {}}
    for algo in ALGORITHMS:
        config = settings(
            args, algorithm=algo, n_meas=TAIL_RUNS * run_length(algo, args.quick), chain="tail"
        )
        chains = cached(
            run_dir(NAME, args) / f"{algo}_tail.npz",
            config,
            lambda missing, algo=algo: parallel_map(
                tail_job, [(algo, i, args.seed, args.quick) for i in missing], workers=args.workers
            ),
            n_temps,
        )
        out["algorithms"][algo] = {}
        for j, obs in enumerate(OBSERVABLES):
            tau, dtau, decay = chains[:, j, :3].T
            found, dfound = chains[:, j, 3 : 3 + n_w], chains[:, j, 3 + n_w :]
            ratio = np.sqrt(tau[:, None] / found)
            dratio = 0.5 * ratio * np.hypot(dtau[:, None] / tau[:, None], dfound / found)
            predicted, dpredicted = pooled(ratio, dratio)
            measured = table[algo][obs]["pooled"]
            # temperature by temperature, at the default window
            by_t, dby_t = table[algo][obs]["error_ratio"], table[algo][obs]["error_ratio_error"]
            z = (by_t - ratio) / np.hypot(dby_t, dratio)
            chi2 = float(np.sum(z[:, DEFAULT] ** 2))
            out["algorithms"][algo][obs] = {
                "tau_int": tau,
                "tau_int_error": dtau,
                "tail_time": decay,
                "tau_int_found": found,
                "predicted_ratio": ratio,
                "predicted_ratio_error": dratio,
                "chi2": chi2,
                "dof": n_temps,
                "p": float(stats.chi2.sf(chi2, n_temps)),
                "pooled": {
                    "predicted_ratio": predicted,
                    "predicted_ratio_error": dpredicted,
                    "measured_ratio": measured["error_ratio"],
                    "measured_ratio_error": measured["error_ratio_error"],
                },
            }
    return out


def single_seed_rows(pulls: np.ndarray, e01: dict | None) -> dict:
    """What one E01 row (the 40 pulls of one seed) looks like across seeds."""
    rows = pulls[..., DEFAULT].reshape(pulls.shape[0], -1)
    mean = rows.mean(axis=1)
    sd = rows.std(axis=1, ddof=1)
    ks = np.array([stats.kstest(r, "norm").pvalue for r in rows])
    out = {
        "n_pulls": int(rows.shape[1]),
        "mean_sd": float(mean.std(ddof=1)),
        "mean_sd_expected": float(1.0 / np.sqrt(rows.shape[1])),
        "sd_median": float(np.median(sd)),
        "share_ks_below_0.05": float(np.mean(ks < 0.05)),
    }
    if e01 is not None:
        out["e01"] = {
            "pull_mean": e01["pull_mean"],
            "share_mean_as_far": float(np.mean(np.abs(mean) >= abs(e01["pull_mean"]))),
            "ks_p": e01["ks_p"],
            "share_ks_as_small": float(np.mean(ks <= e01["ks_p"])),
        }
    return out


def published_e01(args) -> dict | None:
    try:
        return load_results(E01, args)["algorithms"]
    except FileNotFoundError:
        return None


def main():
    args = parse_args(__doc__)
    if args.replot:
        return plot(load_results(NAME, args))
    started = time.time()
    n_seeds = 8 if args.quick else N_SEEDS
    temps = temperatures()
    exact = kaufman.thermodynamics(1.0 / temps, L)
    e01 = published_e01(args)
    banner(f"{NAME}: {len(ALGORITHMS)} algorithms x {n_seeds} seeds x {len(temps)} temperatures")

    table = {}
    for algo in ALGORITHMS:
        runs = collect(algo, n_seeds, args)
        table[algo], pulls = analyse(runs, exact)
        table[algo]["single_seed"] = single_seed_rows(pulls, None if e01 is None else e01[algo])
        print(f"  {LABELS[algo]}")
        for obs, label in OBSERVABLES.items():
            r, p = table[algo][obs], table[algo][obs]["pooled"]
            print(
                f"    {label:>13}: bias chi2/dof {r['bias_chi2']:5.1f}/{r['bias_dof']}"
                f" (p = {r['bias_p']:.2f})   scatter/error "
                f"{p['error_ratio'][DEFAULT]:.3f} +- {p['error_ratio_error'][DEFAULT]:.3f}"
                f"   pulls mean {p['pull_mean'][DEFAULT]:+.3f} +- "
                f"{p['pull_mean_error'][DEFAULT]:.3f}, sd {p['pull_sd'][DEFAULT]:.3f} +- "
                f"{p['pull_sd_error'][DEFAULT]:.3f}"
            )
    tail = tail_study(table, args)
    print("  scatter/error predicted from the tail of the autocorrelation function")
    for algo in ALGORITHMS:
        for obs, label in OBSERVABLES.items():
            c = tail["algorithms"][algo][obs]
            p = c["pooled"]
            slow = int(np.argmax(c["tau_int"]))
            print(
                f"    {LABELS[algo]:>13}, {label:>13}: predicted "
                f"{p['predicted_ratio'][DEFAULT]:.3f} +- {p['predicted_ratio_error'][DEFAULT]:.3f}"
                f", measured {p['measured_ratio'][DEFAULT]:.3f} +- "
                f"{p['measured_ratio_error'][DEFAULT]:.3f}, by temperature chi2/dof "
                f"{c['chi2']:.1f}/{c['dof']}   (slowest: T = {temps[slow]:.3g}, "
                f"tau_int {c['tau_int'][slow]:.2f}, tail decay {c['tail_time'][slow]:.2f})"
            )
    summary = {
        "L": L,
        "n_seeds": n_seeds,
        "temperatures": temps,
        "windows": WINDOWS,
        "default_window": WINDOWS[DEFAULT],
        "n_meas": {a: run_length(a, args.quick) for a in ALGORITHMS},
        "exact": {"u": exact["u"], "c": exact["c"]},
        "algorithms": table,
        "tail": tail,
    }
    path = save_results(NAME, summary, args, started)
    print(f"  results -> {path}")
    if not args.no_plots:
        plot(to_jsonable(summary))


def plot(r):
    """Figure from the (JSON-compatible) summary alone, so ``--replot`` needs no simulation."""
    temps = np.array(r["temperatures"])
    windows = np.array(r["windows"])
    k0 = r["windows"].index(r["default_window"])
    n_seeds = r["n_seeds"]
    fig, axes = P.figure(3, 1, width=3.3, height=2.8)
    offsets = np.linspace(-0.018, 0.018, len(ALGORITHMS))
    markers = ["o", "s", "D", "^"]

    ax = axes[0]
    ax.axhspan(-2, 2, color=P.CATEGORICAL[0], alpha=0.07, linewidth=0, zorder=0)
    ax.axhline(0, color=P.AXIS, linewidth=0.8, zorder=1)
    for k, algo in enumerate(ALGORITHMS):
        for obs, alpha in (("e", 1.0), ("c", 0.55)):
            ax.plot(
                temps + offsets[k],
                r["algorithms"][algo][obs]["bias_z"],
                marker=markers[k],
                linestyle="none",
                markersize=4.2,
                color=P.CATEGORICAL[k],
                markeredgecolor=P.SURFACE,
                markeredgewidth=0.9,
                alpha=alpha,
                label=LABELS[algo] if obs == "e" else None,
            )
    ax.set_ylim(-4.2, 6.4)
    ax.set_yticks([-4, -2, 0, 2, 4])
    ax.set_xlabel("temperature T")
    ax.set_ylabel(f"(mean of {n_seeds} runs − exact) / its error")
    ax.set_title("a  Bias of e (solid) and c (faded)")
    ax.legend(loc="upper center", ncols=2, fontsize=7.5)

    ax = axes[1]
    for k, algo in enumerate(ALGORITHMS):
        c = r["algorithms"][algo]["c"]
        P.errorbar(
            ax,
            temps + offsets[k],
            np.array(c["pull_sd"])[:, k0],
            np.array(c["pull_sd_error"])[:, k0],
            color=P.CATEGORICAL[k],
            label=LABELS[algo],
            line=False,
            marker=markers[k],
            markersize=4.2,
        )
    P.reference_line(ax, y=1.0)
    ax.set_ylim(0.78, 1.42)
    ax.set_xlabel("temperature T")
    ax.set_ylabel("sd of specific-heat pulls (1 = honest)")
    ax.set_title(f"b  Width of the pulls of c, S = {r['default_window']:g}")
    ax.legend(loc="upper center", ncols=2, fontsize=7.5)

    ax = axes[2]
    shift = np.linspace(-0.06, 0.06, len(ALGORITHMS))
    for k, algo in enumerate(ALGORITHMS):
        p = r["tail"]["algorithms"][algo]["c"]["pooled"]
        ax.plot(
            windows, p["predicted_ratio"], color=P.CATEGORICAL[k], linewidth=1.2, linestyle="--"
        )
        P.errorbar(
            ax,
            windows + shift[k],
            p["measured_ratio"],
            p["measured_ratio_error"],
            color=P.CATEGORICAL[k],
            label=LABELS[algo],
            line=False,
            marker=markers[k],
            markersize=4.2,
        )
    P.reference_line(ax, y=1.0)
    ax.axvline(r["default_window"], color=P.AXIS, linewidth=0.8, zorder=1)
    ax.set_xlabel("windowing parameter S")
    ax.set_ylabel("scatter / quoted error of c")
    ax.set_title("c  Measured (points), from the tail (dashed)")
    ax.legend(loc="upper right", fontsize=7.5)
    P.save(fig, figure_path(NAME, "pulls"))


if __name__ == "__main__":
    main()
