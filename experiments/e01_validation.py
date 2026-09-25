"""E01 - Validation of every update algorithm against the exact finite-lattice solution.

For the 16 x 16 Ising torus the energy and specific heat are known exactly at
every temperature (Kaufman's formula).  Each algorithm is run at 20
temperatures; errors come from the Gamma-method.  If the samplers are
unbiased and the error bars honest, the pulls ``z = (MC - exact) / sigma`` are
standard normal - which is tested with a Kolmogorov-Smirnov test and a
chi-square test per algorithm.
"""

from __future__ import annotations

import time

import numpy as np
from common import banner, data_dir, figure_path, parse_args, save_results
from scipy import stats

from critmc import plotting as P
from critmc.exact import kaufman, onsager
from critmc.runner import Task, run_tasks
from critmc.stats import gamma_method

NAME = "e01_validation"
L = 16
ALGORITHMS = ["metropolis", "heatbath", "sw", "wolff"]
LABELS = {
    "metropolis": "Metropolis",
    "heatbath": "Heat bath",
    "sw": "Swendsen-Wang",
    "wolff": "Wolff",
}


def tasks(args):
    temps = np.sort(np.append(np.linspace(1.6, 3.4, 19), onsager.T_C))
    local = 10_000 if args.quick else 100_000
    cluster = 5_000 if args.quick else 40_000
    out = []
    for algo in ALGORITHMS:
        n_meas = cluster if algo in ("sw", "wolff") else local
        for i, T in enumerate(temps):
            out.append(
                Task(
                    {"model": "ising", "lattice": "square", "L": L},
                    1.0 / T,
                    algo,
                    n_therm=n_meas // 10,
                    n_meas=n_meas,
                    seed=args.seed,
                    keys=(NAME, algo, i),
                    name=f"{algo}_{i:02d}",
                )
            )
    return temps, out


def analyse(ts):
    beta = ts.beta
    n = ts.n_sites
    e = np.asarray(ts["E"], dtype=float)
    ge = gamma_method(e / n)
    gc = gamma_method(np.column_stack([e, e * e]), lambda a: beta**2 * (a[1] - a[0] ** 2) / n)
    return {"e": ge.value, "de": ge.error, "c": gc.value, "dc": gc.error, "tau_e": ge.tau_int}


def main():
    args = parse_args(__doc__)
    started = time.time()
    temps, ts_tasks = tasks(args)
    banner(f"{NAME}: {len(ts_tasks)} runs")
    runs = run_tasks(
        ts_tasks, workers=args.workers, out_dir=data_dir(NAME + ("_q" if args.quick else ""))
    )

    exact = kaufman.thermodynamics(1.0 / temps, L)
    table = {}
    all_pulls = []
    for k, algo in enumerate(ALGORITHMS):
        rows = [analyse(ts) for ts in runs[k * len(temps) : (k + 1) * len(temps)]]
        e = np.array([r["e"] for r in rows])
        de = np.array([r["de"] for r in rows])
        c = np.array([r["c"] for r in rows])
        dc = np.array([r["dc"] for r in rows])
        z = np.concatenate([(e - exact["u"]) / de, (c - exact["c"]) / dc])
        all_pulls.append(z)
        table[algo] = {
            "e": e,
            "de": de,
            "c": c,
            "dc": dc,
            "tau_e": [r["tau_e"] for r in rows],
            "pull_mean": float(z.mean()),
            "pull_std": float(z.std(ddof=1)),
            "chi2": float(z @ z),
            "dof": int(z.size),
            "chi2_p": float(stats.chi2.sf(z @ z, z.size)),
            "ks_p": float(stats.kstest(z, "norm").pvalue),
        }
        print(
            f"  {LABELS[algo]:>14}: pulls mean {z.mean():+.3f} sd {z.std(ddof=1):.3f}  "
            f"chi2/dof {z @ z:.1f}/{z.size}  KS p = {table[algo]['ks_p']:.3f}"
        )
    pooled = np.concatenate(all_pulls)
    summary = {
        "L": L,
        "temperatures": temps,
        "exact": {"u": exact["u"], "c": exact["c"]},
        "algorithms": table,
        "pooled": {
            "mean": float(pooled.mean()),
            "std": float(pooled.std(ddof=1)),
            "n": int(pooled.size),
            "ks_p": float(stats.kstest(pooled, "norm").pvalue),
        },
    }
    path = save_results(NAME, summary, args, started)
    print(
        f"  pooled pulls: mean {pooled.mean():+.3f}, sd {pooled.std(ddof=1):.3f}, "
        f"KS p = {summary['pooled']['ks_p']:.3f}\n  results -> {path}"
    )
    if not args.no_plots:
        plot(temps, exact, table, pooled)


def plot(temps, exact, table, pooled):
    fig, axes = P.figure(3, 1, width=3.3, height=2.8)
    ax = axes[0]
    tt = np.linspace(temps.min(), temps.max(), 400)
    ex = kaufman.thermodynamics(1.0 / tt, L)
    ax.plot(tt, ex["c"], color=P.INK_2, linewidth=1.0, label="exact (Kaufman)", zorder=2)
    offsets = np.linspace(-0.018, 0.018, len(ALGORITHMS))
    markers = ["o", "s", "D", "^"]
    for k, algo in enumerate(ALGORITHMS):
        r = table[algo]
        P.errorbar(
            ax,
            temps + offsets[k],
            r["c"],
            r["dc"],
            color=P.CATEGORICAL[k],
            label=LABELS[algo],
            line=False,
            marker=markers[k],
            markersize=4.2,
        )
    ax.set_xlabel("temperature T")
    ax.set_ylabel("specific heat per site c")
    ax.set_title(f"a  Specific heat, {L}×{L} torus")
    ax.legend(loc="upper right")

    ax = axes[1]
    ax.axhspan(-2, 2, color=P.CATEGORICAL[0], alpha=0.07, linewidth=0, zorder=0)
    ax.axhline(0, color=P.AXIS, linewidth=0.8, zorder=1)
    for k, algo in enumerate(ALGORITHMS):
        r = table[algo]
        ze = (r["e"] - exact["u"]) / r["de"]
        zc = (r["c"] - exact["c"]) / r["dc"]
        ax.plot(
            temps + offsets[k],
            ze,
            marker=markers[k],
            linestyle="none",
            markersize=4.2,
            color=P.CATEGORICAL[k],
            markeredgecolor=P.SURFACE,
            markeredgewidth=0.9,
            label=LABELS[algo],
        )
        ax.plot(
            temps + offsets[k],
            zc,
            marker=markers[k],
            linestyle="none",
            markersize=4.2,
            color=P.CATEGORICAL[k],
            markeredgecolor=P.SURFACE,
            markeredgewidth=0.9,
            alpha=0.55,
        )
    ax.set_ylim(-4.2, 4.2)
    ax.set_xlabel("temperature T")
    ax.set_ylabel("pull  (MC − exact) / σ")
    ax.set_title("b  Pulls of e (solid) and c (faded)")
    ax.legend(loc="lower right", ncols=2, fontsize=7.5)

    ax = axes[2]
    bins = np.linspace(-4, 4, 17)
    ax.hist(
        pooled,
        bins=bins,
        density=True,
        color=P.CATEGORICAL[0],
        alpha=0.85,
        edgecolor=P.SURFACE,
        linewidth=1.5,
        label=f"{pooled.size} pulls",
    )
    x = np.linspace(-4, 4, 300)
    ax.plot(x, stats.norm.pdf(x), color=P.INK_2, linewidth=1.2, label="N(0, 1)")
    ax.set_xlabel("pull")
    ax.set_ylabel("density")
    ax.set_title(f"c  Pooled pulls: mean {pooled.mean():+.2f}, sd {pooled.std(ddof=1):.2f}")
    ax.legend(loc="upper right")
    P.save(fig, figure_path(NAME, "pulls"))


if __name__ == "__main__":
    main()
