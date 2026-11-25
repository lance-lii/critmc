"""E04 - Critical slowing down and dynamic critical exponents.

At the exact critical point of the 2D Ising model the integrated
autocorrelation time of an observable grows like tau_int ~ L^z.  Local
algorithms (Metropolis, heat bath) have z ~ 2.17; the cluster algorithms of
Swendsen-Wang and Wolff reduce it to z ~ 0.25 or less.  Times are measured in
sweeps (Wolff: flipped spins / N), with Wolff's Gamma-method errors.

For the local algorithms the energy gives a visibly smaller exponent than m^2.
The second part of the analysis shows why: the energy's autocorrelation
function ends in the same slow decay as that of m^2, but that decay carries
only a share ~ 1/c(L) of the energy's fluctuations, c being the specific heat,
which grows like ln L.  So tau_int(E) ~ L^z / ln L, and tau_int(E) c(L) grows
with the same exponent as tau_int(m^2).
"""

from __future__ import annotations

import time

import numpy as np
from common import banner, figure_path, load_results, parse_args, run_dir, save_results

from critmc import plotting as P
from critmc.exact import kaufman, onsager
from critmc.exact.constants import reference
from critmc.fss import fit_power_law
from critmc.runner import Task, run_tasks
from critmc.stats import gamma_method

NAME = "e04_dynamics"
BC = onsager.BETA_C
ALGOS = {
    "metropolis": "Metropolis",
    "heatbath": "Heat bath",
    "sw": "Swendsen-Wang",
    "wolff": "Wolff",
}
LOCAL_SIZES = [8, 16, 32, 64, 128]
CLUSTER_SIZES = [8, 16, 32, 64, 128, 256]
# sweeps between measurements for the local algorithms (keeps N_meas ~ 1000 tau)
STRIDE = {8: 1, 16: 1, 32: 2, 64: 8, 128: 32}
LOCAL = ("metropolis", "heatbath")
# the slow decay is fitted at lags between these multiples of tau_int(m2) ...
TAIL = (0.75, 2.5)
# ... with errors from a jackknife over this many blocks of the series
TAIL_BLOCKS = 8


def slow_mode(series, tau_ref: float) -> dict:
    """Late-time decay ``rho(t) = A exp(-t / tau_exp)`` of an autocorrelation function.

    ``tau_exp`` (in measurements) is the relaxation time of the slowest mode the
    observable couples to and ``A`` the share of the observable's variance that
    this mode carries.  The fit is a straight line through ``ln rho`` at lags
    between ``TAIL[0]`` and ``TAIL[1]`` times ``tau_ref``; errors come from a
    jackknife over ``TAIL_BLOCKS`` blocks.  Returns NaNs if the series is too
    short for the tail to stand out of the noise.
    """
    x = np.asarray(series, dtype=float)
    x = x - x.mean()
    lags = np.unique(np.linspace(TAIL[0] * tau_ref, TAIL[1] * tau_ref, 60).astype(int))
    lags = lags[lags >= 1]
    n = len(x) - int(lags[-1])
    # block means of x_i x_{i+t}; row 0 is the variance
    products = np.array(
        [
            [block.mean() for block in np.array_split(x[:n] * x[t : t + n], TAIL_BLOCKS)]
            for t in (0, *lags)
        ]
    )

    def estimate(blocks):
        gamma = blocks.mean(axis=1)
        rho = gamma[1:] / gamma[0]
        ok = rho > 0
        if ok.sum() < 3:
            return np.array([np.nan, np.nan])
        slope, intercept = np.polyfit(lags[ok], np.log(rho[ok]), 1)
        if slope >= 0:
            return np.array([np.nan, np.nan])
        return np.array([-1.0 / slope, np.exp(intercept)])

    full = estimate(products)
    reps = np.array([estimate(np.delete(products, b, axis=1)) for b in range(TAIL_BLOCKS)])
    err = np.sqrt((TAIL_BLOCKS - 1) / TAIL_BLOCKS * np.sum((reps - reps.mean(axis=0)) ** 2, axis=0))
    return {"tau_exp": full[0], "dtau_exp": err[0], "share": full[1], "dshare": err[1]}


def power_law(L, y, dy, l_min: float) -> dict | None:
    """Fit ``y = a L^z`` to the sizes ``L >= l_min``; ``None`` if a point is unusable."""
    L, y, dy = (np.asarray(v, dtype=float) for v in (L, y, dy))
    sel = L >= l_min
    if sel.sum() < 3 or not (np.all(np.isfinite(y[sel])) and np.all(dy[sel] > 0)):
        return None
    f = fit_power_law(L[sel], y[sel], dy[sel])
    return {
        "z": f["exponent"].value,
        "dz": f["exponent"].error,
        "chi2": f.chi2,
        "dof": f.dof,
        "params": f.params,
        "L_min": float(L[sel][0]),
    }


def plan(args):
    tasks, index = [], []
    for algo in ALGOS:
        local = algo in ("metropolis", "heatbath")
        sizes = LOCAL_SIZES if local else CLUSTER_SIZES
        if args.quick:
            sizes = sizes[:3]
        for L in sizes:
            k = STRIDE[L] if local else 1
            updates = f"{k}*{algo}" if local else algo
            n_meas = 200_000 // (20 if args.quick else 1)
            tasks.append(
                Task(
                    {"model": "ising", "lattice": "square", "L": L},
                    BC,
                    updates,
                    n_therm=20_000 // max(1, k) if local else 5_000,
                    n_meas=n_meas,
                    seed=args.seed,
                    keys=(NAME, algo, L),
                    name=f"{algo}_L{L}",
                )
            )
            index.append((algo, L))
    return tasks, index


def main():
    args = parse_args(__doc__)
    if args.replot:
        r = load_results(NAME, args)
        plot(r["table"], r["fits"])
        return
    started = time.time()
    tasks, index = plan(args)
    banner(f"{NAME}: {len(tasks)} runs")
    runs = run_tasks(tasks, workers=args.workers, out_dir=run_dir(NAME, args))

    table: dict = {
        a: {"L": [], "tau_e": [], "dtau_e": [], "tau_m2": [], "dtau_m2": [], "sweeps_per_meas": []}
        for a in ALGOS
    }
    slow_keys = ("tau_exp_e", "dtau_exp_e", "share_e", "dshare_e")
    slow_keys += ("tau_exp_m2", "dtau_exp_m2", "share_m2", "dshare_m2")
    slow: dict = {a: {"L": [], "c": [], **{k: [] for k in slow_keys}} for a in LOCAL if a in ALGOS}
    for (algo, L), ts in zip(index, runs):
        spm = float(ts.meta["sweeps_per_meas"])
        ge = gamma_method(np.asarray(ts["e"]))
        gm = gamma_method(np.asarray(ts["m2"]))
        row = table[algo]
        row["L"].append(L)
        row["tau_e"].append(ge.tau_int * spm)
        row["dtau_e"].append(ge.dtau_int * spm)
        row["tau_m2"].append(gm.tau_int * spm)
        row["dtau_m2"].append(gm.dtau_int * spm)
        row["sweeps_per_meas"].append(spm)
        if algo in slow:
            srow = slow[algo]
            srow["L"].append(L)
            srow["c"].append(float(kaufman.specific_heat(BC, L)))
            for obs in ("e", "m2"):
                mode = slow_mode(ts[obs], gm.tau_int)
                srow[f"tau_exp_{obs}"].append(mode["tau_exp"] * spm)
                srow[f"dtau_exp_{obs}"].append(mode["dtau_exp"] * spm)
                srow[f"share_{obs}"].append(mode["share"])
                srow[f"dshare_{obs}"].append(mode["dshare"])
        print(
            f"  {ALGOS[algo]:>14} L={L:3d}: tau_E = {ge.tau_int * spm:9.2f} +- {ge.dtau_int * spm:7.2f}   "
            f"tau_m2 = {gm.tau_int * spm:9.2f} +- {gm.dtau_int * spm:7.2f}  sweeps"
        )

    fits = {}
    for algo, row in table.items():
        L = np.array(row["L"], dtype=float)
        fits[algo] = {}
        for obs in ("e", "m2"):
            tau = np.array(row[f"tau_{obs}"])
            dtau = np.array(row[f"dtau_{obs}"])
            sel = L >= (16 if len(L) > 3 else 8)
            f = fit_power_law(L[sel], tau[sel], dtau[sel])
            fits[algo][obs] = {
                "z": f["exponent"].value,
                "dz": f["exponent"].error,
                "chi2": f.chi2,
                "dof": f.dof,
                "params": f.params,
                "L_min": float(L[sel][0]),
            }
            print(
                f"  {ALGOS[algo]:>14} z_int,{obs:<2} = {f['exponent']}  (chi2/dof {f.chi2:.1f}/{f.dof})"
            )
    # why the energy's exponent is low for the local algorithms
    banner("slowest mode and its share of the energy fluctuations")
    for algo, srow in slow.items():
        L = np.array(srow["L"], dtype=float)
        c = np.array(srow["c"])
        l_min = 16 if len(L) > 3 else 8
        tau_e = np.array(table[algo]["tau_e"])
        dtau_e = np.array(table[algo]["dtau_e"])
        srow["share_e_times_c"] = list(np.array(srow["share_e"]) * c)
        srow["dshare_e_times_c"] = list(np.array(srow["dshare_e"]) * c)
        srow["tau_e_times_c"] = list(tau_e * c)
        srow["dtau_e_times_c"] = list(dtau_e * c)
        srow["fits"] = {
            "tau_e_times_c": power_law(L, tau_e * c, dtau_e * c, l_min),
            "tau_exp_m2": power_law(L, srow["tau_exp_m2"], srow["dtau_exp_m2"], l_min),
            "tau_exp_e": power_law(L, srow["tau_exp_e"], srow["dtau_exp_e"], l_min),
        }
        # the exponent a pure L^z / c(L) law would show over the fitted sizes
        z_ref = reference("ising2d", "z_metropolis").value
        sel = L >= l_min
        srow["z_expected_from_c"] = float(
            z_ref - np.log(c[sel][-1] / c[sel][0]) / np.log(L[sel][-1] / L[sel][0])
        )
        for i, size in enumerate(srow["L"]):
            print(
                f"  {ALGOS[algo]:>14} L={size:3d}: c = {c[i]:.3f}  tau_exp(m2) = "
                f"{srow['tau_exp_m2'][i]:8.1f} +- {srow['dtau_exp_m2'][i]:6.1f}  share of E = "
                f"{srow['share_e'][i]:.3f} +- {srow['dshare_e'][i]:.3f}  share x c = "
                f"{srow['share_e_times_c'][i]:.3f}"
            )
        for key, label in (("tau_e_times_c", "tau_int(E) c"), ("tau_exp_m2", "tau_exp(m2)")):
            f = srow["fits"][key]
            if f is not None:
                print(
                    f"  {ALGOS[algo]:>14} z from {label:<12} = {f['z']:.3f} +- {f['dz']:.3f}"
                    f"  (chi2/dof {f['chi2']:.1f}/{f['dof']})"
                )
        print(f"  {ALGOS[algo]:>14} z_int,e expected from L^z / c: {srow['z_expected_from_c']:.3f}")
    res = {
        "table": table,
        "fits": fits,
        "slow_mode": {"tail": list(TAIL), "blocks": TAIL_BLOCKS, "algorithms": slow},
        "literature": {
            k: list(reference("ising2d", f"z_{k}")) for k in ("metropolis", "wolff", "sw")
        },
    }
    path = save_results(NAME, res, args, started)
    print(f"  results -> {path}")
    if not args.no_plots:
        plot(table, fits)


def plot(table, fits):
    fig, axes = P.figure(2, 1, width=3.6, height=3.0)
    markers = ["o", "s", "D", "^"]
    for ax, obs, title in ((axes[0], "e", "a  Energy"), (axes[1], "m2", "b  Magnetisation m²")):
        for k, algo in enumerate(ALGOS):
            row = table[algo]
            L = np.array(row["L"], dtype=float)
            tau = np.array(row[f"tau_{obs}"])
            dtau = np.array(row[f"dtau_{obs}"])
            f = fits[algo][obs]
            P.errorbar(
                ax,
                L,
                tau,
                dtau,
                color=P.CATEGORICAL[k],
                line=False,
                marker=markers[k],
                label=f"{ALGOS[algo]}: z = {f['z']:.2f}({int(round(f['dz'] * 100))})",
            )
            xx = np.geomspace(f["L_min"], L.max(), 40)
            a, z = f["params"]
            ax.plot(xx, a * xx**z, color=P.CATEGORICAL[k], linewidth=1.2, alpha=0.8)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        P.plain_log_ticks(ax)
        ax.set_xlabel("linear size L")
        ax.set_ylabel("τ_int  (sweeps)")
        ax.set_title(f"{title}: τ_int ~ L^z at β_c")
        ax.legend(loc="upper left", fontsize=7.5)
    P.save(fig, figure_path(NAME, "tau"))


if __name__ == "__main__":
    main()
