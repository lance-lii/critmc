"""E04 - Critical slowing down and dynamic critical exponents.

At the exact critical point of the 2D Ising model the integrated
autocorrelation time of an observable grows like tau_int ~ L^z.  Local
algorithms (Metropolis, heat bath) have z ~ 2.17; the cluster algorithms of
Swendsen-Wang and Wolff reduce it to z ~ 0.25 or less.  Times are measured in
sweeps (Wolff: flipped spins / N), with Wolff's Gamma-method errors.
"""

from __future__ import annotations

import time

import numpy as np
from common import banner, data_dir, figure_path, load_results, parse_args, save_results

from critmc import plotting as P
from critmc.exact import onsager
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
    runs = run_tasks(
        tasks, workers=args.workers, out_dir=data_dir(NAME + ("_q" if args.quick else ""))
    )

    table: dict = {
        a: {"L": [], "tau_e": [], "dtau_e": [], "tau_m2": [], "dtau_m2": [], "sweeps_per_meas": []}
        for a in ALGOS
    }
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
    res = {
        "table": table,
        "fits": fits,
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
