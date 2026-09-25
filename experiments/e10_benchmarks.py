"""E10 - Throughput of the compiled kernels (single core).

Site updates per second for every model/algorithm pair of the default suite,
plus the scaling of throughput with lattice size for the 2D Ising kernels.
Numbers depend on the machine; the relative costs are what matter.
"""

from __future__ import annotations

import platform
import time

import numpy as np
from common import banner, figure_path, load_results, parse_args, save_results

from critmc import Ising, square
from critmc import plotting as P
from critmc.bench import benchmark, default_suite
from critmc.exact.onsager import BETA_C

NAME = "e10_benchmarks"


def main():
    args = parse_args(__doc__)
    if args.replot:
        r = load_results(NAME, args)
        plot(r["sizes"], r["scaling"])
        return
    started = time.time()
    banner("default suite (L = 64)")
    suite = default_suite(64, 60 if args.quick else 300)
    for r in suite:
        print(
            f"  {r.label:>14} {r.updates:>18} N={r.n_sites:6d}: {r.site_updates_per_s:9.3e} site updates/s"
        )

    banner("size dependence, 2D Ising at beta_c")
    sizes = [8, 16, 32, 64, 128, 256] if not args.quick else [8, 32, 128]
    scaling = {}
    for upd in ("metropolis", "heatbath", "wolff", "sw"):
        rates = []
        for L in sizes:
            n = max(50, int(4e6 / (L * L))) // (5 if args.quick else 1)
            rates.append(benchmark("Ising 2D", Ising(square(L)), BETA_C, upd, n).site_updates_per_s)
        scaling[upd] = rates
        print(f"  {upd:>10}: " + "  ".join(f"{r:8.2e}" for r in rates))

    res = {
        "machine": {"processor": platform.processor(), "python": platform.python_version()},
        "suite": [r.__dict__ for r in suite],
        "sizes": sizes,
        "scaling": scaling,
    }
    path = save_results(NAME, res, args, started)
    print(f"  results -> {path}")
    if not args.no_plots:
        plot(sizes, scaling)


def plot(sizes, scaling):
    fig, ax = P.figure(1, 1, width=4.0, height=2.8)
    labels = {
        "metropolis": "Metropolis",
        "heatbath": "Heat bath",
        "wolff": "Wolff",
        "sw": "Swendsen-Wang",
    }
    markers = ["o", "s", "^", "D"]
    for k, (upd, rates) in enumerate(scaling.items()):
        P.errorbar(
            ax,
            sizes,
            np.array(rates) / 1e6,
            None,
            color=P.CATEGORICAL[k],
            marker=markers[k],
            label=labels[upd],
        )
    ax.set_xscale("log", base=2)
    P.plain_log_ticks(ax, "x")
    ax.set_xlabel("linear size L")
    ax.set_ylabel("million site updates / s")
    ax.set_title("Single-core throughput, 2D Ising at β_c")
    ax.set_ylim(bottom=0)
    ax.legend(loc="center right", fontsize=7.5, ncols=2)
    P.save(fig, figure_path(NAME, "throughput"))


if __name__ == "__main__":
    main()
