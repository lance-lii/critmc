"""E09 - The 3D Edwards-Anderson spin glass with Gaussian couplings.

Parallel tempering with two replica sets per disorder sample (L = 4 ... 10).
Disorder averages of the dimensionless correlation length xi_L / L and the
Binder ratio g are computed with bootstrap errors over samples; their curves
for different L cross near the spin-glass transition (T_c = 0.951(9) in the
literature).  Equilibration is verified with the Gaussian-disorder identity
[e] = -(z/2)(1 - [q_l]) / T in logarithmic time bins.  Every sample is cached
on disk, so the campaign can be interrupted and resumed.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from common import banner, data_dir, figure_path, parse_args, save_results

from critmc import cubic
from critmc import plotting as P
from critmc.exact.constants import reference
from critmc.fss import fit
from critmc.glass import DisorderAverage, EdwardsAnderson, ParallelTempering, TemperingRun
from critmc.runner import parallel_map

NAME = "e09_spin_glass"
T_MIN, T_MAX = 0.6, 2.0
CONFIG = {  # L: (temperatures, samples, log2 of the number of PT steps)
    4: (16, 600, 13),
    6: (16, 400, 15),
    8: (20, 200, 16),
    10: (24, 100, 17),
}


def temperatures(L):
    m = CONFIG[L][0]
    return T_MIN * (T_MAX / T_MIN) ** (np.arange(m) / (m - 1))


def run_sample(job) -> TemperingRun:
    L, s, n_bins, seed, out_dir = job
    path = Path(out_dir) / f"L{L}_s{s:04d}.npz"
    temps = temperatures(L)
    if path.exists():
        z = np.load(path)
        return TemperingRun(
            z["temperatures"], L**3, 6, z["acc"], z["swap_rate"], z["label_hist"], z["trips"]
        )
    ss = np.random.SeedSequence(seed, spawn_key=(9, L, s))
    disorder, chain = ss.spawn(2)
    model = EdwardsAnderson(cubic(L), seed=np.random.default_rng(disorder))
    run = ParallelTempering(model, temps, seed=chain).run(n_bins)
    tmp = path.with_suffix(".part.npz")
    np.savez(
        tmp,
        temperatures=run.temperatures,
        acc=run.acc,
        swap_rate=run.swap_rate,
        label_hist=run.label_hist,
        trips=run.trips,
    )
    tmp.replace(path)
    return run


CROSS_WINDOW = (0.75, 1.40)


def crossing(temps, a, b, window=CROSS_WINDOW):
    """Crossing temperature of curve a (smaller L) and curve b (larger L).

    The difference D(T) = b - a is fitted with a quadratic inside ``window`` and
    the root where D decreases (larger L above below T_c, below above T_c) is
    returned.  Fitting the whole window instead of taking the first sign change
    keeps nearly tangent, noisy curves from producing spurious crossings.
    """
    sel = (temps >= window[0]) & (temps <= window[1])
    if sel.sum() < 5:
        return np.nan
    coef = np.polyfit(temps[sel], (b - a)[sel], 2)
    roots = np.roots(coef)
    roots = roots[np.isreal(roots)].real
    roots = roots[(roots >= window[0]) & (roots <= window[1])]
    slope = np.polyval(np.polyder(coef), roots)
    roots = roots[slope < 0]
    if not roots.size:
        return np.nan
    return float(roots[np.argmin(np.abs(roots - np.mean(window)))])


FIT_WINDOW = (0.80, 1.30)


def curves_xi(averages, sizes, idx=None):
    """xi_L / L of every size (optionally from resampled disorder indices), ascending T."""
    out = {}
    for L in sizes:
        avg = averages[L]
        sel = np.arange(len(avg.runs)) if idx is None else idx[L]
        s = {k: v[sel] for k, v in avg.samples.items()}
        val = DisorderAverage._observables(s, avg.n_sites, avg.L)["xi_L"][::-1]
        out[L] = (avg.temperatures[::-1], val)
    return out


def scaling_fit(curves, errors, start=(0.95, 0.4)):
    """Joint fit xi_L/L = a0 + a1 x + a2 x^2 with x = (T - T_c) L^(1/nu) inside FIT_WINDOW."""
    t_all, l_all, y_all, dy_all = [], [], [], []
    for L, (t, y) in curves.items():
        sel = (t >= FIT_WINDOW[0]) & (t <= FIT_WINDOW[1])
        t_all.append(t[sel])
        l_all.append(np.full(sel.sum(), float(L)))
        y_all.append(y[sel])
        dy_all.append(errors[L][sel])
    t = np.concatenate(t_all)
    lv = np.concatenate(l_all)
    y = np.concatenate(y_all)
    dy = np.concatenate(dy_all)

    def model(xs, tc, inv_nu, a0, a1, a2):
        x = (xs[0] - tc) * xs[1] ** inv_nu
        return a0 + a1 * x + a2 * x * x

    f = fit(
        model,
        np.vstack([t, lv]),
        y,
        dy,
        [start[0], start[1], 0.6, -0.3, 0.0],
        ("T_c", "inv_nu", "a0", "a1", "a2"),
    )
    return f


def main():
    args = parse_args(__doc__)
    started = time.time()
    sizes = [4, 6] if args.quick else sorted(CONFIG)
    out_dir = data_dir(NAME + ("_q" if args.quick else ""))
    averages = {}
    for L in sizes:
        m, n_samples, n_bins = CONFIG[L]
        if args.quick:
            n_samples, n_bins = 40, n_bins - 3
        banner(f"L = {L}: {n_samples} samples, {m} temperatures, {2**n_bins - 1} PT steps")
        jobs = [(L, s, n_bins, args.seed, str(out_dir)) for s in range(n_samples)]
        runs = parallel_map(run_sample, jobs, workers=args.workers)
        averages[L] = DisorderAverage(runs, L)
        rates = np.array([r.swap_rate for r in runs])
        print(f"  swap acceptance: min {rates.min():.2f}, median {np.median(rates):.2f}")

    obs = {L: averages[L].observables(n_boot=400, seed=args.seed) for L in sizes}
    # crossings between successive sizes, with bootstrap errors over disorder samples
    rng = np.random.default_rng(args.seed)
    cross = {}
    for L1, L2 in zip(sizes[:-1], sizes[1:]):
        t1, t2 = averages[L1].temperatures[::-1], averages[L2].temperatures[::-1]
        grid = np.linspace(max(t1.min(), t2.min()), min(t1.max(), t2.max()), 400)
        for key in ("xi_L", "g"):

            def curve(avg, idx, key=key, grid=grid):
                s = {k: v[idx] for k, v in avg.samples.items()}
                val = DisorderAverage._observables(s, avg.n_sites, avg.L)[key][::-1]
                return np.interp(grid, avg.temperatures[::-1], val)

            n1, n2 = len(averages[L1].runs), len(averages[L2].runs)
            full = crossing(
                grid, curve(averages[L1], np.arange(n1)), curve(averages[L2], np.arange(n2))
            )
            boots = []
            for _ in range(300):
                boots.append(
                    crossing(
                        grid,
                        curve(averages[L1], rng.integers(0, n1, n1)),
                        curve(averages[L2], rng.integers(0, n2, n2)),
                    )
                )
            boots = np.array(boots)
            ok = np.isfinite(boots)
            # robust spread: half the 68 % interval of the bootstrap distribution
            spread = np.percentile(boots[ok], [15.87, 84.13]) if ok.sum() > 10 else [np.nan] * 2
            cross[(L1, L2, key)] = (full, float((spread[1] - spread[0]) / 2), float(ok.mean()))
            print(
                f"  crossing ({L1},{L2}) of {key:>4}: T = {full:.3f} +- {cross[(L1, L2, key)][1]:.3f}"
            )

    # joint finite-size-scaling fit of xi_L / L, errors from resampling disorder samples
    errs = {L: obs[L]["xi_L"][1][::-1] for L in sizes}
    f_full = scaling_fit(curves_xi(averages, sizes), errs)
    boots = []
    for _ in range(200):
        idx = {L: rng.integers(0, len(averages[L].runs), len(averages[L].runs)) for L in sizes}
        try:
            fb = scaling_fit(
                curves_xi(averages, sizes, idx), errs, start=(f_full.params[0], f_full.params[1])
            )
            boots.append(fb.params[:2])
        except RuntimeError:
            continue
    boots = np.array(boots)
    tc_err, inv_nu_err = np.std(boots, axis=0, ddof=1)
    nu = 1.0 / f_full.params[1]
    print(
        f"  scaling fit of xi_L/L, T in {FIT_WINDOW}: T_c = {f_full.params[0]:.3f}({tc_err:.3f}), "
        f"nu = {nu:.2f}({inv_nu_err * nu**2:.2f}); chi2/dof = {f_full.chi2:.1f}/{f_full.dof} "
        f"({len(boots)} bootstrap fits)"
    )
    eq = averages[sizes[-1]].equilibration()
    res = {
        "scaling_fit": {
            "T_c": [float(f_full.params[0]), float(tc_err)],
            "nu": [float(nu), float(inv_nu_err * nu**2)],
            "chi2": f_full.chi2,
            "dof": f_full.dof,
            "window": FIT_WINDOW,
            "n_boot": len(boots),
            "params": f_full.params,
        },
        "sizes": sizes,
        "config": {
            L: {"temperatures": averages[L].temperatures, "samples": len(averages[L].runs)}
            for L in sizes
        },
        "observables": {
            L: {k: {"value": v[0], "error": v[1]} for k, v in obs[L].items()} for L in sizes
        },
        "crossings": [
            {"L1": a, "L2": b, "observable": k, "T": v[0], "error": v[1], "fraction_found": v[2]}
            for (a, b, k), v in cross.items()
        ],
        "equilibration_largest_L": {
            "e": eq["e"],
            "e_from_ql": eq["e_from_ql"],
            "error": eq["residual_error"],
        },
        "reference": {
            "T_c": reference("ea3d_gauss", "T_c").value,
            "source": reference("ea3d_gauss", "T_c").source,
        },
    }
    path = save_results(NAME, res, args, started)
    print(f"  results -> {path}")
    if not args.no_plots:
        plot(sizes, averages, obs, eq, res["scaling_fit"])


def plot(sizes, averages, obs, eq, sfit):
    fig, axes = P.figure(4, 1, width=3.05, height=2.8)
    cols = dict(zip(sizes, P.ordinal_colors(len(sizes))))
    t_ref = reference("ea3d_gauss", "T_c").value
    for ax, key, ylabel, title in (
        (axes[0], "xi_L", "ξ_L / L", "a  Correlation length"),
        (axes[2], "g", "Binder ratio g", "c  Binder ratio"),
    ):
        for L in sizes:
            v, e = obs[L][key]
            P.errorbar(
                ax, averages[L].temperatures, v, e, color=cols[L], label=f"L = {L}", markersize=3.6
            )
        ax.axvline(t_ref, color=P.INK_2, linewidth=0.9)
        ax.annotate(
            f"T_c = {t_ref} (lit.)",
            (t_ref, ax.get_ylim()[1]),
            xytext=(3, -10),
            textcoords="offset points",
            fontsize=7.5,
            color=P.INK_2,
        )
        ax.set_xlim(0.55, 1.6)
        ax.set_xlabel("temperature T")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc="upper right", fontsize=7.5)

    # b: collapse of xi_L / L with the fitted scaling function
    ax = axes[1]
    tc, inv_nu = sfit["params"][0], sfit["params"][1]
    a0, a1, a2 = sfit["params"][2:]
    lo, hi = sfit["window"]
    for L in sizes:
        t = averages[L].temperatures
        v, e = obs[L]["xi_L"]
        sel = (t >= lo) & (t <= hi)
        x = (t[sel] - tc) * L**inv_nu
        P.errorbar(
            ax, x, v[sel], e[sel], color=cols[L], line=False, label=f"L = {L}", markersize=4.0
        )
    xs = np.linspace(*ax.get_xlim(), 100)
    ax.plot(xs, a0 + a1 * xs + a2 * xs * xs, color=P.INK_2, linewidth=1.0, label="fit")
    ax.set_xlabel(r"$(T - T_c)\,L^{1/\nu}$")
    ax.set_ylabel("ξ_L / L")
    ax.set_title(
        f"b  Collapse: T_c = {sfit['T_c'][0]:.3f}({int(round(sfit['T_c'][1] * 1e3))}), "
        f"ν = {sfit['nu'][0]:.2f}"
    )
    ax.legend(loc="upper right", fontsize=7.5)

    ax = axes[3]
    L = sizes[-1]
    temps = averages[L].temperatures
    k = int(np.argmin(temps))
    steps = 2.0 ** np.arange(eq["e"].shape[0])
    P.errorbar(ax, steps, eq["e"][:, k], None, color=P.CATEGORICAL[0], label="[e] measured")
    P.errorbar(
        ax,
        steps,
        eq["e_from_ql"][:, k],
        None,
        color=P.CATEGORICAL[1],
        marker="s",
        label="−3(1 − [q_l]) / T",
    )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("PT steps (log bins)")
    ax.set_ylabel("energy per spin")
    ax.set_title(f"d  Equilibration, L = {L}, T = {temps[k]:.2f}")
    ax.legend(loc="upper right", fontsize=7.5)
    P.save(fig, figure_path(NAME, "overview"))


if __name__ == "__main__":
    main()
