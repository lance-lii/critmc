"""E07 - The Berezinskii-Kosterlitz-Thouless transition of the 2D XY model.

There is no order parameter and no power-law divergence to fit; instead the
helicity modulus (spin stiffness) jumps from 2 T_BKT / pi to zero at the
transition (Nelson & Kosterlitz 1977).  Two estimators of T_BKT are used:

* the crossings T*(L) of Upsilon(T, L) with the line 2T/pi, extrapolated with
  the logarithmic correction T*(L) = T_BKT + a / (ln L + c)^2 (c = -ln L0 sets
  the RG scale; the plain a / (ln L)^2 form is reported for comparison);
* the Weber-Minnhagen criterion: at T_BKT the finite-size modulus follows
  Upsilon(L) = (2T/pi) (1 + 1 / (2 ln L + C)); the temperature at which this
  one-parameter form fits the data best estimates T_BKT.  The scan runs on a
  0.001 grid with Upsilon(T) interpolated between the simulated temperatures,
  and its error comes from a parametric bootstrap of the Upsilon values.

``--replot`` recomputes this analysis from the stored per-temperature data.

Updates: Metropolis + 2 over-relaxation sweeps + Wolff clusters; Gamma-method errors.
"""

from __future__ import annotations

import time

import numpy as np
from common import banner, data_dir, figure_path, load_results, parse_args, save_results
from scipy import optimize
from scipy.interpolate import PchipInterpolator

from critmc import plotting as P
from critmc.exact.constants import reference
from critmc.fss import fit
from critmc.runner import Task, run_tasks
from critmc.stats import gamma_method

NAME = "e07_xy_bkt"
TEMPS = np.array(
    [
        0.70,
        0.75,
        0.80,
        0.83,
        0.85,
        0.86,
        0.87,
        0.88,
        0.89,
        0.90,
        0.91,
        0.92,
        0.93,
        0.95,
        0.98,
        1.00,
        1.05,
        1.10,
        1.20,
    ]
)
T_REF = reference("xy2d", "T_bkt").value


def plan(args):
    sizes = [8, 16, 32] if args.quick else [8, 16, 32, 64, 128]
    meas = {8: 40_000, 16: 40_000, 32: 40_000, 64: 30_000, 128: 20_000}
    tasks, index = [], []
    for L in sizes:
        for i, T in enumerate(TEMPS):
            n = meas[L] // (10 if args.quick else 1)
            tasks.append(
                Task(
                    {"model": "on", "lattice": "square", "L": L, "n": 2},
                    1.0 / T,
                    "metropolis+2*overrelax+wolff",
                    n_therm=max(500, n // 10),
                    n_meas=n,
                    seed=args.seed,
                    keys=(NAME, L, i),
                    name=f"L{L}_T{i:02d}",
                )
            )
            index.append((L, i))
    return sizes, tasks, index


def helicity(ts):
    beta = ts.beta
    n = ts.n_sites
    c = ts["stiff_c"].mean(axis=1)
    s2 = (ts["stiff_s"] ** 2).mean(axis=1)
    g = gamma_method(np.column_stack([c, s2]), lambda a: (a[0] - beta * a[1]) / n)
    chi = gamma_method(np.asarray(ts["m2"]), lambda a: beta * n * a[0])
    return g.value, g.error, chi.value, chi.error


def crossing(T, ups, dups):
    """Temperature where Upsilon(T) = 2T/pi by linear interpolation, with a propagated error."""
    diff = ups - 2 * T / np.pi
    idx = np.flatnonzero(np.diff(np.sign(diff)) < 0)
    if not idx.size:
        return np.nan, np.nan
    i = idx[0]
    t0, t1 = T[i], T[i + 1]
    d0, d1 = diff[i], diff[i + 1]
    tc = t0 + (t1 - t0) * d0 / (d0 - d1)
    slope = (d1 - d0) / (t1 - t0)
    err = np.hypot(dups[i], dups[i + 1]) / 2 / abs(slope)
    return tc, err


def wm_scan(sizes, ups, dups, grid):
    """Weber-Minnhagen chi^2/dof of Upsilon(L) = (2T/pi)(1 + 1/(2 ln L + C)) on a fine grid.

    Upsilon(T) of each size is interpolated monotonically (PCHIP) between the
    simulated temperatures; errors are interpolated linearly.
    """
    wm_l = np.array([L for L in sizes if L >= 16], dtype=float)
    interp = {L: PchipInterpolator(TEMPS, ups[L]) for L in sizes if L >= 16}
    out = np.full(len(grid), np.inf)
    for k, T in enumerate(grid):
        y = np.array([float(interp[int(L)](T)) for L in wm_l])
        dy = np.array([np.interp(T, TEMPS, dups[int(L)]) for L in wm_l])

        def chi2(c, y=y, dy=dy, T=T):
            model = 2 * T / np.pi * (1 + 1 / (2 * np.log(wm_l) + c))
            return float(np.sum(((y - model) / dy) ** 2))

        res = optimize.minimize_scalar(chi2, bounds=(-5.0, 60.0), method="bounded")
        out[k] = res.fun / max(len(wm_l) - 1, 1)
    return out


def refine_minimum(grid, values):
    """Parabolic refinement of the minimum of a curve sampled on a uniform grid."""
    j = int(np.clip(np.argmin(values), 1, len(grid) - 2))
    y0, y1, y2 = values[j - 1], values[j], values[j + 1]
    denom = y0 - 2 * y1 + y2
    h = grid[1] - grid[0]
    return float(grid[j] + (0.5 * h * (y0 - y2) / denom if denom > 0 else 0.0))


def analyse(sizes, ups, dups, chi, seed):
    """Crossings, their extrapolations and the Weber-Minnhagen estimate (with bootstrap errors)."""
    cross = {L: crossing(TEMPS, ups[L], dups[L]) for L in sizes}
    lc = np.array([L for L in sizes if np.isfinite(cross[L][0]) and L >= 16], dtype=float)
    tcs = np.array([cross[int(L)][0] for L in lc])
    dtcs = np.array([cross[int(L)][1] for L in lc])
    f2 = fit(lambda x, t, a: t + a / np.log(x) ** 2, lc, tcs, dtcs, [0.89, 1.0], ("T_bkt", "a"))
    # the RG scale enters through ln(L / L0) = ln L + c
    f3 = None
    if len(lc) >= 4:
        f3 = fit(
            lambda x, t, a, c: t + a / (np.log(x) + c) ** 2,
            lc,
            tcs,
            dtcs,
            [0.893, 1.0, 0.0],
            ("T_bkt", "a", "c"),
        )

    grid = np.round(np.arange(0.860, 0.9301, 0.001), 4)
    chi2 = wm_scan(sizes, ups, dups, grid)
    t_wm = refine_minimum(grid, chi2)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(200):
        pert = {L: ups[L] + rng.normal(0.0, dups[L]) for L in sizes}
        boots.append(refine_minimum(grid, wm_scan(sizes, pert, dups, grid)))
    dt_wm = float(np.std(boots, ddof=1))

    i_ref = int(np.argmin(np.abs(TEMPS - T_REF)))
    lchi = np.array([L for L in sizes if L >= 16], dtype=float)
    ychi = np.array([chi[int(L)][i_ref][0] for L in lchi])
    dychi = np.array([chi[int(L)][i_ref][1] for L in lchi])
    f_chi = fit(lambda x, a, e: a * x ** (2 - e), lchi, ychi, dychi, [1.0, 0.25], ("a", "eta"))
    return {
        "crossings": {L: list(cross[L]) for L in sizes},
        "T_bkt_crossing_fit": [f2["T_bkt"].value, f2["T_bkt"].error, f2.chi2, f2.dof],
        "T_bkt_crossing_fit_lnL0": None
        if f3 is None
        else [f3["T_bkt"].value, f3["T_bkt"].error, f3.chi2, f3.dof],
        "crossing_params": {
            "two": list(f2.params),
            "three": None if f3 is None else list(f3.params),
        },
        "weber_minnhagen": {"T": grid, "chi2_dof": chi2, "T_bkt": t_wm, "error": dt_wm},
        "eta_eff": [f_chi["eta"].value, f_chi["eta"].error, float(TEMPS[i_ref])],
    }


def report(sizes, a):
    for L in sizes:
        tc, dtc = a["crossings"][L]
        print(f"  L={L:3d}: T*(L) = {tc:.4f} +- {dtc:.4f}")
    t2 = a["T_bkt_crossing_fit"]
    print(
        f"  T* = T_BKT + a/(ln L)^2:        T_BKT = {t2[0]:.5f}({t2[1]:.5f})  chi2 {t2[2]:.1f}/{t2[3]}"
    )
    t3 = a["T_bkt_crossing_fit_lnL0"]
    if t3 is not None:
        print(
            f"  T* = T_BKT + a/(ln L + c)^2:    T_BKT = {t3[0]:.5f}({t3[1]:.5f})  chi2 {t3[2]:.1f}/{t3[3]}"
        )
    wm = a["weber_minnhagen"]
    print(f"  Weber-Minnhagen (fine grid):   T_BKT = {wm['T_bkt']:.5f}({wm['error']:.5f})")
    e = a["eta_eff"]
    print(f"  chi ~ L^(2 - eta) at T = {e[2]}: eta_eff = {e[0]:.4f}({e[1]:.4f})  (1/4 at T_BKT)")


def main():
    args = parse_args(__doc__)
    started = time.time()
    if args.replot:
        # recompute the analysis from the stored per-temperature data, no simulation
        stored = load_results(NAME, args)
        sizes = [int(L) for L in stored["sizes"]]
        ups = {L: np.asarray(stored["helicity"][str(L)]["value"]) for L in sizes}
        dups = {L: np.asarray(stored["helicity"][str(L)]["error"]) for L in sizes}
        chi = {L: np.asarray(stored["chi"][str(L)]) for L in sizes}
    else:
        sizes, tasks, index = plan(args)
        banner(f"{NAME}: {len(tasks)} runs")
        runs = run_tasks(
            tasks, workers=args.workers, out_dir=data_dir(NAME + ("_q" if args.quick else ""))
        )
        ups = {L: np.zeros(len(TEMPS)) for L in sizes}
        dups = {L: np.zeros(len(TEMPS)) for L in sizes}
        chi = {L: np.zeros((len(TEMPS), 2)) for L in sizes}
        for (L, i), ts in zip(index, runs):
            u, du, c, dc = helicity(ts)
            ups[L][i], dups[L][i] = u, du
            chi[L][i] = c, dc

    banner("analysis")
    a = analyse(sizes, ups, dups, chi, args.seed)
    report(sizes, a)
    res = {
        "temperatures": TEMPS,
        "sizes": sizes,
        "helicity": {L: {"value": ups[L], "error": dups[L]} for L in sizes},
        "chi": {L: chi[L] for L in sizes},
        **a,
        "reference": {"T_bkt": T_REF, "source": reference("xy2d", "T_bkt").source},
    }
    path = save_results(NAME, res, args, started)
    print(f"  results -> {path}")
    if not args.no_plots:
        plot(sizes, ups, dups, a)


def plot(sizes, ups, dups, a):
    fig, axes = P.figure(3, 1, width=3.3, height=2.8)
    ax = axes[0]
    shown = sizes[-5:]
    cols = dict(zip(shown, P.ordinal_colors(len(shown))))
    for L in shown:
        P.errorbar(ax, TEMPS, ups[L], dups[L], color=cols[L], label=f"L = {L}", markersize=3.8)
    tt = np.linspace(TEMPS.min(), TEMPS.max(), 10)
    ax.plot(tt, 2 * tt / np.pi, color=P.INK_2, linewidth=1.0, label="2T/π")
    ax.set_xlabel("temperature T")
    ax.set_ylabel("helicity modulus Υ")
    ax.set_title("a  Stiffness jump at the BKT point")
    ax.legend(loc="lower left", fontsize=7.5)

    ax = axes[1]
    cross = a["crossings"]
    ls = np.array([L for L in sizes if np.isfinite(cross[L][0])], dtype=float)
    x = 1 / np.log(ls) ** 2
    y = np.array([cross[int(L)][0] for L in ls])
    dy = np.array([cross[int(L)][1] for L in ls])
    P.errorbar(ax, x, y, dy, color=P.CATEGORICAL[0], line=False, label="T*(L): Υ = 2T/π")
    big_l = np.geomspace(16, 1e12, 200)
    t2, a2 = a["crossing_params"]["two"]
    ax.plot(
        1 / np.log(big_l) ** 2,
        t2 + a2 / np.log(big_l) ** 2,
        color=P.CATEGORICAL[1],
        linewidth=1.1,
        label=f"a/(ln L)²: {t2:.4f}",
    )
    if a["crossing_params"]["three"] is not None:
        t3, a3, c3 = a["crossing_params"]["three"]
        e3 = a["T_bkt_crossing_fit_lnL0"][1]
        ax.plot(
            1 / np.log(big_l) ** 2,
            t3 + a3 / (np.log(big_l) + c3) ** 2,
            color=P.CATEGORICAL[0],
            linewidth=1.1,
            label=f"a/(ln L + c)²: {t3:.4f}({int(round(e3 * 1e4))})",
        )
    ax.axhline(T_REF, color=P.INK_2, linewidth=0.9)
    ax.annotate(
        f"literature {T_REF:.4f}",
        (x.max() * 1.05, T_REF),
        xytext=(-2, -11),
        textcoords="offset points",
        fontsize=7.5,
        color=P.INK_2,
        ha="right",
    )
    ax.set_xlim(0, x.max() * 1.08)
    ax.set_xlabel("1 / (ln L)²")
    ax.set_ylabel("crossing temperature")
    ax.set_title("b  Logarithmic finite-size drift")
    ax.legend(loc="upper left", fontsize=7.5)

    ax = axes[2]
    wm = a["weber_minnhagen"]
    grid = np.asarray(wm["T"], dtype=float)
    ax.plot(
        grid, wm["chi2_dof"], color=P.CATEGORICAL[0], linewidth=1.6, label="χ²/dof of the KT form"
    )
    ax.axvline(T_REF, color=P.INK_2, linewidth=0.9)
    ax.axvspan(
        wm["T_bkt"] - wm["error"],
        wm["T_bkt"] + wm["error"],
        color=P.CATEGORICAL[0],
        alpha=0.15,
        linewidth=0,
        label=f"minimum {wm['T_bkt']:.4f}({int(round(wm['error'] * 1e4))})",
    )
    ax.set_yscale("log")
    P.plain_log_ticks(ax)
    ax.set_xlabel("temperature T")
    ax.set_ylabel("χ² / dof")
    ax.set_title("c  Weber-Minnhagen criterion")
    ax.legend(loc="upper right", fontsize=7.5)
    P.save(fig, figure_path(NAME, "overview"))


if __name__ == "__main__":
    main()
