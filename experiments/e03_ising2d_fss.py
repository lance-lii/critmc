"""E03 - Finite-size-scaling study of the 2D Ising model, L = 8 ... 256.

For every size three Wolff runs at beta_c and beta_c +- 1/sigma_E(L) are
combined by multi-histogram reweighting.  From them we estimate, without using
any exact input in the fits:

* beta_c from the drift of Binder-cumulant crossings (L, 2L),
* 1/nu from dU/dbeta and d ln<m^2>/dbeta at the critical point,
* gamma/nu and beta/nu from the susceptibility and magnetisation,
* gamma/nu and 1/nu again from the height and location of the susceptibility peak,
* the log-divergence amplitude of the specific heat,
* (beta_c, 1/nu, gamma/nu) from data collapses,

and compare them with the exact values.  All errors are jackknife errors that
propagate through the reweighting.
"""

from __future__ import annotations

import time

import numpy as np
from common import banner, data_dir, figure_path, parse_args, save_results
from scipy import optimize

from critmc import plotting as P
from critmc.exact import kaufman, onsager
from critmc.exact.constants import reference
from critmc.fss import (
    SizeData,
    _jackknife,
    binder_crossing,
    data_collapse,
    fit,
    fit_power_law,
    fit_shift,
    peak,
)
from critmc.runner import Task, run_tasks

NAME = "e03_ising2d_fss"
BC = onsager.BETA_C


def exact_cmax(L: int) -> float:
    """Maximum of Kaufman's exact specific heat for the L x L torus."""
    res = optimize.minimize_scalar(
        lambda b: -kaufman.specific_heat(b, L),
        bounds=(0.3, 0.6),
        method="bounded",
        options={"xatol": 1e-12},
    )
    return float(-res.fun)


def plan(args):
    sizes = [8, 16, 32, 64] if args.quick else [8, 16, 32, 64, 128, 256]
    meas = {8: 500_000, 16: 500_000, 32: 500_000, 64: 400_000, 128: 250_000, 256: 125_000}
    tasks, layout = [], {}
    for L in sizes:
        c = kaufman.specific_heat(BC, L)
        delta = BC / np.sqrt(c * L * L)  # one standard deviation of the energy in beta
        betas = BC + delta * np.array([-1.0, 0.0, 1.0])
        n_meas = meas[L] // (20 if args.quick else 1)
        layout[L] = []
        for k, b in enumerate(betas):
            layout[L].append(len(tasks))
            tasks.append(
                Task(
                    {"model": "ising", "lattice": "square", "L": L},
                    float(b),
                    "wolff",
                    n_therm=5000,
                    n_meas=n_meas,
                    seed=args.seed,
                    keys=(NAME, L, k),
                    name=f"L{L}_{k}",
                )
            )
    return sizes, tasks, layout


def main():
    args = parse_args(__doc__)
    started = time.time()
    sizes, tasks, layout = plan(args)
    banner(f"{NAME}: {len(tasks)} runs")
    runs = run_tasks(
        tasks, workers=args.workers, out_dir=data_dir(NAME + ("_q" if args.quick else ""))
    )
    data = {L: SizeData.from_runs([runs[i] for i in layout[L]], n_blocks=32) for L in sizes}
    # beta intervals in which the reweighted ensembles are trustworthy
    window = {L: data[L].reliable_window() for L in sizes}
    Ls = np.array(sizes, dtype=float)
    res: dict = {
        "sizes": sizes,
        "reliable_windows": window,
        "runs": {
            L: {"betas": data[L].betas, "n_meas": [len(runs[i]) for i in layout[L]]} for L in sizes
        },
    }

    banner("observables at the exact critical point")
    at_c = {L: data[L].at(BC, ["U", "dU", "dlogm2", "dlogm", "chi0", "absm", "c"]) for L in sizes}
    for L in sizes:
        o = at_c[L]
        print(
            f"  L={L:3d}: U={o['U']}  dU={o['dU']}  chi0={o['chi0']}  |m|={o['absm']}  c={o['c']}  "
            f"(exact c={kaufman.specific_heat(BC, L):.5f})"
        )
    res["at_beta_c"] = {L: {k: [v.value, v.error] for k, v in at_c[L].items()} for L in sizes}
    res["c_exact_at_beta_c"] = {L: kaufman.specific_heat(BC, L) for L in sizes}

    def series(key):
        return (
            np.array([at_c[L][key].value for L in sizes]),
            np.array([at_c[L][key].error for L in sizes]),
        )

    fits = {}
    lmin = 16 if not args.quick else 8
    sel = Ls >= lmin
    for key, label in (
        ("dU", "1/nu"),
        ("dlogm2", "1/nu"),
        ("chi0", "gamma/nu"),
        ("absm", "-beta/nu"),
    ):
        y, dy = series(key)
        f = fit_power_law(Ls[sel], np.abs(y[sel]), dy[sel])
        fits[key] = {
            "quantity": label,
            "exponent": f["exponent"].value,
            "error": f["exponent"].error,
            "chi2": f.chi2,
            "dof": f.dof,
            "L_min": lmin,
            "params": f.params,
        }
        msg = f"  {key:>7} ~ L^x: x = {f['exponent']}  ({label}; chi2/dof = {f.chi2:.1f}/{f.dof})"
        if key != "absm" and len(Ls) >= 4:
            # regular (background) part of the free energy: y = a L^x + b, all sizes
            fb = fit(
                lambda x, a, e, b: a * x**e + b,
                Ls,
                np.abs(y),
                dy,
                [f.params[0], f.params[1], 0.0],
                ("a", "exponent", "b"),
            )
            fits[key]["background"] = [fb["exponent"].value, fb["exponent"].error, fb.chi2, fb.dof]
            msg += f";  with background: x = {fb['exponent']} (chi2/dof = {fb.chi2:.1f}/{fb.dof})"
        print(msg)
    res["fits_at_beta_c"] = fits

    banner("Binder cumulant crossings")
    crossings = []
    for L1, L2 in zip(sizes[:-1], sizes[1:]):
        lo = max(window[L1][0], window[L2][0], BC - 1.5 / L1)
        hi = min(window[L1][1], window[L2][1], BC + 1.5 / L1)
        cr = binder_crossing(data[L1], data[L2], (lo, hi))
        crossings.append({"L": L1, "L2": L2, "beta": list(cr["beta"]), "U": list(cr["U"])})
        print(f"  ({L1:3d},{L2:3d}): beta* = {cr['beta']}  U* = {cr['U']}")
    res["crossings"] = crossings
    lc = np.array([c["L"] for c in crossings], dtype=float)
    bstar = np.array([c["beta"][0] for c in crossings])
    dbstar = np.array([c["beta"][1] for c in crossings])
    if len(lc) >= 3:
        # 2D Ising: corrections to the crossing drift ~ L^-(1/nu + omega) with omega = 2 (analytic)
        fs = fit_shift(lc[1:], bstar[1:], dbstar[1:], exponent=3.0)
        res["beta_c_from_crossings"] = {
            "value": fs["beta_c"].value,
            "error": fs["beta_c"].error,
            "chi2": fs.chi2,
            "dof": fs.dof,
        }
        print(f"  beta_c from crossing drift ~ L^-3: {fs['beta_c']}  (exact {BC:.7f})")
    ustar = np.array([c["U"][0] for c in crossings])
    dustar = np.array([c["U"][1] for c in crossings])
    res["U_star_largest"] = [ustar[-1], dustar[-1]]

    banner("susceptibility and specific-heat peaks")
    peaks = {}
    for L in sizes:
        lo, hi = window[L]
        pk = peak(data[L], "chi", (max(lo, BC - 1.2 / L), min(hi, BC + 0.1 / L)))
        pc = peak(data[L], "c", (max(lo, BC - 0.8 / L), min(hi, BC + 0.8 / L)))
        peaks[L] = {
            "beta_chi": list(pk["beta"]),
            "chi_max": list(pk["chi"]),
            "beta_c": list(pc["beta"]),
            "c_max": list(pc["c"]),
        }
        print(f"  L={L:3d}: chi' max {pk['chi']} at {pk['beta']};  c max {pc['c']} at {pc['beta']}")
    res["peaks"] = peaks
    chi_max = np.array([peaks[L]["chi_max"] for L in sizes])
    f_chi = fit_power_law(Ls[sel], chi_max[sel, 0], chi_max[sel, 1])
    b_chi = np.array([peaks[L]["beta_chi"] for L in sizes])
    f_shift = fit_shift(Ls[sel], b_chi[sel, 0], b_chi[sel, 1])
    c_max = np.array([peaks[L]["c_max"] for L in sizes])
    c_exact = np.array([exact_cmax(L) for L in sizes])
    c_pulls = (c_max[:, 0] - c_exact) / c_max[:, 1]
    f_c = fit(
        lambda x, a, b, d: a * np.log(x) + b + d / x,
        Ls,
        c_max[:, 0],
        c_max[:, 1],
        [0.5, 0.1, 0.0],
        ("a", "b", "d"),
    )
    res["peak_fits"] = {
        "gamma_over_nu": [f_chi["exponent"].value, f_chi["exponent"].error, f_chi.chi2, f_chi.dof],
        "shift": {
            "beta_c": list(f_shift["beta_c"]),
            "inv_nu": list(f_shift["x"]),
            "chi2": f_shift.chi2,
            "dof": f_shift.dof,
        },
        "c_log_amplitude": [f_c["a"].value, f_c["a"].error, f_c.chi2, f_c.dof],
        "c_max_exact": c_exact,
        "c_max_pulls": c_pulls,
    }
    print(
        f"  chi'_max ~ L^{f_chi['exponent']}   beta_max shift: beta_c={f_shift['beta_c']}, "
        f"1/nu={f_shift['x']}   c_max = a ln L + b + d/L: a={f_c['a']}"
    )
    print(f"  c_max pulls against the exact finite-L maxima: {np.round(c_pulls, 2)}")

    banner("data collapse")
    collapse_sizes = [L for L in sizes if L >= (16 if args.quick else 32)]
    # scaling variable (beta - beta_c) L in [-0.35, 0.35], clipped to the reliable windows
    grids = {L: np.clip(BC + np.linspace(-0.35, 0.35, 21) / L, *window[L]) for L in collapse_sizes}
    curves_u, curves_chi = {}, {}
    for L in collapse_sizes:
        curves_u[L] = data[L].curve(grids[L], "U")
        curves_chi[L] = data[L].curve(grids[L], "chi0")

    def run_collapse(rep, start=(BC + 1e-4, 0.95, 1.7)):
        cu = {
            L: (
                grids[L],
                curves_u[L][2][rep] if rep is not None else curves_u[L][0],
                curves_u[L][1],
            )
            for L in collapse_sizes
        }
        cc = {
            L: (
                grids[L],
                curves_chi[L][2][rep] if rep is not None else curves_chi[L][0],
                curves_chi[L][1],
            )
            for L in collapse_sizes
        }
        ru = data_collapse(cu, start[0], start[1], 0.0, fixed=("kappa",))
        # near beta_c the chi collapse alone cannot separate beta_c from gamma/nu, so it is
        # done at the Binder-collapse beta_c of the same jackknife replicate
        rc = data_collapse(cc, ru.beta_c, ru.inv_nu, start[2], fixed=("beta_c",))
        return np.array(
            [ru.beta_c, ru.inv_nu, ru.quality, rc.beta_c, rc.inv_nu, rc.kappa, rc.quality]
        )

    full = run_collapse(None)
    # Replicates start from the full-sample optimum.  The estimate is the full-sample
    # collapse and the jackknife provides only its error: the jackknife bias correction
    # n*full - (n-1)*mean would amplify the optimiser's convergence noise n-fold.
    reps = [run_collapse(r, start=(full[0], full[1], full[5])) for r in range(32)]
    _, err = _jackknife(full, reps)
    val = full
    res["collapse"] = {
        "binder": {"beta_c": [val[0], err[0]], "inv_nu": [val[1], err[1]], "quality": full[2]},
        "chi": {
            "beta_c": [val[3], err[3]],
            "inv_nu": [val[4], err[4]],
            "gamma_over_nu": [val[5], err[5]],
            "quality": full[6],
        },
        "sizes": collapse_sizes,
    }
    print(
        f"  Binder collapse: beta_c={val[0]:.6f}({err[0]:.1e}) 1/nu={val[1]:.4f}({err[1]:.1e}) S={full[2]:.2f}"
    )
    print(
        f"  chi collapse:    beta_c={val[3]:.6f}({err[3]:.1e}) 1/nu={val[4]:.4f}({err[4]:.1e}) "
        f"gamma/nu={val[5]:.4f}({err[5]:.1e}) S={full[6]:.2f}"
    )

    res["exact"] = {
        "beta_c": BC,
        "nu": 1.0,
        "gamma_over_nu": 1.75,
        "beta_over_nu": 0.125,
        "U_star": reference("ising2d", "binder_U4").value,
        "c_amplitude": onsager.heat_capacity_amplitude(),
    }
    path = save_results(NAME, res, args, started)
    print(f"  results -> {path}")
    if not args.no_plots:
        plot(
            sizes,
            data,
            at_c,
            fits,
            crossings,
            peaks,
            f_c,
            curves_u,
            curves_chi,
            grids,
            full,
            collapse_sizes,
        )


def plot(
    sizes,
    data,
    at_c,
    fits,
    crossings,
    peaks,
    f_c,
    curves_u,
    curves_chi,
    grids,
    full,
    collapse_sizes,
):
    shown = sizes[-5:]
    colors = dict(zip(shown, P.ordinal_colors(len(shown))))
    Ls = np.array(sizes, dtype=float)
    u_star = reference("ising2d", "binder_U4").value

    # Figure 1: Binder cumulant and crossings
    fig, axes = P.figure(2, 1, width=3.5, height=2.9)
    ax = axes[0]
    for L in shown:
        lo, hi = data[L].reliable_window()
        betas = np.linspace(max(lo, BC - 0.03), min(hi, BC + 0.03), 41)
        v, e, _ = data[L].curve(betas, "U")
        ax.fill_between(betas, v - e, v + e, color=colors[L], alpha=0.25, linewidth=0)
        ax.plot(betas, v, color=colors[L], linewidth=1.5, label=f"L = {L}")
    ax.set_xlim(BC - 0.03, BC + 0.03)
    ax.set_ylim(0.45, 0.68)
    P.reference_line(ax, x=BC, label="exact β_c")
    ax.axhline(u_star, color=P.AXIS, linewidth=0.8)
    ax.annotate(
        f"U* = {u_star:.4f}",
        (BC + 0.03, u_star),
        xytext=(-2, 3),
        textcoords="offset points",
        fontsize=7.5,
        color=P.INK_2,
        ha="right",
    )
    ax.set_xlabel("inverse temperature β")
    ax.set_ylabel("Binder cumulant U₄")
    ax.set_title("a  Binder cumulants cross at β_c")
    ax.legend(loc="lower right")

    ax = axes[1]
    lc = np.array([c["L"] for c in crossings], dtype=float)
    b = np.array([c["beta"] for c in crossings])
    P.errorbar(
        ax,
        lc,
        (b[:, 0] - BC) * 1e4,
        b[:, 1] * 1e4,
        color=P.CATEGORICAL[0],
        line=False,
        label="crossing of (L, 2L)",
    )
    ax.axhline(0, color=P.INK_2, linewidth=0.9)
    ax.set_xscale("log", base=2)
    big = lc >= 16
    if big.any() and not big.all():
        lo = ((b[big, 0] - b[big, 1] - BC) * 1e4).min()
        hi = ((b[big, 0] + b[big, 1] - BC) * 1e4).max()
        pad = 0.25 * (hi - lo)
        ax.set_ylim(min(lo - pad, -pad), hi + pad)
        ax.set_xlim(lc[big].min() / 1.5, lc.max() * 1.5)
        small = lc[~big][0]
        off = (b[~big, 0][0] - BC) * 1e4
        ax.annotate(
            f"({int(small)}, {int(2 * small)}): {off:+.1f} (off scale)",
            (0.03, 0.95),
            xycoords="axes fraction",
            fontsize=7.5,
            color=P.INK_2,
            va="top",
        )
    ax.set_xlabel("smaller size L of the pair")
    ax.set_ylabel("(β* − β_c) × 10⁴")
    ax.set_title("b  Crossing drift towards the exact β_c")
    ax.legend(loc="upper right")
    P.save(fig, figure_path(NAME, "binder"))

    # Figure 2: scaling at beta_c (three small multiples + specific heat)
    fig, axes = P.figure(4, 1, width=2.75, height=2.6)
    panels = [
        ("dU", "|dU/dβ| at β_c", "1/ν", 1.0),
        ("chi0", "χ at β_c", "γ/ν", 1.75),
        ("absm", "<|m|> at β_c", "−β/ν", -0.125),
    ]
    for ax, (key, ylabel, sym, exact) in zip(axes[:3], panels):
        y = np.abs([at_c[L][key].value for L in sizes])
        dy = np.array([at_c[L][key].error for L in sizes])
        P.errorbar(ax, Ls, y, dy, color=P.CATEGORICAL[0], line=False, label="Monte Carlo")
        f = fits[key]
        a, x = f["params"]
        xx = np.geomspace(Ls.min(), Ls.max(), 50)
        ax.plot(
            xx,
            a * xx**x,
            color=P.INK_2,
            linewidth=1.0,
            label=f"{sym} = {f['exponent']:.4f}({int(round(f['error'] * 1e4))})",
        )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        P.plain_log_ticks(ax)
        ax.set_xlabel("L")
        ax.set_title(f"{ylabel}  (exact {exact:g})", fontsize=9.5)
        ax.legend(loc="best", fontsize=7.5)
    ax = axes[3]
    c_mc = np.array([at_c[L]["c"].value for L in sizes])
    dc_mc = np.array([at_c[L]["c"].error for L in sizes])
    P.errorbar(ax, Ls, c_mc, dc_mc, color=P.CATEGORICAL[0], line=False, label="Monte Carlo at β_c")
    every_l = np.arange(int(Ls.min()), int(Ls.max()) + 1)  # exact values at every integer size
    ax.plot(
        every_l,
        [kaufman.specific_heat(BC, int(L)) for L in every_l],
        color=P.INK_2,
        linewidth=1.0,
        label="exact (Kaufman)",
    )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("L")
    ax.set_title("c at β_c vs exact", fontsize=9.5)
    ax.legend(loc="upper left", fontsize=7.5)
    P.save(fig, figure_path(NAME, "scaling"))

    # Figure 3: data collapse
    fig, axes = P.figure(2, 1, width=3.5, height=2.8)
    bc_u, inv_nu_u = full[0], full[1]
    bc_c, inv_nu_c, kappa = full[3], full[4], full[5]
    cols = dict(zip(collapse_sizes, P.ordinal_colors(len(collapse_sizes))))
    n_c = len(collapse_sizes)
    handles = {0: {}, 1: {}}
    # largest size first with the biggest marker, smaller sizes on top: where the curves
    # coincide (a good collapse) every size stays visible as nested markers
    for rank, L in enumerate(reversed(collapse_sizes)):
        style = dict(
            color=cols[L],
            line=False,
            label=f"L = {L}",
            markersize=7.0 - 3.6 * rank / max(1, n_c - 1),
        )
        x = (grids[L] - bc_u) * L**inv_nu_u
        v, e, _ = curves_u[L]
        handles[0][L] = P.errorbar(axes[0], x, v, e, **style)
        x = (grids[L] - bc_c) * L**inv_nu_c
        v, e, _ = curves_chi[L]
        handles[1][L] = P.errorbar(axes[1], x, v * L ** (-kappa), e * L ** (-kappa), **style)
    labels = [f"L = {L}" for L in collapse_sizes]
    axes[0].set_xlabel(r"$(\beta - \beta_c)\,L^{1/\nu}$")
    axes[0].set_ylabel("U₄")
    axes[0].set_title(f"a  Binder collapse: 1/ν = {inv_nu_u:.3f}")
    axes[0].legend([handles[0][L] for L in collapse_sizes], labels, loc="lower right")
    axes[1].set_xlabel(r"$(\beta - \beta_c)\,L^{1/\nu}$")
    axes[1].set_ylabel(r"$\chi\,L^{-\gamma/\nu}$")
    axes[1].set_title(f"b  χ collapse at the Binder β_c: γ/ν = {kappa:.3f}")
    axes[1].legend([handles[1][L] for L in collapse_sizes], labels, loc="lower right")
    P.save(fig, figure_path(NAME, "collapse"))

    # Figure 4: specific-heat maximum
    fig, ax = P.figure(1, 1, width=3.5, height=2.7)
    cm = np.array([peaks[L]["c_max"] for L in sizes])
    P.errorbar(
        ax,
        np.log(Ls),
        cm[:, 0],
        cm[:, 1],
        color=P.CATEGORICAL[0],
        line=False,
        label="Monte Carlo c_max(L)",
    )
    every_l = np.unique(np.geomspace(Ls.min(), Ls.max(), 40).astype(int))
    ax.plot(
        np.log(every_l),
        [exact_cmax(int(L)) for L in every_l],
        color=P.INK_2,
        linewidth=1.0,
        label="exact c_max(L) (Kaufman)",
    )
    ax.set_xlabel("ln L")
    ax.set_ylabel("specific-heat maximum")
    err = int(round(f_c["a"].error * 1e3))
    ax.set_title(f"c_max = A ln L + B + D/L:  A = {f_c['a'].value:.3f}({err}), exact 0.4945")
    ax.legend(loc="upper left")
    P.save(fig, figure_path(NAME, "cmax"))


if __name__ == "__main__":
    main()
