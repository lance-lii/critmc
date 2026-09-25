"""E08 - The 3D Ising model: critical point and exponents without an exact solution.

Wolff simulations on L^3 tori (L = 8 ... 48) at three couplings around the
critical region are combined by multi-histogram reweighting.  The critical
coupling follows from Binder-cumulant crossings, the exponents from
finite-size scaling at the estimated beta_c with corrections to scaling
(omega = 0.83).  The results are compared with the best literature values
(conformal bootstrap and high-precision Monte Carlo).
"""

from __future__ import annotations

import time

import numpy as np
from common import banner, data_dir, figure_path, parse_args, save_results

from critmc import plotting as P
from critmc.exact.constants import reference
from critmc.fss import Estimate, SizeData, binder_crossing, fit, fit_power_law
from critmc.runner import Task, run_tasks

NAME = "e08_ising3d_fss"
B0 = 0.2216546  # simulation point near the literature critical coupling
OMEGA = reference("ising3d", "omega").value


def plan(args):
    sizes = [8, 12, 16] if args.quick else [8, 12, 16, 24, 32, 48]
    meas = {8: 400_000, 12: 400_000, 16: 400_000, 24: 300_000, 32: 200_000, 48: 100_000}
    tasks, layout = [], {}
    for L in sizes:
        delta = B0 / np.sqrt(1.5 * L**3)  # ~ one standard deviation of the energy
        layout[L] = []
        for k, b in enumerate(B0 + delta * np.array([-1.0, 0.0, 1.0])):
            layout[L].append(len(tasks))
            tasks.append(
                Task(
                    {"model": "ising", "lattice": "cubic", "L": L},
                    float(b),
                    "wolff",
                    n_therm=5000,
                    n_meas=meas[L] // (20 if args.quick else 1),
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

    banner("Binder crossings")
    pairs = [(L, 2 * L) for L in sizes if 2 * L in sizes]
    crossings = []
    for L1, L2 in pairs:
        w1, w2 = data[L1].reliable_window(), data[L2].reliable_window()
        lo = max(w1[0], w2[0], B0 - 0.6 / L1**1.6)
        hi = min(w1[1], w2[1], B0 + 0.6 / L1**1.6)
        cr = binder_crossing(data[L1], data[L2], (lo, hi))
        crossings.append({"L": L1, "L2": L2, "beta": list(cr["beta"]), "U": list(cr["U"])})
        print(f"  ({L1:2d},{L2:2d}): beta* = {cr['beta']}  U* = {cr['U']}")
    lc = np.array([c["L"] for c in crossings], dtype=float)
    b = np.array([c["beta"] for c in crossings])
    u = np.array([c["U"] for c in crossings])
    if len(lc) >= 3:
        f_bc = fit(
            lambda x, bc, a: bc + a * x ** (-(1.587 + OMEGA)),
            lc,
            b[:, 0],
            b[:, 1],
            [B0, 0.0],
            ("beta_c", "a"),
        )
        bc_est = f_bc["beta_c"]
        f_u = fit(
            lambda x, us, a: us + a * x ** (-OMEGA), lc, u[:, 0], u[:, 1], [0.46, 0.0], ("U", "a")
        )
        u_est = f_u["U"]
    else:  # too few pairs for an extrapolation (quick mode): use the largest pair
        bc_est = Estimate(float(b[-1, 0]), float(b[-1, 1]))
        u_est = Estimate(float(u[-1, 0]), float(u[-1, 1]))
    print(
        f"  beta_c = {bc_est.value:.7f} +- {bc_est.error:.1e} (ref {reference('ising3d', 'beta_c').value})"
    )
    print(
        f"  U*     = {u_est.value:.4f} +- {u_est.error:.4f} (ref {1 - reference('ising3d', 'binder_ratio').value / 3:.4f})"
    )

    banner("exponents at the estimated beta_c")
    at = {L: data[L].at(bc_est.value, ["dU", "dlogm2", "chi0", "absm", "U"]) for L in sizes}
    Ls = np.array(sizes, dtype=float)
    fits = {}
    for key, name in (
        ("dU", "1/nu"),
        ("dlogm2", "1/nu"),
        ("chi0", "gamma/nu"),
        ("absm", "beta/nu"),
    ):
        y = np.abs([at[L][key].value for L in sizes])
        dy = np.array([at[L][key].error for L in sizes])
        plain = fit_power_law(Ls[1:], y[1:], dy[1:])
        corr = fit_power_law(Ls, y, dy, omega=OMEGA)
        fits[key] = {
            "quantity": name,
            "plain": [plain["exponent"].value, plain["exponent"].error, plain.chi2, plain.dof],
            "corrected": [corr["exponent"].value, corr["exponent"].error, corr.chi2, corr.dof],
            "params_plain": plain.params,
            "params_corr": corr.params,
        }
        sign = -1 if key == "absm" else 1
        print(
            f"  {key:>7}: {name} = {sign * plain['exponent'].value:.4f}({plain['exponent'].error:.4f}) plain, "
            f"{sign * corr['exponent'].value:.4f}({corr['exponent'].error:.4f}) with L^-omega "
            f"(chi2 {corr.chi2:.1f}/{corr.dof})"
        )
    ref = {k: reference("ising3d", k).value for k in ("beta_c", "nu", "gamma/nu", "beta/nu")}
    res = {
        "sizes": sizes,
        "crossings": crossings,
        "beta_c": [bc_est.value, bc_est.error],
        "U_star": [u_est.value, u_est.error],
        "at_beta_c": {L: {k: list(v) for k, v in at[L].items()} for L in sizes},
        "fits": fits,
        "reference": {
            **ref,
            "1/nu": 1 / ref["nu"],
            "U_star": 1 - reference("ising3d", "binder_ratio").value / 3,
        },
    }
    path = save_results(NAME, res, args, started)
    print(f"  results -> {path}")
    if not args.no_plots:
        plot(sizes, data, crossings, bc_est, at, fits)


def plot(sizes, data, crossings, bc_est, at, fits):
    fig, axes = P.figure(3, 1, width=3.3, height=2.8)
    ax = axes[0]
    shown = sizes[-5:]
    cols = dict(zip(shown, P.ordinal_colors(len(shown))))
    for L in shown:
        betas = B0 + np.linspace(-1.0, 1.0, 31) * 0.5 / L**1.59
        v, e, _ = data[L].curve(betas, "U")
        ax.fill_between(betas, v - e, v + e, color=cols[L], alpha=0.25, linewidth=0)
        ax.plot(betas, v, color=cols[L], linewidth=1.5, label=f"L = {L}")
    ax.axvline(reference("ising3d", "beta_c").value, color=P.INK_2, linewidth=0.9)
    ax.set_xlim(B0 - 0.0035, B0 + 0.0035)
    ax.set_xlabel("coupling β")
    ax.set_ylabel("Binder cumulant U₄")
    ax.set_title("a  3D Ising Binder cumulants")
    ax.legend(loc="lower right", fontsize=7.5)

    ax = axes[1]
    lc = np.array([c["L"] for c in crossings], dtype=float)
    bb = np.array([c["beta"] for c in crossings])
    P.errorbar(
        ax,
        lc,
        (bb[:, 0] - reference("ising3d", "beta_c").value) * 1e5,
        bb[:, 1] * 1e5,
        color=P.CATEGORICAL[0],
        line=False,
        label="crossing (L, 2L)",
    )
    ax.axhline(0, color=P.INK_2, linewidth=0.9)
    ax.axhspan(
        (bc_est.value - bc_est.error - reference("ising3d", "beta_c").value) * 1e5,
        (bc_est.value + bc_est.error - reference("ising3d", "beta_c").value) * 1e5,
        color=P.CATEGORICAL[0],
        alpha=0.12,
        linewidth=0,
        label="extrapolated β_c ± 1σ",
    )
    ax.set_xscale("log", base=2)
    P.plain_log_ticks(ax, "x")
    ax.set_xlabel("smaller size L")
    ax.set_ylabel("(β* − β_c,lit) × 10⁵")
    ax.set_title("b  Crossings vs literature β_c")
    ax.legend(loc="upper right", fontsize=7.5)

    ax = axes[2]
    Ls = np.array(sizes, dtype=float)
    for k, (key, lab) in enumerate((("dU", "|dU/dβ|"), ("chi0", "χ"))):
        y = np.abs([at[L][key].value for L in sizes])
        dy = np.array([at[L][key].error for L in sizes])
        norm = y[0]
        P.errorbar(
            ax,
            Ls,
            y / norm,
            dy / norm,
            color=P.CATEGORICAL[k],
            line=False,
            label=f"{lab}: exponent {fits[key]['corrected'][0]:.3f}",
        )
        a, x, c = fits[key]["params_corr"]
        xx = np.geomspace(Ls.min(), Ls.max(), 50)
        ax.plot(
            xx,
            a * xx**x * (1 + c * xx ** (-OMEGA)) / norm,
            color=P.CATEGORICAL[k],
            linewidth=1.1,
            alpha=0.8,
        )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    P.plain_log_ticks(ax, "both")
    ax.set_xlabel("L")
    ax.set_ylabel("value / value at L = 8")
    ax.set_title("c  Scaling at β_c (ω = 0.83)")
    ax.legend(loc="upper left", fontsize=7.5)
    P.save(fig, figure_path(NAME, "overview"))


if __name__ == "__main__":
    main()
