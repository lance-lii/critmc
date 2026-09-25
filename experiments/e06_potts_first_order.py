"""E06 - The first-order transition of the 2D 10-state Potts model.

Canonical simulations of a strongly first-order transition are exponentially
slow, because the mixed-phase states with two interfaces between the ordered
and disordered phases are suppressed by exp(-2 sigma L).  Here Wang-Landau
provides weights 1/g(E) on the energy window of the transition, a
multicanonical run samples that window uniformly, and reweighting recovers
the canonical energy distribution at any beta.  From the equal-height and
equal-weight points we extract the latent heat, the infinite-volume
transition point and the order-disorder interface tension, and compare with
Baxter's exact latent heat and the self-dual beta_c = ln(1 + sqrt 10).
Errors: jackknife over 20 blocks of each multicanonical time series.
"""

from __future__ import annotations

import time

import numpy as np
from common import banner, figure_path, parse_args, save_results
from scipy import optimize

from critmc import Potts, square
from critmc import plotting as P
from critmc.exact import potts as xpotts
from critmc.flathist import MucaResult, Multicanonical, WangLandau
from critmc.fss import _jackknife, fit
from critmc.runner import parallel_map

NAME = "e06_potts_first_order"
Q = 10
BC = xpotts.beta_c(Q)
E_SPLIT = xpotts.critical_energy(Q)  # separates the two phases
WINDOW = (-1.95, -0.60)
N_BLOCKS = 20
SMOOTH = 0.01  # Gaussian smoothing width (energy per site) for peak heights


def c_peak(res: MucaResult, center: float, half_width: float) -> tuple[float, float]:
    """Location and height of the specific-heat maximum (per site) near ``center``."""

    def neg_c(b):
        return -float(res.canonical(b)["c"][0])

    grid = np.linspace(center - half_width, center + half_width, 31)
    i = int(np.clip(np.argmin([neg_c(b) for b in grid]), 1, len(grid) - 2))
    r = optimize.minimize_scalar(
        neg_c, bounds=(grid[i - 1], grid[i + 1]), method="bounded", options={"xatol": 1e-10}
    )
    return float(r.x), float(-r.fun)


def analyse(res: MucaResult) -> np.ndarray:
    """Estimators from one multicanonical time series.

    Returns (beta_eqh, beta_eqw, <e>_ordered, <e>_disordered, ln(P_max/P_min), e_o peak,
    e_d peak, beta at c_max, c_max / N).  The phase-conditional mean energies are taken
    at equal weight, peak heights and the valley depth from the lightly smoothed
    distribution at equal height.  For a first-order transition
    c_max / N -> (beta_c Delta e / 2)^2 + O(1/N) (Challa, Landau & Binder 1986).
    """
    n = res.levels.n_sites
    bracket = (BC - 0.10, BC + 0.05)
    b_eqw = res.equal_weight_beta(bracket, split=E_SPLIT)
    pm = res.phase_means(b_eqw, split=E_SPLIT)
    b_eqh = res.equal_height_beta(bracket, split=E_SPLIT, smooth=SMOOTH)
    dp = res.double_peak(b_eqh, split=E_SPLIT, smooth=SMOOTH)
    b_c, c_max = c_peak(res, b_eqw, 0.002 + 4.0 / n)
    return np.array(
        [
            b_eqh,
            b_eqw,
            pm["e_ordered"],
            pm["e_disordered"],
            np.log(dp["p_ordered"] / dp["p_min"]),
            dp["e_ordered"],
            dp["e_disordered"],
            b_c,
            c_max / n,
        ]
    )


def _subset(res: MucaResult, keep: np.ndarray) -> MucaResult:
    levels = res.level_series[keep]
    hist = np.bincount(levels, minlength=len(res.hist)).astype(np.int64)
    return MucaResult(res.levels, res.ln_g, levels, res.order_series[keep], res.window, hist)


def run_size(job):
    L, seed, quick = job
    model = Potts(square(L), Q)
    t0 = time.time()
    wl = WangLandau(model, e_window=WINDOW, seed=np.random.SeedSequence(seed, spawn_key=(6, L, 0)))
    wlr = wl.run(3e-6 if quick else (1e-6 if L <= 24 else 2e-6))
    t_wl = time.time() - t0
    n_meas = 20_000 if quick else (300_000 if L < 32 else 600_000)
    mu = Multicanonical(
        model, wlr.ln_g, e_window=WINDOW, seed=np.random.SeedSequence(seed, spawn_key=(6, L, 1))
    )
    raw = mu.run(n_meas, sweeps_per_meas=1, n_therm=2000)
    # use the measured series for both the full estimate and the jackknife replicates
    full_res = _subset(raw, np.ones(n_meas, dtype=bool))
    full = analyse(full_res)
    size = n_meas // N_BLOCKS
    reps = []
    for b in range(N_BLOCKS):
        keep = np.ones(n_meas, dtype=bool)
        keep[b * size : (b + 1) * size] = False
        reps.append(analyse(_subset(raw, keep)))
    value, err = _jackknife(full, reps)
    e_grid, p = full_res.energy_distribution(full[0], SMOOTH)
    # number of round trips between the two phases (tunnelling events)
    phase = np.sign(full_res.energy / (L * L) - E_SPLIT)
    flips = int(np.count_nonzero(np.diff(phase[phase != 0])))
    return {
        "L": L,
        "value": value,
        "error": err,
        "e": e_grid,
        "p": p,
        "tunnel_half_trips": flips,
        "wl_trials": wlr.trials,
        "wl_seconds": t_wl,
        "seconds": time.time() - t0,
        "n_meas": n_meas,
    }


def main():
    args = parse_args(__doc__)
    started = time.time()
    sizes = [8, 12, 16, 20, 24] if args.quick else [8, 12, 16, 20, 24, 32, 40, 48]
    banner(f"{NAME}: Wang-Landau + multicanonical for L = {sizes}")
    out = parallel_map(run_size, [(L, args.seed, args.quick) for L in sizes], workers=args.workers)
    out.sort(key=lambda o: o["L"])
    L = np.array([o["L"] for o in out], dtype=float)
    val = np.array([o["value"] for o in out])
    err = np.array([o["error"] for o in out])
    names = [
        "beta_eqh",
        "beta_eqw",
        "e_ordered",
        "e_disordered",
        "log_ratio",
        "e_o_peak",
        "e_d_peak",
        "beta_cmax",
        "cmax_per_site",
    ]
    table = {
        o["L"]: {n: [float(v), float(e)] for n, v, e in zip(names, o["value"], o["error"])}
        for o in out
    }
    for o in out:
        v, e = o["value"], o["error"]
        print(
            f"  L={o['L']:3d}: beta_eqh={v[0]:.6f}({e[0]:.1e}) beta_eqw={v[1]:.6f}({e[1]:.1e}) "
            f"<e>_o={v[2]:.4f}({e[2]:.4f}) <e>_d={v[3]:.4f}({e[3]:.4f}) peaks {v[5]:.3f}/{v[6]:.3f} "
            f"ln(Pmax/Pmin)={v[4]:.3f}({e[4]:.2f}) "
            f"tunnel={o['tunnel_half_trips'] // 2} ({o['seconds']:.0f} s)"
        )

    latent = val[:, 3] - val[:, 2]
    dlatent = np.hypot(err[:, 3], err[:, 2])
    exact_dl = xpotts.latent_heat(Q)
    sel = L >= 16
    # (a) phase-conditional means: plateau of the largest sizes
    big24 = L >= 24
    w = 1.0 / dlatent[big24] ** 2
    plateau = float(np.sum(w * latent[big24]) / w.sum())
    dplateau = float(1.0 / np.sqrt(w.sum()))
    chi2_plateau = float(np.sum(w * (latent[big24] - plateau) ** 2))
    f_lat = fit(
        lambda x, a, b: a + b / x, L[sel], latent[sel], dlatent[sel], [0.7, 1.0], ("dE", "b")
    )
    # (b) specific-heat maximum: c_max / N = (beta Delta e / 2)^2 + B / N
    cm, dcm = val[:, 8], err[:, 8]
    n_sites = L**2
    f_cm = fit(lambda x, a, b: a + b / x, n_sites[sel], cm[sel], dcm[sel], [0.12, 1.0], ("A", "B"))
    a_cm, da_cm = f_cm["A"].value, f_cm["A"].error
    dl_c = 2.0 * np.sqrt(a_cm) / BC
    ddl_c = da_cm / (BC * np.sqrt(a_cm))
    dl_c_L = 2.0 * np.sqrt(cm) / val[:, 7]
    ddl_c_L = dcm / (val[:, 7] * np.sqrt(cm))
    f_bw = fit(
        lambda x, a, b: a + b / x**2, L[sel], val[sel, 1], err[sel, 1], [BC, -1.0], ("beta_c", "b")
    )
    f_bh = fit(
        lambda x, a, b, c: a + b / x**2 + c / x**4,
        L[sel],
        val[sel, 0],
        err[sel, 0],
        [BC, -1.0, 0.0],
        ("beta_c", "b", "c"),
    )
    sigma = val[:, 4] / (2 * L)
    dsigma = err[:, 4] / (2 * L)
    big = L >= 16
    f_sig = fit(
        lambda x, a, b: a + b / x, L[big], sigma[big], dsigma[big], [0.05, 0.1], ("sigma", "b")
    )
    print(
        f"  latent heat: c_max route {dl_c:.4f}({ddl_c:.4f}) [chi2 {f_cm.chi2:.1f}/{f_cm.dof}]; "
        f"conditional means, plateau L>=24: {plateau:.4f}({dplateau:.4f}) [chi2 {chi2_plateau:.1f}/"
        f"{int(big24.sum()) - 1}]; a + b/L: {f_lat['dE']}; exact {exact_dl:.6f}"
    )
    print(
        f"  beta_c from equal weight: {f_bw['beta_c']} (chi2 {f_bw.chi2:.1f}/{f_bw.dof}); "
        f"from equal height: {f_bh['beta_c']}; exact {BC:.7f}"
    )
    print(
        f"  interface tension 2 sigma_od = {2 * f_sig['sigma'].value:.4f}({2 * f_sig['sigma'].error:.4f})"
    )
    res = {
        "q": Q,
        "window": WINDOW,
        "table": table,
        "latent_heat": {
            "L": L,
            "value": latent,
            "error": dlatent,
            "fit_1overL": [f_lat["dE"].value, f_lat["dE"].error, f_lat.chi2, f_lat.dof],
            "plateau_L24": [plateau, dplateau, chi2_plateau, int(big24.sum()) - 1],
            "cmax_route": [float(dl_c), float(ddl_c), f_cm.chi2, f_cm.dof],
            "cmax_per_L": {"value": dl_c_L, "error": ddl_c_L},
            "exact": exact_dl,
        },
        "beta_c": {
            "equal_weight_fit": [f_bw["beta_c"].value, f_bw["beta_c"].error, f_bw.chi2, f_bw.dof],
            "equal_height_fit": [f_bh["beta_c"].value, f_bh["beta_c"].error, f_bh.chi2, f_bh.dof],
            "exact": BC,
        },
        "interface": {
            "sigma_L": sigma,
            "error": dsigma,
            "two_sigma_fit": [
                2 * f_sig["sigma"].value,
                2 * f_sig["sigma"].error,
                f_sig.chi2,
                f_sig.dof,
            ],
        },
        "runs": [
            {
                "L": o["L"],
                "wl_trials": o["wl_trials"],
                "wl_seconds": o["wl_seconds"],
                "seconds": o["seconds"],
                "n_meas": o["n_meas"],
                "tunnel_round_trips": o["tunnel_half_trips"] // 2,
            }
            for o in out
        ],
    }
    path = save_results(NAME, res, args, started)
    print(f"  results -> {path}")
    if not args.no_plots:
        plot(out, L, val, err, latent, dlatent, dl_c_L, ddl_c_L, f_cm, f_bw)


def plot(out, L, val, err, latent, dlatent, dl_c_L, ddl_c_L, f_cm, f_bw):
    fig, axes = P.figure(3, 1, width=3.3, height=2.8)
    ax = axes[0]
    shown = [o for o in out if o["L"] in (8, 16, 24, 32, 48)] or out[-5:]
    for o, col in zip(shown, P.ordinal_colors(len(shown))):
        ok = o["p"] > 0
        ax.plot(
            o["e"][ok], o["p"][ok] * (o["L"] ** 2), color=col, linewidth=1.4, label=f"L = {o['L']}"
        )
    eo, ed = xpotts.phase_energies(Q)
    for x in (eo, ed):
        ax.axvline(x, color=P.INK_2, linewidth=0.9)
    ax.annotate(
        "exact e_o, e_d",
        (eo, 2e-4),
        xytext=(4, 0),
        textcoords="offset points",
        fontsize=7.5,
        color=P.INK_2,
    )
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 300)
    ax.set_xlim(-1.9, -0.65)
    ax.set_xlabel("energy per site e")
    ax.set_ylabel("P(e) at equal height")
    ax.set_title("a  Two coexisting phases, q = 10")
    ax.legend(loc="upper center", fontsize=7.5, ncols=3)

    ax = axes[1]
    x = 1.0 / L
    P.errorbar(
        ax,
        x,
        dl_c_L,
        ddl_c_L,
        color=P.CATEGORICAL[0],
        line=False,
        label="from c_max: (2/β)√(c_max/N)",
    )
    P.errorbar(
        ax,
        x,
        latent,
        dlatent,
        color=P.CATEGORICAL[1],
        line=False,
        marker="s",
        label="phase-conditional means",
    )
    xx = np.linspace(1e-3, x.max() * 1.05, 100)
    a, b = f_cm.params
    ax.plot(xx, 2.0 * np.sqrt(a + b * xx**2) / BC, color=P.CATEGORICAL[0], linewidth=1.1, alpha=0.8)
    exact_dl = xpotts.latent_heat(Q)
    ax.axhline(exact_dl, color=P.INK_2, linewidth=0.9)
    ax.annotate(
        f"Baxter {exact_dl:.4f}",
        (x.max() * 1.05, exact_dl),
        xytext=(-2, -11),
        textcoords="offset points",
        fontsize=7.5,
        color=P.INK_2,
        ha="right",
    )
    ax.set_xlim(0, x.max() * 1.08)
    ax.set_xlabel("1 / L")
    ax.set_ylabel("latent heat Δe")
    ax.set_title("b  Latent heat")
    ax.legend(loc="upper left", fontsize=7.5)

    ax = axes[2]
    x2 = 1.0 / L**2
    P.errorbar(
        ax,
        x2 * 1e3,
        (val[:, 1] - BC) * 1e3,
        err[:, 1] * 1e3,
        color=P.CATEGORICAL[0],
        line=False,
        label="equal-weight β(L)",
    )
    xx = np.linspace(0, x2.max() * 1.05, 100)
    ax.plot(
        xx * 1e3,
        (f_bw.params[0] + f_bw.params[1] * xx - BC) * 1e3,
        color=P.CATEGORICAL[0],
        linewidth=1.1,
        alpha=0.8,
        label="β_c + a / L²",
    )
    ax.axhline(0, color=P.INK_2, linewidth=0.9)
    ax.annotate(
        "exact β_c = ln(1 + √10)",
        (x2.max() * 1e3 * 1.05, 0),
        xytext=(-2, -11),
        textcoords="offset points",
        fontsize=7.5,
        color=P.INK_2,
        ha="right",
    )
    ax.set_xlabel("1 / L²  (× 10⁻³)")
    ax.set_ylabel("(β − β_c) × 10³")
    ax.set_title("c  Transition point")
    ax.legend(loc="lower left", fontsize=7.5)
    P.save(fig, figure_path(NAME, "overview"))


if __name__ == "__main__":
    main()
