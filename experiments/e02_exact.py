"""E02 - Exact finite-size results for the square-lattice Ising model.

* Exact integer densities of states up to L = 64 (Kaufman's formula continued
  to complex temperature + arbitrary-precision FFT) and the resulting entropy
  densities, which converge to Onsager's s(e).
* Complete Fisher-zero maps, which approach the circles |u +- 1| = sqrt 2.
* Leading Fisher zeros up to L = 512: |u_1(L) - u_c| ~ L^(-1/nu) gives nu
  without any simulation.
* Specific-heat maxima c_max(L) ~ A ln L with the exact A = 8 K_c^2 / pi.
"""

from __future__ import annotations

import time

import numpy as np
from common import banner, data_dir, figure_path, parse_args, save_results
from scipy import optimize

from critmc import plotting as P
from critmc.exact import dos, kaufman, onsager, zeros
from critmc.fss import fit, fit_power_law

NAME = "e02_exact"


def cmax(L):
    res = optimize.minimize_scalar(
        lambda b: -kaufman.specific_heat(b, L),
        bounds=(0.3, 0.6),
        method="bounded",
        options={"xatol": 1e-12},
    )
    return float(res.x), float(-res.fun)


def main():
    args = parse_args(__doc__)
    started = time.time()
    cache = data_dir("exact_dos")
    sizes = [4, 8, 16, 32] if args.quick else [4, 8, 16, 32, 64]

    banner("exact densities of states")
    dos_by_L = {}
    dos_info = {}
    for L in sizes:
        t0 = time.time()
        path = cache / f"ising_dos_{L}x{L}.txt"
        cached = path.exists() and not args.fresh
        d = dos.ising_dos(L, cache_dir=cache) if cached else dos.ising_dos(L)
        if not cached:
            d.save(path)
        dos_by_L[L] = d
        gmax = max(d.g)
        dos_info[L] = {
            "levels": int(sum(1 for g in d.g if g)),
            "log10_gmax": float(len(str(gmax)) - 1),
            "seconds": round(time.time() - t0, 2),
            "cached": cached,
            "check_sum_rule": sum(d.g) == 2 ** (L * L),
        }
        print(
            f"  L={L:3d}: {dos_info[L]['levels']} levels, g_max ~ 1e{dos_info[L]['log10_gmax']:.0f}, "
            f"{dos_info[L]['seconds']} s"
        )

    banner("Fisher zeros")
    maps = {}
    for L in [6, 8] if args.quick else [8, 12]:
        t0 = time.time()
        maps[L] = zeros.fisher_zeros(dos.ising_dos(L, cache_dir=cache))
        print(f"  all {len(maps[L])} zeros for L={L}: {time.time() - t0:.1f} s")
    lead_sizes = [4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]
    if args.quick:
        lead_sizes = lead_sizes[:10]
    lead = zeros.leading_zeros(lead_sizes)
    Ls = np.array([z.L for z in lead], dtype=float)
    dist = np.array([z.distance for z in lead])
    local = -np.diff(np.log(dist)) / np.diff(np.log(Ls))
    sel = Ls >= 32
    # exact data: fit with a small nominal relative error to obtain a least-squares slope
    fit_nu = fit_power_law(Ls[sel], dist[sel], 1e-6 * dist[sel], omega=1.0)
    print(
        f"  1/nu from leading zeros (L >= 32, with 1/L correction): {fit_nu['exponent'].value:+.6f}"
    )
    print(f"  local exponents: {np.round(local, 5)}")

    banner("specific-heat maxima")
    c_sizes = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    peaks = np.array([cmax(L) for L in c_sizes])
    lc = np.array(c_sizes, dtype=float)
    # Ferdinand-Fisher form: c_max = A ln L + B + (C ln L + D) / L
    fit_a = fit(
        lambda x, a, b, c, d: a * np.log(x) + b + (c * np.log(x) + d) / x,
        lc[1:],
        peaks[1:, 1],
        np.full(len(lc) - 1, 1e-7),
        [0.5, 0.1, 0.0, 0.0],
        ("a", "b", "c", "d"),
    )
    naive = (peaks[-1, 1] - peaks[-2, 1]) / np.log(2)
    print(
        f"  c_max(L) = A ln L + B + (C ln L + D)/L: A = {fit_a['a'].value:.6f}; "
        f"slope of the last pair {naive:.6f}; exact {onsager.heat_capacity_amplitude():.6f}"
    )

    summary = {
        "dos": dos_info,
        "leading_zeros": [
            {"L": z.L, "re": z.u.real, "im": z.u.imag, "distance": z.distance} for z in lead
        ],
        "local_inv_nu": local,
        "inv_nu_fit": {"value": fit_nu["exponent"].value, "L_min": 32},
        "c_max": [{"L": L, "beta_max": b, "c_max": c} for L, (b, c) in zip(c_sizes, peaks)],
        "c_amplitude_fit": fit_a["a"].value,
        "c_amplitude_last_pair": float(naive),
        "c_amplitude_exact": onsager.heat_capacity_amplitude(),
    }
    path = save_results(NAME, summary, args, started)
    print(f"  results -> {path}")
    if not args.no_plots:
        plot(dos_by_L, maps, lead, Ls, dist, fit_nu, c_sizes, peaks, fit_a)


def plot(dos_by_L, maps, lead, Ls, dist, fit_nu, c_sizes, peaks, fit_a):
    fig, axes = P.figure(2, 2, width=3.5, height=2.9)

    # a: entropy density
    ax = axes[0, 0]
    sizes = sorted(dos_by_L)[-4:]
    colors = P.ordinal_colors(len(sizes))
    for L, col in zip(sizes, colors):
        d = dos_by_L[L]
        lg = d.log_g
        ok = np.isfinite(lg)
        ax.plot(
            d.energies[ok] / d.n_sites,
            lg[ok] / d.n_sites,
            color=col,
            linewidth=1.4,
            label=f"L = {L}",
        )
    temps = np.concatenate([np.linspace(0.6, 2.25, 80), np.linspace(2.29, 60, 120)])
    u = onsager.energy(temps)
    f = onsager.free_energy(temps)
    s = (u - f) / temps
    ax.plot(u, s, color=P.INK_2, linewidth=1.0, label="Onsager (L → ∞)")
    ax.set_xlim(-2.05, 0.05)
    ax.set_ylim(-0.02, 0.72)
    ax.set_xlabel("energy per site e")
    ax.set_ylabel("entropy per site ln g / N")
    ax.set_title("a  Exact microcanonical entropy")
    ax.legend(loc="lower right")

    # b: Fisher zeros
    ax = axes[0, 1]
    phi = np.linspace(0, 2 * np.pi, 400)
    for sign in (1, -1):
        ax.plot(
            sign + np.sqrt(2) * np.cos(phi),
            np.sqrt(2) * np.sin(phi),
            color=P.AXIS,
            linewidth=0.9,
            zorder=1,
        )
    map_sizes = sorted(maps)
    for L, col, mk in zip(map_sizes, [P.ORDINAL[1], P.ORDINAL[3]], ["o", "o"]):
        z = maps[L]
        ax.plot(
            z.real,
            z.imag,
            linestyle="none",
            marker=mk,
            markersize=3.2 if L == map_sizes[-1] else 4.2,
            color=col,
            markeredgecolor=P.SURFACE,
            markeredgewidth=0.5,
            label=f"L = {L}",
            zorder=3,
        )
    ax.plot(
        [zeros.U_C],
        [0],
        marker="*",
        markersize=9,
        color=P.CATEGORICAL[1],
        markeredgecolor=P.SURFACE,
        linestyle="none",
        zorder=4,
    )
    ax.annotate(
        "u_c = √2 − 1",
        (zeros.U_C, 0),
        xytext=(8, -12),
        textcoords="offset points",
        fontsize=8,
        color=P.INK_2,
    )
    ax.set_aspect("equal")
    ax.set_xlim(-2.6, 2.6)
    ax.set_ylim(-1.6, 1.6)
    ax.set_xlabel("Re u")
    ax.set_ylabel("Im u")
    ax.set_title("b  Fisher zeros in u = exp(−2K)")
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.2), ncols=2, fontsize=7.5, markerscale=1.4
    )

    # c: leading zeros
    ax = axes[1, 0]
    P.errorbar(ax, Ls, dist, color=P.CATEGORICAL[0], line=False, label="|u₁(L) − u_c|")
    xx = np.geomspace(Ls.min(), Ls.max(), 50)
    a, b, c = fit_nu.params
    ax.plot(
        xx,
        a * xx**b * (1 + c / xx),
        color=P.INK_2,
        linewidth=1.0,
        label=f"fit L ≥ 32: 1/ν = {-b:.5f}",
    )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("linear size L")
    ax.set_ylabel("distance of leading zero")
    ax.set_title("c  Leading-zero scaling (exact)")
    ax.legend(loc="lower left")

    # d: specific heat
    ax = axes[1, 1]
    tt = np.linspace(1.9, 3.0, 500)
    shown = [8, 16, 32, 64, 128]
    for L, col in zip(shown, P.ordinal_colors(len(shown))):
        ax.plot(tt, kaufman.specific_heat(1.0 / tt, L), color=col, linewidth=1.4, label=f"L = {L}")
    tt2 = tt[np.abs(tt - onsager.T_C) > 0.004]
    ax.plot(
        tt2[tt2 < onsager.T_C],
        onsager.specific_heat(tt2[tt2 < onsager.T_C]),
        color=P.INK_2,
        linewidth=1.0,
        label="L → ∞",
    )
    ax.plot(
        tt2[tt2 > onsager.T_C],
        onsager.specific_heat(tt2[tt2 > onsager.T_C]),
        color=P.INK_2,
        linewidth=1.0,
    )
    ax.set_ylim(0, 3.2)
    ax.set_xlabel("temperature T")
    ax.set_ylabel("specific heat per site c")
    ax.set_title(f"d  c_max ~ A ln L: A = {fit_a['a'].value:.4f}, exact 0.4945")
    ax.legend(loc="upper right", fontsize=7.5)
    P.save(fig, figure_path(NAME, "overview"))


if __name__ == "__main__":
    main()
