"""Generate docs/results_tables.md from results/*.json.

Every number quoted in the report is produced here from the stored results,
so nothing is transcribed by hand.

    python experiments/make_tables.py [--quick]
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from common import RESULTS, ROOT

from critmc.fss import format_value as fv


def load(name: str, quick: bool):
    path = RESULTS / f"{name}{'_quick' if quick else ''}.json"
    return json.loads(path.read_text()) if path.exists() else None


def table(header: list[str], rows: list[list[str]]) -> str:
    def cell(c) -> str:
        return str(c).replace("|", "\\|")  # literal bars (as in |m|) would split the column

    out = [
        "| " + " | ".join(cell(h) for h in header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
    ]
    out += ["| " + " | ".join(cell(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def pv(pair) -> str:
    return fv(float(pair[0]), float(pair[1]))


def dev(value, error, exact) -> str:
    """Deviation from a reference value in units of the statistical error."""
    value, error = float(value), float(error)  # non-finite values are stored as strings
    if not np.isfinite(error) or error <= 0 or not np.isfinite(value):
        return ""
    return f"{(value - exact) / error:+.1f}σ"


def e01(r) -> str:
    rows = []
    names = {
        "metropolis": "Metropolis",
        "heatbath": "Heat bath",
        "sw": "Swendsen–Wang",
        "wolff": "Wolff",
    }
    for key, a in r["algorithms"].items():
        rows.append(
            [
                names.get(key, key),
                f"{a['pull_mean']:+.2f}",
                f"{a['pull_std']:.2f}",
                f"{a['chi2']:.1f} / {a['dof']}",
                f"{a['ks_p']:.2f}",
            ]
        )
    p = r["pooled"]
    rows.append(
        ["**pooled**", f"{p['mean']:+.2f}", f"{p['std']:.2f}", f"— / {p['n']}", f"{p['ks_p']:.2f}"]
    )
    return table(["algorithm", "mean pull", "sd of pulls", "χ² / dof", "KS p-value"], rows)


def e02(r) -> str:
    s = "**Exact densities of states**\n\n"
    rows = [
        [
            L,
            d["levels"],
            f"10^{d['log10_gmax']:.0f}",
            "cached" if d.get("cached") else f"{d['seconds']:.1f}",
            "✓" if d["check_sum_rule"] else "✗",
        ]
        for L, d in r["dos"].items()
    ]
    s += table(["L", "occupied levels", "largest g(E)", "time (s)", "Σg = 2^N"], rows)
    s += "\n**Leading Fisher zeros** (local exponent from consecutive sizes)\n\n"
    zs = r["leading_zeros"]
    loc = [None] + list(r["local_inv_nu"])
    rows = [
        [
            z["L"],
            f"{z['re']:.10f}",
            f"{z['im']:.10f}",
            f"{z['distance']:.6e}",
            "" if lo is None else f"{lo:.5f}",
        ]
        for z, lo in zip(zs, loc)
    ]
    s += table(["L", "Re u₁", "Im u₁", "|u₁ − u_c|", "local 1/ν"], rows)
    # the same maxima fitted without the 1/L corrections, to show their size
    peaks = [c for c in r["c_max"] if c["L"] >= 16]
    naive = np.polyfit(np.log([c["L"] for c in peaks]), [c["c_max"] for c in peaks], 1)[0]
    s += (
        f"\nSpecific-heat amplitude from exact c_max(L), L = 16–2048, fitted with "
        f"A ln L + B + (C ln L + D)/L: **A = {r['c_amplitude_fit']:.6f}**; exact 8K_c²/π = "
        f"{r['c_amplitude_exact']:.6f}; a plain A ln L + B fit to the same maxima gives "
        f"{naive:.4f}.\n"
    )
    return s


def e03(r) -> str:
    ex = r["exact"]
    rows = []

    def add(q, method, pair, exact):
        rows.append([q, method, pv(pair), f"{exact:.6g}", dev(pair[0], pair[1], exact)])

    if "beta_c_from_crossings" in r:
        b = r["beta_c_from_crossings"]
        add("β_c", "Binder crossings, drift ∝ L⁻³", [b["value"], b["error"]], ex["beta_c"])
    add("β_c", "χ′ peak shift", r["peak_fits"]["shift"]["beta_c"], ex["beta_c"])
    add("β_c", "Binder collapse", r["collapse"]["binder"]["beta_c"], ex["beta_c"])
    f = r["fits_at_beta_c"]
    add("1/ν", "dU/dβ at β_c", [f["dU"]["exponent"], f["dU"]["error"]], 1.0)
    add("1/ν", "d ln⟨m²⟩/dβ at β_c", [f["dlogm2"]["exponent"], f["dlogm2"]["error"]], 1.0)
    if "background" in f["dlogm2"]:
        add("1/ν", "d ln⟨m²⟩/dβ, with background", f["dlogm2"]["background"][:2], 1.0)
    add("1/ν", "χ′ peak shift", r["peak_fits"]["shift"]["inv_nu"], 1.0)
    add("1/ν", "Binder collapse", r["collapse"]["binder"]["inv_nu"], 1.0)
    add("γ/ν", "χ at β_c", [f["chi0"]["exponent"], f["chi0"]["error"]], ex["gamma_over_nu"])
    if "background" in f["chi0"]:
        add("γ/ν", "χ at β_c, with background", f["chi0"]["background"][:2], ex["gamma_over_nu"])
    g = r["peak_fits"]["gamma_over_nu"]
    add("γ/ν", "χ′ peak height", [g[0], g[1]], ex["gamma_over_nu"])
    add("γ/ν", "χ collapse", r["collapse"]["chi"]["gamma_over_nu"], ex["gamma_over_nu"])
    add("β/ν", "⟨|m|⟩ at β_c", [-f["absm"]["exponent"], f["absm"]["error"]], ex["beta_over_nu"])
    add("U*", "largest crossing pair", r["U_star_largest"], ex["U_star"])
    c = r["peak_fits"]["c_log_amplitude"]
    add("A (c ~ A ln L)", "specific-heat maxima, + D/L", [c[0], c[1]], ex["c_amplitude"])
    s = table(["quantity", "estimator", "Monte Carlo", "exact", "deviation"], rows)
    s += "\n**Observables at the exact β_c** (c compared with Kaufman's exact finite-L value)\n\n"
    rows = []
    for L, o in r["at_beta_c"].items():
        c_ex = r["c_exact_at_beta_c"][L]
        rows.append(
            [
                L,
                pv(o["U"]),
                pv(o["dU"]),
                pv(o["chi0"]),
                pv(o["absm"]),
                pv(o["c"]),
                f"{c_ex:.5f}",
                dev(o["c"][0], o["c"][1], c_ex),
            ]
        )
    s += table(["L", "U₄", "dU/dβ", "χ", "⟨|m|⟩", "c (MC)", "c (exact)", "dev."], rows)
    if "c_max_pulls" in r["peak_fits"]:
        pulls = ", ".join(f"{v:+.2f}" for v in r["peak_fits"]["c_max_pulls"])
        s += f"\nSpecific-heat maxima against Kaufman's exact c_max(L), pulls by size: {pulls}.\n"
    s += "\n**Binder crossings**\n\n"
    rows = [[f"({c['L']}, {c['L2']})", pv(c["beta"]), pv(c["U"])] for c in r["crossings"]]
    s += table(["pair", "β*", "U*"], rows)
    return s


def e04(r) -> str:
    names = {
        "metropolis": "Metropolis",
        "heatbath": "Heat bath",
        "sw": "Swendsen–Wang",
        "wolff": "Wolff",
    }
    rows = []
    for algo, f in r["fits"].items():
        lit = r["literature"].get(algo)
        rows.append(
            [
                names[algo],
                fv(f["e"]["z"], f["e"]["dz"]),
                fv(f["m2"]["z"], f["m2"]["dz"]),
                f"{int(f['e']['L_min'])}",
                "" if lit is None else pv(lit),
            ]
        )
    s = table(["algorithm", "z_int (energy)", "z_int (m²)", "L_min of fit", "literature z"], rows)
    s += "\n**Integrated autocorrelation times (sweeps)**\n\n"
    sizes = sorted({L for t in r["table"].values() for L in t["L"]})
    rows = []
    for algo, t in r["table"].items():
        d = dict(zip(t["L"], zip(t["tau_e"], t["dtau_e"])))
        rows.append([names[algo]] + [fv(*d[L]) if L in d else "—" for L in sizes])
    s += table(["τ_int(E)"] + [f"L={L}" for L in sizes], rows)
    return s


def e05(r) -> str:
    rows = []
    for v, label in (("original", "Wang–Landau"), ("1/t", "1/t refinement")):
        d = r[v]
        rows.append(
            [
                label,
                f"{d['final_mean_error']:.2e}",
                f"{d['late_slope']:+.2f}",
                f"{d['c_max_abs_dev']:.1e}",
            ]
        )
    return table(
        [
            "algorithm",
            f"mean |Δ ln g| at t = {r['t_max']:.0e}",
            "late slope of error",
            "max |Δc| (T = 1.5–3.5)",
        ],
        rows,
    )


def e06(r) -> str:
    lh = r["latent_heat"]
    per_l = dict(
        zip([int(v) for v in lh["L"]], zip(lh["cmax_per_L"]["value"], lh["cmax_per_L"]["error"]))
    )
    rows = []
    for L, t in r["table"].items():
        lat = [
            t["e_disordered"][0] - t["e_ordered"][0],
            np.hypot(t["e_disordered"][1], t["e_ordered"][1]),
        ]
        rows.append(
            [
                L,
                pv(t["beta_eqw"]),
                pv(t["e_ordered"]),
                pv(t["e_disordered"]),
                pv(lat),
                pv(per_l[int(L)]),
                pv(t["log_ratio"]),
            ]
        )
    s = table(
        [
            "L",
            "β_eqw",
            "⟨e⟩_ordered",
            "⟨e⟩_disordered",
            "Δe (means)",
            "Δe (c_max)",
            "ln(P_max/P_min)",
        ],
        rows,
    )
    bc = r["beta_c"]
    ex = lh["exact"]

    def row(q, method, pair, exact, fmt):
        return [q, method, pv(pair), fmt.format(exact), dev(pair[0], pair[1], exact)]

    s += "\n"
    s += table(
        ["quantity", "estimator", "Monte Carlo", "exact", "deviation"],
        [
            row("latent heat", "c_max/N = (βΔe/2)² + B/N, L ≥ 16", lh["cmax_route"], ex, "{:.6f}"),
            row(
                "latent heat",
                "phase-conditional means, L ≥ 24 plateau",
                lh["plateau_L24"],
                ex,
                "{:.6f}",
            ),
            row("latent heat", "phase-conditional means, Δe + b/L", lh["fit_1overL"], ex, "{:.6f}"),
            row("β_c", "equal weight, β_c + b/L²", bc["equal_weight_fit"], bc["exact"], "{:.7f}"),
            row(
                "β_c", "equal height, + b/L² + c/L⁴", bc["equal_height_fit"], bc["exact"], "{:.7f}"
            ),
            ["2σ_od", "ln(P_max/P_min)/L, + b/L", pv(r["interface"]["two_sigma_fit"]), "—", ""],
        ],
    )
    return s


def e07(r) -> str:
    rows = [[L, pv(c)] for L, c in r["crossings"].items()]
    s = table(["L", "T*(L): Υ = 2T/π"], rows)
    ref = r["reference"]["T_bkt"]
    t2 = r["T_bkt_crossing_fit"]
    rows = [
        [
            "crossings, T* = T_BKT + a/(ln L)²",
            pv(t2),
            f"{t2[2]:.1f} / {t2[3]}",
            dev(t2[0], t2[1], ref),
        ]
    ]
    t3 = r.get("T_bkt_crossing_fit_lnL0")
    if t3:
        rows.append(
            [
                "crossings, T* = T_BKT + a/(ln L + c)²",
                pv(t3),
                f"{t3[2]:.1f} / {t3[3]}",
                dev(t3[0], t3[1], ref),
            ]
        )
    wm = r["weber_minnhagen"]
    if "error" in wm:
        rows.append(
            [
                "Weber–Minnhagen, fine grid + bootstrap",
                fv(wm["T_bkt"], wm["error"]),
                "—",
                dev(wm["T_bkt"], wm["error"], ref),
            ]
        )
    s += "\n" + table(["estimator", "T_BKT", "χ² / dof", f"deviation from {ref}"], rows)
    e = r["eta_eff"]
    s += (
        f"\nEffective η from χ ~ L^(2−η) at T = {e[2]}: {fv(e[0], e[1])} "
        "(1/4 at T_BKT, with logarithmic corrections).\n"
    )
    return s


def e08(r) -> str:
    ref = r["reference"]
    f = r["fits"]
    rows = [
        [
            "β_c",
            "Binder crossings",
            pv(r["beta_c"]),
            f"{ref['beta_c']:.9f}",
            dev(r["beta_c"][0], r["beta_c"][1], ref["beta_c"]),
        ],
        [
            "U*",
            "crossings, + L^−ω",
            pv(r["U_star"]),
            f"{ref['U_star']:.4f}",
            dev(r["U_star"][0], r["U_star"][1], ref["U_star"]),
        ],
    ]
    for key, q, refv, sign in (
        ("dU", "1/ν", ref["1/nu"], 1),
        ("dlogm2", "1/ν", ref["1/nu"], 1),
        ("chi0", "γ/ν", ref["gamma/nu"], 1),
        ("absm", "β/ν", ref["beta/nu"], -1),
    ):
        for kind, label in (("plain", "L ≥ 12"), ("corrected", "all L, + L^−ω")):
            v, e = sign * f[key][kind][0], f[key][kind][1]
            rows.append([q, f"{key}, {label}", fv(v, e), f"{refv:.5f}", dev(v, e, refv)])
    s = table(["quantity", "estimator", "Monte Carlo", "literature", "deviation"], rows)
    s += "\n"
    s += table(
        ["pair", "β*", "U*"],
        [[f"({c['L']}, {c['L2']})", pv(c["beta"]), pv(c["U"])] for c in r["crossings"]],
    )
    return s


def e09(r) -> str:
    rows = [[L, c["samples"], len(c["temperatures"])] for L, c in r["config"].items()]
    s = table(["L", "disorder samples", "temperatures"], rows)
    s += "\n"
    rows = []
    for c in r["crossings"]:
        t, e = float(c["T"]), float(c["error"])
        cell = (
            fv(t, e)
            if np.isfinite(t) and np.isfinite(e)
            else ("not found" if not np.isfinite(t) else f"{t:.3f}")
        )
        rows.append([f"({c['L1']}, {c['L2']})", c["observable"], cell])
    s += table(["pair", "observable", "crossing temperature"], rows)
    if "scaling_fit" in r:
        sf = r["scaling_fit"]
        lo, hi = sf["window"]
        s += "\n" + table(
            ["joint scaling fit of ξ_L/L", "Monte Carlo", "literature", "deviation"],
            [
                [f"T_c (T ∈ [{lo}, {hi}])", pv(sf["T_c"]), "0.951(9)", dev(*sf["T_c"], 0.951)],
                ["ν", pv(sf["nu"]), "2.44(9)", dev(*sf["nu"], 2.44)],
                ["χ² / dof", f"{sf['chi2']:.1f} / {sf['dof']}", "", ""],
            ],
        )
    s += f"\nLiterature: {r['reference']['source']}.\n"
    return s


def e10(r) -> str:
    rows = [
        [b["label"], b["updates"], b["n_sites"], f"{b['site_updates_per_s'] / 1e6:.1f}"]
        for b in r["suite"]
    ]
    return table(["system", "updates", "N", "10⁶ site updates / s"], rows)


SECTIONS = [
    ("e01_validation", "E01 — validation against the exact finite-lattice solution", e01),
    ("e02_exact", "E02 — exact results", e02),
    ("e03_ising2d_fss", "E03 — finite-size scaling of the 2D Ising model", e03),
    ("e04_dynamics", "E04 — dynamic critical exponents", e04),
    ("e05_wang_landau", "E05 — Wang–Landau convergence", e05),
    ("e06_potts_first_order", "E06 — first-order transition of the 10-state Potts model", e06),
    ("e07_xy_bkt", "E07 — BKT transition of the XY model", e07),
    ("e08_ising3d_fss", "E08 — 3D Ising model", e08),
    ("e09_spin_glass", "E09 — 3D Edwards–Anderson spin glass", e09),
    ("e10_benchmarks", "E10 — kernel throughput", e10),
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    parts = [
        "# Results tables\n",
        "Generated by `experiments/make_tables.py` from `results/*.json`; do not edit by hand.\n",
    ]
    for name, title, fn in SECTIONS:
        r = load(name, args.quick)
        if r is None:
            continue
        meta = r["meta"]
        parts.append(f"\n## {title}\n")
        # no wall time here: re-runs reuse cached raw data, so it says little about the cost
        parts.append(f"_critmc {meta['critmc_version']}, seed {meta['seed']}._\n\n")
        parts.append(fn(r))
    out = ROOT / "docs" / f"results_tables{'_quick' if args.quick else ''}.md"
    out.write_text("".join(parts), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
