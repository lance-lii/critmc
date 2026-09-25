"""E05 - Convergence of Wang-Landau sampling against the exact density of states.

The error of the estimated ln g(E) of the 16 x 16 Ising model is tracked as a
function of Monte Carlo time for the original algorithm (flatness 80 %,
ln f halved at every stage) and for the 1/t refinement of Belardinelli and
Pereyra.  The original algorithm saturates at a finite error, the 1/t
variant keeps converging like t^(-1/2).  Eight independent runs each.
"""

from __future__ import annotations

import time

import numpy as np
from common import banner, data_dir, figure_path, load_results, parse_args, save_results

from critmc import Ising, square
from critmc import plotting as P
from critmc.exact import dos, kaufman
from critmc.flathist import WangLandau
from critmc.runner import parallel_map

NAME = "e05_wang_landau"
L = 16


def run_one(job):
    variant, rep, seed, t_max, exact = job
    wl = WangLandau(
        Ising(square(L)),
        seed=np.random.SeedSequence(seed, spawn_key=(5, rep, int(variant == "1/t"))),
    )
    checkpoints = np.unique(np.geomspace(1e5, t_max, 36).astype(np.int64))
    res = wl.run(
        1e-15,
        one_over_t=(variant == "1/t"),
        max_trials=t_max,
        checkpoints=checkpoints,
        check_sweeps=50,
    )
    v = np.isfinite(exact)
    errors = []
    for t, ln_f, ln_g in res.checkpoints:
        est = res.normalized(ln_g)
        ok = v & np.isfinite(est)
        # levels not yet discovered count with the error of a flat guess
        err = np.abs(np.where(ok, est, 0.0) - exact)[v]
        errors.append((t, ln_f, float(err.mean()), float(err.max()), int(ok.sum())))
    final = res.normalized()
    return {
        "variant": variant,
        "rep": rep,
        "errors": errors,
        "stages": res.stages,
        "final_ln_g": final,
        "trials": res.trials,
    }


def main():
    args = parse_args(__doc__)
    if args.replot:
        plot(load_results(NAME, args))
        return
    started = time.time()
    exact = dos.ising_dos(L, cache_dir=data_dir("exact_dos")).log_g
    t_max = int(1e8 if args.quick else 1e10)
    n_rep = 4 if args.quick else 8
    jobs = [
        (variant, rep, args.seed, t_max, exact)
        for variant in ("original", "1/t")
        for rep in range(n_rep)
    ]
    banner(f"{NAME}: {len(jobs)} Wang-Landau runs up to {t_max:.0e} trials")
    out = parallel_map(run_one, jobs, workers=args.workers)

    summary = {"L": L, "t_max": t_max, "n_rep": n_rep, "levels": int(np.isfinite(exact).sum())}
    for variant in ("original", "1/t"):
        runs = [o for o in out if o["variant"] == variant]
        t = np.array([e[0] for e in runs[0]["errors"]], dtype=float)
        err = np.array([[e[2] for e in r["errors"]] for r in runs])
        late = t >= t_max / 100
        slope = np.polyfit(np.log(t[late]), np.log(err.mean(axis=0)[late]), 1)[0]
        final = np.array([r["final_ln_g"] for r in runs])
        summary[variant] = {
            "t": t,
            "mean_error": err.mean(axis=0),
            "sd_error": err.std(axis=0, ddof=1),
            "final_mean_error": float(err[:, -1].mean()),
            "late_slope": float(slope),
            "stages": len(runs[0]["stages"]),
        }
        # thermodynamics from the final estimates
        v = np.isfinite(exact)
        e = -2.0 * L * L + 4.0 * np.arange(len(exact))
        betas = 1.0 / np.linspace(1.5, 3.5, 81)
        cs = []
        for lg in final:
            a = lg[v][None, :] - betas[:, None] * e[v][None, :]
            p = np.exp(a - a.max(axis=1, keepdims=True))
            p /= p.sum(axis=1, keepdims=True)
            m1 = p @ e[v]
            cs.append(betas**2 * (p @ e[v] ** 2 - m1**2) / (L * L))
        cs = np.array(cs)
        summary[variant]["c_mean"] = cs.mean(axis=0)
        summary[variant]["c_sd"] = cs.std(axis=0, ddof=1)
        summary[variant]["c_max_abs_dev"] = float(
            np.abs(cs.mean(axis=0) - kaufman.specific_heat(betas, L)).max()
        )
        print(
            f"  {variant:>8}: final mean |d ln g| = {summary[variant]['final_mean_error']:.2e}, "
            f"late slope {slope:+.3f}, max |dc| = {summary[variant]['c_max_abs_dev']:.2e}"
        )
    summary["temperatures"] = 1.0 / betas
    summary["c_exact"] = kaufman.specific_heat(betas, L)
    path = save_results(NAME, summary, args, started)
    print(f"  results -> {path}")
    if not args.no_plots:
        plot(summary)


def plot(summary):
    """Figures from the (JSON-compatible) summary alone, so ``--replot`` needs no simulation."""
    n_levels = summary["levels"]
    labels = {"original": "Wang-Landau (ln f → ln f / 2)", "1/t": "1/t refinement"}
    fig, axes = P.figure(2, 1, width=3.6, height=2.9)
    ax = axes[0]
    for k, variant in enumerate(("original", "1/t")):
        d = summary[variant]
        mc_time = np.asarray(d["t"], dtype=float) / n_levels
        mean = np.asarray(d["mean_error"], dtype=float)
        sd = np.asarray(d["sd_error"], dtype=float)
        ax.fill_between(
            mc_time,
            np.clip(mean - sd, mean * 0.05, None),
            mean + sd,
            color=P.CATEGORICAL[k],
            alpha=0.15,
            linewidth=0,
        )
        ax.plot(mc_time, mean, color=P.CATEGORICAL[k], linewidth=1.6, label=labels[variant])
    d = summary["1/t"]
    t = np.asarray(d["t"], dtype=float) / n_levels
    ref_t = t[t >= t[-1] / 300]
    anchor = float(np.asarray(d["mean_error"])[-1])
    ax.plot(
        ref_t,
        1.8 * anchor * (ref_t / ref_t[-1]) ** -0.5,
        color=P.INK_2,
        linewidth=1.0,
        label="slope −1/2",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Monte Carlo time t  (trials / energy levels)")
    ax.set_ylabel("mean |ln g − ln g_exact|")
    ax.set_title("a  Error of ln g(E), 16×16 Ising")
    ax.legend(loc="lower left", fontsize=7.5)

    ax = axes[1]
    temps = np.asarray(summary["temperatures"], dtype=float)
    exact = np.asarray(summary["c_exact"], dtype=float)
    ax.axhline(0, color=P.INK_2, linewidth=0.9)
    for k, variant in enumerate(("original", "1/t")):
        d = summary[variant]
        dev = (np.asarray(d["c_mean"]) - exact) * 1e3
        sd = np.asarray(d["c_sd"]) * 1e3
        ax.fill_between(temps, dev - sd, dev + sd, color=P.CATEGORICAL[k], alpha=0.15, linewidth=0)
        ax.plot(temps, dev, color=P.CATEGORICAL[k], linewidth=1.4, label=labels[variant])
    ax.set_xlabel("temperature T")
    ax.set_ylabel("(c − c_exact) × 10³")
    ax.set_title("b  Specific heat from g(E): deviation")
    ax.legend(loc="upper right", fontsize=7.5)
    P.save(fig, figure_path(NAME, "convergence"))


if __name__ == "__main__":
    main()
