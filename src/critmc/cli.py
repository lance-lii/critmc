"""Command-line interface: ``critmc <command> ...``.

Commands
--------
``simulate``  run one Markov chain and print observables with Gamma-method errors
``analyze``   re-analyse a saved time series
``exact``     exact results: finite-torus thermodynamics, density of states, Fisher zeros
``bench``     kernel throughput
``info``      versions and available models
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from . import __version__
from .lattice import make_lattice
from .models import ON, Ising, Potts
from .sim import Simulation, TimeSeries


def _model(args):
    lat = make_lattice(args.lattice, args.L)
    if args.model == "ising":
        return Ising(lat, h=args.field)
    if args.model == "potts":
        return Potts(lat, args.q)
    if args.model in ("xy", "heisenberg", "on"):
        n = {"xy": 2, "heisenberg": 3}.get(args.model, args.n)
        return ON(lat, n)
    raise SystemExit(f"unknown model {args.model}")


def _summary(ts: TimeSeries, out=sys.stdout) -> None:
    from .fss import format_value
    from .stats import gamma_method

    beta = ts.beta
    n = ts.n_sites
    rows = []
    for key in ("e", "absm", "m2"):
        r = gamma_method(ts[key])
        rows.append((f"<{key}>", format_value(r.value, r.error), f"{r.tau_int:.2f}"))
    e = np.asarray(ts["E"], dtype=float)
    stacked = np.column_stack([e, e * e])
    c = gamma_method(stacked, lambda a: beta**2 * (a[1] - a[0] ** 2) / n)
    rows.append(("c", format_value(c.value, c.error), f"{c.tau_int:.2f}"))
    m = np.column_stack([ts["m2"], ts["m4"]])
    u = gamma_method(m, lambda a: 1.0 - a[1] / (3.0 * a[0] ** 2))
    rows.append(("U4", format_value(u.value, u.error), f"{u.tau_int:.2f}"))
    chi = gamma_method(m[:, :1], lambda a: beta * n * a[0])
    rows.append(("chi0", format_value(chi.value, chi.error), f"{chi.tau_int:.2f}"))
    if "stiff_c" in ts.raw:
        sc = ts["stiff_c"].mean(axis=1)
        ss2 = (ts["stiff_s"] ** 2).mean(axis=1)
        ups = gamma_method(np.column_stack([sc, ss2]), lambda a: (a[0] - beta * a[1]) / n)
        rows.append(("Upsilon", format_value(ups.value, ups.error), f"{ups.tau_int:.2f}"))
    meta = ts.meta
    name = {"on": f"O({meta.get('n')})", "potts": f"Potts(q={meta.get('q')})"}.get(
        meta["model"], meta["model"].capitalize()
    )
    print(
        f"{name} on {meta['lattice']} L={meta['L']}  beta={beta:.6f}  "
        f"updates={meta['updates']}  n={len(ts)}",
        file=out,
    )
    print(f"{'observable':>10} {'value':>18} {'tau_int':>9}", file=out)
    for name, val, tau in rows:
        print(f"{name:>10} {val:>18} {tau:>9}", file=out)
    if meta.get("site_updates_per_s"):
        print(f"throughput: {meta['site_updates_per_s']:.3g} site updates/s", file=out)


def cmd_simulate(args) -> int:
    model = _model(args)
    beta = 1.0 / args.T if args.T is not None else args.beta
    sim = Simulation(model, beta, args.updates, seed=args.seed, start=args.start)
    sim.thermalize(args.therm)
    ts = sim.run(args.meas)
    _summary(ts)
    if args.out:
        print(f"saved to {ts.save(args.out)}")
    return 0


def cmd_analyze(args) -> int:
    for path in args.files:
        _summary(TimeSeries.load(path))
    return 0


def cmd_exact(args) -> int:
    from .exact import dos, kaufman, onsager, zeros

    if args.what == "thermo":
        temps = np.asarray(args.T, dtype=float)
        print(f"{'T':>10} {'u':>16} {'c':>16} {'f':>16}")
        if args.L:
            th = kaufman.thermodynamics(1.0 / temps, args.L)
            for T, u, c, f in zip(temps, th["u"], th["c"], th["f"]):
                print(f"{T:10.6f} {u:16.12f} {c:16.12f} {f:16.12f}")
        else:
            for T in temps:
                print(
                    f"{T:10.6f} {float(onsager.energy(T)):16.12f} "
                    f"{float(onsager.specific_heat(T)):16.12f} {onsager.free_energy(T):16.12f}"
                )
    elif args.what == "dos":
        d = dos.ising_dos(args.L)
        if args.out:
            d.save(args.out)
            print(f"saved {len(d.g)} levels to {args.out}")
        else:
            for e, g in zip(d.energies.astype(int), d.g):
                if g:
                    print(e, g)
    elif args.what == "zeros":
        print(f"{'L':>6} {'Re u1':>14} {'Im u1':>14} {'|u1 - u_c|':>14}")
        for z in zeros.leading_zeros(args.sizes):
            print(f"{z.L:6d} {z.u.real:14.10f} {z.u.imag:14.10f} {z.distance:14.8e}")
    return 0


def cmd_bench(args) -> int:
    from .bench import default_suite

    print(f"{'system':>14} {'updates':>18} {'N':>7} {'site updates/s':>16}")
    for r in default_suite(args.L, args.meas):
        print(f"{r.label:>14} {r.updates:>18} {r.n_sites:7d} {r.site_updates_per_s:16.3e}")
    return 0


def cmd_info(args) -> int:
    import numba
    import scipy

    print(
        f"critmc {__version__}  (numpy {np.__version__}, scipy {scipy.__version__}, "
        f"numba {numba.__version__})"
    )
    print("models: ising, potts, xy, heisenberg (O(n)), Edwards-Anderson spin glass")
    print("updates:", ", ".join(sorted({*Ising.updates, *Potts.updates, *ON.updates})))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="critmc", description=__doc__.splitlines()[0])
    p.add_argument("--version", action="version", version=f"critmc {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("simulate", help="run a single Markov chain")
    s.add_argument("--model", default="ising", choices=["ising", "potts", "xy", "heisenberg", "on"])
    s.add_argument(
        "--lattice", default="square", choices=["chain", "square", "cubic", "triangular"]
    )
    s.add_argument("--L", type=int, default=32)
    s.add_argument("--q", type=int, default=3, help="Potts states")
    s.add_argument("--n", type=int, default=2, help="O(n) components")
    s.add_argument("--field", type=float, default=0.0, help="Ising magnetic field")
    temp = s.add_mutually_exclusive_group()
    temp.add_argument("--T", type=float, help="temperature")
    temp.add_argument("--beta", type=float, default=0.4406867935, help="inverse temperature")
    s.add_argument("--updates", default="wolff")
    s.add_argument("--therm", type=int, default=2000)
    s.add_argument("--meas", type=int, default=20000)
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--start", default="random", choices=["random", "ordered"])
    s.add_argument("--out", type=Path)
    s.set_defaults(func=cmd_simulate)

    a = sub.add_parser("analyze", help="analyse saved time series")
    a.add_argument("files", nargs="+", type=Path)
    a.set_defaults(func=cmd_analyze)

    e = sub.add_parser("exact", help="exact results for the square-lattice Ising model")
    esub = e.add_subparsers(dest="what", required=True)
    t = esub.add_parser("thermo", help="u, c, f of an L x L torus (or L -> infinity)")
    t.add_argument("--L", type=int, default=0, help="linear size (0 = thermodynamic limit)")
    t.add_argument("--T", type=float, nargs="+", required=True)
    d = esub.add_parser("dos", help="exact integer density of states (even L)")
    d.add_argument("--L", type=int, required=True)
    d.add_argument("--out", type=Path)
    z = esub.add_parser("zeros", help="leading Fisher zeros for a sequence of even sizes")
    z.add_argument("sizes", type=int, nargs="+")
    e.set_defaults(func=cmd_exact)

    b = sub.add_parser("bench", help="kernel throughput")
    b.add_argument("--L", type=int, default=64)
    b.add_argument("--meas", type=int, default=200)
    b.set_defaults(func=cmd_bench)

    i = sub.add_parser("info", help="versions and capabilities")
    i.set_defaults(func=cmd_info)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
