"""E11 - Equilibrium configurations: what the transitions look like.

Ising configurations on a 256 x 256 torus below, at and above T_c (Wolff
updates), and XY configurations on a 64 x 64 torus below and above T_BKT with
their vortices (plaquettes whose angle differences wind by +-2 pi) marked.
Below T_BKT vortices are rare and bound in pairs; above it they proliferate.
"""

from __future__ import annotations

import time

import numpy as np
from common import banner, figure_path, parse_args, save_results

from critmc import XY, Ising, Simulation, square
from critmc import plotting as P
from critmc.exact import onsager


def vortices(theta: np.ndarray) -> np.ndarray:
    """Winding number of every plaquette of an L x L angle field (indexed [x, y])."""

    def wrap(d):
        return (d + np.pi) % (2 * np.pi) - np.pi

    a = theta
    b = np.roll(theta, -1, axis=0)  # x + 1
    c = np.roll(b, -1, axis=1)  # x + 1, y + 1
    d = np.roll(theta, -1, axis=1)  # y + 1
    w = wrap(b - a) + wrap(c - b) + wrap(d - c) + wrap(a - d)
    return np.rint(w / (2 * np.pi)).astype(int)


def main():
    args = parse_args(__doc__)
    started = time.time()
    L = 128 if args.quick else 256
    banner("Ising configurations")
    ising = {}
    for label, T in (("T = 2.0", 2.0), ("T = T_c", onsager.T_C), ("T = 2.6", 2.6)):
        sim = Simulation(Ising(square(L)), 1.0 / T, "wolff", seed=(args.seed, 11, int(100 * T)))
        sim.thermalize(500 if args.quick else 3000)
        ising[label] = sim.state.reshape(L, L, order="F").copy()
        print(f"  {label}: m = {sim.state.mean():+.3f}")
    banner("XY configurations")
    Lxy = 64
    xy = {}
    counts = {}
    for label, T in (("T = 0.75", 0.75), ("T = 1.10", 1.10)):
        sim = Simulation(
            XY(square(Lxy)),
            1.0 / T,
            "metropolis+2*overrelax+wolff",
            seed=(args.seed, 12, int(100 * T)),
        )
        sim.thermalize(500 if args.quick else 3000)
        theta = np.arctan2(sim.state[:, 1], sim.state[:, 0]).reshape(Lxy, Lxy, order="F")
        xy[label] = theta
        v = vortices(theta)
        counts[label] = {"vortices": int((v > 0).sum()), "antivortices": int((v < 0).sum())}
        print(f"  {label}: {counts[label]}")
    save_results(
        "e11_snapshots", {"xy_vortex_counts": counts, "ising_L": L, "xy_L": Lxy}, args, started
    )
    if not args.no_plots:
        plot(ising, xy)


def plot(ising, xy):
    from matplotlib.colors import ListedColormap

    fig, axes = P.figure(5, 1, width=2.3, height=2.45)
    two_tone = ListedColormap([P.SURFACE, P.ORDINAL[3]])
    for ax, (label, s) in zip(axes[:3], ising.items()):
        ax.imshow(s.T, cmap=two_tone, vmin=-1, vmax=1, interpolation="nearest", origin="lower")
        ax.set_title(f"Ising, {label}", fontsize=9.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    for ax, (label, theta) in zip(axes[3:], xy.items()):
        ax.imshow(
            theta.T,
            cmap="twilight",
            vmin=-np.pi,
            vmax=np.pi,
            interpolation="nearest",
            origin="lower",
        )
        v = vortices(theta)
        px, py = np.nonzero(v > 0)
        mx, my = np.nonzero(v < 0)
        ax.plot(
            px + 0.5,
            py + 0.5,
            "o",
            markersize=4.2,
            markerfacecolor=P.INK,
            markeredgecolor=P.SURFACE,
            markeredgewidth=0.8,
            linestyle="none",
            label="vortex",
        )
        ax.plot(
            mx + 0.5,
            my + 0.5,
            "s",
            markersize=4.0,
            markerfacecolor=P.SURFACE,
            markeredgecolor=P.INK,
            markeredgewidth=0.9,
            linestyle="none",
            label="antivortex",
        )
        ax.set_title(f"XY, {label}", fontsize=9.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    axes[4].legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncols=2, fontsize=7.5)
    P.save(fig, figure_path("e11_snapshots", "configurations"))


if __name__ == "__main__":
    main()
