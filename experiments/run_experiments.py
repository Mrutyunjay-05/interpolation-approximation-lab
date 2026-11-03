"""Run experiments A-D and save the figures and error tables.

Usage (from the project root):
    python experiments/run_experiments.py
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")  # draw to files, no window needed

import matplotlib.pyplot as plt
import numpy as np

# Make the project root importable when running this file directly.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from interp.errors import (  # noqa: E402
    chebyshev_nodes,
    equispaced_nodes,
    error_table,
    format_error_table,
)
from interp.plotting import (  # noqa: E402
    plot_error_curves,
    plot_error_vs_n,
    plot_interpolation,
    save_figure,
)

OUT = os.path.join(ROOT, "figures")


def runge(x):
    return 1.0 / (1.0 + 25.0 * x**2)


def wide_runge(x):
    return 1.0 / (1.0 + x**2)


def save_table(rows, name):
    text = format_error_table(rows)
    with open(os.path.join(OUT, name), "w") as fh:
        fh.write(text + "\n")
    print(text)


def experiment_a():
    """A: smooth function (sin) with a few node counts."""
    a, b = 0.0, 2 * np.pi
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, n in zip(axes, [4, 6, 8]):
        plot_interpolation(np.sin, equispaced_nodes(a, b, n), a, b, ax=ax,
                           title=f"sin(x), n = {n} equispaced nodes")
    save_figure(fig, os.path.join(OUT, "A_smooth_function.png"))


def experiment_b():
    """B: maximum error as the number of nodes grows."""
    a, b = 0.0, 2 * np.pi
    rows = error_table(np.sin, a, b, range(3, 31, 3))
    print("Experiment B: sin(x) on [0, 2*pi]")
    save_table(rows, "B_error_table.txt")
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_error_vs_n(rows, ax=ax, title="Error vs number of nodes: sin(x)")
    save_figure(fig, os.path.join(OUT, "B_error_vs_n.png"))


def experiment_c():
    """C: equispaced vs Chebyshev nodes on the same function."""
    a, b, n = -5.0, 5.0, 11
    eq, ch = equispaced_nodes(a, b, n), chebyshev_nodes(a, b, n)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    plot_interpolation(wide_runge, eq, a, b, ax=axes[0], ylim=(-0.6, 1.4),
                       title=f"1/(1+x^2): {n} equispaced nodes")
    plot_interpolation(wide_runge, ch, a, b, ax=axes[1], ylim=(-0.6, 1.4),
                       title=f"1/(1+x^2): {n} Chebyshev nodes")
    plot_error_curves(wide_runge, a, b, {"equispaced": eq, "Chebyshev": ch},
                      ax=axes[2], title="Error |f - p| (log scale)")
    save_figure(fig, os.path.join(OUT, "C_node_placement.png"))


def experiment_d():
    """D: Runge function, equispaced (top row) vs Chebyshev (bottom row)."""
    a, b = -1.0, 1.0
    n_values = [5, 10, 15, 20]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for col, n in enumerate(n_values):
        plot_interpolation(runge, equispaced_nodes(a, b, n), a, b, ax=axes[0, col],
                           ylim=(-1, 2), title=f"Equispaced, n = {n}")
        plot_interpolation(runge, chebyshev_nodes(a, b, n), a, b, ax=axes[1, col],
                           ylim=(-1, 2), title=f"Chebyshev, n = {n}")
    save_figure(fig, os.path.join(OUT, "D_runge.png"))

    rows = error_table(runge, a, b, n_values)
    print("Experiment D: Runge function 1/(1+25x^2) on [-1, 1]")
    save_table(rows, "D_error_table.txt")
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_error_vs_n(rows, ax=ax, title="Runge function: error vs n")
    save_figure(fig, os.path.join(OUT, "D_runge_error.png"))


def main():
    os.makedirs(OUT, exist_ok=True)
    experiment_a()
    experiment_b()
    experiment_c()
    experiment_d()
    print(f"Figures saved in {OUT}")


if __name__ == "__main__":
    main()
