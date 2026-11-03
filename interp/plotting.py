"""Plotting helpers for the interpolation experiments."""

import os

import matplotlib.pyplot as plt
import numpy as np

from interp.evaluate import interpolate


def plot_interpolation(f, x_nodes, a, b, ax=None, title=None, ylim=None, num=1000):
    """Plot f, the interpolating polynomial and the nodes on [a, b]."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    x_nodes = np.asarray(x_nodes, dtype=float)
    grid = np.linspace(a, b, num)
    p = interpolate(x_nodes, f(x_nodes), grid)

    ax.plot(grid, f(grid), linewidth=2, label="f(x)")
    ax.plot(grid, p, "--", linewidth=1.5, label="interpolating polynomial")
    ax.plot(x_nodes, f(x_nodes), "o", color="black", markersize=4, label="nodes")
    if ylim is not None:
        ax.set_ylim(*ylim)
    if title:
        ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return ax


def plot_error_curves(f, a, b, node_sets, ax=None, title=None, num=1000):
    """Plot |f - p| on a log scale for several node sets.

    node_sets is a dict {label: nodes}.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    grid = np.linspace(a, b, num)
    for label, nodes in node_sets.items():
        nodes = np.asarray(nodes, dtype=float)
        err = np.abs(f(grid) - interpolate(nodes, f(nodes), grid))
        ax.semilogy(grid, np.maximum(err, 1e-17), label=label)
    if title:
        ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("|f(x) - p(x)|")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    return ax


def plot_error_vs_n(rows, ax=None, title=None):
    """Semilog plot of the maximum error against n, from error_table rows."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    n = [r["n"] for r in rows]
    ax.semilogy(n, [r["equispaced"] for r in rows], "o-", label="equispaced nodes")
    ax.semilogy(n, [r["chebyshev"] for r in rows], "s-", label="Chebyshev nodes")
    if title:
        ax.set_title(title)
    ax.set_xlabel("number of nodes n")
    ax.set_ylabel("maximum absolute error")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    return ax


def save_figure(fig, path):
    """Save a figure to path (creating the folder if needed) and close it."""
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
