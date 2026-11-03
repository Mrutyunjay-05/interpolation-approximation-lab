import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from interp.errors import equispaced_nodes, error_table
from interp.plotting import (
    plot_error_curves,
    plot_error_vs_n,
    plot_interpolation,
    save_figure,
)


def test_plot_interpolation_draws_three_lines():
    fig, ax = plt.subplots()
    plot_interpolation(np.sin, equispaced_nodes(0, 3, 5), 0, 3, ax=ax)
    assert len(ax.lines) == 3  # function, polynomial, nodes
    plt.close(fig)


def test_plot_error_curves_one_line_per_node_set():
    fig, ax = plt.subplots()
    nodes = {"a": equispaced_nodes(0, 3, 5), "b": equispaced_nodes(0, 3, 6)}
    plot_error_curves(np.sin, 0, 3, nodes, ax=ax)
    assert len(ax.lines) == 2
    plt.close(fig)


def test_plot_error_vs_n_and_save(tmp_path):
    rows = error_table(np.sin, 0, 3, [3, 5, 7])
    fig, ax = plt.subplots()
    plot_error_vs_n(rows, ax=ax)
    out = tmp_path / "sub" / "error.png"
    save_figure(fig, str(out))
    assert out.exists() and out.stat().st_size > 0
