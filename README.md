# Interpolation and Approximation Laboratory

Scientific Computing, Project 9 (team project).

A small toolkit for approximating functions from sampled data. It implements
polynomial interpolation with **Newton divided differences** (no black-box
interpolation call) and uses it to study how the number and placement of
interpolation nodes affect the approximation, including the Runge phenomenon.

## Method

Given distinct nodes `x_0, ..., x_{n-1}` and values `y_0, ..., y_{n-1}`, the
interpolating polynomial in Newton form is

    p(x) = c_0 + c_1 (x - x_0) + c_2 (x - x_0)(x - x_1) + ...
           + c_{n-1} (x - x_0) ... (x - x_{n-2})

The coefficients `c_k` are the divided differences `f[x_0, ..., x_k]`, computed
in `O(n^2)` operations from the recurrence

    f[x_i, ..., x_{i+k}] = (f[x_{i+1}, ..., x_{i+k}] - f[x_i, ..., x_{i+k-1}]) / (x_{i+k} - x_i)

The polynomial is evaluated with a Horner-style nested scheme, which needs only
`O(n)` operations per query point.

### Interpolation vs approximation

- **Interpolation** builds a function that passes *exactly* through the given
  data points. Between the nodes it can still differ from the underlying
  function, and outside the node range (extrapolation) it can be far off.
- **Approximation** only asks for a function that is *close* to the data or to
  the underlying function (for example a least-squares fit). It need not pass
  through any of the points.

Interpolating polynomial through `n` points has degree at most `n - 1`. More
nodes do not always give a better result: with equally spaced nodes, high-degree
interpolation can oscillate strongly near the endpoints (Runge phenomenon).

## Project structure

    interp/
        core.py        Newton divided differences and Horner-style evaluation
        evaluate.py    input validation and the public interpolate() function
        errors.py      equispaced and Chebyshev nodes, max absolute error, error tables
        plotting.py    plotting helpers
    experiments/
        run_experiments.py    runs experiments A-D and saves figures and tables
    tests/                    pytest tests for every module
    figures/                  output of the experiments

## Setup

Requires Python 3.9 or newer.

    git clone https://github.com/Mrutyunjay-05/interpolation-approximation-lab.git
    cd interpolation-approximation-lab
    python -m pip install -r requirements.txt pytest

## Usage

    import numpy as np
    from interp.evaluate import interpolate

    x_nodes = [0.0, 1.0, 2.0]
    y_nodes = [1.0, 2.0, 5.0]          # values of x**2 + 1

    interpolate(x_nodes, y_nodes, 1.5)               # 3.25 (a float)
    interpolate(x_nodes, y_nodes, np.linspace(0, 2, 5))   # array, same shape as the query

Invalid input (different lengths, repeated nodes, empty or non-1D arrays,
NaN/infinity, non-numeric values) raises a `ValueError` with a clear message.

Error measurement and node generators:

    from interp.errors import chebyshev_nodes, equispaced_nodes, fine_grid, max_abs_error

    f = lambda x: 1.0 / (1.0 + 25.0 * x**2)
    grid = fine_grid(-1, 1)
    max_abs_error(f, equispaced_nodes(-1, 1, 15), grid)   # large (Runge phenomenon)
    max_abs_error(f, chebyshev_nodes(-1, 1, 15), grid)    # much smaller

## Running the tests

From the project root:

    python -m pytest

## Reproducing the experiments

From the project root:

    python experiments/run_experiments.py

This writes figures and error tables to `figures/`:

| Experiment | Description | Output |
|---|---|---|
| A | sin(x) on [0, 2*pi] with 4, 6 and 8 equispaced nodes | `A_smooth_function.png` |
| B | maximum error against the number of nodes (3 to 30) | `B_error_vs_n.png`, `B_error_table.txt` |
| C | equispaced vs Chebyshev nodes for 1/(1+x^2) on [-5, 5] | `C_node_placement.png` |
| D | Runge function 1/(1+25x^2) on [-1, 1], n = 5, 10, 15, 20 | `D_runge.png`, `D_runge_error.png`, `D_error_table.txt` |

### Main results (Runge function, maximum absolute error on [-1, 1])

| n | equispaced nodes | Chebyshev nodes |
|---|---|---|
| 5 | 0.44 | 0.40 |
| 10 | 0.30 | 0.27 |
| 15 | 7.2 | 0.047 |
| 20 | 8.6 | 0.038 |

With equally spaced nodes the error grows with `n` and the polynomial
oscillates near `x = +-1`. With Chebyshev nodes, which cluster near the
endpoints, the error keeps decreasing.


## Design decisions

- Newton divided differences were chosen over the Lagrange form because new
  nodes can be added cheaply and evaluation by nested multiplication is simple
  and efficient.
- Validation lives in `interp/evaluate.py`, so the core algorithm stays small
  and every public call goes through the same checks.
- The error is measured on a fine grid of 2001 points by default.
- Experiments run from one command and write all outputs to `figures/`, so the
  results are reproducible.

## Team and workflow

| Member | Role | Main files |
|---|---|---|
| Mrutyunjay Malik (team leader) | Interpolation core, integration, README | `interp/core.py` |
| Rahul Kumar | Evaluation and input validation | `interp/evaluate.py` |
| Satrudhan Kumar | Error analysis | `interp/errors.py` |
| Vanlalthazama | Experiments and visualization | `interp/plotting.py`, `experiments/run_experiments.py` |

Work was organised with GitHub issues, one feature branch per task, and pull
requests that were reviewed and approved by another team member before merging
into `main` (`main` is protected).
