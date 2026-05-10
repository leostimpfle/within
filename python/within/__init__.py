"""High-performance fixed-effects solver for econometric panel data.

``within`` solves the normal equations arising from multi-way fixed-effect
models (D^T W D x = D^T W y) using modified LSMR with a domain-decomposition
(Schwarz) preconditioner. The heavy lifting is done in Rust; this package
provides the Python API.

Quick start::

    import numpy as np
    import within

    # Two factors, four observations
    categories = np.array(
        [[0, 0],
         [0, 1],
         [1, 0],
         [1, 1]],
        dtype=np.uint32,
    )
    y = np.array([1.0, 2.0, 3.0, 4.0])

    result = within.solve(categories, y)
    print(result.x)         # fixed-effect coefficients
    print(result.converged) # True

For repeated solves on the same panel structure, use the persistent
``Solver`` class to amortise the preconditioner setup::

    solver = within.Solver(categories)
    r1 = solver.solve(y1)
    r2 = solver.solve(y2)

Key exports:

- :func:`solve` / :func:`solve_batch` -- one-shot solve functions
- :class:`Solver` -- persistent solver with reusable preconditioner
- :class:`LSMR` -- LSMR solver configuration
- :class:`Preconditioner` -- quick preconditioner selection enum
- :class:`AdditiveSchwarz` -- fine-grained preconditioner configuration

For Rust-level internals, build the API docs with ``cargo doc --open``.
"""

from within._within import (
    Preconditioner,
    SolveResult,
    BatchSolveResult,
    LSMR,
    FePreconditioner,
    Solver,
    solve,
    solve_batch,
    AdditiveSchwarz,
    AdditiveSchwarzDiagnostics,
    ReductionStrategy,
)

__all__ = [
    "Preconditioner",
    "SolveResult",
    "BatchSolveResult",
    "LSMR",
    "FePreconditioner",
    "Solver",
    "solve",
    "solve_batch",
    "AdditiveSchwarz",
    "AdditiveSchwarzDiagnostics",
    "ReductionStrategy",
]
