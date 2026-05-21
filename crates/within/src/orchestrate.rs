//! Public entry points: [`solve`] / [`solve_batch`] convenience wrappers around [`crate::Solver`].

use std::time::Instant;

use ndarray::ArrayView2;

use crate::config::LsmrOptions;
use crate::WithinError;

/// Common solve output for all orchestration entry points.
#[derive(Debug, Clone)]
#[must_use]
pub struct SolveResult {
    /// Fixed-effect coefficients (length = total DOFs across all factors).
    pub x: Vec<f64>,
    /// Demeaned response: `y - D x` (length = n_obs).
    pub demeaned: Vec<f64>,
    /// Whether the iterative solver converged within `maxiter` iterations.
    pub converged: bool,
    /// Number of LSMR iterations used.
    pub iterations: usize,
    /// Final relative residual norm `‖r‖ / ‖b‖`.
    pub residual: f64,
    /// Wall-clock time for the entire solve (setup + LSMR), in seconds.
    pub time_total: f64,
    /// Wall-clock time for preconditioner construction, in seconds.
    pub time_setup: f64,
    /// Wall-clock time for the LSMR solve phase, in seconds.
    pub time_solve: f64,
}

/// Result of a batch solve across multiple RHS vectors.
#[derive(Debug, Clone)]
pub struct BatchSolveResult {
    /// All coefficient vectors concatenated (length = n_dofs * n_rhs).
    pub x: Vec<f64>,
    /// All demeaned responses concatenated (length = n_obs * n_rhs).
    pub demeaned: Vec<f64>,
    /// Per-RHS convergence flags.
    pub converged: Vec<bool>,
    /// Per-RHS iteration counts.
    pub iterations: Vec<usize>,
    /// Per-RHS final relative residual norms.
    pub residual: Vec<f64>,
    /// Per-RHS solve times in seconds.
    pub time_solve: Vec<f64>,
    /// Total wall-clock time for the entire batch (setup + all solves), in seconds.
    pub time_total: f64,
    /// Number of coefficients per RHS (rows of the underlying design).
    pub n_dofs: usize,
    /// Number of observations (columns of the underlying design).
    pub n_obs: usize,
}

impl BatchSolveResult {
    /// Coefficient vector for the `i`-th RHS.
    pub fn x(&self, i: usize) -> &[f64] {
        &self.x[i * self.n_dofs..(i + 1) * self.n_dofs]
    }
    /// Demeaned response for the `i`-th RHS.
    pub fn demeaned(&self, i: usize) -> &[f64] {
        &self.demeaned[i * self.n_obs..(i + 1) * self.n_obs]
    }
}

// ===========================================================================
// High-level API
// ===========================================================================

/// Solve fixed-effects least squares from raw category data.
///
/// `categories` is an observation-major `(n_obs, n_factors)` array where
/// `categories[[i, q]]` is the level of observation `i` in factor `q`.
/// Levels must be `0..max_level` per factor; the number of levels is inferred.
/// `y` is the response vector (length = n_obs).
///
/// Zero-copy: the category array is borrowed, not copied.
///
/// `preconditioner` accepts the same input shapes as [`crate::Solver::new`]:
/// `None`, a [`crate::PreconditionerConfig`] by reference or value, an owned
/// [`crate::Preconditioner`], or a `&Preconditioner` for amortized reuse.
///
/// This is a convenience wrapper around [`crate::Solver::new`] + [`crate::Solver::solve`].
pub fn solve(
    categories: ArrayView2<u32>,
    y: &[f64],
    weights: Option<&[f64]>,
    lsmr: &LsmrOptions,
    preconditioner: impl Into<crate::solver::PreconditionerInput>,
) -> Result<SolveResult, WithinError> {
    let t_start = Instant::now();
    let solver = crate::solver::Solver::new(categories, weights, preconditioner)?;
    let time_setup = t_start.elapsed().as_secs_f64();
    let mut result = solver.solve(y, lsmr)?;
    // Include solver construction (preconditioner build) in setup time
    result.time_setup += time_setup;
    result.time_total = t_start.elapsed().as_secs_f64();
    Ok(result)
}

/// Solve fixed-effects least squares for multiple response vectors.
///
/// Same as [`solve`] but solves all RHS vectors in parallel (via rayon),
/// reusing the preconditioner across all solves.
pub fn solve_batch(
    categories: ArrayView2<u32>,
    ys: &[&[f64]],
    weights: Option<&[f64]>,
    lsmr: &LsmrOptions,
    preconditioner: impl Into<crate::solver::PreconditionerInput>,
) -> Result<BatchSolveResult, WithinError> {
    let t_start = Instant::now();
    let solver = crate::solver::Solver::new(categories, weights, preconditioner)?;
    let mut result = solver.solve_batch(ys, lsmr)?;
    result.time_total = t_start.elapsed().as_secs_f64();
    Ok(result)
}
