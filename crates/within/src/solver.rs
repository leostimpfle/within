//! Persistent solver that caches the preconditioner for reuse across multiple
//! right-hand sides.
//!
//! # Motivation
//!
//! Building the Schwarz preconditioner is the most expensive step in a
//! fixed-effects solve: it scans observations to build subdomains, assembles
//! local operators, and computes approximate Cholesky factorizations. For a
//! single right-hand side (RHS) this cost is unavoidable, but econometric
//! workflows frequently solve the same design matrix with many different
//! response vectors (e.g., multiple dependent variables, bootstrap replications,
//! or iteratively reweighted least squares). [`Solver`] lets callers pay the
//! preconditioner cost once and amortize it across all subsequent solves.
//!
//! # Usage
//!
//! ```no_run
//! use within::{Solver, SolverParams, Preconditioner, LocalSolverConfig};
//! use ndarray::Array2;
//!
//! let categories = Array2::<u32>::zeros((1000, 2));
//! let params = SolverParams::default();
//! let precond = Preconditioner::Additive(
//!     LocalSolverConfig::solver_default(),
//!     Default::default(),
//! );
//!
//! // Build once — expensive
//! let solver = Solver::new(categories.view(), None, &params, Some(&precond)).unwrap();
//!
//! // Solve many — cheap (reuses preconditioner)
//! let y1 = vec![1.0; 1000];
//! let y2 = vec![2.0; 1000];
//! let r1 = solver.solve(&y1).unwrap();
//! let r2 = solver.solve(&y2).unwrap();
//! ```

use std::time::Instant;

use ndarray::ArrayView2;
use rayon::prelude::*;
use schwarz_precond::{lsmr, mlsmr};

use crate::operator::WeightedDesignOperator;

use crate::config::{Preconditioner, SolverParams};
use crate::domain::WeightedDesign;
use crate::observation::{ArrayStore, ObservationStore, ObservationWeights};
use crate::operator::preconditioner::{build_preconditioner, FePreconditioner};
use crate::orchestrate::{BatchSolveResult, SolveResult};
use crate::WithinResult;

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Persistent solver that owns its preconditioner for reuse across multiple solves.
///
/// Build once with [`Solver::new`] or [`Solver::from_design`], then call
/// [`Solver::solve`] or [`Solver::solve_batch`] repeatedly with different RHS
/// vectors. The expensive preconditioner factorization happens only at
/// construction time.
pub struct Solver<S: ObservationStore> {
    design: WeightedDesign<S>,
    preconditioner: Option<FePreconditioner>,
    tol: f64,
    maxiter: usize,
    local_size: Option<usize>,
}

impl<S: ObservationStore> Solver<S> {
    /// Build from an existing [`WeightedDesign`].
    pub fn from_design(
        design: WeightedDesign<S>,
        params: &SolverParams,
        preconditioner: Option<&Preconditioner>,
    ) -> WithinResult<Self> {
        let built_precond = match preconditioner {
            Some(config) => Some(build_preconditioner(&design, config)?),
            None => None,
        };

        Ok(Self {
            design,
            preconditioner: built_precond,
            tol: params.tol,
            maxiter: params.maxiter,
            local_size: params.local_size,
        })
    }

    /// Build from a design with a pre-built preconditioner (e.g. deserialized).
    pub fn from_design_with_preconditioner(
        design: WeightedDesign<S>,
        params: &SolverParams,
        preconditioner: FePreconditioner,
    ) -> WithinResult<Self> {
        Ok(Self {
            design,
            preconditioner: Some(preconditioner),
            tol: params.tol,
            maxiter: params.maxiter,
            local_size: params.local_size,
        })
    }

    /// Solve for a single RHS vector.
    pub fn solve(&self, y: &[f64]) -> WithinResult<SolveResult> {
        let t_start = Instant::now();
        let t_setup_start = Instant::now();

        let rect_op = WeightedDesignOperator::new(&self.design);
        let b = rect_op.weighted_rhs(y);

        let t_solve_start = Instant::now();
        let time_setup = t_solve_start.duration_since(t_setup_start).as_secs_f64();

        let r = match self.preconditioner.as_ref() {
            Some(p) => mlsmr(&rect_op, &b, p, self.tol, self.maxiter, self.local_size)?,
            None => lsmr(&rect_op, &b, self.tol, self.maxiter, self.local_size)?,
        };

        let time_solve = t_solve_start.elapsed().as_secs_f64();

        let mut demeaned = vec![0.0; self.design.n_rows];
        self.design.matvec_d(&r.x, &mut demeaned);
        for (d, &yi) in demeaned.iter_mut().zip(y.iter()) {
            *d = yi - *d;
        }

        let mut rhs = vec![0.0; self.design.n_dofs];
        self.design.rmatvec_wdt(y, &mut rhs);
        let rhs_norm = norm(&rhs).max(1e-15);
        let mut residual_dof = vec![0.0; self.design.n_dofs];
        self.design.rmatvec_wdt(&demeaned, &mut residual_dof);
        let final_residual = norm(&residual_dof) / rhs_norm;

        Ok(SolveResult {
            x: r.x,
            demeaned,
            converged: r.converged,
            iterations: r.iterations,
            final_residual,
            time_total: t_start.elapsed().as_secs_f64(),
            time_setup,
            time_solve,
        })
    }

    /// Solve for multiple RHS vectors in parallel.
    pub fn solve_batch(&self, ys: &[&[f64]]) -> WithinResult<BatchSolveResult> {
        let t_start = Instant::now();
        let n_rhs = ys.len();

        let results: Vec<WithinResult<SolveResult>> =
            ys.par_iter().map(|y| self.solve(y)).collect();

        let mut x = Vec::with_capacity(self.design.n_dofs * n_rhs);
        let mut demeaned = Vec::with_capacity(self.design.n_rows * n_rhs);
        let mut converged = Vec::with_capacity(n_rhs);
        let mut iterations = Vec::with_capacity(n_rhs);
        let mut final_residual = Vec::with_capacity(n_rhs);
        let mut time_solve = Vec::with_capacity(n_rhs);

        for r in results {
            let r = r?;
            x.extend_from_slice(&r.x);
            demeaned.extend_from_slice(&r.demeaned);
            converged.push(r.converged);
            iterations.push(r.iterations);
            final_residual.push(r.final_residual);
            time_solve.push(r.time_solve);
        }

        Ok(BatchSolveResult::new(
            x,
            demeaned,
            converged,
            iterations,
            final_residual,
            time_solve,
            t_start.elapsed().as_secs_f64(),
        ))
    }

    /// Access the preconditioner (for serialization).
    pub fn preconditioner(&self) -> Option<&FePreconditioner> {
        self.preconditioner.as_ref()
    }

    /// Number of DOFs (coefficients).
    pub fn n_dofs(&self) -> usize {
        self.design.n_dofs
    }

    /// Number of observations.
    pub fn n_obs(&self) -> usize {
        self.design.n_rows
    }
}

// Convenience constructors for ArrayStore
impl<'a> Solver<ArrayStore<'a>> {
    /// Build a solver from raw category data (zero-copy).
    pub fn new(
        categories: ArrayView2<'a, u32>,
        weights: Option<&[f64]>,
        params: &SolverParams,
        preconditioner: Option<&Preconditioner>,
    ) -> WithinResult<Self> {
        let weights = match weights {
            Some(w) => ObservationWeights::Dense(w.to_vec()),
            None => ObservationWeights::Unit,
        };
        let store = ArrayStore::new(categories, weights)?;
        let design = WeightedDesign::from_store(store)?;
        Self::from_design(design, params, preconditioner)
    }

    /// Build a solver with a pre-built preconditioner (e.g. deserialized).
    pub fn with_preconditioner(
        categories: ArrayView2<'a, u32>,
        weights: Option<&[f64]>,
        params: &SolverParams,
        preconditioner: FePreconditioner,
    ) -> WithinResult<Self> {
        let weights = match weights {
            Some(w) => ObservationWeights::Dense(w.to_vec()),
            None => ObservationWeights::Unit,
        };
        let store = ArrayStore::new(categories, weights)?;
        let design = WeightedDesign::from_store(store)?;
        Self::from_design_with_preconditioner(design, params, preconditioner)
    }
}
