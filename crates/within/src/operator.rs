//! Linear algebra layer: operator representations and preconditioner wiring.
//!
//! This module is the hub between the [`domain`](crate::domain) layer (which
//! builds subdomains from panel data) and the [`orchestrate`](crate::orchestrate)
//! layer (the public solve API). It provides the rectangular design operator
//! that LSMR consumes and the Schwarz preconditioner construction.
//!
//! # Operators
//!
//! LSMR works directly on `sqrt(W) D` rather than the assembled Gramian
//! `G = D^T W D`. The matvec primitive is:
//!
//! | Operator | Type | Description |
//! |---|---|---|
//! | **sqrt(W) D** | [`DesignOperator`] | Rectangular operator used by LSMR. Implements `sqrt(W) D x` and `D^T sqrt(W) x` via gather/scatter on the observation store |
//!
//! # Submodules
//!
//! - [`gramian::cross_tab`](gramian) — Per-pair `CrossTab` blocks driving
//!   factor-pair subdomain construction (the only Gramian-shaped artefact
//!   the solver still needs)
//! - [`schwarz`] — Schwarz preconditioner construction: bridges fixed-effects
//!   types to the generic `schwarz-precond` API
//! - `local_solver` — Local subdomain solvers: approximate Cholesky (SDDM)
//!   and block-elimination backends
//! - `schur_complement` — Exact and approximate Schur complement computation
//!   for block-elimination local solves
//! - [`preconditioner`] — [`FePreconditioner`](preconditioner::FePreconditioner)
//!   wrapping the additive Schwarz variant
//! - `csr_block` — Internal rectangular CSR block used in bipartite blocks

pub(crate) mod csr_block;
pub(crate) mod gramian;
pub(crate) mod local_solver;
pub mod preconditioner;
pub(crate) mod schur_complement;
pub(crate) mod schwarz;

#[cfg(test)]
mod tests;

// ---------------------------------------------------------------------------
// DesignOperator — rectangular, W^{1/2}·D·x / D^T·W^{1/2}·x
// ---------------------------------------------------------------------------

use std::sync::Mutex;

use schwarz_precond::Operator;

use crate::domain::Design;
use crate::observation::Store;

/// Weighted rectangular design operator: `A = W^{1/2} D`.
///
/// `apply` = `W^{1/2} D x` (observation space), `apply_adjoint` = `D^T W^{1/2} x` (DOF space).
/// For unweighted designs, delegates directly to `D x` / `D^T x` with no extra work.
///
/// The normal equations of this operator give `A^T A = D^T W D = G` (the Gramian),
/// so the existing Schwarz preconditioner approximating `G^{-1}` can be used directly.
pub struct DesignOperator<'a, S: Store> {
    design: &'a Design<S>,
    /// Pre-computed `sqrt(w_i)` per observation. `None` when unweighted.
    sqrt_weights: Option<Vec<f64>>,
    /// Scratch for the adjoint path: stores `sqrt(w_i) * u_i`.
    scratch: Mutex<Vec<f64>>,
}

impl<'a, S: Store> DesignOperator<'a, S> {
    /// Create from a weighted design matrix and optional observation weights.
    ///
    /// `weights = None` selects the unweighted fast-path: `sqrt_weights` is
    /// `None` and `apply` / `apply_adjoint` skip the per-row scaling entirely.
    pub fn new(design: &'a Design<S>, weights: Option<&[f64]>) -> Self {
        let sqrt_weights = weights.map(|w| w.iter().map(|wi| wi.sqrt()).collect::<Vec<f64>>());
        Self {
            scratch: Mutex::new(vec![0.0; design.n_rows]),
            design,
            sqrt_weights,
        }
    }

    /// Compute the observation-space RHS `b = W^{1/2} y`.
    ///
    /// For unweighted designs, returns a copy of `y`.
    pub fn weighted_rhs(&self, y: &[f64]) -> Vec<f64> {
        match &self.sqrt_weights {
            None => y.to_vec(),
            Some(sw) => y.iter().zip(sw).map(|(&yi, &swi)| swi * yi).collect(),
        }
    }
}
impl<S: Store> Operator for DesignOperator<'_, S> {
    fn nrows(&self) -> usize {
        self.design.n_rows
    }

    fn ncols(&self) -> usize {
        self.design.n_dofs
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), schwarz_precond::SolveError> {
        // y = W^{1/2} (D x)
        self.design.matvec_d(x, y);
        if let Some(sw) = &self.sqrt_weights {
            for (yi, &swi) in y.iter_mut().zip(sw) {
                *yi *= swi;
            }
        }
        Ok(())
    }

    fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) -> Result<(), schwarz_precond::SolveError> {
        // y = D^T (W^{1/2} x)
        match &self.sqrt_weights {
            None => self.design.rmatvec_dt(x, y),
            Some(sw) => {
                let mut tmp = self.scratch.lock().unwrap();
                for (ti, (&xi, &swi)) in tmp.iter_mut().zip(x.iter().zip(sw)) {
                    *ti = swi * xi;
                }
                self.design.rmatvec_dt(&tmp, y);
            }
        }
        Ok(())
    }
}
