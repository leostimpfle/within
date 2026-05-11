//! Schwarz preconditioner: FE-specific construction helpers.
//!
//! This module bridges the fixed-effects domain types ([`Design`],
//! [`Subdomain`], `CrossTab`) to the generic `schwarz-precond` crate API.
//! The generic crate knows nothing about panel data — it operates on abstract
//! [`SubdomainEntry`] values containing a local solver and a set of global DOF
//! indices. This module handles the translation.
//!
//! # Local solver
//!
//! Each subdomain needs a local solver that can approximately invert the
//! restricted Gramian on that subdomain. The solver eliminates one factor
//! block via exact diagonal inversion, then factors the reduced Schur
//! complement (see `schur_complement`).
//!
//! # Builder pattern
//!
//! Construction flows through a layered builder:
//!
//! 1. **Domain acquisition** — `(Subdomain, CrossTab)` pairs come from
//!    [`build_local_domains`](crate::domain) via a single observation scan.
//! 2. **Entry construction** — each `(Subdomain, CrossTab)` pair is
//!    converted into a `SubdomainEntry<BlockElimSolver>` in parallel via
//!    `build_entry`, which dispatches on the config
//! 3. **Schwarz assembly** — entries are passed to the generic
//!    `SchwarzPreconditioner` constructor from `schwarz-precond`.

use approx_chol::low_level::Builder;
use approx_chol::CsrRef;
use rayon::prelude::*;
use schwarz_precond::{SchwarzPreconditioner, SubdomainEntry};
use serde::{Deserialize, Serialize};

use super::gramian::CrossTab;
use super::local_solver::{BlockElimSolver, ReducedFactor};
use super::schur_complement::{
    ApproxSchurComplement, EliminationInfo, ExactSchurComplement, SchurComplement, SchurResult,
};
use crate::config::{ApproxCholConfig, ApproxSchurConfig, LocalSolverConfig};
use crate::domain::Subdomain;
use crate::BuildError;

/// Concrete additive Schwarz type used in the parent crate.
#[derive(Clone, Serialize, Deserialize)]
pub struct FeSchwarz(SchwarzPreconditioner<BlockElimSolver>);

impl FeSchwarz {
    pub(crate) fn new(inner: SchwarzPreconditioner<BlockElimSolver>) -> Self {
        Self(inner)
    }

    /// Subdomain entries with their local solvers.
    pub fn subdomains(&self) -> &[SubdomainEntry<BlockElimSolver>] {
        self.0.subdomains()
    }

    /// Current reduction strategy (may be `Auto`).
    pub fn reduction_strategy(&self) -> schwarz_precond::ReductionStrategy {
        self.0.reduction_strategy()
    }

    /// Resolved reduction strategy (`Auto` replaced by the detected choice).
    pub fn resolved_reduction_strategy(&self) -> schwarz_precond::ReductionStrategy {
        self.0.resolved_reduction_strategy()
    }

    /// Apply the preconditioner, returning an error on local-solver failure.
    pub fn apply(&self, r: &[f64], z: &mut [f64]) -> Result<(), schwarz_precond::SolveError> {
        self.0.apply(r, z)
    }

    #[cfg(test)]
    pub fn with_reduction_strategy(&self, strategy: schwarz_precond::ReductionStrategy) -> Self {
        Self(self.0.with_reduction_strategy(strategy))
    }
}

impl schwarz_precond::Operator for FeSchwarz {
    fn nrows(&self) -> usize {
        self.0.nrows()
    }

    fn ncols(&self) -> usize {
        self.0.ncols()
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), schwarz_precond::SolveError> {
        self.0.apply(x, y)
    }

    fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) -> Result<(), schwarz_precond::SolveError> {
        self.0.apply_adjoint(x, y)
    }
}

// ---------------------------------------------------------------------------
// Crate-internal builders
// ---------------------------------------------------------------------------

/// Build additive Schwarz with an explicit reduction strategy.
pub(crate) fn build_additive_with_strategy(
    domains: Vec<(Subdomain, CrossTab)>,
    n_dofs: usize,
    config: &LocalSolverConfig,
    strategy: schwarz_precond::ReductionStrategy,
) -> Result<FeSchwarz, BuildError> {
    let entries = build_entries_from_pairs(domains, config)?;
    Ok(FeSchwarz::new(
        SchwarzPreconditioner::with_strategy(entries, n_dofs, strategy)
            .map_err(BuildError::Preconditioner)?,
    ))
}

fn build_entries_from_pairs(
    domain_pairs: Vec<(Subdomain, CrossTab)>,
    config: &LocalSolverConfig,
) -> Result<Vec<SubdomainEntry<BlockElimSolver>>, BuildError> {
    domain_pairs
        .into_par_iter()
        .map(|(domain, cross_tab)| build_entry(domain, cross_tab, config))
        .collect()
}

// ---------------------------------------------------------------------------
// Helper: build SubdomainEntry from FE types
// ---------------------------------------------------------------------------

/// Build a single `SubdomainEntry<BlockElimSolver>` from a pre-built CrossTab.
pub(crate) fn build_entry(
    domain: Subdomain,
    cross_tab: CrossTab,
    config: &LocalSolverConfig,
) -> Result<SubdomainEntry<BlockElimSolver>, BuildError> {
    let schur_config = ReducedSchurConfig {
        approx_chol: config.approx_chol,
        approx_schur: config.approx_schur,
        dense_threshold: config.dense_threshold,
    };
    let reduced = build_reduced_schur_factor(&cross_tab, &schur_config)?;
    let solver = BlockElimSolver::new(
        cross_tab,
        reduced.elimination.inv_diag_elim,
        reduced.factor,
        reduced.elimination.eliminate_q,
    );
    SubdomainEntry::try_new(domain.core, solver).map_err(BuildError::Preconditioner)
}

pub(crate) struct ReducedSchurBuild {
    pub(crate) factor: ReducedFactor,
    pub(crate) elimination: EliminationInfo,
}

fn dense_fast_path_enabled(n_keep: usize, threshold: usize) -> bool {
    threshold > 0 && n_keep <= threshold
}

fn compute_schur(
    cross_tab: &CrossTab,
    approx_schur: Option<ApproxSchurConfig>,
) -> Result<SchurResult, BuildError> {
    match approx_schur {
        None => ExactSchurComplement.compute(cross_tab),
        Some(cfg) => ApproxSchurComplement::new(cfg).compute(cross_tab),
    }
}

fn build_sparse_reduced_factor(
    matrix: &schwarz_precond::SparseMatrix,
    approx_chol: ApproxCholConfig,
) -> Result<ReducedFactor, BuildError> {
    let schur_builder = Builder::new(approx_chol.to_approx_chol());
    let csr = CsrRef::new(
        matrix.indptr(),
        matrix.indices(),
        matrix.data(),
        matrix.n() as u32,
    )
    .map_err(|e| BuildError::LocalSolverBuild(format!("invalid Schur complement CSR: {e}")))?;
    schur_builder
        .build(csr)
        .map(ReducedFactor::approx)
        .map_err(|e| {
            BuildError::LocalSolverBuild(format!("failed Schur complement factorization: {e}"))
        })
}

/// Configuration for building a reduced Schur factor.
pub(crate) struct ReducedSchurConfig {
    pub approx_chol: ApproxCholConfig,
    pub approx_schur: Option<ApproxSchurConfig>,
    pub dense_threshold: usize,
}

pub(crate) fn build_reduced_schur_factor(
    cross_tab: &CrossTab,
    config: &ReducedSchurConfig,
) -> Result<ReducedSchurBuild, BuildError> {
    let n_keep = cross_tab.n_q().min(cross_tab.n_r());
    let prefer_dense = dense_fast_path_enabled(n_keep, config.dense_threshold);

    // Below the dense threshold the reduced system is tiny — always use exact
    // Schur complement (cheap at this size) and dense Cholesky factorization.
    if prefer_dense {
        let dense = ExactSchurComplement.compute_dense_anchored(cross_tab)?;
        if let Some(factor) =
            ReducedFactor::try_dense_laplacian_minor(dense.anchored_minor, dense.n)
        {
            return Ok(ReducedSchurBuild {
                factor,
                elimination: dense.elimination,
            });
        }
    }

    // General path (exact or approximate): sparse Schur assembly.
    let schur = compute_schur(cross_tab, config.approx_schur)?;

    let factor = build_sparse_reduced_factor(&schur.matrix, config.approx_chol)?;
    Ok(ReducedSchurBuild {
        factor,
        elimination: schur.elimination,
    })
}
