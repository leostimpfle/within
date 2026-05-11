//! Pre-built fixed-effects preconditioner for LSMR.
//!
//! [`FePreconditioner`] is the top-level preconditioner type used by the
//! [`orchestrate`](crate::orchestrate) layer. It currently has a single
//! variant for one-level additive Schwarz; the enum + `#[non_exhaustive]`
//! shape leaves room for future variants (e.g. two-level Schwarz with a
//! coarse space) without breaking existing call sites.
//!
//! # Integration with `schwarz-precond`
//!
//! The enum implements the [`Operator`] trait from the `schwarz-precond`
//! crate, so it can be passed directly to LSMR as a preconditioner. The
//! `apply` method is fallible — local-solver failures propagate to the
//! caller as `SolveError`.

use schwarz_precond::{LocalSolver, Operator, ReductionStrategy};
use serde::{Deserialize, Serialize};

use crate::config::Preconditioner;
use crate::domain::Design;
use crate::observation::{validate_weights, Store};
use crate::operator::schwarz::{build_additive_with_strategy, FeSchwarz};
use crate::BuildError;

/// A pre-built preconditioner ready for use in LSMR solves.
///
/// Marked `#[non_exhaustive]`: future variants (e.g. two-level Schwarz)
/// may be added without requiring a major version bump.
#[derive(Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum FePreconditioner {
    /// One-level additive Schwarz over factor-pair subdomains.
    Additive(FeSchwarz),
}

impl FePreconditioner {
    /// Number of Schwarz subdomains in the built preconditioner.
    pub fn n_subdomains(&self) -> usize {
        match self {
            Self::Additive(p) => p.subdomains().len(),
        }
    }

    /// Estimated nested-parallel work per subdomain.
    pub fn subdomain_inner_parallel_work(&self) -> Vec<usize> {
        match self {
            Self::Additive(p) => p
                .subdomains()
                .iter()
                .map(|entry| entry.solver().inner_parallelism_work_estimate())
                .collect(),
        }
    }

    /// Configured additive reduction strategy, if this is an additive variant.
    pub fn additive_reduction_strategy(&self) -> Option<ReductionStrategy> {
        match self {
            Self::Additive(p) => Some(p.reduction_strategy()),
        }
    }

    /// Concrete additive backend resolved for the current Rayon thread-pool width.
    pub fn resolved_additive_reduction_strategy(&self) -> Option<ReductionStrategy> {
        match self {
            Self::Additive(p) => Some(p.resolved_reduction_strategy()),
        }
    }
}

impl Operator for FePreconditioner {
    fn nrows(&self) -> usize {
        match self {
            Self::Additive(p) => p.nrows(),
        }
    }

    fn ncols(&self) -> usize {
        match self {
            Self::Additive(p) => p.ncols(),
        }
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), schwarz_precond::SolveError> {
        match self {
            Self::Additive(p) => p.apply(x, y),
        }
    }

    fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) -> Result<(), schwarz_precond::SolveError> {
        match self {
            Self::Additive(p) => p.apply_adjoint(x, y),
        }
    }
}

/// Build a [`FePreconditioner`] from a design, optional observation weights,
/// and configuration.
pub fn build_preconditioner<S: Store>(
    design: &Design<S>,
    weights: Option<&[f64]>,
    config: &Preconditioner,
) -> Result<FePreconditioner, BuildError> {
    use crate::domain::build_local_domains;

    validate_weights(weights, design.n_rows)?;
    match config {
        Preconditioner::Additive(local, reduction) => {
            let domains = build_local_domains(design, weights);
            let p = build_additive_with_strategy(domains, design.n_dofs, local, *reduction)?;
            Ok(FePreconditioner::Additive(p))
        }
    }
}
