//! Preconditioner enum dispatch and fused build paths.
//!
//! [`FePreconditioner`] is the top-level preconditioner type used by the
//! [`orchestrate`](crate::orchestrate) layer. It currently wraps a single
//! Schwarz variant:
//!
//! - **Additive** ([`FeSchwarz`]) — symmetric. Subdomains contribute
//!   independently and their corrections are summed.
//!
//! # Integration with `schwarz-precond`
//!
//! The enum implements the [`Operator`] trait from the `schwarz-precond`
//! crate, so it can be passed directly to LSMR as a preconditioner. Error
//! handling flows through `try_apply` for graceful reporting of local-solver
//! failures.

use schwarz_precond::{AdditiveSchwarzDiagnostics, LocalSolver, Operator, ReductionStrategy};
use serde::{Deserialize, Serialize};

use crate::config::Preconditioner;
use crate::domain::{Subdomain, WeightedDesign};
use crate::observation::ObservationStore;
use crate::operator::gramian::CrossTab;
use crate::operator::schwarz::{build_additive_with_strategy, FeSchwarz};
use crate::WithinResult;

/// A pre-built preconditioner ready for use in LSMR solves.
///
/// Implements [`Operator`] via enum dispatch to the inner variant.
#[derive(Clone, Serialize, Deserialize)]
pub enum FePreconditioner {
    /// Additive Schwarz.
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
}

/// Configured additive reduction strategy.
pub fn additive_reduction_strategy(preconditioner: &FePreconditioner) -> ReductionStrategy {
    match preconditioner {
        FePreconditioner::Additive(p) => p.reduction_strategy(),
    }
}

/// Concrete additive backend selected for the current Rayon thread-pool width.
pub fn resolved_additive_reduction_strategy(
    preconditioner: &FePreconditioner,
) -> ReductionStrategy {
    match preconditioner {
        FePreconditioner::Additive(p) => p.resolved_reduction_strategy(),
    }
}

/// Build-time additive Schwarz scheduling diagnostics.
pub fn additive_schwarz_diagnostics(
    preconditioner: &FePreconditioner,
) -> AdditiveSchwarzDiagnostics {
    match preconditioner {
        FePreconditioner::Additive(p) => p.diagnostics(),
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

    fn apply(&self, x: &[f64], y: &mut [f64]) {
        match self {
            Self::Additive(p) => p.apply(x, y),
        }
    }

    fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) {
        match self {
            Self::Additive(p) => p.apply_adjoint(x, y),
        }
    }

    fn try_apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), schwarz_precond::ApplyError> {
        match self {
            Self::Additive(p) => p.try_apply(x, y),
        }
    }

    fn try_apply_adjoint(
        &self,
        x: &[f64],
        y: &mut [f64],
    ) -> Result<(), schwarz_precond::ApplyError> {
        match self {
            Self::Additive(p) => p.try_apply_adjoint(x, y),
        }
    }
}

/// Build a [`FePreconditioner`] from pre-built domains and configuration.
fn build_from_domains(
    domains: Vec<(Subdomain, CrossTab)>,
    n_dofs: usize,
    config: &Preconditioner,
) -> WithinResult<FePreconditioner> {
    match config {
        Preconditioner::Additive(solver_config, strategy) => {
            let p = build_additive_with_strategy(domains, n_dofs, solver_config, *strategy)?;
            Ok(FePreconditioner::Additive(p))
        }
    }
}

/// Build a [`FePreconditioner`] from a design and configuration.
pub fn build_preconditioner<S: ObservationStore>(
    design: &WeightedDesign<S>,
    preconditioner_config: &Preconditioner,
) -> WithinResult<FePreconditioner> {
    use crate::domain::build_local_domains;

    let domains = build_local_domains(design);
    build_from_domains(domains, design.n_dofs, preconditioner_config)
}
