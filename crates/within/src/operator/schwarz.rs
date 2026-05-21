//! Schwarz preconditioner: bridges FE domain types to the generic
//! `schwarz-precond` API, plus the opaque public [`Preconditioner`] handle.

use rayon::prelude::*;
use schwarz_precond::{Operator, SchwarzPreconditioner, SubdomainEntry};
use serde::{Deserialize, Serialize};

use crate::block_elim::BlockElimSolver;
use crate::config::{LocalSolverConfig, PreconditionerConfig};
use crate::domain::CrossTab;
use crate::domain::{Design, Subdomain};
use crate::observation::{validate_weights, Store};
use crate::BuildError;

/// Concrete additive Schwarz type used in the parent crate.
#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct FeSchwarz(SchwarzPreconditioner<BlockElimSolver>);

impl std::fmt::Debug for FeSchwarz {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FeSchwarz")
            .field("n_subdomains", &self.0.subdomains().len())
            .finish()
    }
}

impl FeSchwarz {
    pub(crate) fn new(inner: SchwarzPreconditioner<BlockElimSolver>) -> Self {
        Self(inner)
    }
}

impl Operator for FeSchwarz {
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
    config: &LocalSolverConfig,
    strategy: schwarz_precond::ReductionStrategy,
) -> Result<FeSchwarz, BuildError> {
    let entries = build_entries_from_pairs(domains, config)?;
    Ok(FeSchwarz::new(SchwarzPreconditioner::new(
        entries, strategy,
    )))
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
    let solver = BlockElimSolver::build(cross_tab, config)?;
    SubdomainEntry::try_new(domain.core, solver).map_err(BuildError::Preconditioner)
}

// ---------------------------------------------------------------------------
// Opaque public preconditioner handle
// ---------------------------------------------------------------------------

/// Opaque handle to a pre-built fixed-effects preconditioner.
///
/// # Cheap clone
///
/// Cloning a `Preconditioner` is O(1): it bumps `Arc` refcounts on the inner
/// subdomain vector (`Arc<Vec<SubdomainEntry<_>>>`) and the scratch
/// `BufferPool` shared by `schwarz_precond::SchwarzPreconditioner`. No
/// factorization state is deep-copied (see that crate's `Clone` impl). This
/// is what makes `From<&Preconditioner> for PreconditionerInput` honest —
/// passing `&precond` to [`Solver::new`] does not duplicate the factorization.
///
/// # State invariants
///
/// Factorization state is immutable after build: the per-subdomain solvers
/// and their backing storage never change once a `Preconditioner` exists.
/// The one piece of interior mutability is a scratch `BufferPool:
/// Arc<Mutex<Vec<SchwarzBuffers>>>` inside the underlying Schwarz executor,
/// used by `apply` for take/put of temporary buffers; it is fully
/// encapsulated and not observable to callers.
///
/// # For future maintainers
///
/// This contract holds for the current [`Variant::Additive`] variant. Any
/// new `Variant` added below MUST preserve both invariants:
/// 1. `Clone` must be O(1) — wrap any heavy per-variant state in an
///    `Arc` rather than deep-copying it.
/// 2. Factorization state must be immutable after build (interior
///    mutability is allowed only for opaque scratch like `BufferPool`).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Preconditioner {
    inner: Variant,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
enum Variant {
    Additive(FeSchwarz),
}

impl Preconditioner {
    pub(crate) fn additive(inner: FeSchwarz) -> Self {
        Self {
            inner: Variant::Additive(inner),
        }
    }

    /// Number of rows of the underlying linear operator.
    pub fn nrows(&self) -> usize {
        <Self as schwarz_precond::Operator>::nrows(self)
    }

    /// Number of columns of the underlying linear operator.
    pub fn ncols(&self) -> usize {
        <Self as schwarz_precond::Operator>::ncols(self)
    }

    /// Apply the preconditioner: writes `M^{-1} x` into `y`.
    pub fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), schwarz_precond::SolveError> {
        <Self as schwarz_precond::Operator>::apply(self, x, y)
    }
}

impl Operator for Preconditioner {
    fn nrows(&self) -> usize {
        match &self.inner {
            Variant::Additive(p) => p.nrows(),
        }
    }

    fn ncols(&self) -> usize {
        match &self.inner {
            Variant::Additive(p) => p.ncols(),
        }
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), schwarz_precond::SolveError> {
        match &self.inner {
            Variant::Additive(p) => p.apply(x, y),
        }
    }

    fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) -> Result<(), schwarz_precond::SolveError> {
        match &self.inner {
            Variant::Additive(p) => p.apply_adjoint(x, y),
        }
    }
}

/// Build a [`Preconditioner`] from a design and optional observation weights.
pub(crate) fn build_preconditioner<S: Store>(
    design: &Design<S>,
    weights: Option<&[f64]>,
    config: Option<&PreconditionerConfig>,
) -> Result<Option<Preconditioner>, BuildError> {
    use crate::domain::build_local_domains;

    validate_weights(weights, design.n_obs)?;

    let default_cfg = PreconditionerConfig::default();
    let resolved = config.unwrap_or(&default_cfg);
    match resolved {
        PreconditionerConfig::Off => Ok(None),
        PreconditionerConfig::Additive {
            local_solver,
            reduction,
        } => {
            let domains = build_local_domains(design, weights);
            if domains.is_empty() {
                // Single-factor designs (and other configurations with no
                // factor-pair subdomains) have no useful additive Schwarz
                // preconditioner. Fall back to unpreconditioned LSMR.
                return Ok(None);
            }
            let p = build_additive_with_strategy(domains, local_solver, *reduction)?;
            Ok(Some(Preconditioner::additive(p)))
        }
    }
}
