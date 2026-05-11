//! Error types for the `within` crate.
//!
//! Errors are partitioned by lifecycle phase:
//!
//! - **Build** ([`BuildError`]) — input validation and operator/preconditioner
//!   construction failures.
//! - **Solve** ([`SolveError`]) — runtime failures during the iterative solve.
//!   Re-exported from [`schwarz_precond`] because this crate adds no new
//!   solve-time failure modes.
//! - **Union** ([`WithinError`]) — a thin convenience union used by
//!   [`crate::solve`] and [`crate::solve_batch`] so callers see a single
//!   error type at the top-level API boundary.
//!
//! Per-phase functions return [`Result<T, BuildError>`] or
//! [`Result<T, SolveError>`] directly. Only the top-level convenience
//! wrappers return [`WithinError`].

use thiserror::Error;

pub use schwarz_precond::SolveError;

/// Errors produced while validating inputs or building solver components.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum BuildError {
    /// No observations provided.
    #[error("no observations provided")]
    EmptyObservations,
    /// One factor column does not match the expected observation count.
    #[error("factor {factor} has {got} observations, expected {expected}")]
    ObservationCountMismatch {
        /// Index of the factor with mismatched length.
        factor: usize,
        /// Expected number of observations.
        expected: usize,
        /// Actual number of observations in this factor's column.
        got: usize,
    },
    /// Weight vector does not match the number of observations.
    #[error("weights has length {got}, expected {expected}")]
    WeightCountMismatch {
        /// Expected number of weights.
        expected: usize,
        /// Actual weight vector length.
        got: usize,
    },
    /// A zero diagonal was encountered during block elimination.
    #[error("zero diagonal in {block} block at index {index}")]
    SingularDiagonal {
        /// Which block contained the zero diagonal ("keep" or "elim").
        block: &'static str,
        /// Row/column index of the zero diagonal entry.
        index: usize,
    },
    /// Local solver construction failed.
    #[error("local solver build failed: {0}")]
    LocalSolverBuild(String),
    /// Schwarz preconditioner structural validation failed.
    ///
    /// Lifted from [`schwarz_precond::BuildError`] explicitly at call sites
    /// via `.map_err(BuildError::Preconditioner)`; no `From` conversion is
    /// provided so the cross-crate boundary stays visible.
    #[error("preconditioner build failed: {0}")]
    Preconditioner(#[source] schwarz_precond::BuildError),
}

/// Top-level error type returned by [`crate::solve`] and [`crate::solve_batch`].
///
/// Lifts the per-phase [`BuildError`] / [`SolveError`] pair into a single
/// union at the convenience-wrapper boundary. Per-phase APIs do not return
/// this type. The wrapping variants are transparent: [`Display`] and
/// [`Error::source`] both forward to the inner error, so the union does not
/// appear in the error chain.
#[derive(Debug, Error)]
pub enum WithinError {
    /// Build-time failure: validation or operator/preconditioner construction.
    #[error(transparent)]
    Build(#[from] BuildError),
    /// Solve-time failure: iterative solver runtime error.
    #[error(transparent)]
    Solve(#[from] SolveError),
}
