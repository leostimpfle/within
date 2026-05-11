//! Error types for the `schwarz-precond` crate.
//!
//! Errors are partitioned by lifecycle phase:
//!
//! - **Build** ([`BuildError`]) — caught during construction, before any
//!   solve begins. Covers partition-weight validation, subdomain DOF/scratch
//!   contracts, and preconditioner-wide index checks.
//! - **Solve** ([`SolveError`]) — runtime failures during a solve, including
//!   operator/preconditioner application (e.g. a local solver diverges) and
//!   iterative-solver input validation.
//!
//! [`LocalSolveError`] is the narrow trait contract returned by
//! [`LocalSolver::solve_local`](crate::LocalSolver::solve_local). The Schwarz
//! executor lifts it into [`SolveError::LocalSolveFailed`] at the one apply
//! site that knows the subdomain index, so there is no `From` chain between
//! the two.

use thiserror::Error;

use crate::local_solve::{LocalSolver, SubdomainEntry};

/// Construction-time validation errors for the Schwarz building blocks.
///
/// Consolidates failures from subdomain core/entry construction and
/// preconditioner-wide index checks into a single flat enum.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum BuildError {
    /// Partition-of-unity weight vector length does not match index count.
    #[error("partition weight count ({weight_count}) does not match index count ({index_count})")]
    PartitionWeightLengthMismatch {
        /// Number of global indices in the subdomain core.
        index_count: usize,
        /// Number of partition weights in the subdomain core.
        weight_count: usize,
    },
    /// Local solver `n_local` does not match the subdomain index count.
    #[error("index count ({index_count}) does not match solver n_local ({solver_n_local})")]
    LocalDofCountMismatch {
        /// Number of global indices in the subdomain core.
        index_count: usize,
        /// Local DOF count reported by the solver implementation.
        solver_n_local: usize,
    },
    /// Local solver scratch size is too small for the subdomain gather/scatter buffers.
    #[error("scratch size ({scratch_size}) is smaller than required minimum ({required_min})")]
    ScratchSizeTooSmall {
        /// Scratch size reported by the local solver.
        scratch_size: usize,
        /// Minimum scratch size required by the subdomain core.
        required_min: usize,
    },
    /// A subdomain references a global DOF outside `[0, n_dofs)`.
    #[error(
        "subdomain {subdomain}: global index at local position {local_index} is out of bounds ({global_index} >= {n_dofs})"
    )]
    GlobalIndexOutOfBounds {
        /// Index of the failing subdomain entry in the provided list.
        subdomain: usize,
        /// Position inside `global_indices` where the invalid DOF was found.
        local_index: usize,
        /// Global DOF index that exceeded the valid range.
        global_index: u32,
        /// Total number of global DOFs configured for the preconditioner.
        n_dofs: usize,
    },
}

/// Runtime error emitted by a local subdomain solver during a solve call.
///
/// Returned by [`LocalSolver::solve_local`](crate::LocalSolver::solve_local).
/// Backend-agnostic by design: backends report a `context` site and a
/// free-form `message` rather than enumerating their internal error modes
/// through this generic crate.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum LocalSolveError {
    /// The backend implementation reported a failure during a local solve.
    #[error("{context}: {message}")]
    BackendFailed {
        /// Context string identifying where the failure occurred.
        context: &'static str,
        /// Backend error text.
        message: String,
    },
}

/// Runtime failure while executing a solve.
///
/// Covers both operator/preconditioner application failures (e.g. a local
/// solver diverges) and iterative-solver input validation.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum SolveError {
    /// A local subdomain solve failed during a preconditioner apply.
    #[error("subdomain {subdomain} local solve failed: {source}")]
    LocalSolveFailed {
        /// Index of the failing subdomain entry in the preconditioner.
        subdomain: usize,
        /// Local solver error.
        #[source]
        source: LocalSolveError,
    },
    /// Internal synchronization failed (e.g. poisoned mutex) during an apply.
    #[error("synchronization failure at {context}")]
    Synchronization {
        /// Context string identifying the lock/synchronization site.
        context: &'static str,
    },
    /// Solver input was invalid before any iteration was attempted.
    #[error("invalid solver input at {context}: {message}")]
    InvalidInput {
        /// Context string identifying the validation site.
        context: &'static str,
        /// Validation failure details.
        message: String,
    },
}

pub(crate) fn validate_entries<S: LocalSolver>(
    entries: &[SubdomainEntry<S>],
    n_dofs: usize,
) -> Result<(), BuildError> {
    for (subdomain, entry) in entries.iter().enumerate() {
        for (local_index, &global_index) in entry.global_indices().iter().enumerate() {
            if (global_index as usize) >= n_dofs {
                return Err(BuildError::GlobalIndexOutOfBounds {
                    subdomain,
                    local_index,
                    global_index,
                    n_dofs,
                });
            }
        }
    }
    Ok(())
}
