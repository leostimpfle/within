//! Additive Schwarz preconditioner
//!
//! Implements the [`Operator`](crate::Operator) trait, so it
//! can be passed directly to an iterative solver as a preconditioner.
//!
//! - [`additive`] — `M⁻¹ = Σ Rᵢᵀ D̃ᵢ Aᵢ⁻¹ D̃ᵢ Rᵢ`: independent local solves
//!   combined via atomic scatter or parallel reduction. Symmetric.

mod additive;
pub use additive::{AdditiveSchwarzDiagnostics, ReductionStrategy, SchwarzPreconditioner};
