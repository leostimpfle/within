//! Additive Schwarz preconditioner: `M⁻¹ = Σ Rᵢᵀ D̃ᵢ Aᵢ⁻¹ D̃ᵢ Rᵢ`.
//!
//! Implements the [`Operator`](crate::Operator) trait, so it can be passed
//! directly to an iterative solver as a preconditioner.
//!
//! All subdomain local solves run in parallel (via Rayon). The per-subdomain
//! corrections are combined into the global output vector by one of two
//! reduction strategies (chosen automatically or configured explicitly):
//!
//! - **Atomic scatter** — each task atomically adds its weighted correction
//!   into a shared `AtomicU64` accumulator. Low memory (`O(n_dofs)` shared),
//!   best when overlap is low.
//! - **Parallel reduction** — each Rayon worker accumulates into a private
//!   `Vec<f64>`, then a final parallel reduction sums them. Higher memory
//!   (`O(P × n_dofs)` where P = active workers) but avoids atomic contention
//!   when overlap is high.
//!
//! Internal structure:
//! - `planning` — [`ReductionStrategy`] enum, `Auto` heuristic, build-time
//!   diagnostics
//! - `executor` — owns the subdomain entries and dispatches `try_apply`
//! - `buffers` — pooled scratch and accumulator buffers for zero-allocation
//!   steady-state apply
//! - `preconditioner` — the public [`SchwarzPreconditioner`] type

mod buffers;
mod executor;
mod planning;
mod preconditioner;

pub use planning::{AdditiveSchwarzDiagnostics, ReductionStrategy};
pub use preconditioner::SchwarzPreconditioner;
