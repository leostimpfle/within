//! Cross-tabulation primitives for factor-pair Schwarz subdomains.
//!
//! Each factor pair `(q, r)` has a bipartite cross-tabulation `C_{qr}` whose
//! entry `[i, j]` counts the (weighted) observations at level `i` of factor
//! `q` and level `j` of factor `r`. These cross-tabs feed the local-solver
//! construction (Schur complement reduction) and the bipartite-component
//! decomposition that splits each factor pair into independent subdomains.
//!
//! The LSMR solver does not assemble the full Gramian `G = D^T W D`; it
//! works on the rectangular operator `sqrt(W) D` directly via
//! [`crate::operator::WeightedDesignOperator`]. Only per-pair `CrossTab`s are
//! needed, and only for preconditioner construction.

mod cross_tab;
#[cfg(test)]
mod tests;

pub(crate) use cross_tab::{find_all_active_levels, BipartiteComponent, CrossTab};
