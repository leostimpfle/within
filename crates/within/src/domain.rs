//! Domain layer: design matrix metadata and factor-pair subdomain construction.
//!
//! This module sits between raw observation storage ([`crate::observation`]) and
//! the linear-algebra operators ([`crate::operator`]).  It answers two questions:
//!
//! 1. **What does the design matrix look like?** — [`Design`] wraps a [`Store`]
//!    with per-factor metadata ([`FactorMeta`]). It is *pure data + layout*: it
//!    knows the number of rows, the number of DOFs, and how to recover factor
//!    levels per observation. The matrix-vector products live next door in
//!    [`crate::operator::DesignOperator`].
//!
//! 2. **How is the problem decomposed into subdomains?** — The `factor_pairs`
//!    submodule builds one [`Subdomain`] per connected component of each factor
//!    pair, with partition-of-unity weights that ensure the additive Schwarz
//!    preconditioner is mathematically correct.
//!
//! # Design matrix structure
//!
//! The design matrix **D** is a block matrix with one block per factor. Each
//! block is a "one-hot" matrix: observation (row) *i* has a single 1
//! corresponding to its level in that factor. With Q factors and `n_q` levels
//! each, D has shape `(n_obs, sum(n_q))` and exactly Q nonzeros per row.
//!
//! ```text
//! D = [ D_1 | D_2 | ... | D_Q ]     (n_obs × n_dofs)
//!
//! where D_q[i, j] = 1  if observation i has level j in factor q
//!                    0  otherwise
//! ```
//!
//! The coefficient vector **x** is laid out as `[x_1, x_2, ..., x_Q]` where
//! `x_q` starts at `factors[q].offset` and has length `factors[q].n_levels`.
//!
//! # Domain decomposition and factor pairs
//!
//! The normal-equation Gramian `G = D^T W D` has a natural block structure:
//! diagonal blocks are diagonal matrices (weighted level counts) and off-diagonal
//! blocks `D_q^T W D_r` capture the co-occurrence between each pair of factors.
//! Each factor pair `(q, r)` defines a subdomain whose DOFs are the union of
//! active levels in factors q and r. When the factor-pair bipartite graph has
//! multiple connected components, each component becomes a separate subdomain.
//!
//! This decomposition maps directly onto the Schwarz method: each subdomain
//! gets a local solver, and the partition-of-unity weights ensure that
//! overlapping DOFs (levels that appear in multiple factor pairs) are correctly
//! scaled. See the `factor_pairs` submodule for details.

pub(crate) mod factor_pairs;

pub(crate) use factor_pairs::build_local_domains;

// Re-exports from schwarz-precond
pub use schwarz_precond::PartitionWeights;
pub use schwarz_precond::SubdomainCore;

/// A local subdomain corresponding to a pair of factors.
#[derive(Clone)]
pub struct Subdomain {
    /// Indices `(q, r)` of the two factors this subdomain covers.
    pub factor_pair: (usize, usize),
    /// Generic subdomain core: global DOF indices, restriction, and partition-of-unity weights.
    pub core: SubdomainCore,
}

impl std::fmt::Debug for Subdomain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Subdomain")
            .field("factor_pair", &self.factor_pair)
            .field("n_dofs", &self.core.n_local())
            .finish()
    }
}

// ===========================================================================
// Design — categorical fixed-effects design (data + layout)
// ===========================================================================

use crate::observation::{FactorMeta, Store};
use crate::{WithinError, WithinResult};

/// Fixed-effects design, generic over observation storage.
///
/// `store` holds per-observation factor levels; `factors` holds per-factor
/// metadata (n_levels, offset). The `Design` itself is pure data + layout —
/// matrix-vector products live in [`crate::operator::DesignOperator`].
pub struct Design<S: Store> {
    /// Observation storage backend (owns or borrows the raw factor levels).
    pub store: S,
    /// Per-factor metadata: level count and global DOF offset.
    pub factors: Vec<FactorMeta>,
    /// Number of observations (rows of D).
    pub n_rows: usize,
    /// Total degrees of freedom (columns of D = sum of levels across factors).
    pub n_dofs: usize,
}

impl<S: Store + Clone> Clone for Design<S> {
    fn clone(&self) -> Self {
        Self {
            store: self.store.clone(),
            factors: self.factors.clone(),
            n_rows: self.n_rows,
            n_dofs: self.n_dofs,
        }
    }
}

impl<S: Store + std::fmt::Debug> std::fmt::Debug for Design<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Design")
            .field("store", &self.store)
            .field("factors", &self.factors)
            .field("n_rows", &self.n_rows)
            .field("n_dofs", &self.n_dofs)
            .finish()
    }
}

impl<S: Store> Design<S> {
    /// Construct from a store, inferring the number of levels per factor
    /// from the maximum observed level in each column (`max + 1`).
    pub fn from_store(store: S) -> WithinResult<Self> {
        if store.n_obs() == 0 {
            return Err(WithinError::EmptyObservations);
        }

        let mut factors = Vec::with_capacity(store.n_factors());
        let mut offset = 0;
        for q in 0..store.n_factors() {
            let n_levels = (0..store.n_obs())
                .map(|uid| store.level(uid, q) as usize + 1)
                .max()
                .unwrap(); // safe: n_obs > 0
            factors.push(FactorMeta { n_levels, offset });
            offset += n_levels;
        }
        let n_rows = store.n_obs();
        Ok(Design {
            store,
            factors,
            n_rows,
            n_dofs: offset,
        })
    }

    /// Number of categorical factors in the design.
    #[inline]
    pub fn n_factors(&self) -> usize {
        self.factors.len()
    }
}
