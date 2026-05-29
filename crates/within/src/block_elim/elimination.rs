//! Block-elimination metadata and star iteration for Schur complement assembly.
//!
//! For the bipartite SDDM `[D_q, -C; -C^T, D_r]`, eliminating the larger
//! diagonal block (exact since it's diagonal) yields a reduced Laplacian-style
//! system on the smaller block. This module owns the block-selection decision,
//! precomputed inverse-diagonals, and zero-copy [`Star`] views used by both
//! Schur complement strategies.

use approx_chol::low_level::{clique_tree_sample, clique_tree_sample_multi};
use rayon::prelude::*;

use crate::config::ApproxSchurConfig;
use crate::csr_block::CsrBlock;
use crate::domain::CrossTab;
use crate::BuildError;

/// Undirected fill edge: `(lo_col, hi_col, weight)` with `lo_col < hi_col`.
pub(crate) type Edge = (u32, u32, f64);

// ===========================================================================
// EliminationInfo — handed to BlockElimSolver after Schur assembly
// ===========================================================================

/// Metadata from the block-elimination step needed by `BlockElimSolver`.
pub(crate) struct EliminationInfo {
    /// 1 / D_elim[k] for the eliminated diagonal block.
    pub(crate) inv_diag_elim: Vec<f64>,
    /// True if the q-block was eliminated (n_q >= n_r).
    pub(crate) eliminate_q: bool,
}

// ===========================================================================
// Star — zero-copy neighborhood view
// ===========================================================================

// The Schur complement S = D_keep - C_keep^T * D_elim^{-1} * C_keep arises from
// block-eliminating the diagonal block of the larger partition in the bipartite
// SDDM system [D_q, -C; -C^T, D_r]. Since D_elim is diagonal, the elimination
// is exact and each eliminated vertex k contributes a rank-1 clique (star) to
// the fill graph: all pairs of k's neighbors in the keep-block get a fill edge.
//
// Two strategies materialize these fill edges:
// - `ExactCliqueEmitter` (not shown here; the exact path uses row-workspace
//   accumulation in `SchurLaplacian::from_elimination` instead)
// - `SampledCliqueEmitter`: uses GKS 2023 clique-tree sampling to approximate
//   high-degree cliques with O(deg) edges instead of O(deg^2), keeping the
//   Schur complement spectrally close to the exact one.

/// One eliminated vertex's neighbors in the keep-block.
///
/// References into [`CsrBlock`]'s arrays for zero-copy access.
pub(crate) struct Star<'a> {
    /// Eliminated vertex index (used for deterministic seeding).
    index: usize,
    /// Neighbor columns in the keep-block.
    col_indices: &'a [u32],
    /// Edge weights to each neighbor.
    weights: &'a [f64],
}

impl Star<'_> {
    pub(crate) fn degree(&self) -> usize {
        self.col_indices.len()
    }
}

/// Emits sampled clique-tree fill edges for every star.
pub(crate) struct SampledCliqueEmitter {
    seed: u64,
    split: u32,
}

impl SampledCliqueEmitter {
    pub(crate) fn new(config: &ApproxSchurConfig) -> Self {
        Self {
            seed: config.seed,
            split: config.split,
        }
    }

    fn emit(&self, star: &Star, edges: &mut Vec<Edge>, scratch: &mut Vec<(u32, f64)>) {
        scratch.clear();
        for (&col, &w) in star.col_indices.iter().zip(star.weights) {
            scratch.push((col, w));
        }
        let seed = self.seed.wrapping_add(star.index as u64);
        if self.split <= 1 {
            clique_tree_sample(scratch, seed, edges);
        } else {
            clique_tree_sample_multi(scratch, self.split, seed, edges);
        }
    }
}

// ===========================================================================
// Elimination — block selection + star iteration
// ===========================================================================

/// Block-selection decision and star iteration for Schur elimination.
///
/// Encapsulates which block to eliminate, precomputed inverse-diagonals,
/// and provides zero-copy [`Star`] views for each eliminated vertex.
pub(crate) struct Elimination<'a> {
    pub(crate) eliminate_q: bool,
    pub(crate) n_keep: usize,
    pub(crate) n_elim: usize,
    pub(crate) inv_diag_elim: Vec<f64>,
    pub(crate) diag_keep: &'a [f64],
    pub(crate) keep_to_elim: &'a CsrBlock,
    pub(crate) elim_to_keep: &'a CsrBlock,
}

impl<'a> Elimination<'a> {
    /// Select which block to eliminate and precompute inverse-diagonals.
    pub(crate) fn new(cross_tab: &'a CrossTab) -> Result<Self, BuildError> {
        let n_q = cross_tab.n_q();
        let n_r = cross_tab.n_r();
        // Eliminate the larger block to minimize the reduced system size.
        let eliminate_q = n_q >= n_r;
        let (n_keep, n_elim) = if eliminate_q { (n_r, n_q) } else { (n_q, n_r) };

        let diag_elim = if eliminate_q {
            &cross_tab.diag_q
        } else {
            &cross_tab.diag_r
        };
        let inv_diag_elim = diag_elim
            .iter()
            .enumerate()
            .map(|(i, &d)| {
                if d > 0.0 {
                    Ok(1.0 / d)
                } else {
                    Err(BuildError::SingularDiagonal {
                        block: if eliminate_q { "q (elim)" } else { "r (elim)" },
                        index: i,
                    })
                }
            })
            .collect::<Result<_, _>>()?;

        let diag_keep = if eliminate_q {
            &cross_tab.diag_r
        } else {
            &cross_tab.diag_q
        };

        let (keep_to_elim, elim_to_keep) = if eliminate_q {
            (&cross_tab.ct, &cross_tab.c)
        } else {
            (&cross_tab.c, &cross_tab.ct)
        };

        Ok(Self {
            eliminate_q,
            n_keep,
            n_elim,
            inv_diag_elim,
            diag_keep,
            keep_to_elim,
            elim_to_keep,
        })
    }

    /// Create a zero-copy [`Star`] view for eliminated vertex `k`.
    fn star(&self, k: usize) -> Star<'_> {
        let start = self.elim_to_keep.indptr[k] as usize;
        let end = self.elim_to_keep.indptr[k + 1] as usize;
        Star {
            index: k,
            col_indices: &self.elim_to_keep.indices[start..end],
            weights: &self.elim_to_keep.data[start..end],
        }
    }

    pub(crate) fn par_emit(&self, emitter: &SampledCliqueEmitter) -> Vec<Edge> {
        (0..self.n_elim)
            .into_par_iter()
            .fold(
                || (Vec::new(), Vec::<(u32, f64)>::new()),
                |(mut edges, mut scratch), k| {
                    let star = self.star(k);
                    if star.degree() > 1 {
                        emitter.emit(&star, &mut edges, &mut scratch);
                    }
                    (edges, scratch)
                },
            )
            .map(|(mut edges, _)| {
                Self::sort_and_dedup(&mut edges);
                edges
            })
            .reduce(Vec::new, Self::merge_dedup)
    }

    /// Package elimination metadata into [`EliminationInfo`] for the solver.
    pub(crate) fn into_info(self) -> EliminationInfo {
        EliminationInfo {
            inv_diag_elim: self.inv_diag_elim,
            eliminate_q: self.eliminate_q,
        }
    }

    /// Sort edges by `(lo, hi)` and merge duplicates by summing weights.
    fn sort_and_dedup(edges: &mut Vec<Edge>) {
        edges.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
        if edges.len() <= 1 {
            return;
        }
        let mut write = 0;
        for read in 1..edges.len() {
            if edges[write].0 == edges[read].0 && edges[write].1 == edges[read].1 {
                edges[write].2 += edges[read].2;
            } else {
                write += 1;
                edges[write] = edges[read];
            }
        }
        edges.truncate(write + 1);
    }

    /// Merge two sorted, deduplicated edge lists, summing weights for duplicates.
    fn merge_dedup(a: Vec<Edge>, b: Vec<Edge>) -> Vec<Edge> {
        if a.is_empty() {
            return b;
        }
        if b.is_empty() {
            return a;
        }
        let mut result = Vec::with_capacity(a.len() + b.len());
        let (mut ia, mut ib) = (0, 0);
        while ia < a.len() && ib < b.len() {
            let ka = (a[ia].0, a[ia].1);
            let kb = (b[ib].0, b[ib].1);
            match ka.cmp(&kb) {
                std::cmp::Ordering::Less => {
                    result.push(a[ia]);
                    ia += 1;
                }
                std::cmp::Ordering::Greater => {
                    result.push(b[ib]);
                    ib += 1;
                }
                std::cmp::Ordering::Equal => {
                    result.push((a[ia].0, a[ia].1, a[ia].2 + b[ib].2));
                    ia += 1;
                    ib += 1;
                }
            }
        }
        if ia < a.len() {
            result.extend_from_slice(&a[ia..]);
        }
        if ib < b.len() {
            result.extend_from_slice(&b[ib..]);
        }
        result
    }
}
