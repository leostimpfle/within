//! Linear algebra layer: operator representations and preconditioner wiring.
//!
//! This module is the hub between the [`domain`](crate::domain) layer (which
//! builds subdomains from panel data) and the [`orchestrate`](crate::orchestrate)
//! layer (the public solve API). It provides the rectangular design operator
//! that LSMR consumes and the Schwarz preconditioner construction.
//!
//! # Operators
//!
//! LSMR works directly on `sqrt(W) D` rather than the assembled Gramian
//! `G = D^T W D`. The matvec primitive is:
//!
//! | Operator | Type | Description |
//! |---|---|---|
//! | **sqrt(W) D** | [`DesignOperator`] | Rectangular operator used by LSMR. Implements `sqrt(W) D x` and `D^T sqrt(W) x` via gather/scatter on the observation store |
//!
//! # Submodules
//!
//! - [`gramian::cross_tab`](gramian) — Per-pair `CrossTab` blocks driving
//!   factor-pair subdomain construction (the only Gramian-shaped artefact
//!   the solver still needs)
//! - [`schwarz`] — Schwarz preconditioner construction: bridges fixed-effects
//!   types to the generic `schwarz-precond` API
//! - `local_solver` — Local subdomain solvers: approximate Cholesky (SDDM)
//!   and block-elimination backends
//! - `schur_complement` — Exact and approximate Schur complement computation
//!   for block-elimination local solves
//! - [`preconditioner`] — [`FePreconditioner`](preconditioner::FePreconditioner)
//!   wrapping the additive Schwarz variant
//! - `csr_block` — Internal rectangular CSR block used in bipartite blocks

pub(crate) mod csr_block;
pub(crate) mod gramian;
pub(crate) mod local_solver;
pub mod preconditioner;
pub(crate) mod schur_complement;
pub(crate) mod schwarz;

#[cfg(test)]
mod tests;

// ---------------------------------------------------------------------------
// DesignOperator — rectangular, W^{1/2}·D·x / D^T·W^{1/2}·x
// ---------------------------------------------------------------------------

use std::sync::atomic::Ordering;
use std::sync::Mutex;

use portable_atomic::AtomicF64;
use rayon::prelude::*;
use schwarz_precond::Operator;

use crate::domain::Design;
use crate::observation::Store;

/// Weighted rectangular design operator: `A = W^{1/2} D`.
///
/// `apply` = `W^{1/2} D x` (observation space), `apply_adjoint` = `D^T W^{1/2} x` (DOF space).
/// For unweighted designs, delegates directly to `D x` / `D^T x` with no extra work.
///
/// The normal equations of this operator give `A^T A = D^T W D = G` (the Gramian),
/// so the existing Schwarz preconditioner approximating `G^{-1}` can be used directly.
pub struct DesignOperator<'a, S: Store> {
    design: &'a Design<S>,
    /// Pre-computed `sqrt(w_i)` per observation. `None` when unweighted.
    sqrt_weights: Option<Vec<f64>>,
    /// Scratch for the adjoint path: stores `sqrt(w_i) * u_i`.
    scratch: Mutex<Vec<f64>>,
}

impl<'a, S: Store> DesignOperator<'a, S> {
    /// Create from a design matrix and optional observation weights.
    ///
    /// `weights = None` selects the unweighted fast-path: `sqrt_weights` is
    /// `None` and `apply` / `apply_adjoint` skip the per-row scaling entirely.
    pub fn new(design: &'a Design<S>, weights: Option<&[f64]>) -> Self {
        let sqrt_weights = weights.map(|w| w.iter().map(|wi| wi.sqrt()).collect::<Vec<f64>>());
        Self {
            scratch: Mutex::new(vec![0.0; design.n_rows]),
            design,
            sqrt_weights,
        }
    }

    /// Compute the observation-space RHS `b = W^{1/2} y`.
    ///
    /// For unweighted designs, returns a copy of `y`.
    pub fn weighted_rhs(&self, y: &[f64]) -> Vec<f64> {
        match &self.sqrt_weights {
            None => y.to_vec(),
            Some(sw) => y.iter().zip(sw).map(|(&yi, &swi)| swi * yi).collect(),
        }
    }
}

impl<S: Store> Operator for DesignOperator<'_, S> {
    fn nrows(&self) -> usize {
        self.design.n_rows
    }

    fn ncols(&self) -> usize {
        self.design.n_dofs
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), schwarz_precond::SolveError> {
        debug_assert_eq!(x.len(), self.design.n_dofs);
        debug_assert_eq!(y.len(), self.design.n_rows);
        // y = W^{1/2} (D x)
        y.fill(0.0);
        gather_add(self.design, x, y);
        if let Some(sw) = &self.sqrt_weights {
            for (yi, &swi) in y.iter_mut().zip(sw) {
                *yi *= swi;
            }
        }
        Ok(())
    }

    fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) -> Result<(), schwarz_precond::SolveError> {
        debug_assert_eq!(x.len(), self.design.n_rows);
        debug_assert_eq!(y.len(), self.design.n_dofs);
        // y = D^T (W^{1/2} x)
        y.fill(0.0);
        match &self.sqrt_weights {
            None => scatter_add(self.design, y, |i| x[i]),
            Some(sw) => {
                let mut tmp = self.scratch.lock().unwrap();
                for (ti, (&xi, &swi)) in tmp.iter_mut().zip(x.iter().zip(sw)) {
                    *ti = swi * xi;
                }
                scatter_add(self.design, y, |i| tmp[i]);
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// gather/scatter helpers — implementation of DesignOperator's apply/apply_adjoint
// ---------------------------------------------------------------------------
//
// These free functions implement the per-row scatter/gather over a `Design`.
// They live in this module (not in `domain.rs`) because the design itself is
// pure data + layout; these helpers compute the linear map, which is an
// operator concern.

/// Minimum number of rows before scatter/gather loops are parallelized.
const PAR_THRESHOLD: usize = 10_000;

/// Factor-level threshold for choosing between fold and atomic scatter-add.
///
/// Factors with fewer than this many levels use thread-local fold/reduce
/// (O(n_levels * n_threads) memory). Larger factors use atomic CAS instead,
/// which has low contention when bins vastly outnumber threads.
/// 100K levels * 8 bytes * ~24 Rayon tasks ~ 19 MB — fits comfortably.
const SCATTER_LOCAL_THRESHOLD: usize = 100_000;

/// Strategy for a single factor's scatter-add loop.
enum ScatterStrategy {
    /// Plain sequential loop — used when n_rows is below `PAR_THRESHOLD`.
    Sequential,
    /// Parallel fold/reduce with thread-local accumulators — for small factors.
    Fold,
    /// Parallel atomic CAS — for large factors with low contention.
    Atomic,
}

#[inline]
fn level_from_column_or_store<S: Store>(
    store: &S,
    levels: Option<&[u32]>,
    row: usize,
    factor: usize,
) -> usize {
    match levels {
        Some(col) => col[row] as usize,
        None => store.level(row, factor) as usize,
    }
}

/// Pre-compute factor column slices for all factors of `design`.
fn factor_columns<S: Store>(design: &Design<S>) -> Vec<Option<&[u32]>> {
    design
        .factors
        .iter()
        .enumerate()
        .map(|(q, _)| design.store.factor_column(q))
        .collect()
}

/// Gather-add: `dst[i] += src[offset_q + level(i, q)]` for each factor `q` and row `i`.
///
/// This is the core loop of `y = D·x`. Loop order is chosen based on the
/// store's preferred iteration pattern for cache locality.
///
/// For large problems (n_rows > 10 000), rows are partitioned into chunks
/// and processed in parallel via Rayon `par_chunks_mut`.
fn gather_add<S: Store>(design: &Design<S>, src: &[f64], dst: &mut [f64]) {
    const CHUNK_SIZE: usize = 4096;
    let factor_columns = factor_columns(design);

    if design.n_rows > PAR_THRESHOLD {
        // Parallel path: each chunk processes its own row range.
        // The inner loop iterates factors inside each chunk, which is optimal for the
        // common case (2-3 factors) where all factor data fits in L1 cache. For many
        // factors (10+) a layout with factors in the outer loop might help, but
        // econometric models typically have 2-5 factors so this isn't worth optimizing.
        dst.par_chunks_mut(CHUNK_SIZE)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let row_start = chunk_idx * CHUNK_SIZE;
                for (q, f) in design.factors.iter().enumerate() {
                    let levels = factor_columns[q];
                    for (local, dst_val) in chunk.iter_mut().enumerate() {
                        let i = row_start + local;
                        let level = level_from_column_or_store(&design.store, levels, i, q);
                        *dst_val += src[f.offset + level];
                    }
                }
            });
    } else {
        // Sequential factor-major: outer loop on factors, inner on observations.
        for (q, f) in design.factors.iter().enumerate() {
            let levels = factor_columns[q];
            for (i, dst_i) in dst.iter_mut().enumerate().take(design.n_rows) {
                let level = level_from_column_or_store(&design.store, levels, i, q);
                *dst_i += src[f.offset + level];
            }
        }
    }
}

/// Scatter-add: `dst[offset_q + level(i, q)] += value_fn(i)` for each factor `q` and row `i`.
///
/// This is the core loop of `x = D^T · r` (and weighted variant `D^T · W · r`).
/// The `value_fn` closure computes the per-row contribution:
/// - unweighted: `|i| r[i]`
/// - weighted:   `|i| w[i] * r[i]`
///
/// For large problems, each factor's row loop is parallelized:
/// - Small factors (< 100K levels): thread-local fold/reduce (avoids CAS contention)
/// - Large factors: atomic CAS scatter (low contention on millions of bins)
///   Factors are processed sequentially so each gets the full thread pool.
fn scatter_add<S: Store>(
    design: &Design<S>,
    dst: &mut [f64],
    value_fn: impl Fn(usize) -> f64 + Sync,
) {
    let factor_columns = factor_columns(design);
    let parallel = design.n_rows > PAR_THRESHOLD;
    let max_levels = design.factors.iter().map(|f| f.n_levels).max().unwrap_or(0);
    let mut atomic_buf: Vec<AtomicF64> = Vec::with_capacity(max_levels);

    for (q, f) in design.factors.iter().enumerate() {
        let slice = &mut dst[f.offset..f.offset + f.n_levels];
        let levels = factor_columns[q];
        let strategy = if !parallel {
            ScatterStrategy::Sequential
        } else if f.n_levels < SCATTER_LOCAL_THRESHOLD {
            ScatterStrategy::Fold
        } else {
            ScatterStrategy::Atomic
        };
        scatter_add_single_factor(
            slice,
            design.n_rows,
            f.n_levels,
            &design.store,
            levels,
            q,
            &value_fn,
            strategy,
            &mut atomic_buf,
        );
    }
}

/// Scatter-add for a single factor, dispatched by strategy.
///
/// Accumulates `value_fn(i)` into `slice[level(i, q)]` for all rows, using the
/// requested parallelization strategy. The `atomic_buf` is reused across calls
/// to avoid repeated allocation in the `Atomic` path.
///
/// All branches share the same per-row `(level, value)` computation via
/// `level_value`; only the accumulation strategy differs.
#[allow(clippy::too_many_arguments)]
fn scatter_add_single_factor<S: Store>(
    slice: &mut [f64],
    n_rows: usize,
    n_levels: usize,
    store: &S,
    levels: Option<&[u32]>,
    q: usize,
    value_fn: &(impl Fn(usize) -> f64 + Sync),
    strategy: ScatterStrategy,
    atomic_buf: &mut Vec<AtomicF64>,
) {
    #[inline(always)]
    fn level_value<S: Store>(
        store: &S,
        levels: Option<&[u32]>,
        q: usize,
        value_fn: &impl Fn(usize) -> f64,
        i: usize,
    ) -> (usize, f64) {
        let level = level_from_column_or_store(store, levels, i, q);
        (level, value_fn(i))
    }

    match strategy {
        ScatterStrategy::Sequential => {
            for i in 0..n_rows {
                let (level, val) = level_value(store, levels, q, value_fn, i);
                slice[level] += val;
            }
        }
        ScatterStrategy::Fold => {
            let min_len = (n_rows / rayon::current_num_threads().max(1)).max(1024);
            let result: Vec<f64> = (0..n_rows)
                .into_par_iter()
                .with_min_len(min_len)
                .fold(
                    || vec![0.0f64; n_levels],
                    |mut acc, i| {
                        let (level, val) = level_value(store, levels, q, value_fn, i);
                        acc[level] += val;
                        acc
                    },
                )
                .reduce(
                    || vec![0.0f64; n_levels],
                    |mut a, b| {
                        for (ai, bi) in a.iter_mut().zip(b.iter()) {
                            *ai += *bi;
                        }
                        a
                    },
                );
            for (d, r) in slice.iter_mut().zip(result.iter()) {
                *d += *r;
            }
        }
        ScatterStrategy::Atomic => {
            atomic_buf.clear();
            atomic_buf.extend(slice.iter().map(|&v| AtomicF64::new(v)));
            (0..n_rows).into_par_iter().for_each(|i| {
                let (level, val) = level_value(store, levels, q, value_fn, i);
                atomic_buf[level].fetch_add(val, Ordering::Relaxed);
            });
            for (d, a) in slice.iter_mut().zip(atomic_buf.iter()) {
                *d = a.load(Ordering::Relaxed);
            }
        }
    }
}
