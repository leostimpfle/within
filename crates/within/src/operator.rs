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
//! | **D** / **W^{1/2} D** | [`DesignOperator`] | Rectangular `D x` (or `W^{1/2} D x`); pass weights to [`DesignOperator::new`] for the weighted variant |
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

use std::sync::atomic::Ordering;

use portable_atomic::AtomicF64;
use rayon::prelude::*;
use schwarz_precond::Operator;

use crate::domain::Design;
use crate::observation::Store;

// ===========================================================================
// Iteration kernels — module-private, shared between apply / apply_adjoint
// ===========================================================================

/// Minimum number of rows before scatter/gather loops are parallelized.
const PAR_THRESHOLD: usize = 10_000;

/// Factor-level threshold for choosing between fold and atomic scatter-add.
///
/// Factors with fewer than this many levels use thread-local fold/reduce
/// (O(n_levels * n_threads) memory). Larger factors use atomic CAS instead,
/// which has low contention when bins vastly outnumber threads.
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

impl ScatterStrategy {
    /// Pick the scatter strategy for one factor.
    fn pick(parallel: bool, n_levels: usize) -> Self {
        match (parallel, n_levels < SCATTER_LOCAL_THRESHOLD) {
            (false, _) => ScatterStrategy::Sequential,
            (true, true) => ScatterStrategy::Fold,
            (true, false) => ScatterStrategy::Atomic,
        }
    }
}

/// Resolve the level for row `i` in factor `q`.
///
/// `levels` is the optional fast-path column (a contiguous `&[u32]` view of the
/// factor's levels); when `None`, fall back to the store's virtual lookup.
/// Hoisted out of inner loops so the compiler keeps the row body branch-free.
#[inline]
fn level_at<S: Store>(store: &S, levels: Option<&[u32]>, i: usize, q: usize) -> usize {
    match levels {
        Some(col) => col[i] as usize,
        None => store.level(i, q) as usize,
    }
}

/// Pre-compute the factor-column fast-path slices for all factors of `design`.
fn factor_columns<S: Store>(design: &Design<S>) -> Vec<Option<&[u32]>> {
    (0..design.factors.len())
        .map(|q| design.store.factor_column(q))
        .collect()
}

/// Gather-apply: `dst[i] = finalize(i, Σ_q src[off_q + level(i, q)])`.
///
/// `finalize` is folded into the LAST factor's pass — exactly Q sweeps over
/// `dst`, no trailing scale loop. The identity finalize (`|_, s| s`) recovers
/// the unweighted gather.
fn gather_apply<S, F>(design: &Design<S>, src: &[f64], dst: &mut [f64], finalize: F)
where
    S: Store,
    F: Fn(usize, f64) -> f64 + Sync,
{
    debug_assert_eq!(src.len(), design.n_dofs);
    debug_assert_eq!(dst.len(), design.n_rows);
    let factors = &design.factors;
    if factors.is_empty() {
        // Q=0 guard — no factors means dst[i] = finalize(i, 0.0).
        for (i, d) in dst.iter_mut().enumerate() {
            *d = finalize(i, 0.0);
        }
        return;
    }
    dst.fill(0.0);
    let factor_columns = factor_columns(design);
    let store = &design.store;
    let last = factors.len() - 1;

    let kernel = |chunk: &mut [f64], row_start: usize| {
        // Accumulate factors 0..last
        for q in 0..last {
            let f = &factors[q];
            let levels = factor_columns[q];
            for (local, dst_val) in chunk.iter_mut().enumerate() {
                let i = row_start + local;
                *dst_val += src[f.offset + level_at(store, levels, i, q)];
            }
        }
        // Last factor: accumulate AND finalize, single store per row.
        // Q=1 is well-defined: this is the only loop that runs.
        let f = &factors[last];
        let levels = factor_columns[last];
        for (local, dst_val) in chunk.iter_mut().enumerate() {
            let i = row_start + local;
            let s = *dst_val + src[f.offset + level_at(store, levels, i, last)];
            *dst_val = finalize(i, s);
        }
    };

    if design.n_rows > PAR_THRESHOLD {
        const CHUNK_SIZE: usize = 4096;
        dst.par_chunks_mut(CHUNK_SIZE)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| kernel(chunk, chunk_idx * CHUNK_SIZE));
    } else {
        kernel(dst, 0);
    }
}

/// Scatter-apply: `dst[off_q + level(i, q)] += value_fn(i)` for each factor q, row i.
///
/// Caller is responsible for any leading `dst.fill(0.0)`. For large problems,
/// each factor's row loop is parallelized:
/// - Small factors (< 100K levels): thread-local fold/reduce
/// - Large factors: atomic CAS scatter
fn scatter_apply<S, F>(design: &Design<S>, dst: &mut [f64], value_fn: F)
where
    S: Store,
    F: Fn(usize) -> f64 + Sync,
{
    debug_assert_eq!(dst.len(), design.n_dofs);
    let factor_columns = factor_columns(design);
    let parallel = design.n_rows > PAR_THRESHOLD;
    let max_levels = design.factors.iter().map(|f| f.n_levels).max().unwrap_or(0);
    let mut atomic_buf: Vec<AtomicF64> = Vec::with_capacity(max_levels);

    for (q, f) in design.factors.iter().enumerate() {
        let slice = &mut dst[f.offset..f.offset + f.n_levels];
        let levels = factor_columns[q];
        let strategy = ScatterStrategy::pick(parallel, f.n_levels);
        scatter_add_single_factor(
            slice,
            design.n_rows,
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
#[allow(clippy::too_many_arguments)]
fn scatter_add_single_factor<S: Store>(
    slice: &mut [f64],
    n_rows: usize,
    store: &S,
    levels: Option<&[u32]>,
    q: usize,
    value_fn: &(impl Fn(usize) -> f64 + Sync),
    strategy: ScatterStrategy,
    atomic_buf: &mut Vec<AtomicF64>,
) {
    let n_levels = slice.len();
    let lvl = |i: usize| level_at(store, levels, i, q);

    match strategy {
        ScatterStrategy::Sequential => {
            for i in 0..n_rows {
                slice[lvl(i)] += value_fn(i);
            }
        }
        ScatterStrategy::Fold => {
            let min_len = (n_rows / rayon::current_num_threads().max(1)).max(1024);

            let identity = || vec![0.0f64; n_levels];

            let fold = |mut acc: Vec<f64>, i| {
                acc[lvl(i)] += value_fn(i);
                acc
            };

            let reduction = |mut a: Vec<f64>, b: Vec<f64>| {
                for (ai, bi) in a.iter_mut().zip(b.iter()) {
                    *ai += *bi;
                }
                a
            };

            let result: Vec<f64> = (0..n_rows)
                .into_par_iter()
                .with_min_len(min_len)
                .fold(identity, fold)
                .reduce(identity, reduction);

            for (d, r) in slice.iter_mut().zip(result.iter()) {
                *d += *r;
            }
        }
        ScatterStrategy::Atomic => {
            atomic_buf.clear();
            atomic_buf.extend(slice.iter().map(|&v| AtomicF64::new(v)));
            (0..n_rows).into_par_iter().for_each(|i| {
                atomic_buf[lvl(i)].fetch_add(value_fn(i), Ordering::Relaxed);
            });
            for (d, a) in slice.iter_mut().zip(atomic_buf.iter()) {
                *d = a.load(Ordering::Relaxed);
            }
        }
    }
}

// ===========================================================================
// DesignOperator — D, optionally rescaled by W^{1/2}
// ===========================================================================

/// Rectangular design operator: `D` (unweighted) or `W^{1/2} D` (weighted).
///
/// `apply` = `D x` / `W^{1/2} D x` (gather), `apply_adjoint` = `D^T x` /
/// `D^T W^{1/2} x` (scatter). For the weighted variant, the normal equations
/// `A^T A = D^T W D = G` recover the Gramian, so the same Schwarz
/// preconditioner approximating `G^{-1}` applies. Pass `None` to
/// [`DesignOperator::new`] for `D`, or `Some(&w)` for `W^{1/2} D`. The branch
/// on weights is hoisted outside the per-row loop — the weighted finalize is
/// fused into the last gather sweep, and the adjoint multiplies inline through
/// a closure, so there is no scratch buffer.
pub struct DesignOperator<'a, S: Store> {
    design: &'a Design<S>,
    sqrt_weights: Option<Vec<f64>>,
}

impl<'a, S: Store> DesignOperator<'a, S> {
    /// Wrap a design matrix as a linear operator.
    ///
    /// Pass `None` for `D`, `Some(&w)` for `W^{1/2} D` (then `w.len()` must
    /// equal `design.n_rows`). Precomputes and stores `sqrt(W)` when weights
    /// are present.
    ///
    /// # Panics
    ///
    /// Panics when `weights.is_some()` and `weights.unwrap().len()` does not
    /// equal `design.n_rows`. The `Solver` entry points perform fallible
    /// validation against `BuildError::WeightCountMismatch` before
    /// construction, so callers that go through `Solver::from_design` or
    /// `solve()` never trigger this panic.
    pub fn new(design: &'a Design<S>, weights: Option<&[f64]>) -> Self {
        let sqrt_weights = weights.map(|w| {
            assert_eq!(
                w.len(),
                design.n_rows,
                "weights length {} does not match design.n_rows {}",
                w.len(),
                design.n_rows
            );
            w.iter().map(|wi| wi.sqrt()).collect()
        });
        Self {
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
        match &self.sqrt_weights {
            Some(sw) => gather_apply(self.design, x, y, |i, s| sw[i] * s),
            None => gather_apply(self.design, x, y, |_, s| s),
        }
        Ok(())
    }

    fn apply_adjoint(&self, x: &[f64], y: &mut [f64]) -> Result<(), schwarz_precond::SolveError> {
        debug_assert_eq!(x.len(), self.design.n_rows);
        debug_assert_eq!(y.len(), self.design.n_dofs);
        y.fill(0.0);
        match &self.sqrt_weights {
            Some(sw) => scatter_apply(self.design, y, |i| sw[i] * x[i]),
            None => scatter_apply(self.design, y, |i| x[i]),
        }
        Ok(())
    }
}
