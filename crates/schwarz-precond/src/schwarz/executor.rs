//! Additive Schwarz execution engine and its scratch-buffer types.
//!
//! [`AdditiveExecutor`] owns the subdomain entries and a [`BufferPool`] that
//! transitively carries the global sizes. Its `apply` method takes a
//! reduction plan from the scheduler and dispatches to either the
//! atomic-scatter or the parallel-reduction backend. Buffers are taken from
//! / returned to the pool so the steady state allocates nothing.
//!
//! Two buffer layouts exist, matching the two reduction strategies:
//!
//! - [`SchwarzBuffers::Atomic`] — a single shared `Vec<AtomicU64>` accumulator
//! - [`SchwarzBuffers::Reduction`] — a pool of per-worker
//!   [`AdditiveSweepBuffers`], each containing a private `Vec<f64>`
//!   accumulator plus local-solve scratch
//!
//! [`WorkerReductionBuffers`] manages the worker-local buffer stacks for
//! the parallel-reduction path, using `ThreadLocal` to give each Rayon
//! worker its own reusable buffer without cross-thread synchronization
//! in the hot loop.

use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use rayon::prelude::*;
use thread_local::ThreadLocal;

use crate::error::SolveError;
use crate::local_solve::{LocalSolver, SubdomainEntry};

use super::planning::{ReductionPlan, ResolvedReductionStrategy};

// ============================================================================
// Buffer pooling
// ============================================================================

pub(super) struct BufferPool {
    n_dofs: usize,
    max_scratch_size: usize,
    inner: Arc<Mutex<Vec<SchwarzBuffers>>>,
}

impl BufferPool {
    const MAX_POOL_SIZE: usize = 4;

    pub(super) fn new(n_dofs: usize, max_scratch_size: usize) -> Self {
        Self {
            n_dofs,
            max_scratch_size,
            inner: Arc::default(),
        }
    }

    pub(super) fn n_dofs(&self) -> usize {
        self.n_dofs
    }

    pub(super) fn max_scratch_size(&self) -> usize {
        self.max_scratch_size
    }

    fn take(&self, strategy: ResolvedReductionStrategy) -> Result<SchwarzBuffers, SolveError> {
        let mut pool = self.inner.lock().map_err(|_| SolveError::Synchronization {
            context: "additive.buf_pool.lock.pop",
        })?;
        if let Some(idx) = pool.iter().position(|bufs| bufs.strategy() == strategy) {
            return Ok(pool.swap_remove(idx));
        }
        Ok(SchwarzBuffers::new(
            strategy,
            self.n_dofs,
            self.max_scratch_size,
        ))
    }

    /// Return a buffer to the pool. Infallible by design: pool bookkeeping
    /// must never mask the caller's real `apply_result`. On the error path the
    /// buffer is dropped (see below); on a poisoned pool lock the buffer is
    /// likewise dropped rather than surfaced as a `Synchronization` error.
    fn put(&self, bufs: SchwarzBuffers, apply_result: &Result<(), SolveError>) {
        // On error, the atomic backend's swap-zero readout pass is skipped,
        // leaving stale partial-write values in the AtomicU64 vec. Drop the
        // buffer rather than pooling it for the next caller to inherit dirty
        // state.
        if apply_result.is_err() {
            return;
        }
        // A poisoned lock means a worker panicked; just drop the buffer (the
        // pool lazily re-allocates on the next `take`) instead of erroring.
        if let Ok(mut pool) = self.inner.lock() {
            if pool.len() < Self::MAX_POOL_SIZE {
                pool.push(bufs);
            }
        }
    }
}

impl Clone for BufferPool {
    fn clone(&self) -> Self {
        Self {
            n_dofs: self.n_dofs,
            max_scratch_size: self.max_scratch_size,
            inner: Arc::clone(&self.inner),
        }
    }
}

struct LocalSolveScratch {
    r_scratch: Vec<f64>,
    z_scratch: Vec<f64>,
}

impl LocalSolveScratch {
    #[inline]
    fn new(max_scratch_size: usize) -> Self {
        Self {
            r_scratch: vec![0.0f64; max_scratch_size],
            z_scratch: vec![0.0f64; max_scratch_size],
        }
    }
}

/// Task-local scratch for the parallel-reduction path.
struct AdditiveSweepBuffers {
    global_accum: Vec<f64>,
    scratch: LocalSolveScratch,
}

impl AdditiveSweepBuffers {
    fn new(n_dofs: usize, max_scratch_size: usize) -> Self {
        Self {
            global_accum: vec![0.0f64; n_dofs],
            scratch: LocalSolveScratch::new(max_scratch_size),
        }
    }
}

/// Pooled buffers that vary by reduction strategy.
enum SchwarzBuffers {
    /// Shared atomic accumulator.
    Atomic { accum: Vec<AtomicU64> },
    /// Reusable task-local buffers for parallel reduction.
    Reduction { pool: Vec<AdditiveSweepBuffers> },
}

impl SchwarzBuffers {
    fn new(strategy: ResolvedReductionStrategy, n_dofs: usize, max_scratch_size: usize) -> Self {
        match strategy {
            ResolvedReductionStrategy::AtomicScatter => Self::Atomic {
                accum: (0..n_dofs).map(|_| AtomicU64::new(0)).collect(),
            },
            ResolvedReductionStrategy::ParallelReduction => Self::Reduction {
                pool: vec![AdditiveSweepBuffers::new(n_dofs, max_scratch_size)],
            },
        }
    }

    fn strategy(&self) -> ResolvedReductionStrategy {
        match self {
            Self::Atomic { .. } => ResolvedReductionStrategy::AtomicScatter,
            Self::Reduction { .. } => ResolvedReductionStrategy::ParallelReduction,
        }
    }
}

/// Worker-local buffer stacks for additive parallel reduction.
///
/// Each Rayon worker reuses its own accumulator buffers across sequential outer
/// tasks. Nested re-entry on the same worker allocates a second buffer only when
/// needed, so the number of retained full-length accumulators tracks re-entry
/// depth rather than Rayon task splitting.
struct WorkerReductionBuffers {
    shared_pool: Mutex<Vec<AdditiveSweepBuffers>>,
    worker_stacks: ThreadLocal<RefCell<Vec<AdditiveSweepBuffers>>>,
    n_dofs: usize,
    max_scratch_size: usize,
}

impl WorkerReductionBuffers {
    fn new(pool: Vec<AdditiveSweepBuffers>, n_dofs: usize, max_scratch_size: usize) -> Self {
        Self {
            shared_pool: Mutex::new(pool),
            worker_stacks: ThreadLocal::with_capacity(rayon::current_num_threads().max(1)),
            n_dofs,
            max_scratch_size,
        }
    }

    fn with_buffer<T>(&self, f: impl FnOnce(&mut AdditiveSweepBuffers) -> T) -> T {
        let worker_stack = self.worker_stacks.get_or(|| RefCell::new(Vec::new()));
        let mut buffers = if let Some(buffers) = worker_stack.borrow_mut().pop() {
            buffers
        } else {
            self.take_or_alloc()
        };

        let result = f(&mut buffers);
        self.worker_stacks
            .get_or(|| RefCell::new(Vec::new()))
            .borrow_mut()
            .push(buffers);
        result
    }

    fn take_or_alloc(&self) -> AdditiveSweepBuffers {
        self.shared_pool
            .lock()
            .ok()
            .and_then(|mut pool| pool.pop())
            .unwrap_or_else(|| AdditiveSweepBuffers::new(self.n_dofs, self.max_scratch_size))
    }

    fn finish_round(
        self,
        z: &mut [f64],
        apply_result: &Result<(), SolveError>,
    ) -> Result<Vec<AdditiveSweepBuffers>, SolveError> {
        // Always leave `z` fully written so a failed apply never exposes a
        // partial accumulation. On the apply-error path zero `z` up front —
        // there is nothing to reduce, and this also covers the case where the
        // subsequent pool recovery fails.
        if apply_result.is_err() {
            z.fill(0.0);
        }
        let mut buffers = self.into_buffers()?;
        // On success, `reduce_into` zeroes-then-sums into `z`.
        if apply_result.is_ok() {
            Self::reduce_into(z, &buffers);
        }
        for b in &mut buffers {
            b.global_accum.fill(0.0);
        }
        Ok(buffers)
    }

    fn into_buffers(mut self) -> Result<Vec<AdditiveSweepBuffers>, SolveError> {
        let mut buffers =
            self.shared_pool
                .into_inner()
                .map_err(|_| SolveError::Synchronization {
                    context: "additive.reduction.pool.into_inner",
                })?;
        for worker_stack in self.worker_stacks.iter_mut() {
            buffers.append(worker_stack.get_mut());
        }
        Ok(buffers)
    }

    fn reduce_into(z: &mut [f64], buffers: &[AdditiveSweepBuffers]) {
        if buffers.is_empty() {
            z.fill(0.0);
            return;
        }

        const REDUCE_CHUNK: usize = 4096;
        z.par_chunks_mut(REDUCE_CHUNK)
            .enumerate()
            .for_each(|(ci, chunk)| {
                let offset = ci * REDUCE_CHUNK;
                chunk.fill(0.0);
                for buffers in buffers {
                    let accum = &buffers.global_accum[offset..offset + chunk.len()];
                    for (zi, &ai) in chunk.iter_mut().zip(accum) {
                        *zi += ai;
                    }
                }
            });
    }
}

// ============================================================================
// Additive executor
// ============================================================================

pub(super) struct AdditiveExecutor<S: LocalSolver> {
    subdomains: Arc<Vec<SubdomainEntry<S>>>,
    buf_pool: BufferPool,
}

impl<S: LocalSolver> AdditiveExecutor<S> {
    pub(super) fn new(
        subdomains: Arc<Vec<SubdomainEntry<S>>>,
        n_dofs: usize,
        max_scratch_size: usize,
    ) -> Self {
        Self {
            subdomains,
            buf_pool: BufferPool::new(n_dofs, max_scratch_size),
        }
    }

    pub(super) fn subdomains(&self) -> &[SubdomainEntry<S>] {
        &self.subdomains
    }

    pub(super) fn n_dofs(&self) -> usize {
        self.buf_pool.n_dofs()
    }

    pub(super) fn n_subdomains(&self) -> usize {
        self.subdomains.len()
    }

    /// Dispatch entry point: take a buffer from the pool, run the backend,
    /// return the buffer. The pool size is bounded, so the steady state
    /// allocates nothing.
    pub(super) fn apply(
        &self,
        plan: ReductionPlan,
        r: &[f64],
        z: &mut [f64],
    ) -> Result<(), SolveError> {
        let mut bufs = self.buf_pool.take(plan.strategy)?;
        let apply_result = match &mut bufs {
            SchwarzBuffers::Atomic { accum } => {
                self.apply_atomic(plan.allow_inner_parallelism, r, z, accum)
            }
            SchwarzBuffers::Reduction { pool } => {
                self.apply_parallel_reduction(plan.allow_inner_parallelism, r, z, pool)
            }
        };
        // `put` is infallible and never overwrites the real `apply_result`.
        self.buf_pool.put(bufs, &apply_result);
        apply_result
    }

    fn apply_atomic(
        &self,
        allow_inner_parallelism: bool,
        r: &[f64],
        z: &mut [f64],
        accum: &[AtomicU64],
    ) -> Result<(), SolveError> {
        let max_scratch_size = self.buf_pool.max_scratch_size();
        self.subdomains.par_iter().enumerate().try_for_each_init(
            || LocalSolveScratch::new(max_scratch_size),
            |scratch, (subdomain, entry)| {
                entry
                    .apply_weighted_into_atomic(
                        r,
                        accum,
                        &mut scratch.r_scratch,
                        &mut scratch.z_scratch,
                        allow_inner_parallelism,
                    )
                    .map_err(|source| SolveError::LocalSolveFailed { subdomain, source })
            },
        )?;

        const READOUT_CHUNK: usize = 4096;
        z.par_chunks_mut(READOUT_CHUNK)
            .enumerate()
            .for_each(|(ci, chunk)| {
                let offset = ci * READOUT_CHUNK;
                for (i, zi) in chunk.iter_mut().enumerate() {
                    let ai = &accum[offset + i];
                    *zi = f64::from_bits(ai.swap(0, Ordering::Relaxed));
                }
            });
        Ok(())
    }

    fn apply_parallel_reduction(
        &self,
        allow_inner_parallelism: bool,
        r: &[f64],
        z: &mut [f64],
        pool: &mut Vec<AdditiveSweepBuffers>,
    ) -> Result<(), SolveError> {
        let worker_buffers = WorkerReductionBuffers::new(
            std::mem::take(pool),
            self.buf_pool.n_dofs(),
            self.buf_pool.max_scratch_size(),
        );
        let apply_result =
            self.subdomains
                .par_iter()
                .enumerate()
                .try_for_each(|(subdomain, entry)| {
                    worker_buffers.with_buffer(|buffers| {
                        entry
                            .apply_weighted_into_with_scratch(
                                r,
                                &mut buffers.global_accum,
                                &mut buffers.scratch.r_scratch,
                                &mut buffers.scratch.z_scratch,
                                allow_inner_parallelism,
                            )
                            .map_err(|source| SolveError::LocalSolveFailed { subdomain, source })
                    })
                });

        // `finish_round` writes `z` (zeroed on the apply-error path) and
        // recovers the pool. A pool-recovery failure must not mask a real
        // `LocalSolveFailed`, so prefer the original error when it is one.
        match worker_buffers.finish_round(z, &apply_result) {
            Ok(recovered) => *pool = recovered,
            Err(finish_err) => return apply_result.and(Err(finish_err)),
        }

        apply_result
    }
}

impl<S: LocalSolver> Clone for AdditiveExecutor<S> {
    fn clone(&self) -> Self {
        Self {
            subdomains: Arc::clone(&self.subdomains),
            buf_pool: self.buf_pool.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// On `put` with `Err`, the pool must drop the buffer so the next caller
    /// gets a freshly-zeroed allocation rather than inheriting stale atomic
    /// state from a partially-completed atomic-scatter pass.
    #[test]
    fn buffer_pool_drops_dirty_buffer_on_error() {
        let pool = BufferPool::new(8, 4);

        let mut bufs = pool
            .take(ResolvedReductionStrategy::AtomicScatter)
            .expect("first take");
        match &mut bufs {
            SchwarzBuffers::Atomic { accum } => {
                for slot in accum {
                    slot.store(0xdead_beef_dead_beef, Ordering::Relaxed);
                }
            }
            _ => panic!("expected atomic buffer"),
        }

        pool.put(
            bufs,
            &Err(SolveError::Synchronization {
                context: "test.simulated_failure",
            }),
        );

        let fresh = pool
            .take(ResolvedReductionStrategy::AtomicScatter)
            .expect("second take");
        match &fresh {
            SchwarzBuffers::Atomic { accum } => {
                for (i, slot) in accum.iter().enumerate() {
                    assert_eq!(
                        slot.load(Ordering::Relaxed),
                        0,
                        "atomic accumulator slot {i} should be freshly zero, not stale dirty bits"
                    );
                }
            }
            _ => panic!("expected atomic buffer"),
        }
    }

    /// Companion: on `put` with `Ok`, the pool retains the buffer and a
    /// subsequent `take` of the same strategy returns the pooled instance.
    #[test]
    fn buffer_pool_retains_clean_buffer_on_success() {
        let pool = BufferPool::new(8, 4);

        let bufs = pool
            .take(ResolvedReductionStrategy::AtomicScatter)
            .expect("first take");
        pool.put(bufs, &Ok(()));

        let _ = pool
            .take(ResolvedReductionStrategy::AtomicScatter)
            .expect("second take");
        let pool_after = pool.inner.lock().expect("pool lock");
        assert!(
            pool_after.is_empty(),
            "second take should have drained the pooled buffer"
        );
    }
}
