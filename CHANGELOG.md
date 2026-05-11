# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

Modified LSMR is now the sole iterative solver, replacing CG and GMRES.

### Added

- Preconditioned modified LSMR on `sqrt(W) D`, avoiding explicit
  normal-equation formation.
- `SolverParams.local_size: Option<usize>` — optional windowed modified
  Gram-Schmidt reorthogonalization for ill-conditioned problems.
- `LsmrStopReason` and `stop_reason` on the LSMR result.
- `lsmr`/`mlsmr` reject non-finite input with `SolveError::InvalidInput`
  (Python: `ValueError`); previously silent NaN propagation.

### Changed

- **BREAKING:** `Preconditioner`, `FePreconditioner`, and
  `schwarz_precond::SolveError` are now `#[non_exhaustive]`. External
  `match` sites need a wildcard arm. `SolveError` gains an
  `InvalidInput { context, message }` variant.
- **BREAKING:** `LocalSolver::solve_local` now takes an
  `allow_inner_parallelism: bool` policy hint. Implementations with no
  nested-parallel region can ignore it (`_allow_inner_parallelism`).
  `SchwarzPreconditioner` no longer carries the `I: LocalSolveInvoker`
  type parameter; the `LocalSolveInvoker` trait, `DefaultLocalSolveInvoker`,
  and `with_strategy_and_invoker` constructor are removed.
- LSMR vector kernels parallelized via Rayon.

### Removed

- **BREAKING:** CG, GMRES, multiplicative Schwarz, iterative refinement,
  and their support types (`KrylovMethod`, `OperatorRepr`,
  `Preconditioner::Multiplicative`, `FePreconditioner::Multiplicative`,
  `SolverParams.max_refinements`, Python `CG`/`GMRES`/
  `MultiplicativeSchwarz`, `MultiplicativeSchwarzPreconditioner`,
  `ResidualUpdater`, `OperatorResidualUpdater`, `IdentityOperator`).
- **BREAKING:** `Gramian`, `GramianOperator`, `DesignOperator`,
  `build_schwarz`, `FeSchwarz`, and `WithinError::Overflow` removed from
  the `within` public surface. LSMR uses `WeightedDesignOperator` directly.
- **BREAKING:** `schwarz_precond::solve::{cg, gmres}` and the `solve`
  module removed; use crate-root `lsmr`/`mlsmr`.
  `schwarz_precond::schwarz::{additive, multiplicative}` flattened into
  `schwarz_precond::schwarz`; crate-root `SchwarzPreconditioner` re-export
  unchanged.
- **BREAKING:** `AdditiveSchwarzDiagnostics` removed from Rust and Python
  public APIs along with `SchwarzPreconditioner::diagnostics`,
  `FePreconditioner::additive_schwarz_diagnostics`, and the Python
  `AdditiveSchwarzDiagnostics` class. Scheduling metrics are now private
  to the `Auto` heuristic (closes #34).
- **BREAKING:** `Operator::apply` / `apply_adjoint` now return
  `Result<(), SolveError>`; the infallible pair and the `try_apply` /
  `try_apply_adjoint` defaults are gone. `ApplyError` is removed; its
  variants moved onto `SolveError`. `PyFePreconditioner.apply` raises
  `RuntimeError` on local-solver failure instead of returning NaNs
  (closes #29).

## [0.1.0] - 2026-03-12

Initial release of `within`, a high-performance fixed-effects solver for
econometric panel data.

### Added

- **Iterative solvers:** Left-preconditioned CG and right-preconditioned
  GMRES(m) with restarts, stagnation detection, and lucky breakdown handling.
- **Schwarz preconditioners:** Additive (CG-compatible, symmetric) and
  multiplicative (sequential sweep with sparse residual update). Additive
  variant auto-selects between atomic scatter and parallel reduction strategies.
- **Domain decomposition:** Bipartite factor-pair subdomains with connected
  component splitting and partition-of-unity weights.
- **Schur complement reduction:** Exact dense path for small subdomains,
  exact sparse path, and approximate path via GKS clique-tree spectral
  sparsification.
- **Approximate Cholesky local solver** via the `approx-chol` crate, with
  block elimination exploiting bipartite Gramian structure.
- **Dual operator representations:** Explicit CSR Gramian (fused sortless
  assembly from pair blocks) and implicit D^T W D (three-pass, no matrix
  stored).
- **Iterative refinement** with adaptive inner tolerance for observation-space
  accuracy.
- **Batch solve** with Rayon parallelism over RHS vectors, sharing the
  precomputed preconditioner.
- **Persistent `Solver` class** for amortizing preconditioner construction
  across multiple solves.
- **Weighted least squares** support via observation weights.
- **Preconditioner serialization** via `postcard` for Python pickle support.
- **Python API** via PyO3/maturin: `solve()`, `solve_batch()`, `Solver`,
  with full type stubs and GIL release during computation.
- **Zero-copy Python boundary** for F-contiguous category arrays and
  contiguous response vectors.
- **Three Rust crates:** `schwarz-precond` (generic, reusable),
  `within` (FE domain), `within-py` (thin PyO3 bridge).
- Benchmark infrastructure: Criterion micro-benchmarks (Rust) and 18-suite
  Python benchmark framework with setup/solve/accuracy measurement.
- CI/CD: Multi-platform testing (Linux, macOS, Windows), clippy with
  `-D warnings`, `#![deny(missing_docs)]` on library crates, and
  multi-architecture wheel builds with build attestation.

### Performance

- Adaptive additive Schwarz reduction scheduling with `Auto`, `AtomicScatter`,
  and `ParallelReduction` backends.
- Worker-local reusable reduction buffers for additive parallel reduction.
- Fused Schur right-hand-side assembly and related reduced-system kernel
  cleanup.

### Fixed

- Nested Rayon deadlocks in additive `ParallelReduction` when local solves
  spawn inner parallel work.
