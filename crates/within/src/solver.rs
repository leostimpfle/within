//! Persistent solver that caches the preconditioner across multiple solves on the same design.

use std::time::Instant;

use ndarray::ArrayView2;
use rayon::prelude::*;
use schwarz_precond::{lsmr as lsmr_solve, mlsmr, Operator as _};

use crate::config::{LsmrOptions, PreconditionerConfig};
use crate::domain::Design;
use crate::observation::{validate_weights, ArrayStore, Store};
use crate::operator::schwarz::{build_preconditioner, Preconditioner};
use crate::operator::DesignOperator;
use crate::orchestrate::{BatchSolveResult, SolveResult};
use crate::{BuildError, SolveError};

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

// ---------------------------------------------------------------------------
// Solver::new inputs (polymorphic via traits)
// ---------------------------------------------------------------------------

/// Fallible conversion into a [`Design`] for [`Solver::new`].
///
/// Implemented for:
/// - `ArrayView2<'a, u32>` — categories matrix; an `ArrayStore`-backed [`Design`] is built
/// - `Design<S>` — pass-through for an already-built design
///
/// Implemented as a custom trait rather than `Into` because the raw-view path
/// is fallible and `From`/`Into` are not.
pub trait IntoDesign<'a> {
    /// Storage backend the resulting [`Design`] uses.
    type Store: Store;
    /// Build the [`Design`], validating inputs along the way.
    fn into_design(self) -> Result<Design<Self::Store>, BuildError>;
}

impl<'a> IntoDesign<'a> for ArrayView2<'a, u32> {
    type Store = ArrayStore<'a>;
    fn into_design(self) -> Result<Design<ArrayStore<'a>>, BuildError> {
        Design::from_store(ArrayStore::new(self)?)
    }
}

impl<S: Store> IntoDesign<'_> for Design<S> {
    type Store = S;
    fn into_design(self) -> Result<Design<S>, BuildError> {
        Ok(self)
    }
}

/// Preconditioner input for [`Solver::new`].
///
/// Constructed implicitly via `From`/`Into` from any of:
/// - bare `None` — build the library default Schwarz preconditioner
/// - `&PreconditionerConfig` or `Some(&PreconditionerConfig)` — build from a tuned config
/// - `PreconditionerConfig` (owned) — same as above
/// - [`Preconditioner`] (owned or `&`) — reuse a previously built (or deserialized) preconditioner
///
/// `None` resolves unambiguously because there is exactly one `From<Option<X>>`
/// impl (with `X = &PreconditionerConfig`).
pub enum PreconditionerInput {
    /// Library default: an additive Schwarz preconditioner with default tuning.
    Default,
    /// Build from this config (`PreconditionerConfig::Off` ⇒ unpreconditioned).
    Config(PreconditionerConfig),
    /// Reuse this pre-built preconditioner (e.g. deserialized or pulled off a previous solver).
    Prebuilt(Preconditioner),
}

impl From<PreconditionerConfig> for PreconditionerInput {
    fn from(c: PreconditionerConfig) -> Self {
        Self::Config(c)
    }
}

impl From<&PreconditionerConfig> for PreconditionerInput {
    fn from(c: &PreconditionerConfig) -> Self {
        Self::Config(c.clone())
    }
}

impl From<Option<&PreconditionerConfig>> for PreconditionerInput {
    fn from(opt: Option<&PreconditionerConfig>) -> Self {
        opt.map_or(Self::Default, |c| Self::Config(c.clone()))
    }
}

impl From<Preconditioner> for PreconditionerInput {
    fn from(p: Preconditioner) -> Self {
        Self::Prebuilt(p)
    }
}

impl From<&Preconditioner> for PreconditionerInput {
    /// Reuse a preconditioner by reference. Cloning is O(1) (see
    /// [`Preconditioner`] docs) so passing `&precond` is essentially free.
    fn from(p: &Preconditioner) -> Self {
        Self::Prebuilt(p.clone())
    }
}

// ---------------------------------------------------------------------------
// Solver
// ---------------------------------------------------------------------------

/// Persistent solver that owns its preconditioner for reuse across multiple solves.
///
/// Build once with [`Solver::new`], then call [`Solver::solve`] or
/// [`Solver::solve_batch`] repeatedly with different RHS vectors. The expensive
/// preconditioner factorization happens only at construction time; LSMR tuning
/// ([`LsmrOptions`]) is supplied per call.
pub struct Solver<S: Store> {
    design: Design<S>,
    weights: Option<Vec<f64>>,
    preconditioner: Option<Preconditioner>,
}

impl<S: Store> std::fmt::Debug for Solver<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Solver")
            .field("n_obs", &self.design.n_obs)
            .field("n_dofs", &self.design.n_dofs)
            .field("has_weights", &self.weights.is_some())
            .field("has_preconditioner", &self.preconditioner.is_some())
            .finish()
    }
}

impl<S: Store> Solver<S> {
    /// Construct a solver.
    ///
    /// `design` accepts raw categories (`ArrayView2<u32>`) or a pre-built
    /// [`Design`]. `preconditioner` accepts:
    /// - `None` — build the library default Schwarz preconditioner
    /// - `&PreconditionerConfig` / `Some(&PreconditionerConfig)` — build from a tuned config
    /// - `PreconditionerConfig::Off` — solve unpreconditioned
    /// - `PreconditionerConfig::Diagonal` — use diagonal/Jacobi preconditioning
    /// - [`Preconditioner`] or `&Preconditioner` — reuse a previously built (or deserialized) preconditioner
    ///
    /// `weights` accepts any `Option<impl Into<Vec<f64>>>` — pass `None` for
    /// unweighted, or an owned `Vec<f64>` / `&[f64]`. The slice form is cloned
    /// once at construction.
    ///
    /// LSMR tuning ([`LsmrOptions`]) is supplied per call to [`Solver::solve`] /
    /// [`Solver::solve_batch`], not at construction; preconditioner factorization
    /// state is the only expensive thing built here.
    pub fn new<'a>(
        design: impl IntoDesign<'a, Store = S>,
        weights: Option<impl Into<Vec<f64>>>,
        preconditioner: impl Into<PreconditionerInput>,
    ) -> Result<Self, BuildError> {
        let design = design.into_design()?;
        let weights = weights.map(Into::into);
        validate_weights(weights.as_deref(), design.n_obs)?;

        let preconditioner = match preconditioner.into() {
            PreconditionerInput::Default => {
                build_preconditioner(&design, weights.as_deref(), None)?
            }
            PreconditionerInput::Config(c) => {
                build_preconditioner(&design, weights.as_deref(), Some(&c))?
            }
            PreconditionerInput::Prebuilt(p) => {
                if p.nrows() != design.n_dofs || p.ncols() != design.n_dofs {
                    return Err(BuildError::PreconditionerDimensionMismatch {
                        expected: design.n_dofs,
                        actual_rows: p.nrows(),
                        actual_cols: p.ncols(),
                    });
                }
                Some(p)
            }
        };

        Ok(Self {
            design,
            weights,
            preconditioner,
        })
    }

    /// Solve for a single RHS vector with the given LSMR tuning.
    pub fn solve(&self, y: &[f64], lsmr: &LsmrOptions) -> Result<SolveResult, SolveError> {
        // Guard the silent-truncation hole: weighted_rhs zips y with sqrt-weights,
        // which would otherwise discard trailing values when y.len() > n_rows.
        if y.len() != self.design.n_obs {
            return Err(SolveError::InvalidInput {
                context: "Solver::solve",
                message: format!(
                    "response vector length ({}) does not match number of observations ({})",
                    y.len(),
                    self.design.n_obs
                ),
            });
        }

        let t_start = Instant::now();
        let t_setup_start = Instant::now();

        let rect_op = DesignOperator::new(&self.design, self.weights.as_deref());
        let b = rect_op.weighted_rhs(y);

        let t_solve_start = Instant::now();
        let time_setup = t_solve_start.duration_since(t_setup_start).as_secs_f64();

        let r = match self.preconditioner.as_ref() {
            Some(p) => mlsmr(&rect_op, &b, p, lsmr.tol, lsmr.maxiter, lsmr.local_size)?,
            None => lsmr_solve(&rect_op, &b, lsmr.tol, lsmr.maxiter, lsmr.local_size)?,
        };

        let time_solve = t_solve_start.elapsed().as_secs_f64();

        // demeaned = y - D x. The bare unweighted `D x` matvec uses an
        // unweighted DesignOperator over the same design.
        let bare_op = DesignOperator::new(&self.design, None);
        let mut demeaned = vec![0.0; self.design.n_obs];
        bare_op.apply(&r.x, &mut demeaned)?;
        for (d, &yi) in demeaned.iter_mut().zip(y.iter()) {
            *d = yi - *d;
        }

        // Relative normal-equation residual: ||D^T W (y - Dx)|| / ||D^T W y||.
        // Compute D^T W v as rect_op.apply_adjoint(W^{1/2} v): apply_adjoint
        // delivers D^T W^{1/2} (·), so feeding W^{1/2} v gives D^T W v.
        let mut rhs = vec![0.0; self.design.n_dofs];
        rect_op.apply_adjoint(&b, &mut rhs)?;
        let rhs_norm = norm(&rhs).max(1e-15);
        let weighted_demeaned = rect_op.weighted_rhs(&demeaned);
        let mut residual_dof = vec![0.0; self.design.n_dofs];
        rect_op.apply_adjoint(&weighted_demeaned, &mut residual_dof)?;
        let residual = norm(&residual_dof) / rhs_norm;

        Ok(SolveResult {
            x: r.x,
            demeaned,
            converged: r.converged,
            iterations: r.iterations,
            residual,
            time_total: t_start.elapsed().as_secs_f64(),
            time_setup,
            time_solve,
        })
    }

    /// Solve for multiple RHS vectors in parallel.
    pub fn solve_batch(
        &self,
        ys: &[&[f64]],
        lsmr: &LsmrOptions,
    ) -> Result<BatchSolveResult, SolveError> {
        let t_start = Instant::now();
        let n_rhs = ys.len();

        let results: Vec<Result<SolveResult, SolveError>> =
            ys.par_iter().map(|y| self.solve(y, lsmr)).collect();

        let mut x = Vec::with_capacity(self.design.n_dofs * n_rhs);
        let mut demeaned = Vec::with_capacity(self.design.n_obs * n_rhs);
        let mut converged = Vec::with_capacity(n_rhs);
        let mut iterations = Vec::with_capacity(n_rhs);
        let mut residual = Vec::with_capacity(n_rhs);
        let mut time_solve = Vec::with_capacity(n_rhs);

        for r in results {
            let r = r?;
            x.extend_from_slice(&r.x);
            demeaned.extend_from_slice(&r.demeaned);
            converged.push(r.converged);
            iterations.push(r.iterations);
            residual.push(r.residual);
            time_solve.push(r.time_solve);
        }

        Ok(BatchSolveResult {
            x,
            demeaned,
            converged,
            iterations,
            residual,
            time_solve,
            time_total: t_start.elapsed().as_secs_f64(),
            n_dofs: self.design.n_dofs,
            n_obs: self.design.n_obs,
        })
    }

    /// Access the preconditioner (for serialization or reuse across solvers).
    pub fn preconditioner(&self) -> Option<&Preconditioner> {
        self.preconditioner.as_ref()
    }

    /// Number of DOFs (coefficients).
    pub fn n_dofs(&self) -> usize {
        self.design.n_dofs
    }

    /// Number of observations.
    pub fn n_obs(&self) -> usize {
        self.design.n_obs
    }
}
