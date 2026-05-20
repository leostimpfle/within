// The various __reduce__ methods return tuples whose Rust type signatures are
// inherently noisy (Bound<'py, PyAny>, (PyO3 fields...)). Suppressing the
// clippy lint keeps the PyO3 boilerplate readable per-method.
#![allow(clippy::type_complexity)]

//! Thin PyO3 bridge exposing the [`within`] Rust crate to Python as `within._within`.
//!
//! This crate is intentionally minimal: it converts between Python/numpy types
//! and the native Rust API, then delegates all computation to [`within`].
//!
//! # GIL release strategy
//!
//! Every call that performs substantial computation ([`solve`], [`solve_batch`],
//! `PySolver::solve_py`, `PySolver::solve_batch_py`, and `PySolver::new`)
//! releases the GIL via [`Python::allow_threads`] before entering the Rust
//! solver. This means Python threads are **not** blocked during solve
//! operations and the solver's internal rayon parallelism can run freely.
//!
//! # Type mapping
//!
//! | Python / numpy              | Rust                              |
//! |-----------------------------|-----------------------------------|
//! | `NDArray[np.uint32]` (2-D)  | `ndarray::ArrayView2<u32>`        |
//! | `NDArray[np.float64]` (1-D) | `&[f64]`                          |
//! | `NDArray[np.float64]` (2-D) | `Vec<Vec<f64>>` (columns)         |
//! | `LsmrOptions`                | [`within::LsmrOptions`]            |
//! | `PreconditionerConfig` enum | [`within::PreconditionerConfig`]  |
//! | `Preconditioner` (built)    | [`within::Preconditioner`]        |
//! | `SolveResult`               | [`within::SolveResult`]           |
//!
//! Category arrays are read directly via numpy's ndarray bridge (zero-copy
//! when F-contiguous). Response vectors and weights are borrowed as slices
//! or copied when non-contiguous. Results are converted to numpy arrays on
//! return.
//!
//! # User-facing documentation
//!
//! For usage examples and the public API surface, see the Python package at
//! `python/within/`. This crate's types are re-exported through
//! `within.__init__` and documented in `within._within.pyi`.

use std::borrow::Cow;

use numpy::ndarray::{Array2, ShapeBuilder};
use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use within::config::{
    ApproxCholConfig, ApproxSchurConfig, LocalSolverConfig, LsmrOptions, PreconditionerConfig,
    ReductionStrategy,
};
use within::observation::FactorMajorStore;
use within::{
    solve as solve_native, solve_batch as solve_batch_native, Design, Preconditioner, SolveResult,
    Solver, WithinError,
};

// ---------------------------------------------------------------------------
// Low-level config classes (available via `_within` for benchmarks)
// ---------------------------------------------------------------------------

#[pyclass(frozen, module = "within._within")]
#[pyo3(name = "ApproxCholConfig")]
pub struct PyApproxCholConfig {
    #[pyo3(get)]
    pub seed: u64,
    #[pyo3(get)]
    pub split_merge: Option<u32>,
}

#[pymethods]
impl PyApproxCholConfig {
    #[new]
    #[pyo3(signature = (seed=0, split_merge=None))]
    fn new(seed: u64, split_merge: Option<u32>) -> Self {
        Self { seed, split_merge }
    }

    /// Pickle support: serialize to ``(class, (seed, split_merge))``.
    fn __reduce__<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyAny>, (u64, Option<u32>))> {
        let cls = py.get_type::<Self>();
        Ok((cls.into_any(), (self.seed, self.split_merge)))
    }
}

impl PyApproxCholConfig {
    fn to_native(&self) -> ApproxCholConfig {
        ApproxCholConfig {
            seed: self.seed,
            split_merge: self.split_merge,
        }
    }
}

#[pyclass(frozen, module = "within._within")]
#[pyo3(name = "ApproxSchurConfig")]
pub struct PyApproxSchurConfig {
    #[pyo3(get)]
    pub seed: u64,
    #[pyo3(get)]
    pub split: u32,
}

#[pymethods]
impl PyApproxSchurConfig {
    #[new]
    #[pyo3(signature = (seed=0, split=1))]
    fn new(seed: u64, split: u32) -> PyResult<Self> {
        if split == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "split must be >= 1",
            ));
        }
        Ok(Self { seed, split })
    }

    /// Pickle support: serialize to ``(class, (seed, split))``.
    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyAny>, (u64, u32))> {
        let cls = py.get_type::<Self>();
        Ok((cls.into_any(), (self.seed, self.split)))
    }
}

impl PyApproxSchurConfig {
    fn to_native(&self) -> ApproxSchurConfig {
        ApproxSchurConfig {
            seed: self.seed,
            split: self.split,
        }
    }
}

// ---------------------------------------------------------------------------
// PreconditionerConfig enum (IntEnum shortcut)
// ---------------------------------------------------------------------------

/// Preconditioner selection shortcut for the LSMR solver.
///
/// - ``PreconditionerConfig.Additive`` — additive Schwarz (default)
/// - ``PreconditionerConfig.Off`` — no preconditioner
#[pyclass(frozen, eq, eq_int, module = "within._within")]
#[pyo3(name = "PreconditionerConfig")]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PyPreconditionerConfig {
    Additive = 0,
    Off = 1,
}

#[pymethods]
impl PyPreconditionerConfig {
    /// Internal: int-to-variant constructor used by ``__reduce__``.
    #[staticmethod]
    fn _from_int(val: i32) -> PyResult<Self> {
        match val {
            0 => Ok(Self::Additive),
            1 => Ok(Self::Off),
            _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "invalid PreconditionerConfig discriminant: {val}"
            ))),
        }
    }

    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyAny>, (i32,))> {
        let cls = py.get_type::<Self>();
        let from_int = cls.getattr("_from_int")?;
        Ok((from_int, (*self as i32,)))
    }
}

#[pyclass(frozen, eq, eq_int, module = "within._within")]
#[pyo3(name = "ReductionStrategy")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyReductionStrategy {
    Auto = 0,
    AtomicScatter = 1,
    ParallelReduction = 2,
}

impl PyReductionStrategy {
    fn to_native(self) -> ReductionStrategy {
        match self {
            Self::Auto => ReductionStrategy::Auto,
            Self::AtomicScatter => ReductionStrategy::AtomicScatter,
            Self::ParallelReduction => ReductionStrategy::ParallelReduction,
        }
    }
}

#[pymethods]
impl PyReductionStrategy {
    /// Internal: int-to-variant constructor used by ``__reduce__``.
    #[staticmethod]
    fn _from_int(val: i32) -> PyResult<Self> {
        match val {
            0 => Ok(Self::Auto),
            1 => Ok(Self::AtomicScatter),
            2 => Ok(Self::ParallelReduction),
            _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "invalid ReductionStrategy discriminant: {val}"
            ))),
        }
    }

    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyAny>, (i32,))> {
        let cls = py.get_type::<Self>();
        let from_int = cls.getattr("_from_int")?;
        Ok((from_int, (*self as i32,)))
    }
}

// ---------------------------------------------------------------------------
// Local solver config (available via `_within` for benchmarks)
// ---------------------------------------------------------------------------

#[pyclass(frozen, subclass, module = "within._within")]
#[pyo3(name = "LocalSolverConfig")]
pub struct PyLocalSolverConfig {
    #[pyo3(get)]
    pub approx_chol: Option<Py<PyApproxCholConfig>>,
    #[pyo3(get)]
    pub approx_schur: Option<Py<PyApproxSchurConfig>>,
    #[pyo3(get)]
    pub dense_threshold: usize,
}

#[pymethods]
impl PyLocalSolverConfig {
    #[new]
    #[pyo3(signature = (approx_chol=None, approx_schur=None, dense_threshold=None))]
    fn new(
        approx_chol: Option<Py<PyApproxCholConfig>>,
        approx_schur: Option<Py<PyApproxSchurConfig>>,
        dense_threshold: Option<usize>,
    ) -> Self {
        Self {
            approx_chol,
            approx_schur,
            dense_threshold: dense_threshold
                .unwrap_or_else(|| LocalSolverConfig::default().dense_threshold),
        }
    }

    /// Pickle support: serialize to ``(class, (approx_chol, approx_schur, dense_threshold))``.
    ///
    /// Nested ``Py<...>`` fields ride through via their own ``__reduce__``.
    fn __reduce__<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(
        Bound<'py, PyAny>,
        (
            Option<Py<PyApproxCholConfig>>,
            Option<Py<PyApproxSchurConfig>>,
            usize,
        ),
    )> {
        let cls = py.get_type::<Self>();
        Ok((
            cls.into_any(),
            (
                self.approx_chol.as_ref().map(|c| c.clone_ref(py)),
                self.approx_schur.as_ref().map(|c| c.clone_ref(py)),
                self.dense_threshold,
            ),
        ))
    }
}

// ---------------------------------------------------------------------------
// Schwarz preconditioner config (available via `_within` for benchmarks)
// ---------------------------------------------------------------------------

#[pyclass(frozen, module = "within._within")]
#[pyo3(name = "AdditiveSchwarz")]
pub struct PyAdditiveSchwarz {
    #[pyo3(get)]
    pub local_solver: Option<PyObject>,
    #[pyo3(get)]
    pub reduction: PyReductionStrategy,
}

#[pymethods]
impl PyAdditiveSchwarz {
    #[new]
    #[pyo3(signature = (local_solver=None, reduction=PyReductionStrategy::Auto))]
    fn new(local_solver: Option<PyObject>, reduction: PyReductionStrategy) -> Self {
        Self {
            local_solver,
            reduction,
        }
    }

    /// Pickle support: serialize to ``(class, (local_solver, reduction))``.
    fn __reduce__<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyAny>, (Option<PyObject>, PyReductionStrategy))> {
        let cls = py.get_type::<Self>();
        Ok((
            cls.into_any(),
            (
                self.local_solver.as_ref().map(|o| o.clone_ref(py)),
                self.reduction,
            ),
        ))
    }
}

// ---------------------------------------------------------------------------
// LSMR config
// ---------------------------------------------------------------------------

#[pyclass(frozen, module = "within._within")]
#[pyo3(name = "LsmrOptions")]
pub struct PyLsmrOptions {
    #[pyo3(get)]
    pub tol: f64,
    #[pyo3(get)]
    pub maxiter: usize,
    #[pyo3(get)]
    pub local_size: Option<usize>,
}

#[pymethods]
impl PyLsmrOptions {
    #[new]
    #[pyo3(signature = (tol=1e-8, maxiter=1000, local_size=None))]
    fn new(tol: f64, maxiter: usize, local_size: Option<usize>) -> Self {
        Self {
            tol,
            maxiter,
            local_size,
        }
    }

    /// Pickle support: serialize to ``(class, (tol, maxiter, local_size))``.
    fn __reduce__<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyAny>, (f64, usize, Option<usize>))> {
        let cls = py.get_type::<Self>();
        Ok((cls.into_any(), (self.tol, self.maxiter, self.local_size)))
    }
}

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

#[pyclass(module = "within._within")]
#[pyo3(name = "SolveResult")]
pub struct PySolveResult {
    #[pyo3(get)]
    pub x: Py<numpy::PyArray1<f64>>,
    #[pyo3(get)]
    pub demeaned: Py<numpy::PyArray1<f64>>,
    #[pyo3(get)]
    pub converged: bool,
    #[pyo3(get)]
    pub iterations: usize,
    #[pyo3(get)]
    pub residual: f64,
    #[pyo3(get)]
    pub time_total: f64,
    #[pyo3(get)]
    pub time_setup: f64,
    #[pyo3(get)]
    pub time_solve: f64,
}

#[pyclass(module = "within._within")]
#[pyo3(name = "BatchSolveResult")]
pub struct PyBatchSolveResult {
    #[pyo3(get)]
    pub x: Py<numpy::PyArray2<f64>>,
    #[pyo3(get)]
    pub demeaned: Py<numpy::PyArray2<f64>>,
    #[pyo3(get)]
    pub converged: Vec<bool>,
    #[pyo3(get)]
    pub iterations: Vec<usize>,
    #[pyo3(get)]
    pub residual: Vec<f64>,
    #[pyo3(get)]
    pub time_solve: Vec<f64>,
    #[pyo3(get)]
    pub time_total: f64,
}

// ---------------------------------------------------------------------------
// Shared conversion helpers
// ---------------------------------------------------------------------------

/// Convert a numpy array view to a contiguous slice, copying only if non-contiguous.
fn coerce_to_slice<'a>(arr: &'a numpy::ndarray::ArrayView1<'_, f64>) -> Cow<'a, [f64]> {
    match arr.as_slice() {
        Some(s) => Cow::Borrowed(s),
        None => Cow::Owned(arr.to_vec()),
    }
}

/// Wrap a display-able error as a `PyValueError`.
fn value_err(e: impl std::fmt::Display) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string())
}

/// Build a slice-of-slices reference view from owned column vectors.
fn column_refs(columns: &[Vec<f64>]) -> Vec<&[f64]> {
    columns.iter().map(|c| c.as_slice()).collect()
}

// ---------------------------------------------------------------------------
// Extraction helpers
// ---------------------------------------------------------------------------

fn extract_preconditioner_config(
    py: Python<'_>,
    preconditioner: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<PreconditionerConfig>> {
    let Some(obj) = preconditioner else {
        return Ok(None);
    };

    // Enum shorthand
    if let Ok(p) = obj.extract::<PyPreconditionerConfig>() {
        return Ok(Some(match p {
            PyPreconditionerConfig::Off => PreconditionerConfig::Off,
            PyPreconditionerConfig::Additive => PreconditionerConfig::default(),
        }));
    }

    // Advanced: AdditiveSchwarz object
    if let Ok(schwarz) = obj.downcast::<PyAdditiveSchwarz>() {
        let s = schwarz.get();
        let local = match &s.local_solver {
            None => LocalSolverConfig::default(),
            Some(obj) => {
                let obj = obj.bind(py);
                let Ok(sc) = obj.downcast::<PyLocalSolverConfig>() else {
                    return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                        "local_solver must be LocalSolverConfig or None",
                    ));
                };
                let sc = sc.get();
                let approx_chol = sc
                    .approx_chol
                    .as_ref()
                    .map(|c| c.bind(py).get().to_native())
                    .unwrap_or_else(|| LocalSolverConfig::default().approx_chol);
                let approx_schur = sc
                    .approx_schur
                    .as_ref()
                    .map(|c| c.bind(py).get().to_native());
                LocalSolverConfig {
                    approx_chol,
                    approx_schur,
                    dense_threshold: sc.dense_threshold,
                }
            }
        };
        let reduction = s.reduction.to_native();
        return Ok(Some(PreconditionerConfig::Additive {
            local_solver: local,
            reduction,
        }));
    }

    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
        "preconditioner must be PreconditionerConfig.Additive, PreconditionerConfig.Off, \
         AdditiveSchwarz(...), Preconditioner(...), or None",
    ))
}

fn resolve_lsmr_config(config: Option<&Bound<'_, PyAny>>) -> PyResult<LsmrOptions> {
    let Some(c) = config else {
        return Ok(LsmrOptions::default());
    };
    if let Ok(lsmr) = c.downcast::<PyLsmrOptions>() {
        let lsmr = lsmr.get();
        return Ok(LsmrOptions {
            tol: lsmr.tol,
            maxiter: lsmr.maxiter,
            local_size: lsmr.local_size,
        });
    }
    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
        "options must be LsmrOptions",
    ))
}

// ---------------------------------------------------------------------------
// Result conversion helpers
// ---------------------------------------------------------------------------

fn into_py_result(py: Python<'_>, result: SolveResult) -> PySolveResult {
    PySolveResult {
        x: result.x.into_pyarray(py).unbind(),
        demeaned: result.demeaned.into_pyarray(py).unbind(),
        converged: result.converged,
        iterations: result.iterations,
        residual: result.residual,
        time_total: result.time_total,
        time_setup: result.time_setup,
        time_solve: result.time_solve,
    }
}

fn into_py_batch_result(
    py: Python<'_>,
    result: within::BatchSolveResult,
    n_dofs: usize,
    n_obs: usize,
) -> PyBatchSolveResult {
    let n_rhs = result.converged.len();

    let x = Array2::from_shape_vec((n_dofs, n_rhs).f(), result.x).expect("shape matches x length");
    let demeaned = Array2::from_shape_vec((n_obs, n_rhs).f(), result.demeaned)
        .expect("shape matches demeaned length");

    PyBatchSolveResult {
        x: x.into_pyarray(py).unbind(),
        demeaned: demeaned.into_pyarray(py).unbind(),
        converged: result.converged,
        iterations: result.iterations,
        residual: result.residual,
        time_solve: result.time_solve,
        time_total: result.time_total,
    }
}

// ---------------------------------------------------------------------------
// Misc helpers
// ---------------------------------------------------------------------------

/// Extract columns from a 2-D array as owned vectors.
///
/// Columns may not be contiguous in memory, so we always copy.
fn extract_columns(arr: &numpy::ndarray::ArrayView2<'_, f64>) -> Vec<Vec<f64>> {
    (0..arr.ncols())
        .map(|j| arr.column(j).iter().copied().collect())
        .collect()
}

fn extract_weight_vec(weights: &Option<PyReadonlyArray1<'_, f64>>) -> Option<Vec<f64>> {
    weights.as_ref().map(|w| w.as_array().to_vec())
}

fn warn_c_contiguous(py: Python<'_>, strides: &[isize]) -> PyResult<()> {
    // strides[0] == 0 occurs for zero-row arrays, which are trivially F-contiguous.
    if strides.len() >= 2 && strides[0] != 1 && strides[0] != 0 {
        PyErr::warn(
            py,
            &py.get_type::<pyo3::exceptions::PyUserWarning>(),
            c"categories array is not F-contiguous (column-major). \
             Use np.asfortranarray(categories) for faster solves.",
            1,
        )?;
    }
    Ok(())
}

/// If the Python preconditioner argument is a pre-built `Preconditioner`,
/// return a clone of the inner native value. Otherwise `None`.
fn extract_prebuilt(preconditioner: Option<&Bound<'_, PyAny>>) -> Option<Preconditioner> {
    let obj = preconditioner?;
    obj.downcast::<PyPreconditioner>()
        .ok()
        .map(|b| b.get().inner.clone())
}

// ---------------------------------------------------------------------------
// Public solve functions
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (categories, y, options=None, weights=None, preconditioner=None))]
pub fn solve<'py>(
    py: Python<'py>,
    categories: PyReadonlyArray2<'py, u32>,
    y: PyReadonlyArray1<'py, f64>,
    options: Option<&Bound<'py, PyAny>>,
    weights: Option<PyReadonlyArray1<'py, f64>>,
    preconditioner: Option<&Bound<'py, PyAny>>,
) -> PyResult<PySolveResult> {
    let cats = categories.as_array();
    warn_c_contiguous(py, cats.strides())?;

    let y_arr = y.as_array();
    let y_cow = coerce_to_slice(&y_arr);
    let w_vec = extract_weight_vec(&weights);
    let w_ref = w_vec.as_deref();
    let params = resolve_lsmr_config(options)?;

    let result = if let Some(built) = extract_prebuilt(preconditioner) {
        py.allow_threads(|| -> Result<SolveResult, WithinError> {
            let solver = Solver::new(cats, w_vec, built)?;
            Ok(solver.solve(&y_cow, &params)?)
        })
        .map_err(value_err)?
    } else {
        let precond = extract_preconditioner_config(py, preconditioner)?;
        py.allow_threads(|| solve_native(cats, &y_cow, w_ref, &params, precond.as_ref()))
            .map_err(value_err)?
    };

    Ok(into_py_result(py, result))
}

#[pyfunction]
#[pyo3(signature = (categories, Y, options=None, weights=None, preconditioner=None))]
pub fn solve_batch<'py>(
    py: Python<'py>,
    categories: PyReadonlyArray2<'py, u32>,
    #[allow(non_snake_case)] Y: PyReadonlyArray2<'py, f64>,
    options: Option<&Bound<'py, PyAny>>,
    weights: Option<PyReadonlyArray1<'py, f64>>,
    preconditioner: Option<&Bound<'py, PyAny>>,
) -> PyResult<PyBatchSolveResult> {
    let cats = categories.as_array();
    warn_c_contiguous(py, cats.strides())?;

    let y_arr = Y.as_array();

    // Validate Y row count against the design up front. Without this, an empty
    // batch (Y.shape[1] == 0) would silently skip the per-column length check
    // inside `Solver::solve`.
    if y_arr.nrows() != cats.nrows() {
        return Err(value_err(format!(
            "Y has {} rows but categories has {} observations",
            y_arr.nrows(),
            cats.nrows()
        )));
    }

    let columns = extract_columns(&y_arr);
    let col_refs = column_refs(&columns);

    let w_vec = extract_weight_vec(&weights);
    let w_ref = w_vec.as_deref();
    let params = resolve_lsmr_config(options)?;

    let result = if let Some(built) = extract_prebuilt(preconditioner) {
        py.allow_threads(|| -> Result<_, WithinError> {
            let solver = Solver::new(cats, w_vec, built)?;
            Ok(solver.solve_batch(&col_refs, &params)?)
        })
        .map_err(value_err)?
    } else {
        let precond = extract_preconditioner_config(py, preconditioner)?;
        py.allow_threads(|| solve_batch_native(cats, &col_refs, w_ref, &params, precond.as_ref()))
            .map_err(value_err)?
    };

    // Use the design dimensions carried by the result rather than inferring
    // them from output lengths — that keeps empty batches well-shaped at
    // (n_dofs, 0) / (n_obs, 0).
    let n_dofs = result.n_dofs;
    let n_obs = result.n_obs;
    Ok(into_py_batch_result(py, result, n_dofs, n_obs))
}

// ---------------------------------------------------------------------------
// Built preconditioner (returned by Solver, picklable)
// ---------------------------------------------------------------------------

/// A pre-built preconditioner that can be pickled and reused.
///
/// Obtained via ``Solver.preconditioner``. Pass it back to a new
/// ``Solver(…, preconditioner=p)`` to skip the expensive factorisation.
#[pyclass(frozen, module = "within._within")]
#[pyo3(name = "Preconditioner")]
pub struct PyPreconditioner {
    inner: Preconditioner,
}

#[pymethods]
impl PyPreconditioner {
    /// Apply the preconditioner: ``y = M⁻¹ x``.
    fn apply<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
        let x_slice = x
            .as_slice()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("x must be contiguous"))?;
        if x_slice.len() != self.inner.ncols() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "x has length {} but preconditioner expects {}",
                x_slice.len(),
                self.inner.ncols()
            )));
        }
        let mut y = vec![0.0; self.inner.nrows()];
        self.inner
            .apply(x_slice, &mut y)
            .map_err(|e: within::SolveError| {
                pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
            })?;
        Ok(numpy::PyArray1::from_vec(py, y))
    }
    /// Number of rows (DOFs).
    #[getter]
    fn nrows(&self) -> usize {
        self.inner.nrows()
    }

    /// Number of columns (DOFs).
    #[getter]
    fn ncols(&self) -> usize {
        self.inner.ncols()
    }

    fn __repr__(&self) -> String {
        format!("Preconditioner(Additive, n={})", self.inner.nrows())
    }

    /// Pickle support: serialize to ``(bytes,)`` constructor arg.
    fn __reduce__<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyAny>, (Bound<'py, pyo3::types::PyBytes>,))> {
        let bytes = postcard::to_stdvec(&self.inner)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let cls = py.get_type::<Self>();
        let py_bytes = pyo3::types::PyBytes::new(py, &bytes);
        Ok((cls.into_any(), (py_bytes,)))
    }

    /// Construct from serialised bytes (used by pickle and for manual persistence).
    #[new]
    fn new(data: &[u8]) -> PyResult<Self> {
        let inner: Preconditioner = postcard::from_bytes(data).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "failed to deserialize preconditioner: {}",
                e
            ))
        })?;
        Ok(Self { inner })
    }
}

// ---------------------------------------------------------------------------
// Persistent Solver
// ---------------------------------------------------------------------------

/// Persistent solver that reuses preconditioners across multiple solves.
///
/// Build once with `Solver(categories, ...)`, then call `solve()` or
/// `solve_batch()` repeatedly. The expensive preconditioner factorization
/// happens only at construction time.
#[pyclass(frozen, module = "within._within")]
#[pyo3(name = "Solver")]
pub struct PySolver {
    solver: Solver<FactorMajorStore>,
}

#[pymethods]
impl PySolver {
    #[new]
    #[pyo3(signature = (categories, weights=None, preconditioner=None))]
    fn new<'py>(
        py: Python<'py>,
        categories: PyReadonlyArray2<'py, u32>,
        weights: Option<PyReadonlyArray1<'py, f64>>,
        preconditioner: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Self> {
        let cats = categories.as_array();
        warn_c_contiguous(py, cats.strides())?;

        // Build owned factor-major store from numpy array
        let n_obs = cats.nrows();
        let n_factors = cats.ncols();
        let factor_levels: Vec<Vec<u32>> = (0..n_factors)
            .map(|f| cats.column(f).iter().copied().collect())
            .collect();
        let store = FactorMajorStore::new(factor_levels, n_obs).map_err(value_err)?;
        let design = Design::from_store(store).map_err(value_err)?;
        let weights_vec: Option<Vec<f64>> = weights
            .as_ref()
            .map(|w| w.as_array().iter().copied().collect());

        // Pre-built Preconditioner uses the reuse path;
        // all other variants go through extract_preconditioner_config.
        let solver = if let Some(built) = extract_prebuilt(preconditioner) {
            py.allow_threads(|| Solver::new(design, weights_vec, built))
                .map_err(value_err)?
        } else {
            let precond = extract_preconditioner_config(py, preconditioner)?;
            py.allow_threads(|| Solver::new(design, weights_vec, precond.as_ref()))
                .map_err(value_err)?
        };

        Ok(Self { solver })
    }

    /// Solve for a single response vector with the given LSMR tuning.
    #[pyo3(name = "solve", signature = (y, options=None))]
    fn solve_py<'py>(
        &self,
        py: Python<'py>,
        y: PyReadonlyArray1<'py, f64>,
        options: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<PySolveResult> {
        let y_arr = y.as_array();
        let y_cow = coerce_to_slice(&y_arr);
        let params = resolve_lsmr_config(options)?;

        let result = py
            .allow_threads(|| self.solver.solve(&y_cow, &params))
            .map_err(value_err)?;

        Ok(into_py_result(py, result))
    }

    /// Solve for multiple response vectors in parallel.
    ///
    /// `Y` is a 2-D array of shape `(n_obs, k)` where each column is a
    /// separate response vector.
    #[pyo3(name = "solve_batch", signature = (Y, options=None))]
    fn solve_batch_py<'py>(
        &self,
        py: Python<'py>,
        #[allow(non_snake_case)] Y: PyReadonlyArray2<'py, f64>,
        options: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<PyBatchSolveResult> {
        let y_arr = Y.as_array();

        let n_obs = self.solver.n_obs();
        if y_arr.nrows() != n_obs {
            return Err(value_err(format!(
                "Y has {} rows but solver has {} observations",
                y_arr.nrows(),
                n_obs
            )));
        }

        let columns = extract_columns(&y_arr);
        let col_refs = column_refs(&columns);

        let n_dofs = self.solver.n_dofs();
        let params = resolve_lsmr_config(options)?;

        let result = py
            .allow_threads(|| self.solver.solve_batch(&col_refs, &params))
            .map_err(value_err)?;

        Ok(into_py_batch_result(py, result, n_dofs, n_obs))
    }

    /// Return the built preconditioner, or ``None`` if unconfigured.
    ///
    /// The returned object is picklable and can be passed to a new
    /// ``Solver(…, preconditioner=p)`` to skip the expensive build step.
    #[getter]
    #[pyo3(name = "preconditioner")]
    fn preconditioner_py(&self) -> PyResult<Option<PyPreconditioner>> {
        match self.solver.preconditioner() {
            None => Ok(None),
            Some(p) => Ok(Some(PyPreconditioner { inner: p.clone() })),
        }
    }

    /// Number of DOFs (coefficients) in the model.
    #[getter]
    fn n_dofs(&self) -> usize {
        self.solver.n_dofs()
    }

    /// Number of observations.
    #[getter]
    fn n_obs(&self) -> usize {
        self.solver.n_obs()
    }
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

#[pymodule]
fn _within(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySolveResult>()?;
    m.add_class::<PyBatchSolveResult>()?;
    m.add_class::<PyLsmrOptions>()?;
    m.add_class::<PyAdditiveSchwarz>()?;
    m.add_class::<PyReductionStrategy>()?;
    m.add_class::<PyPreconditionerConfig>()?;
    m.add_class::<PyApproxCholConfig>()?;
    m.add_class::<PyApproxSchurConfig>()?;
    m.add_class::<PyLocalSolverConfig>()?;
    m.add_class::<PyPreconditioner>()?;
    m.add_class::<PySolver>()?;
    m.add_function(wrap_pyfunction!(solve, m)?)?;
    m.add_function(wrap_pyfunction!(solve_batch, m)?)?;
    Ok(())
}
