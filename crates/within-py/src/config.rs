//! PyO3 config wrapper classes exposed via `within._within`.
//!
//! These mirror the native [`within::config`] types, provide pickle support,
//! and host the Python→native config conversions (`to_native`,
//! `extract_preconditioner_config`, `resolve_lsmr_config`).

use pyo3::prelude::*;

use within::config::{
    ApproxCholConfig, ApproxSchurConfig, LocalSolverConfig, LsmrOptions, PreconditionerConfig,
    ReductionStrategy,
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
    pub(crate) fn to_native(&self) -> ApproxCholConfig {
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
    pub(crate) fn to_native(&self) -> ApproxSchurConfig {
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
    pub(crate) fn to_native(self) -> ReductionStrategy {
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
// Python → native config conversion
// ---------------------------------------------------------------------------

fn build_local_solver_config(py: Python<'_>, sc: &PyLocalSolverConfig) -> LocalSolverConfig {
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

pub(crate) fn extract_preconditioner_config(
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
                build_local_solver_config(py, sc.get())
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

pub(crate) fn resolve_lsmr_config(config: Option<&Bound<'_, PyAny>>) -> PyResult<LsmrOptions> {
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
