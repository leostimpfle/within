//! Shared conversion and extraction helpers bridging numpy/Python types to the
//! native [`within`] API.

use std::borrow::Cow;

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

use within::config::{LocalSolverConfig, LsmrOptions, PreconditionerConfig};
use within::Preconditioner;

use crate::config::{
    PyAdditiveSchwarz, PyLocalSolverConfig, PyLsmrOptions, PyPreconditionerConfig,
};
use crate::objects::PyPreconditioner;

// ---------------------------------------------------------------------------
// Shared conversion helpers
// ---------------------------------------------------------------------------

/// Convert a numpy array view to a contiguous slice, copying only if non-contiguous.
pub(crate) fn coerce_to_slice<'a>(arr: &'a numpy::ndarray::ArrayView1<'_, f64>) -> Cow<'a, [f64]> {
    match arr.as_slice() {
        Some(s) => Cow::Borrowed(s),
        None => Cow::Owned(arr.to_vec()),
    }
}

/// Wrap a display-able error as a `PyValueError`.
pub(crate) fn value_err(e: impl std::fmt::Display) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string())
}

/// Build a slice-of-slices reference view from owned column vectors.
pub(crate) fn column_refs(columns: &[Vec<f64>]) -> Vec<&[f64]> {
    columns.iter().map(|c| c.as_slice()).collect()
}

// ---------------------------------------------------------------------------
// Extraction helpers
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

// ---------------------------------------------------------------------------
// Misc helpers
// ---------------------------------------------------------------------------

/// Extract columns from a 2-D array as owned vectors.
///
/// Columns may not be contiguous in memory, so we always copy.
pub(crate) fn extract_columns(arr: &numpy::ndarray::ArrayView2<'_, f64>) -> Vec<Vec<f64>> {
    (0..arr.ncols())
        .map(|j| arr.column(j).iter().copied().collect())
        .collect()
}

pub(crate) fn extract_weight_vec(weights: &Option<PyReadonlyArray1<'_, f64>>) -> Option<Vec<f64>> {
    weights.as_ref().map(|w| w.as_array().to_vec())
}

pub(crate) fn warn_c_contiguous(py: Python<'_>, strides: &[isize]) -> PyResult<()> {
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
pub(crate) fn extract_prebuilt(
    preconditioner: Option<&Bound<'_, PyAny>>,
) -> Option<Preconditioner> {
    let obj = preconditioner?;
    obj.downcast::<PyPreconditioner>()
        .ok()
        .map(|b| b.get().inner.clone())
}
