//! Shared numpy/Python coercion helpers bridging array and error types to the
//! native [`within`] API.

use std::borrow::Cow;

use pyo3::prelude::*;

use within::{BatchSolveResult, SolveResult};

use crate::results::{into_py_batch_result, into_py_result, PyBatchSolveResult, PySolveResult};

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

/// Build a slice-of-slices reference view from borrowed-or-owned columns.
pub(crate) fn column_refs<'a>(columns: &'a [Cow<'_, [f64]>]) -> Vec<&'a [f64]> {
    columns.iter().map(|c| &**c).collect()
}

// ---------------------------------------------------------------------------
// Misc helpers
// ---------------------------------------------------------------------------

/// Extract the columns of a 2-D array as borrowed-or-owned slices.
///
/// A column of an F-contiguous (column-major) array is contiguous in memory and
/// is borrowed directly (`Cow::Borrowed`); a strided column (e.g. from C-order
/// input) is copied (`Cow::Owned`). Borrows are tied to the view's data lifetime.
pub(crate) fn extract_columns<'a>(
    arr: &numpy::ndarray::ArrayView2<'a, f64>,
) -> Vec<Cow<'a, [f64]>> {
    (0..arr.ncols())
        .map(|j| {
            let col = arr.index_axis_move(numpy::ndarray::Axis(1), j);
            match col.to_slice() {
                Some(s) => Cow::Borrowed(s),
                None => Cow::Owned(col.iter().copied().collect()),
            }
        })
        .collect()
}

pub(crate) fn warn_c_contiguous(
    py: Python<'_>,
    cats: &numpy::ndarray::ArrayView2<'_, u32>,
) -> PyResult<()> {
    // Warn only when the per-factor columns are NOT readable as contiguous
    // slices -- i.e. exactly when `ArrayStore::factor_column` rejects the fast
    // path: row stride != 1, or a non-positive (reversed) column stride. A
    // single row or empty input is trivially contiguous regardless of strides.
    let strides = cats.strides();
    if cats.nrows() > 1 && (strides[0] != 1 || strides[1] < 1) {
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

// ---------------------------------------------------------------------------
// Off-GIL solve orchestration
// ---------------------------------------------------------------------------

/// Run a native single-response solve with the GIL released, then convert.
///
/// The closure produces a [`SolveResult`] off-GIL (`allow_threads`); its native
/// error is mapped to a `PyValueError` and the result to its Python wrapper.
/// Shared by the free `solve` function and the persistent `Solver.solve`.
pub(crate) fn run_solve<E, F>(py: Python<'_>, solve: F) -> PyResult<PySolveResult>
where
    E: std::fmt::Display + Send,
    F: Send + FnOnce() -> Result<SolveResult, E>,
{
    let result = py.allow_threads(solve).map_err(value_err)?;
    Ok(into_py_result(py, result))
}

/// Run a native batch solve with the GIL released, then convert.
///
/// Batch counterpart to [`run_solve`]; the conversion itself is fallible
/// (re-shaping the flat column buffers into 2-D arrays).
pub(crate) fn run_batch<E, F>(py: Python<'_>, solve: F) -> PyResult<PyBatchSolveResult>
where
    E: std::fmt::Display + Send,
    F: Send + FnOnce() -> Result<BatchSolveResult, E>,
{
    let result = py.allow_threads(solve).map_err(value_err)?;
    into_py_batch_result(py, result)
}
