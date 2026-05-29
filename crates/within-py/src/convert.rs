//! Shared numpy/Python coercion helpers bridging array and error types to the
//! native [`within`] API.

use std::borrow::Cow;

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

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
