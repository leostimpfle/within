//! Square `f64`/`u32` CSR matrix with adaptive-parallelism `matvec` and a
//! principal-submatrix extraction used on the subdomain build path.
//!
//! Constructable from `faer::sparse::SparseRowMatRef<u32, f64>` for interop with
//! the approximate Cholesky factor produced by the `within` solver.

use std::collections::HashMap;

use rayon::prelude::*;

use crate::error::BuildError;

// ---------------------------------------------------------------------------
// IndexLookup: adaptive global-to-local index mapping
// ---------------------------------------------------------------------------

/// Adaptive global-to-local index lookup used during submatrix extraction.
///
/// Picks a dense `Vec` when indices are tightly packed, or a `HashMap` when
/// the index range is sparse relative to the subset size.
enum IndexLookup {
    /// Dense lookup: `table[global]` gives `local` (or `usize::MAX` if absent).
    Dense(Vec<usize>),
    /// Sparse lookup: HashMap from global index to local index.
    Sparse(HashMap<usize, usize>),
}

impl IndexLookup {
    /// Build the appropriate lookup variant based on index density.
    fn new(subset: &[usize], max_idx: usize, m: usize) -> Self {
        if max_idx + 1 > 4 * m {
            let map = subset
                .iter()
                .enumerate()
                .map(|(local, &global)| (global, local))
                .collect();
            IndexLookup::Sparse(map)
        } else {
            let mut table = vec![usize::MAX; max_idx + 1];
            for (local, &global) in subset.iter().enumerate() {
                table[global] = local;
            }
            IndexLookup::Dense(table)
        }
    }

    /// Look up the local index for a global column index.
    #[inline]
    fn get(&self, global_col: usize) -> Option<usize> {
        match self {
            IndexLookup::Dense(table) => {
                if global_col < table.len() {
                    let local = table[global_col];
                    if local != usize::MAX {
                        Some(local)
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            IndexLookup::Sparse(map) => map.get(&global_col).copied(),
        }
    }
}

/// Square sparse matrix in Compressed Sparse Row (CSR) format.
///
/// Used internally for explicit Gramian submatrices, Laplacian representations,
/// and BFS adjacency graphs. All methods are pure Rust (`Send + Sync`).
///
/// # CSR Invariants
///
/// A well-formed `CsrMatrix` satisfies:
/// - `indptr.len() == n + 1`
/// - `indptr[0] == 0` and `indptr` is non-decreasing
/// - `indices.len() == data.len() == indptr[n] as usize` (the number of non-zeros)
/// - All column indices in `indices` are in `0..n`
/// - Within each row, column indices are sorted ascending with no duplicates
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone)]
pub struct CsrMatrix {
    indptr: Vec<u32>,
    indices: Vec<u32>,
    data: Vec<f64>,
    n: usize,
}

impl CsrMatrix {
    /// Create a new `CsrMatrix` from raw CSR components, validating the
    /// documented invariants.
    ///
    /// # Arguments
    ///
    /// * `indptr` — Row pointer array (length `n + 1`), non-decreasing with `indptr[0] == 0`.
    /// * `indices` — Column indices (length `nnz`), sorted ascending per row.
    /// * `data` — Non-zero values (length `nnz`), parallel to `indices`.
    /// * `n` — Matrix dimension (square: `n` rows x `n` columns).
    ///
    /// # Errors
    ///
    /// Returns [`BuildError::InvalidCsr`] when any documented invariant is
    /// violated: shape mismatch between `indptr`/`indices`/`data`, out-of-bounds
    /// column indices, non-monotonic `indptr`, or unsorted/duplicate column
    /// indices within a row.
    pub fn new(
        indptr: Vec<u32>,
        indices: Vec<u32>,
        data: Vec<f64>,
        n: usize,
    ) -> Result<Self, BuildError> {
        if indptr.len() != n + 1 {
            return Err(BuildError::InvalidCsr {
                reason: "indptr length must equal n + 1",
            });
        }
        if indptr[0] != 0 {
            return Err(BuildError::InvalidCsr {
                reason: "indptr[0] must be 0",
            });
        }
        if indices.len() != data.len() {
            return Err(BuildError::InvalidCsr {
                reason: "indices and data must have equal length",
            });
        }
        let nnz = indptr[n] as usize;
        if indices.len() != nnz {
            return Err(BuildError::InvalidCsr {
                reason: "indices/data length must equal indptr[n]",
            });
        }
        for row in 0..n {
            let start = indptr[row] as usize;
            let end = indptr[row + 1] as usize;
            if end < start {
                return Err(BuildError::InvalidCsr {
                    reason: "indptr is not non-decreasing",
                });
            }
            let row_cols = &indices[start..end];
            for (k, &col) in row_cols.iter().enumerate() {
                if (col as usize) >= n {
                    return Err(BuildError::InvalidCsr {
                        reason: "column index out of bounds",
                    });
                }
                if k > 0 && col <= row_cols[k - 1] {
                    return Err(BuildError::InvalidCsr {
                        reason: "row column indices must be strictly ascending",
                    });
                }
            }
        }
        Ok(Self {
            indptr,
            indices,
            data,
            n,
        })
    }

    /// Row pointer array (length `n + 1`).
    #[inline]
    pub fn indptr(&self) -> &[u32] {
        &self.indptr
    }

    /// Column indices (length `nnz`).
    #[inline]
    pub fn indices(&self) -> &[u32] {
        &self.indices
    }

    /// Non-zero values (length `nnz`), parallel to `indices`.
    #[inline]
    pub fn data(&self) -> &[f64] {
        &self.data
    }

    /// Matrix dimension (the matrix is always square).
    #[inline]
    pub fn n(&self) -> usize {
        self.n
    }

    /// Sparse matrix-vector multiply: y += A @ x.
    ///
    /// Does NOT zero y first -- caller controls initialization.
    /// For large matrices (> 10 000 rows), rows are processed in parallel
    /// via Rayon `par_chunks_mut`. Each row's dot product reads from shared
    /// `x` and writes to its own `y[row]` — no conflicts.
    fn matvec_add(&self, x: &[f64], y: &mut [f64]) {
        const PAR_THRESHOLD: usize = 10_000;
        const CHUNK_SIZE: usize = 4096;

        debug_assert_eq!(x.len(), self.n);
        debug_assert_eq!(y.len(), self.n);

        if self.n > PAR_THRESHOLD {
            y[..self.n]
                .par_chunks_mut(CHUNK_SIZE)
                .enumerate()
                .for_each(|(chunk_idx, chunk)| {
                    let row_start = chunk_idx * CHUNK_SIZE;
                    for (local, yi) in chunk.iter_mut().enumerate() {
                        let row = row_start + local;
                        let start = self.indptr[row] as usize;
                        let end = self.indptr[row + 1] as usize;
                        let row_data = &self.data[start..end];
                        let row_idx = &self.indices[start..end];
                        let mut acc = 0.0;
                        for (&val, &col) in row_data.iter().zip(row_idx) {
                            acc += val * x[col as usize];
                        }
                        *yi += acc;
                    }
                });
        } else {
            for (row, yi) in y[..self.n].iter_mut().enumerate() {
                let start = self.indptr[row] as usize;
                let end = self.indptr[row + 1] as usize;
                let row_data = &self.data[start..end];
                let row_idx = &self.indices[start..end];
                let mut acc = 0.0;
                for (&val, &col) in row_data.iter().zip(row_idx) {
                    acc += val * x[col as usize];
                }
                *yi += acc;
            }
        }
    }

    /// Sparse matrix-vector multiply: y = A @ x (zeroes y first).
    pub fn matvec(&self, x: &[f64], y: &mut [f64]) {
        y.fill(0.0);
        self.matvec_add(x, y);
    }

    /// Extract diagonal of the matrix. Returns zero for rows with no diagonal entry.
    pub fn diagonal(&self) -> Vec<f64> {
        let mut diag = vec![0.0; self.n];
        for (row, di) in diag.iter_mut().enumerate() {
            let start = self.indptr[row] as usize;
            let end = self.indptr[row + 1] as usize;
            let row_idx = &self.indices[start..end];
            let row_data = &self.data[start..end];
            for (&col, &val) in row_idx.iter().zip(row_data) {
                if col as usize == row {
                    *di = val;
                    break;
                }
            }
        }
        diag
    }

    /// Number of stored non-zero entries.
    pub fn nnz(&self) -> usize {
        self.data.len()
    }

    /// Extract principal submatrix G[subset, subset] with global->local index remapping.
    ///
    /// Returns a new CsrMatrix of size `subset.len()` x `subset.len()`.
    /// `subset` must contain valid row/column indices into `self`.
    ///
    /// Adaptively picks a dense `Vec` or `HashMap` for the global-to-local map
    /// depending on the ratio of `max(subset)` to `subset.len()`.
    pub fn extract_submatrix(&self, subset: &[usize]) -> CsrMatrix {
        let m = subset.len();
        if m == 0 {
            return CsrMatrix {
                indptr: vec![0],
                indices: Vec::new(),
                data: Vec::new(),
                n: 0,
            };
        }

        let max_idx = subset.iter().copied().max().unwrap_or(0);

        debug_assert!(m <= u32::MAX as usize);

        let lookup = IndexLookup::new(subset, max_idx, m);

        let mut new_indptr = Vec::with_capacity(m + 1);
        let mut new_indices = Vec::new();
        let mut new_data = Vec::new();
        new_indptr.push(0u32);

        for &global_row in subset {
            let start = self.indptr[global_row] as usize;
            let end = self.indptr[global_row + 1] as usize;
            for idx in start..end {
                let global_col = self.indices[idx] as usize;
                if let Some(local_col) = lookup.get(global_col) {
                    new_indices.push(local_col as u32);
                    new_data.push(self.data[idx]);
                }
            }
            new_indptr
                .push(u32::try_from(new_indices.len()).expect("submatrix nnz exceeds u32::MAX"));
        }

        CsrMatrix {
            indptr: new_indptr,
            indices: new_indices,
            data: new_data,
            n: m,
        }
    }
}

// ---------------------------------------------------------------------------
// Sparse interop: faer
// ---------------------------------------------------------------------------

impl From<faer::sparse::SparseRowMatRef<'_, u32, f64>> for CsrMatrix {
    /// Convert a `faer` sparse row-major matrix view into an owned `CsrMatrix`.
    ///
    /// # Panics
    ///
    /// Panics if the matrix is not square.
    fn from(mat: faer::sparse::SparseRowMatRef<'_, u32, f64>) -> Self {
        assert_eq!(
            mat.nrows(),
            mat.ncols(),
            "CsrMatrix::from(faer): matrix must be square"
        );
        let symbolic = mat.symbolic();
        Self {
            indptr: symbolic.row_ptr().to_vec(),
            indices: symbolic.col_idx().to_vec(),
            data: mat.val().to_vec(),
            n: mat.nrows(),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_3x3_laplacian() -> CsrMatrix {
        CsrMatrix {
            indptr: vec![0, 3, 5, 7],
            indices: vec![0, 1, 2, 0, 1, 0, 2],
            data: vec![2.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0],
            n: 3,
        }
    }

    #[test]
    fn test_extract_submatrix_full() {
        let m = make_3x3_laplacian();
        let sub = m.extract_submatrix(&[0, 1, 2]);
        assert_eq!(sub.n, 3);
        assert_eq!(sub.data.len(), m.data.len());
    }

    #[test]
    fn test_extract_submatrix_partial() {
        let m = make_3x3_laplacian();
        let sub = m.extract_submatrix(&[0, 2]);
        assert_eq!(sub.n, 2);
        assert_eq!(sub.indptr, vec![0, 2, 4]);
    }

    #[test]
    fn test_extract_submatrix_empty() {
        let m = make_3x3_laplacian();
        let sub = m.extract_submatrix(&[]);
        assert_eq!(sub.n, 0);
        assert_eq!(sub.indptr, vec![0]);
    }

    #[test]
    fn test_matvec_known_result() {
        let m = make_3x3_laplacian();
        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![0.0; 3];
        m.matvec(&x, &mut y);
        // [2, -1, -1] * [1,2,3] = 2-2-3 = -3
        // [-1, 1, 0]  * [1,2,3] = -1+2   = 1
        // [-1, 0, 1]  * [1,2,3] = -1+3   = 2
        assert!((y[0] - (-3.0)).abs() < 1e-14);
        assert!((y[1] - 1.0).abs() < 1e-14);
        assert!((y[2] - 2.0).abs() < 1e-14);
    }

    #[test]
    fn test_matvec_add() {
        let m = make_3x3_laplacian();
        let x = vec![1.0, 0.0, 0.0];
        let mut y = vec![10.0, 20.0, 30.0];
        m.matvec_add(&x, &mut y);
        // adds first column [2, -1, -1] to y
        assert!((y[0] - 12.0).abs() < 1e-14);
        assert!((y[1] - 19.0).abs() < 1e-14);
        assert!((y[2] - 29.0).abs() < 1e-14);
    }

    #[test]
    fn test_diagonal_missing_entry() {
        // Matrix with no diagonal for any row
        let m = CsrMatrix::new(vec![0, 1, 2, 3], vec![1, 0, 0], vec![5.0, 3.0, 7.0], 3)
            .expect("valid CSR");
        let d = m.diagonal();
        assert_eq!(d[0], 0.0); // no diagonal
        assert_eq!(d[1], 0.0); // no diagonal
        assert_eq!(d[2], 0.0); // no diagonal
    }

    #[test]
    fn test_nnz() {
        let m = make_3x3_laplacian();
        assert_eq!(m.nnz(), 7);
    }

    #[test]
    fn test_extract_submatrix_hashmap_path() {
        // Create a matrix where max_idx+1 > 4*m to trigger the HashMap path
        let n = 100;
        let mut indptr = vec![0u32; n + 1];
        let mut indices = Vec::new();
        let mut data = Vec::new();
        // Only put diagonal entries
        for i in 0..n {
            indices.push(i as u32);
            data.push((i + 1) as f64);
            indptr[i + 1] = (i + 1) as u32;
        }
        let m = CsrMatrix::new(indptr, indices, data, n).expect("valid CSR");
        // Extract subset [0, 50, 99] — max_idx=99, m=3, 100 > 4*3=12 → HashMap
        let sub = m.extract_submatrix(&[0, 50, 99]);
        assert_eq!(sub.n(), 3);
        let d = sub.diagonal();
        assert!((d[0] - 1.0).abs() < 1e-14);
        assert!((d[1] - 51.0).abs() < 1e-14);
        assert!((d[2] - 100.0).abs() < 1e-14);
    }

    #[test]
    fn test_new_rejects_invalid_csr() {
        let assert_invalid = |result: Result<CsrMatrix, BuildError>| match result {
            Err(BuildError::InvalidCsr { .. }) => {}
            Err(other) => panic!("expected InvalidCsr, got {other:?}"),
            Ok(_) => panic!("expected InvalidCsr, got Ok"),
        };

        // Out-of-bounds column: indices[0] = 5 with n = 3.
        assert_invalid(CsrMatrix::new(vec![0, 1, 1, 1], vec![5], vec![1.0], 3));

        // Unsorted (and duplicated) columns within a row.
        assert_invalid(CsrMatrix::new(
            vec![0, 2, 2, 2],
            vec![1, 0],
            vec![1.0, 2.0],
            3,
        ));

        // Length mismatch between indices and data.
        assert_invalid(CsrMatrix::new(vec![0, 1, 1, 1], vec![0], vec![1.0, 2.0], 3));

        // indptr.len() != n + 1.
        assert_invalid(CsrMatrix::new(vec![0, 1], vec![0], vec![1.0], 3));
    }
}
