//! Iterative solvers.
//!
//! - **`lsmr`** — Modified LSMR for rectangular least-squares. Uses the
//!   Modified Golub-Kahan bidiagonalization, which requires only one
//!   `M⁻¹` application per iteration (no square-root factorization).
//!   `M` approximates `AᵀA`.

/// Modified LSMR for rectangular least-squares.
pub mod lsmr;

/// Inner product of two vectors.
#[inline]
pub(crate) fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

/// Euclidean norm of a vector.
#[inline]
pub fn vec_norm(v: &[f64]) -> f64 {
    let mut s = 0.0f64;
    for &x in v {
        s += x * x;
    }
    s.sqrt()
}
