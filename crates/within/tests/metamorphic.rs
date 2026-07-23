use proptest::prelude::*;
use within::{solve, solve_batch, LsmrOptions};

#[path = "common/property_strategies.rs"]
mod strategies;
use strategies::{additive_precond, random_fe_problem_strategy};

// Drive both arms of each metamorphic pair to tight first-order optimality so
// the gauge-invariant residual agrees to well below the assertion tolerance;
// draws where either arm fails to converge are rejected.
fn tight_params() -> LsmrOptions {
    LsmrOptions {
        tol: 1e-11,
        maxiter: 3000,
        local_size: Some(10),
    }
}

fn rel_l2_diff(actual: &[f64], expected: &[f64]) -> f64 {
    let num = actual
        .iter()
        .zip(expected)
        .map(|(a, e)| (a - e).powi(2))
        .sum::<f64>()
        .sqrt();
    let den = expected.iter().map(|e| e * e).sum::<f64>().sqrt();
    num / den.max(1e-12)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(12))]

    /// Response-scaling equivariance: the residual `r = y − Dx` is linear in the
    /// response, so `r(c·y) = c·r(y)` for any nonzero scalar `c`.
    #[test]
    fn prop_response_scaling_equivariance(
        (cats, y) in random_fe_problem_strategy(),
        c in prop_oneof![-8.0f64..=-0.25, 0.25f64..=8.0],
    ) {
        let params = tight_params();
        let precond = additive_precond();

        let base = solve(cats.view(), &y, None, &params, &precond).unwrap();
        prop_assume!(base.converged);

        let y_scaled: Vec<f64> = y.iter().map(|v| c * v).collect();
        let scaled = solve(cats.view(), &y_scaled, None, &params, &precond).unwrap();
        prop_assume!(scaled.converged);

        let expected: Vec<f64> = base.demeaned.iter().map(|v| c * v).collect();
        let rel = rel_l2_diff(&scaled.demeaned, &expected);
        prop_assert!(
            rel <= 1e-6,
            "response-scaling equivariance violated: rel L2 = {rel:.3e} (c={c})"
        );
    }

    /// Uniform weight-scaling invariance: `argmin ∑ k·wᵢ rᵢ² = argmin ∑ wᵢ rᵢ²`
    /// for any `k > 0`, so scaling every weight by a constant leaves the fit —
    /// and hence the residual — unchanged.
    #[test]
    fn prop_weight_scaling_invariance(
        (cats, y, w) in random_fe_problem_strategy().prop_flat_map(|(cats, y)| {
            let n = y.len();
            (Just(cats), Just(y), proptest::collection::vec(0.2f64..3.0, n))
        }),
        k in 0.25f64..=6.0,
    ) {
        let params = tight_params();
        let precond = additive_precond();

        let base = solve(cats.view(), &y, Some(w.as_slice()), &params, &precond).unwrap();
        prop_assume!(base.converged);

        let w_scaled: Vec<f64> = w.iter().map(|v| k * v).collect();
        let scaled = solve(cats.view(), &y, Some(w_scaled.as_slice()), &params, &precond).unwrap();
        prop_assume!(scaled.converged);

        let rel = rel_l2_diff(&scaled.demeaned, &base.demeaned);
        prop_assert!(
            rel <= 1e-6,
            "weight-scaling invariance violated: rel L2 = {rel:.3e} (k={k})"
        );
    }

    /// `solve_batch` must agree column-for-column with independent `solve` calls
    /// on the same design: batching only shares the preconditioner, it must not
    /// change the fit of any single RHS.
    #[test]
    fn prop_batch_matches_columnwise_solve(
        (cats, ys) in random_fe_problem_strategy().prop_flat_map(|(cats, y0)| {
            let n = y0.len();
            (
                Just(cats),
                proptest::collection::vec(proptest::collection::vec(-10.0f64..10.0, n), 2..=4),
            )
        }),
    ) {
        let params = tight_params();
        let precond = additive_precond();

        let refs: Vec<&[f64]> = ys.iter().map(Vec::as_slice).collect();
        let batch = solve_batch(cats.view(), &refs, None, &params, &precond).unwrap();
        prop_assume!(batch.converged.iter().all(|&c| c));

        for (j, y) in ys.iter().enumerate() {
            let single = solve(cats.view(), y, None, &params, &precond).unwrap();
            prop_assume!(single.converged);
            let rel = rel_l2_diff(batch.demeaned(j), &single.demeaned);
            prop_assert!(
                rel <= 1e-6,
                "batch vs column-wise residual mismatch (column {j}): rel L2 = {rel:.3e}"
            );
        }
    }
}
