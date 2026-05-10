use within::{LocalSolverConfig, Preconditioner, ReductionStrategy, Solver, SolverParams};

#[path = "common/orchestrate_helpers.rs"]
mod common;

#[test]
fn test_lsmr_unpreconditioned() {
    let design = common::make_test_design();
    let y = common::make_y_from_unit_solution(&design);

    let params = SolverParams {
        tol: 1e-8,
        maxiter: 1000,
        ..Default::default()
    };
    let solver = Solver::from_design(design, &params, None).expect("build solver");
    let result = solver.solve(&y).expect("solve");
    common::assert_converged_with_small_residual(&result, 1e-6);
}

#[test]
fn test_lsmr_preconditioned() {
    let design = common::make_test_design();
    let y = common::make_y_from_unit_solution(&design);

    let params = SolverParams {
        tol: 1e-8,
        maxiter: 1000,
        ..Default::default()
    };
    let precond = Preconditioner::Additive(LocalSolverConfig::default(), ReductionStrategy::Auto);
    let solver = Solver::from_design(design, &params, Some(&precond)).expect("build solver");
    let result = solver.solve(&y).expect("solve");
    common::assert_converged_with_small_residual(&result, 1e-6);
}

#[test]
fn test_lsmr_least_squares() {
    let design = common::make_test_design();
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let params = SolverParams {
        tol: 1e-8,
        maxiter: 1000,
        ..Default::default()
    };
    let solver = Solver::from_design(design, &params, None).expect("build solver");
    let result = solver.solve(&y).expect("solve");
    assert!(result.converged, "LSMR LS did not converge");
    common::assert_solution_finite(&result);
}

#[test]
fn test_lsmr_least_squares_weighted_preconditioned() {
    let design = common::make_weighted_design(
        vec![vec![0, 1, 0, 1, 2], vec![0, 0, 1, 1, 0]],
        within::ObservationWeights::Dense(vec![1.0, 2.0, 1.5, 0.5, 3.0]),
    )
    .expect("valid weighted design");
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let params = SolverParams {
        tol: 1e-8,
        maxiter: 1000,
        ..Default::default()
    };
    let precond =
        Preconditioner::Additive(LocalSolverConfig::solver_default(), ReductionStrategy::Auto);
    let solver = Solver::from_design(design, &params, Some(&precond)).expect("build solver");
    let result = solver.solve(&y).expect("solve");
    common::assert_converged_with_small_residual(&result, 1e-6);
    common::assert_solution_finite(&result);
}

#[test]
fn test_lsmr_weighted() {
    let design = common::make_weighted_design(
        vec![vec![0, 1, 0, 1, 2], vec![0, 0, 1, 1, 0]],
        within::ObservationWeights::Dense(vec![1.0, 2.0, 1.5, 0.5, 3.0]),
    )
    .expect("valid weighted design");
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let params = SolverParams {
        tol: 1e-8,
        maxiter: 1000,
        ..Default::default()
    };
    let precond =
        Preconditioner::Additive(LocalSolverConfig::solver_default(), ReductionStrategy::Auto);
    let solver = Solver::from_design(design, &params, Some(&precond)).expect("build solver");
    let result = solver.solve(&y).expect("solve");
    common::assert_converged_with_small_residual(&result, 1e-6);
    common::assert_solution_finite(&result);
}
