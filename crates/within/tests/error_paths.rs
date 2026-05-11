use std::error::Error;

use ndarray::Array2;
use schwarz_precond::SolveError;
use within::observation::FactorMajorStore;
use within::{solve, BuildError, Design, Preconditioner, Solver, SolverParams, WithinError};

#[test]
fn test_empty_observations_error() {
    // FactorMajorStore::new allows 0 rows; EmptyObservations is raised by Design::from_store
    let store = FactorMajorStore::new(vec![vec![], vec![]], 0).expect("store ok");
    let result = Design::from_store(store);
    assert!(result.is_err());
    match result.unwrap_err() {
        BuildError::EmptyObservations => {}
        other => panic!("Expected EmptyObservations, got: {:?}", other),
    }
}

#[test]
fn test_observation_count_mismatch_error() {
    // Factor columns have different lengths
    let result = FactorMajorStore::new(vec![vec![0, 1, 2], vec![0, 1]], 3);
    assert!(result.is_err());
    match result.unwrap_err() {
        BuildError::ObservationCountMismatch { .. } => {}
        other => panic!("Expected ObservationCountMismatch, got: {:?}", other),
    }
}

#[test]
fn test_weight_count_mismatch_error() {
    // Weights of wrong length are caught at Solver construction time.
    let store = FactorMajorStore::new(vec![vec![0, 1, 2], vec![0, 1, 0]], 3).expect("store ok");
    let design = Design::from_store(store).expect("valid design");
    let params = SolverParams::default();
    let result = Solver::from_design(design, Some(vec![1.0, 2.0]), &params, None);
    let err = result
        .err()
        .expect("expected WeightCountMismatch error, got Ok");
    match err {
        BuildError::WeightCountMismatch { .. } => {}
        other => panic!("Expected WeightCountMismatch, got: {:?}", other),
    }
}

#[test]
fn test_empty_categories_via_solve() {
    let cats = Array2::<u32>::zeros((0, 2));
    let y: Vec<f64> = vec![];
    let params = SolverParams::default();
    let precond = Preconditioner::default();
    let result = solve(cats.view(), &y, None, &params, Some(&precond));
    assert!(result.is_err());
    match result.unwrap_err() {
        WithinError::Build(BuildError::EmptyObservations) => {}
        other => panic!(
            "Expected Build(EmptyObservations) via solve(), got: {:?}",
            other
        ),
    }
}

// ---------------------------------------------------------------------------
// Display tests for BuildError variants
// ---------------------------------------------------------------------------

#[test]
fn test_build_error_display_empty_observations() {
    let e = BuildError::EmptyObservations;
    assert_eq!(e.to_string(), "no observations provided");
}

#[test]
fn test_build_error_display_observation_count_mismatch() {
    let e = BuildError::ObservationCountMismatch {
        factor: 1,
        expected: 10,
        got: 5,
    };
    let s = e.to_string();
    assert!(s.contains("factor 1"));
    assert!(s.contains("5"));
    assert!(s.contains("10"));
}

#[test]
fn test_build_error_display_weight_count_mismatch() {
    let e = BuildError::WeightCountMismatch {
        expected: 10,
        got: 5,
    };
    let s = e.to_string();
    assert!(s.contains("5"));
    assert!(s.contains("10"));
}

#[test]
fn test_build_error_display_singular_diagonal() {
    let e = BuildError::SingularDiagonal {
        block: "test_block",
        index: 42,
    };
    let s = e.to_string();
    assert!(s.contains("test_block"));
    assert!(s.contains("42"));
}

#[test]
fn test_build_error_display_local_solver_build() {
    let e = BuildError::LocalSolverBuild("factorization failed".to_string());
    assert!(e.to_string().contains("factorization failed"));
}

#[test]
fn test_build_error_display_preconditioner() {
    let inner = schwarz_precond::BuildError::GlobalIndexOutOfBounds {
        subdomain: 0,
        local_index: 1,
        global_index: 5,
        n_dofs: 3,
    };
    let e = BuildError::Preconditioner(inner);
    let s = e.to_string();
    assert!(s.contains("5"));
    assert!(s.contains("3"));
}

// ---------------------------------------------------------------------------
// Display tests for WithinError union
// ---------------------------------------------------------------------------

#[test]
fn test_within_error_display_build() {
    let e = WithinError::Build(BuildError::EmptyObservations);
    assert_eq!(e.to_string(), "no observations provided");
}

#[test]
fn test_within_error_display_solve() {
    let inner = SolveError::Synchronization { context: "test" };
    let e = WithinError::Solve(inner);
    assert!(e.to_string().contains("test"));
}

// ---------------------------------------------------------------------------
// Error::source() tests
// ---------------------------------------------------------------------------

#[test]
fn test_build_error_source_leaf_variants_have_no_source() {
    let variants: Vec<BuildError> = vec![
        BuildError::EmptyObservations,
        BuildError::ObservationCountMismatch {
            factor: 0,
            expected: 1,
            got: 2,
        },
        BuildError::WeightCountMismatch {
            expected: 1,
            got: 2,
        },
        BuildError::SingularDiagonal {
            block: "b",
            index: 0,
        },
        BuildError::LocalSolverBuild("x".to_string()),
    ];
    for e in &variants {
        assert!(e.source().is_none(), "expected None source for {:?}", e);
    }
}

#[test]
fn test_build_error_source_preconditioner_chains() {
    let inner = schwarz_precond::BuildError::GlobalIndexOutOfBounds {
        subdomain: 0,
        local_index: 1,
        global_index: 5,
        n_dofs: 3,
    };
    let e = BuildError::Preconditioner(inner);
    assert!(e.source().is_some());
}

#[test]
fn test_within_error_build_leaf_variant_has_no_source() {
    // WithinError::Build is transparent, so source() forwards to the inner
    // BuildError's source. A leaf variant has no underlying source.
    let e = WithinError::Build(BuildError::EmptyObservations);
    assert!(e.source().is_none());
}

#[test]
fn test_within_error_build_preconditioner_chains_through_transparent_wrapper() {
    // Transparent: WithinError -> (BuildError::Preconditioner via #[source]) -> schwarz_precond::BuildError
    let inner = schwarz_precond::BuildError::GlobalIndexOutOfBounds {
        subdomain: 0,
        local_index: 1,
        global_index: 5,
        n_dofs: 3,
    };
    let e = WithinError::Build(BuildError::Preconditioner(inner));
    assert!(e.source().is_some());
}

#[test]
fn test_within_error_solve_leaf_variant_has_no_source() {
    let inner = SolveError::Synchronization { context: "test" };
    let e = WithinError::Solve(inner);
    assert!(e.source().is_none());
}

// ---------------------------------------------------------------------------
// Convenience-wrapper From conversions
// ---------------------------------------------------------------------------

#[test]
fn test_within_error_from_build_error() {
    let inner = BuildError::EmptyObservations;
    let e: WithinError = inner.into();
    match e {
        WithinError::Build(BuildError::EmptyObservations) => {}
        other => panic!("expected Build(EmptyObservations), got: {:?}", other),
    }
}

#[test]
fn test_within_error_from_solve_error() {
    let inner = SolveError::Synchronization { context: "test" };
    let e: WithinError = inner.into();
    match e {
        WithinError::Solve(_) => {}
        other => panic!("expected Solve, got: {:?}", other),
    }
}
