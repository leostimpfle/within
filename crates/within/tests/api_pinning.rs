//! Compile-time pinning of [`within::Solver::new`]'s `impl Into<PreconditionerInput>`
//! argument shapes: if any call form below stops compiling, the public surface shifted.

use ndarray::Array2;
use within::{
    solve, solve_batch, Design, DesignOptions, LsmrOptions, PreconditionerConfig, Solver,
};

fn cats() -> Array2<u32> {
    Array2::from_shape_vec((4, 2), vec![0u32, 0, 1, 1, 2, 0, 3, 1]).expect("cats")
}

fn design(categories: &Array2<u32>) -> Design<'_> {
    Design::from_categories(categories.view(), DesignOptions::default()).expect("design")
}

#[test]
fn design_options_fields_are_public() {
    let options = DesignOptions {
        drop_singletons: true,
        weights: Some(vec![1.0, 2.0, 3.0].into()),
    };

    assert!(options.drop_singletons);
    assert_eq!(options.weights.as_deref(), Some(&[1.0, 2.0, 3.0][..]));
    assert!(options.drop_singletons());
    assert_eq!(options.weights(), Some(&[1.0, 2.0, 3.0][..]));
}

#[test]
fn precond_input_call_shapes_compile() {
    let categories = cats();
    let cfg = PreconditionerConfig::default();

    // Obtain a prebuilt preconditioner through the public path.
    let setup = Solver::new(design(&categories), &cfg).unwrap();
    let prec = setup
        .preconditioner()
        .expect("default solver has a preconditioner")
        .clone();

    // Shape 1: bare `None` — resolves through `From<Option<&PreconditionerConfig>>`.
    let _ = Solver::new(design(&categories), None).expect("None form");

    // Shape 2: `&PreconditionerConfig` — resolves through `From<&PreconditionerConfig>`.
    let _ = Solver::new(design(&categories), &cfg).expect("&Cfg form");

    // Shape 3: `Option<&PreconditionerConfig>` — explicit `Some(&cfg)`.
    let _ = Solver::new(design(&categories), Some(&cfg)).expect("Some(&Cfg) form");

    // Shape 4: owned `Preconditioner` — resolves through `From<Preconditioner>`.
    let _ = Solver::new(design(&categories), prec.clone()).expect("owned Prec form");

    // Shape 5: `&Preconditioner` — resolves through `From<&Preconditioner>` (cheap clone).
    let _ = Solver::new(design(&categories), &prec).expect("&Prec form");

    // Shape 6: owned `Preconditioner` again, consumed last so `prec` is moved here.
    let _ = Solver::new(design(&categories), prec).expect("owned Prec form (move)");
}

#[test]
fn precond_input_owned_config_form() {
    // Owned `PreconditionerConfig` resolves through `From<PreconditionerConfig>`.
    // Kept separate so the main shape sweep stays focused on the &/None variants.
    let categories = cats();
    let cfg = PreconditionerConfig::default();
    let _ = Solver::new(design(&categories), cfg).expect("owned Cfg form");
}

#[test]
fn solve_free_function_precond_call_shapes_compile() {
    let categories = cats();
    let y = vec![0.0; 4];
    let lsmr = LsmrOptions::default();
    let cfg = PreconditionerConfig::default();

    let setup = Solver::new(design(&categories), &cfg).unwrap();
    let prec = setup
        .preconditioner()
        .expect("default solver has a preconditioner")
        .clone();

    let _ = solve(design(&categories), &y, &lsmr, None).expect("None form");
    let _ = solve(design(&categories), &y, &lsmr, &cfg).expect("&Cfg form");
    let _ = solve(design(&categories), &y, &lsmr, Some(&cfg)).expect("Some(&Cfg) form");
    let _ = solve(design(&categories), &y, &lsmr, prec.clone()).expect("owned Prec form");
    let _ = solve(design(&categories), &y, &lsmr, &prec).expect("&Prec form");
    let _ = solve(design(&categories), &y, &lsmr, cfg.clone()).expect("owned Cfg form");

    // solve_batch accepts the same shapes.
    let ys: Vec<&[f64]> = vec![&y];
    let _ = solve_batch(design(&categories), &ys, &lsmr, None).expect("batch None form");
    let _ = solve_batch(design(&categories), &ys, &lsmr, &cfg).expect("batch &Cfg form");
    let _ =
        solve_batch(design(&categories), &ys, &lsmr, Some(&cfg)).expect("batch Some(&Cfg) form");
    let _ =
        solve_batch(design(&categories), &ys, &lsmr, prec.clone()).expect("batch owned Prec form");
    let _ = solve_batch(design(&categories), &ys, &lsmr, &prec).expect("batch &Prec form");
    let _ = solve_batch(design(&categories), &ys, &lsmr, cfg).expect("batch owned Cfg form");
}

#[test]
fn solver_new_weights_call_shapes_compile() {
    let categories = cats();
    let w_vec: Vec<f64> = vec![1.0; 4];

    // Bare `None` — infers, because `weights` is the concrete `Option<Vec<f64>>`
    // (no turbofish needed). The persistent solver owns its weights; borrow-based
    // one-shot weighting lives on the free `solve` function instead.
    let _ = Solver::new(design(&categories), None).expect("None weights");

    // Owned `Vec<f64>` weights — moved into the solver.
    let weighted = Design::from_categories(
        categories.view(),
        DesignOptions {
            weights: Some(w_vec.into()),
            ..Default::default()
        },
    )
    .expect("weighted design");
    let _ = Solver::new(weighted, None).expect("Vec<f64> weights");
}

#[test]
fn options_are_optional_and_tuned_additive_builds_from_crate_root() {
    // A tuned Additive preconditioner is constructible from crate-root imports
    // alone — no `within::config::` paths — and LSMR options are optional (#105).
    use within::{
        ApproxCholConfig, ApproxSchurConfig, LocalSolverConfig, PreconditionerConfig,
        ReductionStrategy, SchurMode,
    };

    let categories = cats();
    let y = vec![0.0; 4];
    let opts = LsmrOptions::default();

    let tuned = PreconditionerConfig::Additive {
        local_solver: LocalSolverConfig {
            approx_chol: ApproxCholConfig::default(),
            schur: SchurMode::Approximate(ApproxSchurConfig::default()),
            ..Default::default()
        },
        reduction: ReductionStrategy::Auto,
    };
    let solver = Solver::new(design(&categories), tuned).expect("tuned Additive builds");

    // `None` accepts the default options; `&opts` still works.
    let _ = solver.solve(&y, None).expect("solve, default options");
    let _ = solver.solve(&y, &opts).expect("solve, explicit options");
    let ys: Vec<&[f64]> = vec![&y];
    let _ = solver
        .solve_batch(&ys, None)
        .expect("solve_batch, default options");

    // Free functions accept `None` options too (previously `&LsmrOptions::default()`).
    let _ = solve(design(&categories), &y, None, None).expect("free solve, None options");
    let _ =
        solve_batch(design(&categories), &ys, None, None).expect("free solve_batch, None options");
}
