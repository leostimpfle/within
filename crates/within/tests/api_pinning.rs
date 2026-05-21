//! Pinning tests for [`within::Solver::new`] argument resolution.
//!
//! The persistent `Solver` constructor accepts the preconditioner argument in
//! several shapes via `impl Into<PreconditionerInput>`. The shapes themselves
//! are part of the public API contract — adding a new `From<Option<X>>` impl
//! to `PreconditionerInput` could make bare `None` ambiguous, and adding a
//! `From<&Foo>` impl that overlaps with the existing chain could silently
//! change which form a call site resolves to.
//!
//! This file is therefore a *compile-time* pinning: if any of the call shapes
//! below stops compiling, the public surface of `Solver::new` has shifted in
//! a way that warrants explicit review.

use ndarray::Array2;
use within::{solve, solve_batch, LsmrOptions, PreconditionerConfig, Solver};

fn cats() -> Array2<u32> {
    Array2::from_shape_vec((4, 2), vec![0u32, 0, 1, 1, 2, 0, 3, 1]).expect("cats")
}

#[test]
fn precond_input_call_shapes_compile() {
    let categories = cats();
    let cfg = PreconditionerConfig::default();

    // Obtain a prebuilt preconditioner through the public path.
    let setup = Solver::new(categories.view(), None::<Vec<f64>>, &cfg).unwrap();
    let prec = setup
        .preconditioner()
        .expect("default solver has a preconditioner")
        .clone();

    // Shape 1: bare `None` — resolves through `From<Option<&PreconditionerConfig>>`.
    let _ = Solver::new(categories.view(), None::<Vec<f64>>, None).expect("None form");

    // Shape 2: `&PreconditionerConfig` — resolves through `From<&PreconditionerConfig>`.
    let _ = Solver::new(categories.view(), None::<Vec<f64>>, &cfg).expect("&Cfg form");

    // Shape 3: `Option<&PreconditionerConfig>` — explicit `Some(&cfg)`.
    let _ = Solver::new(categories.view(), None::<Vec<f64>>, Some(&cfg)).expect("Some(&Cfg) form");

    // Shape 4: owned `Preconditioner` — resolves through `From<Preconditioner>`.
    let _ =
        Solver::new(categories.view(), None::<Vec<f64>>, prec.clone()).expect("owned Prec form");

    // Shape 5: `&Preconditioner` — resolves through `From<&Preconditioner>` (cheap clone).
    let _ = Solver::new(categories.view(), None::<Vec<f64>>, &prec).expect("&Prec form");

    // Shape 6: owned `Preconditioner` again, consumed last so `prec` is moved here.
    let _ = Solver::new(categories.view(), None::<Vec<f64>>, prec).expect("owned Prec form (move)");
}

#[test]
fn precond_input_owned_config_form() {
    // Owned `PreconditionerConfig` resolves through `From<PreconditionerConfig>`.
    // Kept separate so the main shape sweep stays focused on the &/None variants.
    let categories = cats();
    let cfg = PreconditionerConfig::default();
    let _ = Solver::new(categories.view(), None::<Vec<f64>>, cfg).expect("owned Cfg form");
}

#[test]
fn solve_free_function_precond_call_shapes_compile() {
    let categories = cats();
    let y = vec![0.0; 4];
    let lsmr = LsmrOptions::default();
    let cfg = PreconditionerConfig::default();

    let setup = Solver::new(categories.view(), None::<Vec<f64>>, &cfg).unwrap();
    let prec = setup
        .preconditioner()
        .expect("default solver has a preconditioner")
        .clone();

    let _ = solve(categories.view(), &y, None, &lsmr, None).expect("None form");
    let _ = solve(categories.view(), &y, None, &lsmr, &cfg).expect("&Cfg form");
    let _ = solve(categories.view(), &y, None, &lsmr, Some(&cfg)).expect("Some(&Cfg) form");
    let _ = solve(categories.view(), &y, None, &lsmr, prec.clone()).expect("owned Prec form");
    let _ = solve(categories.view(), &y, None, &lsmr, &prec).expect("&Prec form");
    let _ = solve(categories.view(), &y, None, &lsmr, cfg.clone()).expect("owned Cfg form");

    // solve_batch accepts the same shapes.
    let ys: Vec<&[f64]> = vec![&y];
    let _ = solve_batch(categories.view(), &ys, None, &lsmr, None).expect("batch None form");
    let _ = solve_batch(categories.view(), &ys, None, &lsmr, &cfg).expect("batch &Cfg form");
    let _ = solve_batch(categories.view(), &ys, None, &lsmr, Some(&cfg))
        .expect("batch Some(&Cfg) form");
    let _ = solve_batch(categories.view(), &ys, None, &lsmr, prec.clone())
        .expect("batch owned Prec form");
    let _ = solve_batch(categories.view(), &ys, None, &lsmr, &prec).expect("batch &Prec form");
    let _ = solve_batch(categories.view(), &ys, None, &lsmr, cfg).expect("batch owned Cfg form");
}

#[test]
fn solver_new_weights_call_shapes_compile() {
    let categories = cats();
    let w_vec: Vec<f64> = vec![1.0; 4];
    let w_slice: &[f64] = &w_vec;

    // None weights.
    let _ = Solver::new(categories.view(), None::<Vec<f64>>, None).expect("None weights");

    // Owned Vec<f64> weights.
    let _ = Solver::new(categories.view(), Some(w_vec.clone()), None).expect("Vec<f64> weights");

    // &[f64] weights — accepted via `impl Into<Vec<f64>>` (clones once).
    let _ = Solver::new(categories.view(), Some(w_slice), None).expect("&[f64] weights");
}
