use super::reparam::SlopeReparam;
use super::{CoefficientAddress, CoefficientLayout, Solver};
use crate::channel::Channel;
use crate::config::{PreconditionerConfig, ScalingConfig, DEFAULT_DENSE_SCHUR_THRESHOLD};
use crate::domain::{build_local_domains, Design, DesignOptions, Grounding, MatrixForm};
use crate::observation::ObservationFrame;
use crate::Effect;

/// DGP kept in lockstep with `surplus_component_sampled_matches_exact_reduction`
/// in `tests/slopes_routing.rs`. A positive slope-only term is not centered by
/// whitening, so the signed pair stays all-positive — balanced — while generic
/// `z` keeps it strictly inside the PSD cone: genuine surplus, grounded.
fn at(term: usize, level: usize, column: usize) -> CoefficientAddress {
    CoefficientAddress {
        channel: Channel { term, column },
        level,
    }
}

fn positive_slope_only_panel() -> (Vec<u32>, Vec<u32>, Vec<f64>) {
    let n = 8000usize;
    let f: Vec<u32> = (0..n).map(|i| (i % 80) as u32).collect();
    let g: Vec<u32> = (0..n).map(|i| ((i / 80) % 40) as u32).collect();
    let z: Vec<f64> = (0..n)
        .map(|i| 0.5 + ((i * 13) % 100) as f64 / 100.0)
        .collect();
    (f, g, z)
}

#[test]
fn positive_slope_only_pair_grounds_beyond_dense_threshold() {
    let (f, g, z) = positive_slope_only_panel();
    let effects = vec![
        Effect::new(&f, false, [&z[..]]).expect("slope effect"),
        Effect::new(&g, true, []).expect("plain effect"),
    ];
    let mut design = Design::new(effects).expect("design");
    let _reparam = SlopeReparam::build(&mut design.frame, &design.terms, None);
    let (domains, warnings) =
        build_local_domains(&design, None, &ScalingConfig::default()).expect("domains");
    assert!(
        domains.iter().any(|ld| {
            let ct = &ld.component.matrix.cross_tab;
            ld.component.form == MatrixForm::Laplacian
                && ld.component.matrix.grounding == Grounding::Grounded
                && ct.n_rows().min(ct.n_cols()) > DEFAULT_DENSE_SCHUR_THRESHOLD
        }),
        "fixture must ground a component past the dense threshold (warnings: {warnings:?})"
    );
}

#[test]
fn coefficient_layout_translates_addresses_both_ways() {
    // term 0: plain 3-level factor; term 1: 2-level factor with intercept and one slope.
    let f = [0u32, 1, 2, 0, 1, 2];
    let g = [0u32, 0, 1, 1, 0, 1];
    let z = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let design = Design::new(vec![
        Effect::new(&f, true, []).expect("plain effect"),
        Effect::new(&g, true, [&z[..]]).expect("slope effect"),
    ])
    .expect("design");
    let layout = CoefficientLayout::from_design(&design);

    assert_eq!(layout.n_terms(), 2);
    assert_eq!(
        (layout.n_levels(0), layout.n_columns(0)),
        (Some(3), Some(1))
    );
    assert_eq!(
        (layout.n_levels(1), layout.n_columns(1)),
        (Some(2), Some(2))
    );
    assert_eq!(layout.n_levels(2), None);

    // Forward matches the documented `offset + column * n_levels + level`.
    assert_eq!(layout.index(at(0, 2, 0)), Some(2));
    assert_eq!(layout.index(at(1, 0, 0)), Some(3)); // term-1 intercept, level 0
    assert_eq!(layout.index(at(1, 1, 1)), Some(6)); // term-1 slope, level 1
    assert_eq!(layout.n_dofs(), 7);

    // Out-of-range coordinates are rejected, not silently wrapped.
    assert_eq!(layout.index(at(1, 2, 0)), None); // level past n_levels
    assert_eq!(layout.index(at(1, 0, 2)), None); // column past n_columns
    assert_eq!(layout.index(at(2, 0, 0)), None); // term past n_terms
    assert_eq!(layout.address(7), None);

    // `address` inverts `index` for every flat slot.
    for i in 0..layout.n_dofs() {
        assert_eq!(layout.index(layout.address(i).expect("in range")), Some(i));
    }
}

#[test]
fn solver_filters_caller_rhs_and_restores_removed_rows_as_nan() {
    let frame = ObservationFrame::new(vec![vec![0u32, 1, 0, 0, 2].into()], Vec::new()).unwrap();
    let design = DesignOptions::default()
        .drop_singletons(true)
        .from_frame(frame)
        .unwrap();

    let solver = Solver::new(design, PreconditionerConfig::default()).unwrap();
    let y = [1.0, f64::NAN, 3.0, 4.0, f64::NAN];
    let result = solver.solve(&y, None).unwrap();

    assert_eq!(solver.n_obs(), 5);
    assert_eq!(solver.n_retained_obs(), 3);
    assert_eq!(result.demeaned.len(), 5);
    assert!((result.demeaned[0] + 5.0 / 3.0).abs() < 1e-10);
    assert!(result.demeaned[1].is_nan());
    assert!((result.demeaned[2] - 1.0 / 3.0).abs() < 1e-10);
    assert!((result.demeaned[3] - 4.0 / 3.0).abs() < 1e-10);
    assert!(result.demeaned[4].is_nan());

    let batch = solver.solve_batch(&[&y, &y], None).unwrap();
    assert_eq!(batch.n_obs, 5);
    for demeaned in [batch.demeaned(0), batch.demeaned(1)] {
        assert!((demeaned[0] + 5.0 / 3.0).abs() < 1e-10);
        assert!(demeaned[1].is_nan());
        assert!((demeaned[2] - 1.0 / 3.0).abs() < 1e-10);
        assert!((demeaned[3] - 4.0 / 3.0).abs() < 1e-10);
        assert!(demeaned[4].is_nan());
    }
}
