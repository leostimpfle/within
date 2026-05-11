use std::time::Duration;

use criterion::measurement::WallTime;
use criterion::{
    criterion_group, criterion_main, BenchmarkGroup, BenchmarkId, Criterion, SamplingMode,
};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use schwarz_precond::Operator;
use within::config::{
    ApproxCholConfig, LocalSolverConfig, Preconditioner, ReductionStrategy, SolverParams,
};
use within::domain::Design;
use within::observation::FactorMajorStore;
use within::operator::DesignOperator;
use within::Solver;

// ===========================================================================
// Shared types and helpers
// ===========================================================================

const MAXITER: usize = 200;
const TOL: f64 = 1e-6;

#[derive(Clone, Copy)]
enum FixestType {
    Simple,
    Difficult,
}

#[derive(Clone, Copy)]
struct Case {
    n_obs: usize,
    dgp_type: FixestType,
    n_fe: usize,
}

impl Case {
    fn label(&self) -> String {
        let kind = match self.dgp_type {
            FixestType::Simple => "simple",
            FixestType::Difficult => "difficult",
        };
        format!("n={} {} {}FE", self.n_obs, kind, self.n_fe)
    }
}

fn generate_fixest_like_case(case: Case, seed: u64) -> (Design<FactorMajorStore>, Vec<f64>) {
    let mut rng = SmallRng::seed_from_u64(seed);
    let n_years = 10usize;
    let n_indiv_per_firm = 23usize;

    let n_indiv = ((case.n_obs as f64 / n_years as f64).round() as usize).max(1);
    let n_firm = ((n_indiv as f64 / n_indiv_per_firm as f64).round() as usize).max(1);

    let mut indiv_id = Vec::with_capacity(case.n_obs);
    let mut year = Vec::with_capacity(case.n_obs);
    let mut firm_id = Vec::with_capacity(case.n_obs);

    for i in 0..case.n_obs {
        indiv_id.push((i / n_years) as u32);
        year.push((i % n_years) as u32);
        let firm = match case.dgp_type {
            FixestType::Simple => rng.random_range(0..n_firm) as u32,
            FixestType::Difficult => (i % n_firm) as u32,
        };
        firm_id.push(firm);
    }

    let factor_levels: Vec<Vec<u32>> = if case.n_fe == 2 {
        vec![indiv_id, year]
    } else {
        vec![indiv_id, year, firm_id]
    };

    let store = FactorMajorStore::new(factor_levels, case.n_obs).expect("valid factor-major store");
    let design = Design::from_store(store).expect("valid design");

    let mut x_true = vec![0.0; design.n_dofs];
    for x in &mut x_true {
        *x = rng.random_range(-1.0..1.0);
    }

    let mut y = vec![0.0; case.n_obs];
    DesignOperator::new(&design, None)
        .apply(&x_true, &mut y)
        .expect("apply succeeds");
    for yi in &mut y {
        *yi += 0.1 * rng.random_range(-1.0..1.0);
    }

    (design, y)
}

fn one_level_local_solver(ac2: bool) -> LocalSolverConfig {
    let mut cfg = LocalSolverConfig::solver_default();
    if ac2 {
        cfg.approx_chol = ApproxCholConfig {
            split_merge: Some(2),
            ..Default::default()
        };
    } else {
        cfg.approx_chol = ApproxCholConfig::default();
    }
    cfg
}

fn configure_group<'a>(
    c: &'a mut Criterion,
    name: &str,
    sample_size: usize,
    measurement_ms: u64,
) -> BenchmarkGroup<'a, WallTime> {
    let mut group = c.benchmark_group(name);
    group.sample_size(sample_size);
    group.measurement_time(Duration::from_millis(measurement_ms));
    group.sampling_mode(SamplingMode::Flat);
    group
}

fn run_smoke(
    group: &mut BenchmarkGroup<'_, WallTime>,
    label: &str,
    design: &Design<FactorMajorStore>,
    y: &[f64],
) {
    group.bench_function(BenchmarkId::new(label, ""), |b| {
        b.iter(|| run_lsmr_one_level(design, y, false))
    });
}

fn run_lsmr_one_level(design: &Design<FactorMajorStore>, y: &[f64], ac2: bool) {
    let params = SolverParams {
        tol: TOL,
        maxiter: MAXITER,
        ..Default::default()
    };
    let cfg = one_level_local_solver(ac2);
    let precond = Preconditioner::Additive(cfg, ReductionStrategy::Auto);
    let solver =
        Solver::from_design(design.clone(), None, &params, Some(&precond)).expect("solver build");
    let _ = solver.solve(y).expect("solve");
}

fn smoke_cases() -> [Case; 8] {
    [
        Case {
            n_obs: 100_000,
            dgp_type: FixestType::Simple,
            n_fe: 2,
        },
        Case {
            n_obs: 100_000,
            dgp_type: FixestType::Difficult,
            n_fe: 2,
        },
        Case {
            n_obs: 100_000,
            dgp_type: FixestType::Simple,
            n_fe: 3,
        },
        Case {
            n_obs: 100_000,
            dgp_type: FixestType::Difficult,
            n_fe: 3,
        },
        Case {
            n_obs: 1_000_000,
            dgp_type: FixestType::Simple,
            n_fe: 2,
        },
        Case {
            n_obs: 1_000_000,
            dgp_type: FixestType::Difficult,
            n_fe: 2,
        },
        Case {
            n_obs: 1_000_000,
            dgp_type: FixestType::Simple,
            n_fe: 3,
        },
        Case {
            n_obs: 1_000_000,
            dgp_type: FixestType::Difficult,
            n_fe: 3,
        },
    ]
}

fn bench_fixest_smoke_lsmr_1l(c: &mut Criterion) {
    let mut group = configure_group(c, "fixest_smoke_lsmr_1l", 100, 200);
    for case in smoke_cases() {
        let label = case.label();
        let (design, y) = generate_fixest_like_case(case, 42);
        group.bench_function(BenchmarkId::new("LSMR-AC", &label), |b| {
            b.iter(|| run_lsmr_one_level(&design, &y, false));
        });
        group.bench_function(BenchmarkId::new("LSMR-AC2", &label), |b| {
            b.iter(|| run_lsmr_one_level(&design, &y, true));
        });
    }
    group.finish();
}

fn mini_cases() -> [Case; 6] {
    [
        Case {
            n_obs: 10_000,
            dgp_type: FixestType::Simple,
            n_fe: 2,
        },
        Case {
            n_obs: 10_000,
            dgp_type: FixestType::Difficult,
            n_fe: 2,
        },
        Case {
            n_obs: 10_000,
            dgp_type: FixestType::Simple,
            n_fe: 3,
        },
        Case {
            n_obs: 10_000,
            dgp_type: FixestType::Difficult,
            n_fe: 3,
        },
        Case {
            n_obs: 50_000,
            dgp_type: FixestType::Simple,
            n_fe: 3,
        },
        Case {
            n_obs: 50_000,
            dgp_type: FixestType::Difficult,
            n_fe: 3,
        },
    ]
}

fn bench_fixest_mini(c: &mut Criterion) {
    let mut group = configure_group(c, "fixest_mini_lsmr_1l", 50, 100);
    for case in mini_cases() {
        let label = case.label();
        let (design, y) = generate_fixest_like_case(case, 42);
        run_smoke(&mut group, &format!("LSMR-{label}"), &design, &y);
    }
    group.finish();
}

fn matvec_cases() -> [Case; 4] {
    [
        Case {
            n_obs: 1_000_000,
            dgp_type: FixestType::Simple,
            n_fe: 2,
        },
        Case {
            n_obs: 1_000_000,
            dgp_type: FixestType::Difficult,
            n_fe: 2,
        },
        Case {
            n_obs: 1_000_000,
            dgp_type: FixestType::Simple,
            n_fe: 3,
        },
        Case {
            n_obs: 1_000_000,
            dgp_type: FixestType::Difficult,
            n_fe: 3,
        },
    ]
}

fn bench_matvec(c: &mut Criterion) {
    let mut group = configure_group(c, "matvec_weighted_design", 50, 200);
    for case in matvec_cases() {
        let label = case.label();
        let (design, _y) = generate_fixest_like_case(case, 42);
        let n_dofs = design.n_dofs;
        let n_obs = design.n_rows;
        let op = DesignOperator::new(&design, None);
        let x: Vec<f64> = (0..n_dofs).map(|i| (i as f64).sin()).collect();
        let mut y = vec![0.0; n_obs];
        group.bench_function(BenchmarkId::new("apply", &label), |b| {
            b.iter(|| op.apply(&x, &mut y).expect("apply succeeds"))
        });
    }
    group.finish();
}

criterion_group!(
    name = smoke_benches;
    config = Criterion::default();
    targets = bench_fixest_smoke_lsmr_1l,
);
criterion_group!(mini_benches, bench_fixest_mini, bench_matvec);
criterion_main!(smoke_benches, mini_benches);
