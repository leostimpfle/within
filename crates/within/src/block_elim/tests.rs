use super::compensated_sum;

// A naive left-to-right fold drops the small middle term (`1e16 + 1.0` rounds
// back to `1e16`), returning 0.0; Neumaier compensation recovers the dropped
// bits and returns exactly 1.0. The two orderings exercise both dominance
// branches of the running compensation (`sum.abs() >= value.abs()` and `else`),
// so each carried term is pinned rather than the naive and compensated results
// coinciding.
#[test]
fn recovers_catastrophic_cancellation_in_both_branches() {
    // Large value first: the `sum.abs() >= value.abs()` branch carries the 1.0.
    let big_first = [1e16, 1.0, -1e16];
    assert_eq!(
        big_first.iter().sum::<f64>(),
        0.0,
        "naive fold loses the term"
    );
    assert_eq!(compensated_sum(&big_first), 1.0);

    // Small value first: the `else` branch carries the 1.0.
    let small_first = [1.0, 1e16, -1e16];
    assert_eq!(
        small_first.iter().sum::<f64>(),
        0.0,
        "naive fold loses the term"
    );
    assert_eq!(compensated_sum(&small_first), 1.0);
}

#[test]
fn handles_empty_and_singleton() {
    assert_eq!(compensated_sum(&[]), 0.0);
    assert_eq!(compensated_sum(&[42.0]), 42.0);
}
