//! Domain layer: [`Design`] (design-matrix metadata) and factor-pair [`Subdomain`] construction.

pub(crate) mod cross_tab;
mod effect;
pub(crate) mod factor_pairs;

pub(crate) use cross_tab::{find_all_active_levels, BlockDiagonals, CrossTab};

pub use effect::Effect;

pub(crate) use factor_pairs::{
    build_local_domains, CoordinateMap, Grounding, LocalComponent, LocalDomain, MatrixForm,
    SddmMatrix,
};

use std::borrow::Cow;

use crate::channel::Channel;
use crate::observation::ObservationFrame;
use crate::BuildError;

/// A slice that is guaranteed non-empty by construction.
#[repr(transparent)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NonEmpty<T>(Box<[T]>);

impl<T> NonEmpty<T> {
    /// `None` if `items` is empty.
    pub fn new(items: impl Into<Box<[T]>>) -> Option<Self> {
        let items = items.into();
        (!items.is_empty()).then(|| Self(items))
    }

    /// A single-element run.
    pub fn of(item: T) -> Self {
        Self(Box::new([item]))
    }

    /// Structure-preserving map; non-emptiness is carried over.
    pub fn map<U>(&self, f: impl FnMut(&T) -> U) -> NonEmpty<U> {
        NonEmpty(self.0.iter().map(f).collect())
    }
}

impl<T> std::ops::Deref for NonEmpty<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        &self.0
    }
}

/// A coefficient column's loading: the intercept's implicit `1.0`, or a covariate as `T`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Loading<T> {
    /// The intercept column; loading value `1.0` at every observation.
    Constant,
    /// A slope column.
    Covariate(T),
}

impl<T> Loading<T> {
    /// The covariate payload; `None` for the constant column.
    pub fn covariate(&self) -> Option<&T> {
        match self {
            Self::Constant => None,
            Self::Covariate(t) => Some(t),
        }
    }

    /// Replace the covariate payload, preserving which variant this is.
    pub fn map<U>(&self, f: impl FnOnce(&T) -> U) -> Loading<U> {
        match self {
            Self::Constant => Loading::Constant,
            Self::Covariate(t) => Loading::Covariate(f(t)),
        }
    }
}

pub(crate) fn map_to_internal_order<'v, T: Clone>(
    rows: Option<&[u32]>,
    n_obs_input: usize,
    values: &'v [T],
) -> Cow<'v, [T]> {
    assert_eq!(
        values.len(),
        n_obs_input,
        "observation vector length must match the design input"
    );
    match rows {
        None => Cow::Borrowed(values),
        Some(rows) => Cow::Owned(
            rows.iter()
                .map(|&caller| values[caller as usize].clone())
                .collect(),
        ),
    }
}

/// Configuration applied while constructing a [`Design`].
///
/// Options operate on the observation data before the design's internal row
/// order is finalized. Use [`DesignOptions::from_effects`] or
/// [`DesignOptions::from_frame`] to construct a configured design; the
/// convenience constructors on [`Design`] use [`DesignOptions::default`].
#[derive(Clone, Debug, PartialEq)]
pub struct DesignOptions<'a> {
    drop_singletons: bool,
    locality_sort: bool,
    weights: Option<Cow<'a, [f64]>>,
}

impl<'a> DesignOptions<'a> {
    /// Remove observations belonging to a singleton level in any fixed-effect
    /// term.
    ///
    /// Removal is iterative: dropping one observation can make another level a
    /// singleton, so processing continues until every retained level occurs at
    /// least twice. The default is `false`.
    #[must_use]
    pub fn drop_singletons(mut self, enabled: bool) -> Self {
        self.drop_singletons = enabled;
        self
    }

    /// Record observation weights in caller row order.
    ///
    /// Borrowed slices remain borrowed by the resulting [`Design`]; owned
    /// vectors are retained without another copy.
    #[must_use]
    pub fn weights(mut self, weights: impl Into<Cow<'a, [f64]>>) -> Self {
        self.weights = Some(weights.into());
        self
    }

    /// Lower effect terms into a configured design.
    pub fn from_effects(
        self,
        effects: impl IntoIterator<Item = Effect<'a>>,
    ) -> Result<Design<'a>, BuildError> {
        let mut categorical: Vec<Cow<'a, [u32]>> = Vec::new();
        let mut continuous: Vec<Cow<'a, [f64]>> = Vec::new();
        let mut structure: Vec<NonEmpty<Loading<u32>>> = Vec::new();
        for effect in effects {
            structure.push(effect.columns().map(|column| {
                column.map(|&z| {
                    continuous.push(Cow::Borrowed(z));
                    (continuous.len() - 1) as u32
                })
            }));
            categorical.push(Cow::Borrowed(effect.levels()));
        }
        let frame = ObservationFrame::new(categorical, continuous)?;
        Design::build(frame, structure, self)
    }

    /// Construct a configured design from a frame of plain factors.
    pub fn from_frame(self, frame: ObservationFrame<'a>) -> Result<Design<'a>, BuildError> {
        let structure = vec![NonEmpty::of(Loading::Constant); frame.n_factors()];
        Design::build(frame, structure, self)
    }

    #[doc(hidden)]
    pub fn with_locality_sort(mut self, enabled: bool) -> Self {
        self.locality_sort = enabled;
        self
    }

    pub(crate) fn weight_values(&self) -> Option<&[f64]> {
        self.weights.as_deref()
    }
}

impl Default for DesignOptions<'_> {
    fn default() -> Self {
        Self {
            drop_singletons: false,
            locality_sort: true,
            weights: None,
        }
    }
}

/// Per-term metadata; coefficient `c` of `level` lives at `offset + c · n_levels + level`.
#[derive(Debug, Clone)]
pub(crate) struct TermMeta {
    pub n_levels: usize,
    pub offset: usize,
    /// Non-decreasing in the design's internal row order (fixed at construction).
    pub sorted: bool,
    /// Coefficient columns in layout order; `Covariate` indexes the frame's continuous columns.
    pub columns: NonEmpty<Loading<u32>>,
}

impl TermMeta {
    pub fn n_columns(&self) -> usize {
        self.columns.len()
    }

    pub fn n_dofs(&self) -> usize {
        self.n_columns() * self.n_levels
    }

    /// Global DOF base of coefficient column `column`.
    pub fn column_base(&self, column: usize) -> usize {
        self.offset + column * self.n_levels
    }
}

/// Stable argsort of observations by a level column, ascending.
///
/// Dense counting sort in `O(n_obs + n_levels)` or sparse comparison sort — same
/// permutation either way, gated as in `schur::sort_and_dedup`. Gappy caller codes
/// can span far more levels than rows, where the bucket array outgrows the output.
fn stable_argsort(key: &[u32], n_levels: usize) -> Vec<u32> {
    let n_obs = key.len();
    debug_assert!(
        u32::try_from(n_obs).is_ok(),
        "observation index must fit the u32 permutation"
    );
    if n_obs < n_levels {
        // MUST be `sort_by_cached_key`: `sort_by_key` re-gathers `key[i]` O(n log n) times.
        let mut perm: Vec<u32> = (0..n_obs as u32).collect();
        perm.sort_by_cached_key(|&i| key[i as usize]);
        return perm;
    }
    let mut cursors = vec![0usize; n_levels + 1];
    for &k in key {
        debug_assert!(
            (k as usize) < n_levels,
            "counting sort key must be a level id (< n_levels)"
        );
        cursors[k as usize + 1] += 1;
    }
    for i in 1..cursors.len() {
        cursors[i] += cursors[i - 1];
    }
    let mut perm = vec![0u32; n_obs];
    for (i, &k) in key.iter().enumerate() {
        let cursor = &mut cursors[k as usize];
        perm[*cursor] = i as u32;
        *cursor += 1;
    }
    perm
}

/// Return caller rows surviving iterative singleton removal.
///
/// Adapted from pyfixest's [`_detect_singletons_rs`](https://github.com/py-econometrics/pyfixest/blob/0f608eb6e13930b355b4dac9b3f34ad5974e95a1/src/detect_singletons.rs#L25-L93)
fn retained_non_singletons(frame: &ObservationFrame<'_>) -> Result<Option<Vec<u32>>, BuildError> {
    let n_obs = frame.n_obs();
    if frame.n_factors() == 0 {
        return Ok(None);
    }
    if u32::try_from(n_obs).is_err() {
        return Err(BuildError::RowIndexSpaceExceedsU32 { n_obs });
    }

    let max_level = (0..frame.n_factors())
        .filter_map(|term| frame.level_column(term).iter().max().copied())
        .max()
        .unwrap_or(0) as usize;
    let mut counts = vec![0u32; max_level + 1];
    let mut retained: Vec<u32> = (0..n_obs as u32).collect();
    let mut n_retained = n_obs;

    loop {
        let previous_n_retained = n_retained;

        for term in 0..frame.n_factors() {
            let levels = frame.level_column(term);
            counts.fill(0);

            let mut n_singleton_levels = 0i32;
            for &row in &retained[..n_retained] {
                let level = levels[row as usize] as usize;
                let count = counts[level];
                n_singleton_levels += i32::from(count == 0) - i32::from(count == 1);
                counts[level] += 1;
            }

            if n_singleton_levels == 0 {
                continue;
            }

            let mut write = 0;
            for read in 0..n_retained {
                let row = retained[read];
                if counts[levels[row as usize] as usize] != 1 {
                    retained[write] = row;
                    write += 1;
                }
            }
            n_retained = write;
        }

        if previous_n_retained == n_retained {
            break;
        }
    }

    if n_retained == n_obs {
        return Ok(None);
    }
    if n_retained == 0 {
        return Err(BuildError::EmptyObservations);
    }

    retained.truncate(n_retained);
    Ok(Some(retained))
}

/// Fixed-effects design: observation columns plus coefficient-space layout.
#[derive(Clone, Debug)]
pub struct Design<'a> {
    /// Columns in internal row order (caller's, or an owned locality-sorted copy).
    pub(crate) frame: ObservationFrame<'a>,
    pub(crate) terms: Vec<TermMeta>,
    /// Number of observations provided by caller.
    pub(crate) n_obs_input: usize,
    /// Number of retained observations.
    pub(crate) n_obs: usize,
    pub(crate) n_dofs: usize,
    /// `rows[k]` = caller's original row represented at internal position `k`.
    ///
    /// This single map composes semantic row selection with locality sorting.
    /// `None` is the identity map and preserves zero-copy observation input.
    pub(crate) rows: Option<Vec<u32>>,
    /// Construction configuration in caller order.
    pub(crate) options: DesignOptions<'a>,
}

impl<'a> Design<'a> {
    /// Lower effect terms into a design, laid out term-major (`offset[t] + c · L_t + level`).
    pub fn new(effects: impl IntoIterator<Item = Effect<'a>>) -> Result<Self, BuildError> {
        DesignOptions::default().from_effects(effects)
    }

    /// Intercept-only factors, level count `max + 1`; locality-sorts an unsorted dominant factor.
    pub fn from_frame(frame: ObservationFrame<'a>) -> Result<Self, BuildError> {
        DesignOptions::default().from_frame(frame)
    }

    /// [`from_frame`](Self::from_frame) without the locality sort — profiling escape hatch.
    #[doc(hidden)]
    pub fn from_frame_unsorted(frame: ObservationFrame<'a>) -> Result<Self, BuildError> {
        DesignOptions::default()
            .with_locality_sort(false)
            .from_frame(frame)
    }

    /// `column_structure[term]` = that term's coefficient columns, aligned with the frame.
    fn build(
        frame: ObservationFrame<'a>,
        column_structure: Vec<NonEmpty<Loading<u32>>>,
        options: DesignOptions<'a>,
    ) -> Result<Self, BuildError> {
        if frame.n_obs() == 0 {
            return Err(BuildError::EmptyObservations);
        }
        debug_assert_eq!(column_structure.len(), frame.n_factors());

        let n_obs_input = frame.n_obs();
        let mut terms = Vec::with_capacity(frame.n_factors());
        let mut offset = 0;
        for (q, columns) in column_structure.into_iter().enumerate() {
            let col = frame.level_column(q);
            let mut max = 0;
            let mut sorted = true;
            let mut prev = 0;
            for &v in col {
                max = max.max(v);
                sorted &= v >= prev;
                prev = v;
            }
            let meta = TermMeta {
                n_levels: max as usize + 1,
                offset,
                sorted,
                columns,
            };
            offset += meta.n_dofs();
            terms.push(meta);
        }

        // Rejected here rather than left to panic in `to_u32`.
        if u32::try_from(offset).is_err() {
            return Err(BuildError::DofSpaceExceedsU32 { n_dofs: offset });
        }

        // Sort by the term contributing the most DOFs (for plain factors, the
        // highest-cardinality one) so its gather/scatter runs sequentially.
        // `rows` indexes observations as u32; beyond u32::MAX rows skip
        // the optimization — the solve itself has no such limit.
        let dominant = (0..terms.len()).max_by_key(|&q| terms[q].n_dofs());
        let mut rows = if options.drop_singletons {
            retained_non_singletons(&frame)?
        } else {
            None
        };

        if let Some(dominant_term) = dominant {
            let should_sort = options.locality_sort
                && !terms[dominant_term].sorted
                && u32::try_from(n_obs_input).is_ok();

            if should_sort {
                let key = frame.level_column(dominant_term);
                match &mut rows {
                    Some(retained) => {
                        retained.sort_by_cached_key(|&caller_row| key[caller_row as usize]);
                    }
                    None => {
                        rows = Some(stable_argsort(key, terms[dominant_term].n_levels));
                    }
                }
            }
        }

        let n_obs_retained = rows.as_ref().map_or(n_obs_input, Vec::len);
        let frame = match rows.as_deref() {
            Some(selected_rows) => frame.permuted(selected_rows),
            None => frame,
        };

        // Filtering and locality sorting can change whether any factor is
        // non-decreasing in internal row order.
        for (term, meta) in terms.iter_mut().enumerate() {
            meta.sorted = frame.level_column(term).is_sorted();
        }

        let design = Design {
            frame,
            terms,
            n_obs_input,
            n_obs: n_obs_retained,
            n_dofs: offset,
            rows,
            options,
        };
        design.validate_weights(design.weights())?;
        Ok(design)
    }

    /// Convert the frame's columns to owned, dropping ties to caller buffers.
    pub fn into_owned(self) -> Design<'static> {
        Design {
            frame: self.frame.into_owned(),
            terms: self.terms,
            n_obs_input: self.n_obs_input,
            n_obs: self.n_obs,
            n_dofs: self.n_dofs,
            rows: self.rows,
            options: DesignOptions {
                drop_singletons: self.options.drop_singletons,
                locality_sort: self.options.locality_sort,
                weights: self
                    .options
                    .weights
                    .map(|weights| Cow::Owned(weights.into_owned())),
            },
        }
    }

    /// Validate that an optional weight slice matches this design's observation count.
    pub(crate) fn validate_weights(&self, weights: Option<&[f64]>) -> Result<(), BuildError> {
        if let Some(w) = weights {
            if w.len() != self.n_obs_input {
                return Err(BuildError::WeightCountMismatch {
                    expected: self.n_obs_input,
                    got: w.len(),
                });
            }
            // `W^{1/2}` is applied to the design, so each weight must be finite and
            // non-negative; otherwise `sqrt(w)` is NaN and the solution is silently
            // corrupted. `wi >= 0.0` already rejects NaN (comparisons with NaN are
            // false); `is_finite` additionally rejects `+∞`.
            let invalid = match &self.rows {
                None => w
                    .iter()
                    .enumerate()
                    .find(|&(_, &value)| !(value >= 0.0 && value.is_finite()))
                    .map(|(caller, &value)| (caller, value)),
                Some(rows) => rows
                    .iter()
                    .filter_map(|&caller| {
                        let value = w[caller as usize];
                        (!(value >= 0.0 && value.is_finite())).then_some((caller as usize, value))
                    })
                    .min_by_key(|&(caller, _)| caller),
            };
            if let Some((index, value)) = invalid {
                return Err(BuildError::InvalidWeight { index, value });
            }
        }
        Ok(())
    }

    /// Caller order → retained internal order: `out[k] = v[rows[k]]`.
    ///
    /// Borrows when every caller row is retained in its original order.
    pub fn to_internal_order<'v, T: Clone>(&self, v: &'v [T]) -> Cow<'v, [T]> {
        map_to_internal_order(self.rows.as_deref(), self.n_obs_input, v)
    }

    /// Retained internal order → original caller shape.
    ///
    /// Dropped caller rows are represented by `NaN`.
    pub fn from_internal_order(&self, v: Vec<f64>) -> Vec<f64> {
        assert_eq!(
            v.len(),
            self.n_obs,
            "internal vector length must match retained observations"
        );
        match &self.rows {
            None => v,
            Some(rows) => {
                let mut out = vec![f64::NAN; self.n_obs_input];
                for (k, &orig) in rows.iter().enumerate() {
                    out[orig as usize] = v[k];
                }
                out
            }
        }
    }

    /// Number of categorical factors in the design.
    #[inline]
    pub fn n_factors(&self) -> usize {
        self.terms.len()
    }

    /// The term's coefficient columns in layout order.
    pub(crate) fn channels(&self, term: usize) -> impl Iterator<Item = Channel> + '_ {
        (0..self.terms[term].n_columns()).map(move |column| Channel { term, column })
    }

    /// How `channel` loads onto each observation.
    pub(crate) fn loading(&self, channel: Channel) -> Loading<u32> {
        self.terms[channel.term].columns[channel.column]
    }

    /// Number of observations (rows of D).
    #[inline]
    pub fn n_obs(&self) -> usize {
        self.n_obs
    }

    /// Number of rows expected in caller-provided observation vectors.
    #[inline]
    pub fn input_n_obs(&self) -> usize {
        self.n_obs_input
    }

    /// Construction options retained by this design.
    #[inline]
    pub fn options(&self) -> &DesignOptions<'a> {
        &self.options
    }

    /// Observation weights in caller row order.
    #[inline]
    pub fn weights(&self) -> Option<&[f64]> {
        self.options.weights.as_deref()
    }

    /// Original caller row represented by an internal observation position.
    #[inline]
    pub(crate) fn caller_row(&self, internal: usize) -> usize {
        self.rows
            .as_ref()
            .map_or(internal, |rows| rows[internal] as usize)
    }

    /// Total degrees of freedom (columns of D).
    #[inline]
    pub fn n_dofs(&self) -> usize {
        self.n_dofs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observation::ObservationFrame;

    fn frame(categorical: Vec<Vec<u32>>, continuous: Vec<Vec<f64>>) -> ObservationFrame<'static> {
        ObservationFrame::new(
            categorical.into_iter().map(Into::into).collect(),
            continuous.into_iter().map(Into::into).collect(),
        )
        .unwrap()
    }

    /// Both `stable_argsort` branches must emit the SAME permutation, or the
    /// gate silently changes summation order and every downstream result drifts.
    #[test]
    fn stable_argsort_branches_agree_with_a_stable_reference() {
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        // Straddles the gate: n_levels below, at, and above n_obs. `key_span`
        // is decoupled so the dense branch also runs on a key that occupies a
        // fraction of its declared level range, leaving empty buckets.
        for (n_obs, n_levels, key_span) in [
            (0usize, 0usize, 1usize),
            (0, 4, 4),
            (1, 1, 1),
            (997, 1, 1),
            (997, 16, 16),
            (997, 996, 996),
            (997, 997, 997),
            (997, 998, 998),
            (997, 50_000, 50_000),
            (4096, 4096, 8),
            (4096, 4096, 4096),
        ] {
            assert!(key_span <= n_levels.max(1), "keys must stay below n_levels");
            let key: Vec<u32> = (0..n_obs)
                .map(|_| (next() % key_span as u64) as u32)
                .collect();
            let mut expected: Vec<u32> = (0..n_obs as u32).collect();
            expected.sort_by_key(|&i| key[i as usize]);
            assert_eq!(
                stable_argsort(&key, n_levels),
                expected,
                "n_obs={n_obs} n_levels={n_levels}"
            );
        }
    }

    #[test]
    fn build_rejects_dof_space_exceeding_u32() {
        // A single code of u32::MAX implies one level past the CSR column-index width.
        let err = Design::from_frame(frame(vec![vec![u32::MAX]], vec![])).unwrap_err();
        assert!(matches!(
            err,
            BuildError::DofSpaceExceedsU32 { n_dofs } if n_dofs == u32::MAX as usize + 1
        ));
    }

    #[test]
    fn validate_weights_checks_count_and_finiteness() {
        let design = Design::from_frame(frame(vec![vec![0, 0, 0, 0, 0]], vec![])).unwrap();
        assert!(design.validate_weights(None).is_ok());
        assert!(design
            .validate_weights(Some(&[1.0, 2.0, 3.0, 4.0, 5.0]))
            .is_ok());
        // Zero weights are valid (an excluded observation).
        assert!(design
            .validate_weights(Some(&[0.0, 1.0, 2.0, 3.0, 4.0]))
            .is_ok());
        // Length mismatch.
        assert!(design.validate_weights(Some(&[1.0, 2.0])).is_err());
        // Negative / non-finite weights are rejected with the offending index.
        assert!(matches!(
            design.validate_weights(Some(&[1.0, -2.0, 3.0, 4.0, 5.0])),
            Err(BuildError::InvalidWeight { index: 1, .. })
        ));
        assert!(matches!(
            design.validate_weights(Some(&[1.0, 2.0, f64::NAN, 4.0, 5.0])),
            Err(BuildError::InvalidWeight { index: 2, .. })
        ));
        assert!(matches!(
            design.validate_weights(Some(&[1.0, 2.0, 3.0, f64::INFINITY, 5.0])),
            Err(BuildError::InvalidWeight { index: 3, .. })
        ));
    }

    #[test]
    fn from_frame_sorts_owned_unsorted_dominant() {
        // Factor 0 (3 levels) dominates and is unsorted; factor 1 starts sorted.
        let design =
            Design::from_frame(frame(vec![vec![2, 0, 1, 0], vec![0, 0, 1, 1]], vec![])).unwrap();

        // Stable argsort of [2,0,1,0] → original indices [1,3,2,0].
        assert_eq!(design.rows.as_deref(), Some(&[1u32, 3, 2, 0][..]));
        assert!(design.terms[0].sorted);
        // Factor 1's permuted column [0,1,1,0] is no longer non-decreasing.
        assert!(!design.terms[1].sorted);

        assert_eq!(design.frame.level_column(0), [0, 0, 1, 2]);
        assert_eq!(design.frame.level_column(1), [0, 1, 1, 0]);
    }

    #[test]
    fn rescan_marks_nested_factor_sorted_after_permutation() {
        // Factor 1 is nested in dominant factor 0, so the rescan must detect it stays sorted.
        let col0 = vec![3u32, 0, 2, 1];
        let col1: Vec<u32> = col0.iter().map(|&v| v / 2).collect();
        let design = Design::from_frame(frame(vec![col0, col1], vec![])).unwrap();
        assert!(design.rows.is_some());
        assert!(design.terms[0].sorted);
        assert!(design.terms[1].sorted);
    }

    #[test]
    fn from_frame_keeps_sorted_input() {
        let design =
            Design::from_frame(frame(vec![vec![0, 0, 1, 2], vec![1, 0, 1, 0]], vec![])).unwrap();
        assert!(design.rows.is_none());
        assert!(design.terms[0].sorted);
        assert!(!design.terms[1].sorted);
    }

    #[test]
    fn design_options_default_preserves_singletons() {
        let design = DesignOptions::default()
            .from_frame(frame(vec![vec![0, 0, 1], vec![0, 1, 1]], vec![]))
            .unwrap();

        assert_eq!(design.input_n_obs(), 3);
        assert_eq!(design.n_obs(), 3);
    }

    #[test]
    fn singleton_detection_matches_pyfixest_fixtures() {
        let cases = [
            (
                vec![
                    vec![0, 0, 0, 0, 0],
                    vec![2, 2, 1, 1, 1],
                    vec![1, 1, 3, 2, 2],
                ],
                Some(vec![0, 1, 3, 4]),
            ),
            (
                vec![
                    vec![0, 0, 3, 0, 0],
                    vec![2, 2, 1, 1, 1],
                    vec![1, 1, 2, 1, 2],
                ],
                Some(vec![0, 1]),
            ),
            (
                vec![
                    vec![0, 0, 0, 0, 0],
                    vec![2, 2, 1, 1, 1],
                    vec![1, 1, 1, 2, 2],
                ],
                None,
            ),
        ];

        for (categorical, expected) in cases {
            let input = frame(categorical, vec![]);
            assert_eq!(retained_non_singletons(&input).unwrap(), expected);
        }
    }

    #[test]
    fn design_options_iteratively_drop_singletons_before_one_final_gather() {
        // Rows 0..4 form a 2-core. Rows 4..7 form a path:
        //
        //     B2 -- A2 -- B3 -- A3
        //
        // Its endpoint singleton levels trigger a cascading removal of all
        // three path observations, while the four-row core survives.
        let design = DesignOptions::default()
            .drop_singletons(true)
            .from_frame(frame(
                vec![vec![1, 0, 1, 0, 2, 2, 3], vec![1, 0, 0, 1, 2, 3, 3]],
                vec![vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]],
            ))
            .unwrap();

        // The surviving caller rows [0,1,2,3] are locality-sorted by the
        // dominant second factor in the same composed mapping.
        assert_eq!(design.rows.as_deref(), Some(&[1, 2, 0, 3][..]));
        assert_eq!(design.input_n_obs(), 7);
        assert_eq!(design.n_obs(), 4);
        assert_eq!(design.frame.level_column(0), &[0, 1, 1, 0]);
        assert_eq!(design.frame.level_column(1), &[0, 0, 1, 1]);
        assert_eq!(design.frame.loading_column(0), &[11.0, 12.0, 10.0, 13.0]);

        let restored = design.from_internal_order(vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(&restored[..4], &[3.0, 1.0, 2.0, 4.0]);
        assert!(restored[4..].iter().all(|value| value.is_nan()));
    }

    #[test]
    fn design_options_reject_when_singletons_remove_every_observation() {
        let result = DesignOptions::default()
            .drop_singletons(true)
            .from_frame(frame(vec![vec![0, 1], vec![0, 1]], vec![]));

        assert!(matches!(result, Err(BuildError::EmptyObservations)));
    }

    #[test]
    fn continuous_column_stays_row_aligned_after_locality_sort() {
        let design = Design::from_frame(frame(
            vec![vec![2, 0, 1, 0]],
            vec![vec![10.0, 20.0, 30.0, 40.0]],
        ))
        .unwrap();

        let perm = design.rows.as_ref().expect("permutation applied");
        assert_eq!(perm, &[1, 3, 2, 0]);
        assert_eq!(design.frame.level_column(0), [0, 0, 1, 2]);
        assert_eq!(design.frame.loading_column(0), [20.0, 40.0, 30.0, 10.0]);
    }

    #[test]
    fn validation_ignores_weights_on_removed_rows() {
        let weights = [1.0, f64::NAN, 1.0, 1.0, f64::NAN];
        let design = DesignOptions::default()
            .drop_singletons(true)
            .weights(&weights[..])
            .from_frame(frame(vec![vec![0, 1, 0, 0, 2]], vec![]))
            .unwrap();

        let stored = design.weights().expect("weights retained");
        assert_eq!(stored.len(), weights.len());
        assert_eq!(stored[0], 1.0);
        assert!(stored[1].is_nan());
        assert_eq!(&stored[2..4], &[1.0, 1.0]);
        assert!(stored[4].is_nan());
        assert!(matches!(
            DesignOptions::default()
                .drop_singletons(true)
                .weights(&[1.0, f64::NAN, -1.0, 1.0, f64::NAN][..])
                .from_frame(frame(vec![vec![0, 1, 0, 0, 2]], vec![])),
            Err(BuildError::InvalidWeight {
                index: 2,
                value: -1.0
            })
        ));
    }

    #[test]
    fn ordering_is_generic_and_borrows_only_for_identity() {
        let identity = Design::from_frame(frame(vec![vec![0, 0, 1]], vec![])).unwrap();
        let values = ["a", "b", "c"];
        assert!(matches!(
            identity.to_internal_order(&values),
            Cow::Borrowed(_)
        ));

        let sorted = Design::from_frame(frame(vec![vec![2, 0, 1, 0]], vec![])).unwrap();
        let values = ["a", "b", "c", "d"];
        assert_eq!(
            sorted.to_internal_order(&values).as_ref(),
            &["b", "d", "c", "a"]
        );
    }

    #[test]
    fn into_owned_detaches_option_weights() {
        let weights = vec![1.0, 2.0, 3.0];
        let design = DesignOptions::default()
            .weights(weights.as_slice())
            .from_frame(frame(vec![vec![0, 0, 1]], vec![]))
            .unwrap()
            .into_owned();
        drop(weights);

        assert_eq!(design.weights(), Some(&[1.0, 2.0, 3.0][..]));
        assert!(matches!(
            design.options().weights.as_ref(),
            Some(Cow::Owned(_))
        ));
    }

    #[test]
    fn new_lays_out_slope_terms_term_major() {
        // Sorted levels keep the locality sort a no-op, so frame columns stay in caller order.
        let f0 = [0u32, 0, 1, 1];
        let f1 = [0u32, 2, 1, 0];
        let z0 = [1.0, 2.0, 3.0, 4.0];
        let z1 = [5.0, 6.0, 7.0, 8.0];
        let effects = vec![
            Effect::new(&f0, true, [&z0[..], &z1[..]]).unwrap(),
            Effect::new(&f1, true, []).unwrap(),
            Effect::new(&f0, false, [&z1[..]]).unwrap(),
        ];
        let design = Design::new(effects).unwrap();

        // term 0: [intercept, z0, z1] over 2 levels; term 1: intercept over 3; term 2: slope.
        assert_eq!(design.terms[0].offset, 0);
        assert_eq!(design.terms[0].n_dofs(), 6);
        assert_eq!(design.terms[1].offset, 6);
        assert_eq!(design.terms[1].n_dofs(), 3);
        assert_eq!(design.terms[2].offset, 9);
        assert!(!matches!(design.terms[2].columns[0], Loading::Constant));
        assert_eq!(design.terms[2].n_dofs(), 2);
        assert_eq!(design.n_dofs, 11);

        // slope indices resolve to the effects' loading columns in the frame.
        assert_eq!(
            &*design.terms[0].columns,
            &[
                Loading::Constant,
                Loading::Covariate(0),
                Loading::Covariate(1)
            ]
        );
        assert_eq!(&*design.terms[2].columns, &[Loading::Covariate(2)]);
        assert_eq!(design.frame.loading_column(0), &z0[..]);
        assert_eq!(design.frame.loading_column(2), &z1[..]);
    }
}
