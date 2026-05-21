//! Factor-pair subdomain construction.
//!
//! Each factor pair `(q, r)` becomes a Schwarz subdomain (one per connected
//! component of its bipartite cross-tab). Overlap is handled by partition-of-unity
//! weights — see [`schwarz_precond::domain`] for the math.
//!
//! Entry point: [`build_local_domains`].

use schwarz_precond::PartitionWeights;

use super::cross_tab::BipartiteComponent;
use super::{find_all_active_levels, CrossTab, Design, Subdomain};
use crate::observation::Store;

/// Build local subdomains (with pre-built CrossTabs) for pairs of factors.
///
/// For each factor pair, builds a fused CrossTab via a single observation scan,
/// detects connected components on the bipartite structure, and creates one
/// subdomain per component. The CrossTab travels with each subdomain to avoid
/// a rebuild.
///
/// Factor pairs are processed in parallel via Rayon. The
/// `compute_partition_weights` step remains sequential after the parallel
/// collect.
pub(crate) fn build_local_domains<S: Store>(
    design: &Design<S>,
    weights: Option<&[f64]>,
) -> Vec<(Subdomain, CrossTab)> {
    use rayon::prelude::*;

    let n_factors = design.n_factors();
    let pairs = build_pairs(n_factors);
    let all_active = find_all_active_levels(design);

    let mut domain_pairs: Vec<(Subdomain, CrossTab)> = pairs
        .par_iter()
        .flat_map(|&(q, r)| domains_for_pair(design, weights, q, r, &all_active))
        .collect();

    compute_partition_weights(&mut domain_pairs, design.n_dofs);

    domain_pairs
}

fn domains_for_pair<S: Store>(
    design: &Design<S>,
    weights: Option<&[f64]>,
    q: usize,
    r: usize,
    all_active: &[Vec<bool>],
) -> Vec<(Subdomain, CrossTab)> {
    let (full_ct, l2g) =
        match CrossTab::build_for_pair_with_active(design, weights, q, r, all_active) {
            Some(pair) => pair,
            None => return Vec::new(),
        };

    let n_q_full = full_ct.n_q();
    split_into_subdomains(full_ct, &l2g, n_q_full, (q, r))
}

/// Split a full CrossTab into per-component subdomains.
///
/// Finds bipartite connected components, extracts a sub-CrossTab for each,
/// and builds a `Subdomain` with uniform partition-of-unity weights.
fn split_into_subdomains(
    full_ct: CrossTab,
    l2g: &[u32],
    n_q_full: usize,
    factor_pair: (usize, usize),
) -> Vec<(Subdomain, CrossTab)> {
    let components = full_ct.bipartite_connected_components();

    let cross_tabs: Vec<CrossTab> = if components.len() == 1 {
        vec![full_ct]
    } else {
        components
            .iter()
            .map(|comp| full_ct.extract_component(comp))
            .collect()
    };

    components
        .iter()
        .zip(cross_tabs)
        .map(|(comp, comp_ct)| {
            let comp_l2g = component_global_indices(comp, l2g, n_q_full);
            let core = schwarz_precond::SubdomainCore::uniform(
                comp_l2g.into_iter().map(|g| g as u32).collect(),
            );
            (Subdomain { factor_pair, core }, comp_ct)
        })
        .collect()
}

/// Compute global DOF indices for a bipartite component.
///
/// Maps the component's compact q/r indices through the local-to-global vector,
/// returning global indices with q-levels first, then r-levels.
fn component_global_indices(comp: &BipartiteComponent, l2g: &[u32], n_q_full: usize) -> Vec<usize> {
    comp.q_indices
        .iter()
        .map(|&i| l2g[i] as usize)
        .chain(comp.r_indices.iter().map(|&i| l2g[n_q_full + i] as usize))
        .collect()
}

fn build_pairs(n_factors: usize) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    for q in 0..n_factors {
        for r in (q + 1)..n_factors {
            pairs.push((q, r));
        }
    }
    pairs
}

/// Compute partition-of-unity weights for overlapping Schwarz subdomains.
///
/// The two-sided additive Schwarz formula `M⁻¹ = Σ Rᵢᵀ D̃ᵢ Aᵢ⁻¹ D̃ᵢ Rᵢ`
/// requires that the squared weights sum to identity at every DOF:
/// `Σ Rᵢᵀ D̃ᵢ² Rᵢ = I`. For a DOF appearing in `c` subdomains, each weight
/// is set to `1/√c`, so that `c × (1/√c)² = 1`.
///
/// In the common (non-overlapping) case where every DOF belongs to exactly one
/// subdomain, all weights are 1.0 and the compact `PartitionWeights::Uniform`
/// representation is used to avoid per-DOF storage.
fn compute_partition_weights(domain_pairs: &mut [(Subdomain, CrossTab)], n_dofs: usize) {
    let mut counts = vec![0u32; n_dofs];
    for (d, _) in domain_pairs.iter() {
        for &idx in d.core.global_indices() {
            debug_assert!((idx as usize) < n_dofs);
            counts[idx as usize] += 1;
        }
    }
    for (d, _) in domain_pairs.iter_mut() {
        let all_unique = d
            .core
            .global_indices()
            .iter()
            .all(|&idx| counts[idx as usize] <= 1);
        if all_unique {
            d.core.set_uniform_partition_weights();
        } else {
            let weights: Vec<f64> = d
                .core
                .global_indices()
                .iter()
                .map(|&idx| {
                    let c = counts[idx as usize];
                    debug_assert!(c > 0);
                    1.0 / (c as f64).sqrt()
                })
                .collect();
            d.core
                .set_partition_weights(PartitionWeights::NonUniform(weights))
                .expect("partition weight count must match index count");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::Design;
    use crate::observation::FactorMajorStore;

    fn make_test_design() -> Design<FactorMajorStore> {
        let store = FactorMajorStore::new(
            vec![
                vec![0, 1, 2, 0, 1, 2],
                vec![0, 1, 0, 1, 0, 1],
                vec![0, 0, 1, 1, 0, 1],
            ],
            6,
        )
        .expect("valid factor-major store");
        Design::from_store(store).expect("valid test design")
    }

    #[test]
    fn test_full_cover_domain_count() {
        let dm = make_test_design();
        let domain_pairs = build_local_domains(&dm, None);
        // 3 factor pairs; each pair may produce multiple components
        assert!(domain_pairs.len() >= 3);
    }

    #[test]
    fn test_partition_of_unity() {
        let dm = make_test_design();
        let domain_pairs = build_local_domains(&dm, None);
        let n_dofs = dm.n_dofs;
        // Two-sided PoU: squared weights must sum to 1 at every DOF.
        let mut weight_sq_sum = vec![0.0; n_dofs];
        for (d, _) in &domain_pairs {
            for (i, &idx) in d.core.global_indices().iter().enumerate() {
                let w = d.core.partition_weights().get(i);
                weight_sq_sum[idx as usize] += w * w;
            }
        }
        for &ws in &weight_sq_sum {
            if ws > 0.0 {
                assert!((ws - 1.0).abs() < 1e-12, "Weight² sum {ws} != 1.0");
            }
        }
    }

    #[test]
    fn test_domains_cover_all_dofs() {
        let dm = make_test_design();
        let domain_pairs = build_local_domains(&dm, None);
        let mut covered = vec![false; dm.n_dofs];
        for (d, _) in &domain_pairs {
            for &idx in d.core.global_indices() {
                covered[idx as usize] = true;
            }
        }
        assert!(covered.iter().all(|&c| c), "Not all DOFs covered");
    }
}
