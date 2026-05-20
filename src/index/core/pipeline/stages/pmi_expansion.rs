use super::super::{ActivationStage, QueryContext, ScoredCandidate};
use super::{sort_candidates, upsert_candidate};
use std::collections::HashSet;

/// Expands matched query terms through PMI neighbor vocabulary.
pub struct PmiExpansionStage;

impl ActivationStage for PmiExpansionStage {
    fn name(&self) -> &'static str {
        "pmi_expansion"
    }

    fn apply(&self, ctx: &QueryContext<'_>, candidates: &mut Vec<ScoredCandidate>) {
        if candidates.is_empty() || ctx.pmi_neighbors.is_empty() {
            return;
        }

        let matched_terms: HashSet<&str> = ctx
            .terms
            .iter()
            .filter_map(|term| {
                candidates
                    .iter()
                    .any(|candidate| {
                        ctx.entry(candidate.entry_idx)
                            .term_freq
                            .contains_key(term.as_str())
                    })
                    .then_some(term.as_str())
            })
            .collect();
        if matched_terms.is_empty() {
            return;
        }

        let mut extra_terms = Vec::new();
        let mut candidate_ids = HashSet::new();
        for term in matched_terms {
            let Some(neighbors) = ctx.pmi_neighbors.get(term) else {
                continue;
            };
            extra_terms.extend(neighbors.iter().cloned());
            for neighbor in neighbors {
                if let Some(indices) = ctx.posting_list.get(neighbor.as_str()) {
                    candidate_ids.extend(indices.iter().copied());
                }
            }
        }

        if candidate_ids.is_empty() {
            return;
        }

        let mut scoring_terms = ctx.ranking_terms.clone();
        scoring_terms.extend(extra_terms);
        scoring_terms.sort();
        scoring_terms.dedup();

        for idx in candidate_ids {
            let entry = ctx.entry(idx);
            if !ctx.kind_matches(entry) || !ctx.module_matches(idx) {
                continue;
            }
            let score = ctx.score_entry_with_terms(&scoring_terms, entry);
            if score > 0.0 {
                upsert_candidate(candidates, idx, score, entry.tokens);
            }
        }

        sort_candidates(candidates);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::core::pipeline::types::{test_entry, QueryContextFixture};
    use crate::neuron::NeuronKind;
    use std::collections::HashMap;

    #[test]
    fn name_returns_expected_string() {
        assert_eq!(PmiExpansionStage.name(), "pmi_expansion");
    }

    #[test]
    fn empty_candidates_is_passthrough() {
        let fixture = QueryContextFixture::new(vec![]);
        let ctx = fixture.ctx("anything");
        let mut candidates = Vec::new();

        PmiExpansionStage.apply(&ctx, &mut candidates);

        assert!(candidates.is_empty());
    }

    #[test]
    fn injects_neighbor_for_matched_term() {
        let base = test_entry("degree.md", NeuronKind::Verbatim, &[("degree", 2.0)]);
        let neighbor = test_entry("education.md", NeuronKind::Verbatim, &[("education", 3.0)]);
        let mut fixture = QueryContextFixture::new(vec![base, neighbor]);
        fixture
            .pmi_neighbors
            .insert("degree".into(), vec!["education".into()]);
        fixture.posting_list.insert("education".into(), vec![1]);
        let mut ctx = fixture.ctx("degree");
        ctx.terms = vec!["degree".into()];
        ctx.ranking_terms = vec!["degree".into()];

        let mut candidates = vec![ScoredCandidate::new(0, 1.0, 32)];
        PmiExpansionStage.apply(&ctx, &mut candidates);

        assert_eq!(candidates.len(), 2);
        assert!(candidates.iter().any(|candidate| candidate.entry_idx == 1));
    }

    #[test]
    fn skips_when_no_matched_terms_have_neighbors() {
        let base = test_entry("degree.md", NeuronKind::Verbatim, &[("degree", 2.0)]);
        let fixture = QueryContextFixture::new(vec![base]);
        let mut ctx = fixture.ctx("degree");
        ctx.terms = vec!["degree".into()];
        ctx.ranking_terms = vec!["degree".into()];

        let mut candidates = vec![ScoredCandidate::new(0, 1.0, 32)];
        PmiExpansionStage.apply(&ctx, &mut candidates);

        assert_eq!(candidates, vec![ScoredCandidate::new(0, 1.0, 32)]);
    }

    #[test]
    fn injected_neighbors_are_sorted_by_score() {
        let base = test_entry("degree.md", NeuronKind::Verbatim, &[("degree", 2.0)]);
        let weaker = test_entry("master.md", NeuronKind::Verbatim, &[("master", 1.0)]);
        let stronger = test_entry("education.md", NeuronKind::Verbatim, &[("education", 3.0)]);
        let mut fixture = QueryContextFixture::new(vec![base, weaker, stronger]);
        fixture
            .pmi_neighbors
            .insert("degree".into(), vec!["master".into(), "education".into()]);
        fixture.posting_list =
            HashMap::from([("master".into(), vec![1]), ("education".into(), vec![2])]);
        let mut ctx = fixture.ctx("degree");
        ctx.terms = vec!["degree".into()];
        ctx.ranking_terms = vec!["degree".into()];

        let mut candidates = vec![ScoredCandidate::new(0, 1.0, 32)];
        PmiExpansionStage.apply(&ctx, &mut candidates);

        assert_eq!(
            candidates
                .iter()
                .map(|candidate| candidate.entry_idx)
                .collect::<Vec<_>>(),
            vec![2, 1, 0]
        );
    }

    #[test]
    fn does_not_reinject_existing_neighbor_candidate() {
        let base = test_entry("degree.md", NeuronKind::Verbatim, &[("degree", 2.0)]);
        let neighbor = test_entry("education.md", NeuronKind::Verbatim, &[("education", 3.0)]);
        let mut fixture = QueryContextFixture::new(vec![base, neighbor]);
        fixture
            .pmi_neighbors
            .insert("degree".into(), vec!["education".into()]);
        fixture.posting_list.insert("education".into(), vec![1]);
        let mut ctx = fixture.ctx("degree");
        ctx.terms = vec!["degree".into()];
        ctx.ranking_terms = vec!["degree".into()];

        let mut candidates = vec![
            ScoredCandidate::new(0, 1.0, 32),
            ScoredCandidate::new(1, 0.5, 32),
        ];
        PmiExpansionStage.apply(&ctx, &mut candidates);

        assert_eq!(candidates.len(), 2);
    }

    #[test]
    fn respects_multiple_seed_terms_when_expanding() {
        let base = test_entry(
            "degree.md",
            NeuronKind::Verbatim,
            &[("degree", 2.0), ("career", 1.0)],
        );
        let neighbor = test_entry("education.md", NeuronKind::Verbatim, &[("education", 3.0)]);
        let mut fixture = QueryContextFixture::new(vec![base, neighbor]);
        fixture
            .pmi_neighbors
            .insert("degree".into(), vec!["education".into()]);
        fixture.posting_list.insert("education".into(), vec![1]);
        let mut ctx = fixture.ctx("degree career");
        ctx.terms = vec!["degree".into(), "career".into()];
        ctx.ranking_terms = vec!["degree".into(), "career".into()];

        let mut candidates = vec![ScoredCandidate::new(0, 1.0, 32)];
        PmiExpansionStage.apply(&ctx, &mut candidates);

        assert!(candidates.iter().any(|candidate| candidate.entry_idx == 1));
    }
}
