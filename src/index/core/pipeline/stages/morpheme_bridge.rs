use super::super::{ActivationStage, QueryContext, ScoredCandidate};
use super::{sort_candidates, upsert_candidate};
use std::collections::HashSet;

/// Augments candidates with morpheme-root expansions for uncovered query terms.
pub struct MorphemeBridgeStage;

impl ActivationStage for MorphemeBridgeStage {
    fn name(&self) -> &'static str {
        "morpheme_bridge"
    }

    fn apply(&self, ctx: &QueryContext<'_>, candidates: &mut Vec<ScoredCandidate>) {
        if ctx.morpheme_map.is_empty() || ctx.terms.is_empty() {
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

        let mut extra_terms = Vec::new();
        let mut candidate_ids = HashSet::new();
        for term in &ctx.terms {
            if matched_terms.contains(term.as_str()) {
                continue;
            }
            let Some(expansions) = ctx.morpheme_map.get(term.as_str()) else {
                continue;
            };
            extra_terms.extend(expansions.iter().cloned());
            for expansion in expansions {
                if let Some(indices) = ctx.posting_list.get(expansion.as_str()) {
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
        assert_eq!(MorphemeBridgeStage.name(), "morpheme_bridge");
    }

    #[test]
    fn empty_candidates_is_passthrough() {
        let fixture = QueryContextFixture::new(vec![]);
        let ctx = fixture.ctx("anything");
        let mut candidates = Vec::new();

        MorphemeBridgeStage.apply(&ctx, &mut candidates);

        assert!(candidates.is_empty());
    }

    #[test]
    fn injects_candidate_for_unmatched_morpheme_term() {
        let entry = test_entry("auth_guard.md", NeuronKind::Core, &[("auth_guard", 2.0)]);
        let mut fixture = QueryContextFixture::new(vec![entry]);
        fixture
            .morpheme_map
            .insert("auth".into(), vec!["auth_guard".into()]);
        fixture.posting_list.insert("auth_guard".into(), vec![0]);
        let mut ctx = fixture.ctx("auth");
        ctx.terms = vec!["auth".into()];
        ctx.ranking_terms = vec!["auth".into()];

        let mut candidates = Vec::new();
        MorphemeBridgeStage.apply(&ctx, &mut candidates);

        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].entry_idx, 0);
    }

    #[test]
    fn skips_terms_already_matched_by_existing_candidates() {
        let entry = test_entry("auth.md", NeuronKind::Core, &[("auth", 2.0)]);
        let mut fixture = QueryContextFixture::new(vec![entry]);
        fixture
            .morpheme_map
            .insert("auth".into(), vec!["auth_guard".into()]);
        fixture.posting_list.insert("auth_guard".into(), vec![0]);
        let mut ctx = fixture.ctx("auth");
        ctx.terms = vec!["auth".into()];
        ctx.ranking_terms = vec!["auth".into()];

        let mut candidates = vec![ScoredCandidate::new(0, 1.0, 32)];
        MorphemeBridgeStage.apply(&ctx, &mut candidates);

        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].entry_idx, 0);
        assert_eq!(candidates[0].score, 1.0);
    }

    #[test]
    fn multiple_injected_candidates_are_sorted_by_score() {
        let weaker = test_entry("auth_guard.md", NeuronKind::Core, &[("auth_guard", 1.0)]);
        let stronger = test_entry("oauth_token.md", NeuronKind::Core, &[("oauth_token", 3.0)]);
        let mut fixture = QueryContextFixture::new(vec![weaker, stronger]);
        fixture.morpheme_map.insert(
            "auth".into(),
            vec!["auth_guard".into(), "oauth_token".into()],
        );
        fixture.posting_list = HashMap::from([
            ("auth_guard".into(), vec![0]),
            ("oauth_token".into(), vec![1]),
        ]);
        let mut ctx = fixture.ctx("auth");
        ctx.terms = vec!["auth".into()];
        ctx.ranking_terms = vec!["auth".into()];

        let mut candidates = Vec::new();
        MorphemeBridgeStage.apply(&ctx, &mut candidates);

        assert_eq!(
            candidates
                .iter()
                .map(|candidate| candidate.entry_idx)
                .collect::<Vec<_>>(),
            vec![1, 0]
        );
    }

    #[test]
    fn empty_morpheme_map_leaves_candidates_unchanged() {
        let entry = test_entry("auth.md", NeuronKind::Core, &[("auth", 1.0)]);
        let fixture = QueryContextFixture::new(vec![entry]);
        let mut ctx = fixture.ctx("auth");
        ctx.terms = vec!["auth".into()];
        ctx.ranking_terms = vec!["auth".into()];

        let mut candidates = vec![ScoredCandidate::new(0, 1.0, 32)];
        MorphemeBridgeStage.apply(&ctx, &mut candidates);

        assert_eq!(candidates, vec![ScoredCandidate::new(0, 1.0, 32)]);
    }

    #[test]
    fn respects_existing_candidates_when_augmenting_multiple_terms() {
        let base = test_entry("base.md", NeuronKind::Core, &[("token", 2.0)]);
        let bridged = test_entry("auth_guard.md", NeuronKind::Core, &[("auth_guard", 2.0)]);
        let mut fixture = QueryContextFixture::new(vec![base, bridged]);
        fixture
            .morpheme_map
            .insert("auth".into(), vec!["auth_guard".into()]);
        fixture.posting_list.insert("auth_guard".into(), vec![1]);
        let mut ctx = fixture.ctx("token auth");
        ctx.terms = vec!["token".into(), "auth".into()];
        ctx.ranking_terms = vec!["token".into(), "auth".into()];

        let mut candidates = vec![ScoredCandidate::new(0, 2.0, 32)];
        MorphemeBridgeStage.apply(&ctx, &mut candidates);

        assert_eq!(candidates.len(), 2);
        assert!(candidates.iter().any(|candidate| candidate.entry_idx == 1));
    }
}
