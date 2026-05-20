use super::super::{ActivationStage, QueryContext, ScoredCandidate};
use super::sort_candidates;

pub(super) const COACTIVATION_BOOST_PER_HIT: f32 = 0.02;
pub(super) const MAX_COACTIVATION_HITS: u32 = 10;

/// Rewards candidates that historically co-activated with the current query vocabulary.
pub struct CoactivationStage;

impl ActivationStage for CoactivationStage {
    fn name(&self) -> &'static str {
        "coactivation"
    }

    fn apply(&self, ctx: &QueryContext<'_>, candidates: &mut Vec<ScoredCandidate>) {
        let query_terms = if ctx.ranking_terms.is_empty() {
            &ctx.terms
        } else {
            &ctx.ranking_terms
        };
        if query_terms.is_empty() {
            return;
        }

        for candidate in candidates.iter_mut() {
            let counts = match ctx
                .feedback
                .coactivation_counts
                .get(&ctx.entry(candidate.entry_idx).neuron_path)
            {
                Some(counts) => counts,
                None => continue,
            };
            let total_hits: u32 = query_terms
                .iter()
                .map(|term| counts.get(term).copied().unwrap_or(0))
                .sum();
            if total_hits == 0 {
                continue;
            }
            let capped_hits = total_hits.min(MAX_COACTIVATION_HITS);
            candidate.score *= 1.0 + (capped_hits as f32 * COACTIVATION_BOOST_PER_HIT);
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
    use std::path::PathBuf;

    #[test]
    fn name_returns_expected_string() {
        assert_eq!(CoactivationStage.name(), "coactivation");
    }

    #[test]
    fn empty_candidates_is_passthrough() {
        let fixture = QueryContextFixture::new(vec![]);
        let ctx = fixture.ctx("anything");
        let mut candidates = Vec::new();

        CoactivationStage.apply(&ctx, &mut candidates);

        assert!(candidates.is_empty());
    }

    #[test]
    fn boosts_candidate_by_matching_term_history() {
        let entry = test_entry("a.md", NeuronKind::Core, &[("foo", 1.0)]);
        let mut fixture = QueryContextFixture::new(vec![entry]);
        fixture
            .coactivation_counts
            .insert(PathBuf::from("a.md"), HashMap::from([("foo".into(), 3u32)]));
        let mut ctx = fixture.ctx("foo");
        ctx.ranking_terms = vec!["foo".into()];
        let mut candidates = vec![ScoredCandidate::new(0, 1.0, 10)];

        CoactivationStage.apply(&ctx, &mut candidates);

        assert!(candidates[0].score > 1.0);
    }

    #[test]
    fn leaves_score_unchanged_without_history() {
        let entry = test_entry("a.md", NeuronKind::Core, &[("foo", 1.0)]);
        let fixture = QueryContextFixture::new(vec![entry]);
        let mut ctx = fixture.ctx("foo");
        ctx.ranking_terms = vec!["foo".into()];
        let mut candidates = vec![ScoredCandidate::new(0, 1.0, 10)];

        CoactivationStage.apply(&ctx, &mut candidates);

        assert_eq!(candidates[0].score, 1.0);
    }

    #[test]
    fn multiple_candidates_reorder_after_boost() {
        let first = test_entry("a.md", NeuronKind::Core, &[("foo", 1.0)]);
        let second = test_entry("b.md", NeuronKind::Core, &[("foo", 1.0)]);
        let mut fixture = QueryContextFixture::new(vec![first, second]);
        fixture
            .coactivation_counts
            .insert(PathBuf::from("b.md"), HashMap::from([("foo".into(), 5u32)]));
        let mut ctx = fixture.ctx("foo");
        ctx.ranking_terms = vec!["foo".into()];
        let mut candidates = vec![
            ScoredCandidate::new(0, 1.0, 10),
            ScoredCandidate::new(1, 0.95, 10),
        ];

        CoactivationStage.apply(&ctx, &mut candidates);

        assert_eq!(candidates[0].entry_idx, 1);
    }

    #[test]
    fn boost_is_capped_for_large_history_counts() {
        let entry = test_entry("a.md", NeuronKind::Core, &[("foo", 1.0)]);
        let mut fixture = QueryContextFixture::new(vec![entry]);
        fixture.coactivation_counts.insert(
            PathBuf::from("a.md"),
            HashMap::from([("foo".into(), 100u32)]),
        );
        let mut ctx = fixture.ctx("foo");
        ctx.ranking_terms = vec!["foo".into()];
        let mut candidates = vec![ScoredCandidate::new(0, 1.0, 10)];

        CoactivationStage.apply(&ctx, &mut candidates);

        assert!(
            (candidates[0].score
                - (1.0 + MAX_COACTIVATION_HITS as f32 * COACTIVATION_BOOST_PER_HIT))
                .abs()
                < 0.0001
        );
    }

    #[test]
    fn falls_back_to_ctx_terms_when_ranking_terms_are_empty() {
        let entry = test_entry("a.md", NeuronKind::Core, &[("foo", 1.0)]);
        let mut fixture = QueryContextFixture::new(vec![entry]);
        fixture
            .coactivation_counts
            .insert(PathBuf::from("a.md"), HashMap::from([("foo".into(), 2u32)]));
        let mut ctx = fixture.ctx("foo");
        ctx.terms = vec!["foo".into()];
        let mut candidates = vec![ScoredCandidate::new(0, 1.0, 10)];

        CoactivationStage.apply(&ctx, &mut candidates);

        assert!(candidates[0].score > 1.0);
    }
}
