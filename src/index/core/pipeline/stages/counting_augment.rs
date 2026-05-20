use super::super::{ActivationStage, QueryContext, ScoredCandidate};
use super::{sort_candidates, upsert_candidate};
use crate::neuron::NeuronKind;

/// Expands counting queries with additional non-aggregate candidates.
pub struct CountingAugmentStage;

impl ActivationStage for CountingAugmentStage {
    fn name(&self) -> &'static str {
        "counting_augment"
    }

    fn apply(&self, ctx: &QueryContext<'_>, candidates: &mut Vec<ScoredCandidate>) {
        if ctx.counting_augment.is_empty() {
            return;
        }

        for &idx in &ctx.counting_augment {
            if candidates
                .iter()
                .any(|candidate| candidate.entry_idx == idx)
            {
                continue;
            }
            let entry = ctx.entry(idx);
            if matches!(entry.kind, NeuronKind::Aggregate) {
                continue;
            }
            let score = ctx.score_index(idx);
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

    #[test]
    fn name_returns_expected_string() {
        assert_eq!(CountingAugmentStage.name(), "counting_augment");
    }

    #[test]
    fn empty_candidates_is_passthrough() {
        let fixture = QueryContextFixture::new(vec![]);
        let ctx = fixture.ctx("anything");
        let mut candidates = Vec::new();

        CountingAugmentStage.apply(&ctx, &mut candidates);

        assert!(candidates.is_empty());
    }

    #[test]
    fn non_counting_query_doesnt_inject_counting_neurons() {
        let entry = test_entry("session.md", NeuronKind::Verbatim, &[("music", 1.0)]);
        let fixture = QueryContextFixture::new(vec![entry]);
        let ctx = fixture.ctx("play music");
        let mut candidates = Vec::new();

        assert!(!ctx.is_counting);
        CountingAugmentStage.apply(&ctx, &mut candidates);

        assert!(candidates.is_empty());
    }

    #[test]
    fn skips_already_scored_candidates() {
        let entry = test_entry("session.md", NeuronKind::Verbatim, &[("music", 1.0)]);
        let fixture = QueryContextFixture::new(vec![entry]);
        let mut ctx = fixture.ctx("how many music");
        ctx.counting_augment = vec![0];
        ctx.ranking_terms = vec!["music".into()];

        let mut candidates = vec![ScoredCandidate::new(0, 0.8, 32)];
        CountingAugmentStage.apply(&ctx, &mut candidates);
        assert_eq!(candidates.len(), 1);
    }

    #[test]
    fn ignores_aggregate_neurons() {
        let aggregate = test_entry(
            "count.aggregate.md",
            NeuronKind::Aggregate,
            &[("music", 2.0)],
        );
        let verbatim = test_entry("session.md", NeuronKind::Verbatim, &[("music", 1.0)]);
        let fixture = QueryContextFixture::new(vec![aggregate, verbatim]);
        let mut ctx = fixture.ctx("how many music");
        ctx.counting_augment = vec![0, 1];
        ctx.ranking_terms = vec!["music".into()];

        let mut candidates = Vec::new();
        CountingAugmentStage.apply(&ctx, &mut candidates);

        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].entry_idx, 1);
    }

    #[test]
    fn sorts_augmented_candidates_by_score_descending() {
        let weaker = test_entry("weak.md", NeuronKind::Verbatim, &[("music", 1.0)]);
        let stronger = test_entry("strong.md", NeuronKind::Verbatim, &[("music", 3.0)]);
        let fixture = QueryContextFixture::new(vec![weaker, stronger]);
        let mut ctx = fixture.ctx("how many music");
        ctx.counting_augment = vec![0, 1];
        ctx.ranking_terms = vec!["music".into()];

        let mut candidates = Vec::new();
        CountingAugmentStage.apply(&ctx, &mut candidates);

        assert_eq!(
            candidates
                .iter()
                .map(|candidate| candidate.entry_idx)
                .collect::<Vec<_>>(),
            vec![1, 0]
        );
    }
}
