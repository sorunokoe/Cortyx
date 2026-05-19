use super::super::{ActivationStage, QueryContext, ScoredCandidate};
use crate::neuron::NeuronKind;

pub struct CountingAugmentStage;

impl ActivationStage for CountingAugmentStage {
    fn name(&self) -> &'static str {
        "counting_augment"
    }

    fn apply(&self, ctx: &QueryContext<'_>, candidates: &mut Vec<ScoredCandidate>) {
        if ctx.counting_augment.is_empty() {
            return;
        }

        let mut existing: std::collections::HashSet<usize> = candidates
            .iter()
            .map(|candidate| candidate.entry_idx)
            .collect();
        for &idx in &ctx.counting_augment {
            if existing.contains(&idx) {
                continue;
            }
            let entry = ctx.entry(idx);
            if matches!(entry.kind, NeuronKind::Aggregate) {
                continue;
            }
            let score = ctx.score_index(idx);
            if score > 0.0 {
                candidates.push(ScoredCandidate::new(idx, score, entry.tokens));
                existing.insert(idx);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::core::pipeline::types::{test_entry, QueryContextFixture};

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
}
