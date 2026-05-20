use super::super::{ActivationStage, QueryContext, ScoredCandidate};
use super::sort_candidates;

/// Applies per-entry staleness multipliers after lexical scoring completes.
pub struct StalenessDecayStage;

impl ActivationStage for StalenessDecayStage {
    fn name(&self) -> &'static str {
        "staleness_decay"
    }

    fn apply(&self, ctx: &QueryContext<'_>, candidates: &mut Vec<ScoredCandidate>) {
        for candidate in candidates.iter_mut() {
            candidate.score *= ctx.entry(candidate.entry_idx).staleness_multiplier;
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
        assert_eq!(StalenessDecayStage.name(), "staleness_decay");
    }

    #[test]
    fn empty_candidates_is_passthrough() {
        let fixture = QueryContextFixture::new(vec![]);
        let ctx = fixture.ctx("anything");
        let mut candidates = Vec::new();

        StalenessDecayStage.apply(&ctx, &mut candidates);

        assert!(candidates.is_empty());
    }

    #[test]
    fn multiplies_score_by_entry_staleness() {
        let mut entry = test_entry("a.md", NeuronKind::Core, &[("foo", 1.0)]);
        entry.staleness_multiplier = 0.5;
        let fixture = QueryContextFixture::new(vec![entry]);
        let ctx = fixture.ctx("foo");
        let mut candidates = vec![ScoredCandidate::new(0, 2.0, 10)];

        StalenessDecayStage.apply(&ctx, &mut candidates);

        assert_eq!(candidates[0].score, 1.0);
    }

    #[test]
    fn multiple_candidates_are_resorted_after_decay() {
        let mut stale = test_entry("stale.md", NeuronKind::Core, &[("foo", 1.0)]);
        stale.staleness_multiplier = 0.4;
        let mut fresh = test_entry("fresh.md", NeuronKind::Core, &[("foo", 1.0)]);
        fresh.staleness_multiplier = 1.0;
        let fixture = QueryContextFixture::new(vec![stale, fresh]);
        let ctx = fixture.ctx("foo");
        let mut candidates = vec![
            ScoredCandidate::new(0, 2.0, 10),
            ScoredCandidate::new(1, 1.0, 10),
        ];

        StalenessDecayStage.apply(&ctx, &mut candidates);

        assert_eq!(
            candidates
                .iter()
                .map(|candidate| candidate.entry_idx)
                .collect::<Vec<_>>(),
            vec![1, 0]
        );
    }

    #[test]
    fn zero_multiplier_zeroes_candidate_score() {
        let mut entry = test_entry("a.md", NeuronKind::Core, &[("foo", 1.0)]);
        entry.staleness_multiplier = 0.0;
        let fixture = QueryContextFixture::new(vec![entry]);
        let ctx = fixture.ctx("foo");
        let mut candidates = vec![ScoredCandidate::new(0, 2.0, 10)];

        StalenessDecayStage.apply(&ctx, &mut candidates);

        assert_eq!(candidates[0].score, 0.0);
    }

    #[test]
    fn neutral_multiplier_preserves_existing_order() {
        let first = test_entry("a.md", NeuronKind::Core, &[("foo", 1.0)]);
        let second = test_entry("b.md", NeuronKind::Core, &[("foo", 1.0)]);
        let fixture = QueryContextFixture::new(vec![first, second]);
        let ctx = fixture.ctx("foo");
        let mut candidates = vec![
            ScoredCandidate::new(0, 2.0, 10),
            ScoredCandidate::new(1, 1.0, 10),
        ];

        StalenessDecayStage.apply(&ctx, &mut candidates);

        assert_eq!(
            candidates
                .iter()
                .map(|candidate| candidate.entry_idx)
                .collect::<Vec<_>>(),
            vec![0, 1]
        );
    }

    #[test]
    fn stage_composes_with_prior_injection() {
        let mut parent = test_entry("parent.md", NeuronKind::Core, &[("foo", 1.0)]);
        parent.staleness_multiplier = 0.8;
        let fixture = QueryContextFixture::new(vec![parent]);
        let ctx = fixture.ctx("foo");
        let mut candidates = vec![ScoredCandidate::new(0, 3.0, 10)];

        StalenessDecayStage.apply(&ctx, &mut candidates);

        assert!((candidates[0].score - 2.4).abs() < f32::EPSILON);
    }
}
