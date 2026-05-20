use super::super::{ActivationStage, QueryContext, ScoredCandidate};
use super::sort_candidates;
use crate::index::core::config::{TEMPORAL_DECAY_WEIGHT, TEMPORAL_HALF_LIFE};

/// Applies an exponential recency boost to timestamped candidates.
pub struct TemporalProximityStage;

impl ActivationStage for TemporalProximityStage {
    fn name(&self) -> &'static str {
        "temporal_proximity"
    }

    fn apply(&self, ctx: &QueryContext<'_>, candidates: &mut Vec<ScoredCandidate>) {
        if ctx.is_knowledge_update {
            return;
        }

        for candidate in candidates.iter_mut() {
            let Some(timestamp_secs) = ctx.entry(candidate.entry_idx).timestamp_secs else {
                continue;
            };
            let age_secs = ctx.now_secs.saturating_sub(timestamp_secs).max(0);
            let age_secs_u32 = u32::try_from(age_secs).unwrap_or(u32::MAX);
            let age_days = age_secs_u32 as f32 / 86_400.0;
            candidate.score *= 1.0
                + ctx.temporal_bias_scale
                    * TEMPORAL_DECAY_WEIGHT
                    * (-(age_days / TEMPORAL_HALF_LIFE)).exp();
        }

        sort_candidates(candidates);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::core::config::{TEMPORAL_DECAY_WEIGHT, TEMPORAL_HALF_LIFE};
    use crate::index::core::pipeline::types::{test_entry, QueryContextFixture};
    use crate::neuron::NeuronKind;

    #[test]
    fn name_returns_expected_string() {
        assert_eq!(TemporalProximityStage.name(), "temporal_proximity");
    }

    #[test]
    fn empty_candidates_is_passthrough() {
        let fixture = QueryContextFixture::new(vec![]);
        let ctx = fixture.ctx("anything");
        let mut candidates = Vec::new();

        TemporalProximityStage.apply(&ctx, &mut candidates);

        assert!(candidates.is_empty());
    }

    #[test]
    fn applies_exponential_recency_boost() {
        let mut entry = test_entry("recent.md", NeuronKind::Verbatim, &[("foo", 1.0)]);
        entry.timestamp_secs = Some(1_999_913_600);
        let fixture = QueryContextFixture::new(vec![entry]);
        let mut ctx = fixture.ctx("foo");
        ctx.now_secs = 2_000_000_000;
        let mut candidates = vec![ScoredCandidate::new(0, 1.0, 10)];

        TemporalProximityStage.apply(&ctx, &mut candidates);

        let age_days =
            (u32::try_from(ctx.now_secs - 1_999_913_600).unwrap_or(u32::MAX) as f32) / 86_400.0;
        let expected = 1.0 + TEMPORAL_DECAY_WEIGHT * (-(age_days / TEMPORAL_HALF_LIFE)).exp();
        assert!((candidates[0].score - expected).abs() < 0.0001);
    }

    #[test]
    fn more_recent_candidate_can_overtake_older_candidate() {
        let mut older = test_entry("older.md", NeuronKind::Verbatim, &[("foo", 1.0)]);
        older.timestamp_secs = Some(1_990_000_000);
        let mut recent = test_entry("recent.md", NeuronKind::Verbatim, &[("foo", 1.0)]);
        recent.timestamp_secs = Some(1_999_913_600);
        let fixture = QueryContextFixture::new(vec![older, recent]);
        let mut ctx = fixture.ctx("foo");
        ctx.now_secs = 2_000_000_000;
        let mut candidates = vec![
            ScoredCandidate::new(0, 1.0, 10),
            ScoredCandidate::new(1, 0.95, 10),
        ];

        TemporalProximityStage.apply(&ctx, &mut candidates);

        assert_eq!(candidates[0].entry_idx, 1);
    }

    #[test]
    fn skips_candidates_without_timestamps() {
        let entry = test_entry("untimed.md", NeuronKind::Core, &[("foo", 1.0)]);
        let fixture = QueryContextFixture::new(vec![entry]);
        let mut ctx = fixture.ctx("foo");
        ctx.now_secs = 2_000_000_000;
        let mut candidates = vec![ScoredCandidate::new(0, 1.0, 10)];

        TemporalProximityStage.apply(&ctx, &mut candidates);

        assert_eq!(candidates[0].score, 1.0);
    }

    #[test]
    fn skips_knowledge_update_queries() {
        let mut entry = test_entry("recent.md", NeuronKind::Verbatim, &[("foo", 1.0)]);
        entry.timestamp_secs = Some(1_999_913_600);
        let fixture = QueryContextFixture::new(vec![entry]);
        let mut ctx = fixture.ctx("foo");
        ctx.now_secs = 2_000_000_000;
        ctx.is_knowledge_update = true;
        let mut candidates = vec![ScoredCandidate::new(0, 1.0, 10)];

        TemporalProximityStage.apply(&ctx, &mut candidates);

        assert_eq!(candidates[0].score, 1.0);
    }

    #[test]
    fn future_timestamps_are_treated_as_zero_age() {
        let mut entry = test_entry("future.md", NeuronKind::Verbatim, &[("foo", 1.0)]);
        entry.timestamp_secs = Some(2_000_100_000);
        let fixture = QueryContextFixture::new(vec![entry]);
        let mut ctx = fixture.ctx("foo");
        ctx.now_secs = 2_000_000_000;
        let mut candidates = vec![ScoredCandidate::new(0, 1.0, 10)];

        TemporalProximityStage.apply(&ctx, &mut candidates);

        assert!((candidates[0].score - (1.0 + TEMPORAL_DECAY_WEIGHT)).abs() < 0.0001);
    }
}
