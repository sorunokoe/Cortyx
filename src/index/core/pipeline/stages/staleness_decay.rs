use super::super::{ActivationStage, QueryContext, ScoredCandidate};

pub struct StalenessDecayStage;

impl ActivationStage for StalenessDecayStage {
    fn name(&self) -> &'static str {
        "staleness_decay"
    }

    fn apply(&self, _ctx: &QueryContext<'_>, _candidates: &mut Vec<ScoredCandidate>) {}
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
    fn nonempty_candidates_unchanged_by_stub() {
        let entry = test_entry("a.md", NeuronKind::Core, &[("foo", 1.0)]);
        let fixture = QueryContextFixture::new(vec![entry]);
        let ctx = fixture.ctx("foo");
        let mut candidates = vec![ScoredCandidate::new(0, 1.0, 10)];

        StalenessDecayStage.apply(&ctx, &mut candidates);

        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].entry_idx, 0);
        assert_eq!(candidates[0].score, 1.0);
        assert_eq!(candidates[0].tokens, 10);
    }
}
